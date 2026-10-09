"""Application d'aide au tri des radiographies pulmonaires (PROTOTYPE DE RECHERCHE).

Trois commandes (à lancer depuis le dossier `application_clinique`, noyau `memoire_cnn`) :

    python application_clinique.py                      # ouvre l'application dans le navigateur (adresse locale uniquement)
    python application_clinique.py analyser --image radio.jpeg          # analyse une image, sans navigateur
    python application_clinique.py calibrer --csv echantillon.csv --images dossier_images   # seuils calibrés localement

Options : --systeme v3:CNN_CBAM (défaut) | v3:CNN_classique | v4:CNN_CBAM | v4:CNN_classique ; --projet DOSSIER ; --sans-journal.

Tout se passe sur l'ordinateur : aucune image n'est envoyée sur internet. Le journal (journal_analyses.csv) ne contient ni image ni
nom de fichier : seulement une empreinte de l'image, le score, la zone et les alertes.

Ne constitue pas un dispositif médical. Ne remplace pas la lecture d'une radiographie par un médecin.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")   # aucune statistique d'usage envoyée

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

sys.path.insert(0, str(Path(__file__).resolve().parent))
import outils_clinique as oc  # noqa: E402

DOSSIER_APPLICATION = Path(__file__).resolve().parent
PROJET_PAR_DEFAUT = DOSSIER_APPLICATION.parent
FICHIER_PARAMETRES = "tri_parametres.json"
FICHIER_JOURNAL = "journal_analyses.csv"
MIN_RADIOS_CALIBRATION = 10     # en dessous, les seuils n'ont aucun sens statistique : refus
NOMS_ZONES = {0: "VERTE", 1: "ORANGE", 2: "ROUGE"}
COULEURS_ZONES = {0: ("#0f5132", "#d1e7dd"), 1: ("#7a4100", "#ffe5b4"), 2: ("#842029", "#f8d7da")}
SYMBOLES_ZONES = {0: "✔", 1: "◆", 2: "▲"}
TITRES_ZONES = {0: "Zone verte : lecture en file standard", 1: "Zone orange : incertain, lecture médicale sans priorité particulière",
                2: "Zone rouge : lecture prioritaire recommandée"}


class ErreurApplication(RuntimeError):
    """Erreur que l'utilisateur peut corriger (fichier manquant, paramètres absents, image illisible)."""


# --------------------------------------------------------------------------------------------------------------
# Paramètres du tri et seuils
# --------------------------------------------------------------------------------------------------------------
def cle_systeme(version_courte: str, nom: str) -> str:
    version = {"v3": oc.V3, "v4": oc.V4}[version_courte]
    return f"{version}|{nom}"


def lire_parametres(dossier: Path) -> dict:
    fichier = dossier / FICHIER_PARAMETRES
    if not fichier.is_file():
        raise ErreurApplication(f"{fichier} est introuvable. Lancer d'abord le notebook 15 : il écrit ce fichier dans ce dossier.")
    return oc.charger_json(fichier)


def fichier_seuils_locaux(dossier: Path, version_courte: str, nom: str) -> Path:
    """Un fichier de seuils locaux par système : calibrer v4 n'écrase pas la calibration de v3."""
    return Path(dossier) / f"seuils_locaux_{version_courte}_{nom}.json"


def choisir_seuils(parametres: dict, cle: str, dossier: Path, version_courte: str, nom: str):
    """Seuils locaux s'ils existent pour ce système, sinon seuils du développement (avec avertissement)."""
    if cle not in parametres["systemes"]:
        raise ErreurApplication(f"Le système {cle} n'est pas dans {FICHIER_PARAMETRES} (systèmes connus : {list(parametres['systemes'])}).")
    fichier = fichier_seuils_locaux(dossier, version_courte, nom)
    if fichier.is_file():
        locaux = oc.charger_json(fichier)
        if locaux.get("systeme") == cle:
            return locaux["seuils"], "locale", locaux
    return parametres["systemes"][cle]["seuils_developpement"], "développement", None


# --------------------------------------------------------------------------------------------------------------
# Analyse d'une radiographie
# --------------------------------------------------------------------------------------------------------------
def ouvrir_image(source) -> Image.Image:
    """Ouvre un chemin ou reçoit un objet PIL. Corrige l'orientation EXIF, refuse les fichiers illisibles."""
    if not isinstance(source, Image.Image) and not Path(source).is_file():
        raise ErreurApplication(f"Fichier introuvable : {source}")
    try:
        image = source if isinstance(source, Image.Image) else Image.open(source)
        image = ImageOps.exif_transpose(image)
        image.load()
    except (UnidentifiedImageError, OSError) as erreur:
        raise ErreurApplication(f"Fichier image illisible ({erreur}). Formats acceptés : JPEG, PNG.") from erreur
    if min(image.size) < 32:
        raise ErreurApplication("Image trop petite pour être une radiographie.")
    return image


def empreinte_image(image: Image.Image) -> str:
    return hashlib.sha256(image.convert("L").resize((64, 64)).tobytes()).hexdigest()[:16]


class Analyseur:
    def __init__(self, projet: Path, version_courte: str, nom: str, dossier_app: Path = DOSSIER_APPLICATION, journal: bool = True):
        self.projet, self.dossier_app, self.journal = Path(projet), Path(dossier_app), journal
        self.version_courte, self.nom = version_courte, nom
        self.version = {"v3": oc.V3, "v4": oc.V4}[version_courte]
        self.cle = cle_systeme(version_courte, nom)
        self.parametres = lire_parametres(self.dossier_app)
        self.seuils, self.source_seuils, self.info_locale = choisir_seuils(self.parametres, self.cle, self.dossier_app, version_courte, nom)
        self.systeme = self.parametres["systemes"][self.cle]
        self.alertes_parametres = self.parametres["alertes"]["parametres_images"]
        self.modeles = []
        for graine in oc.GRAINES:
            chemin = oc.dossier_modele(self.projet, self.version, nom, graine) / "meilleur.pt"
            if not chemin.is_file():
                raise ErreurApplication(f"Modèle introuvable : {chemin}. Le notebook {'12' if version_courte == 'v3' else '16'} est-il terminé ?")
            self.modeles.append(oc.charger_modele(chemin, self.version, nom, graine))
        self.segmenteur, self.sens_lateralite = None, 1.0
        if version_courte == "v4":
            termine = self.projet / "optimisation" / "poumons_v4" / "termine.json"
            if termine.is_file():
                self.sens_lateralite = float(oc.charger_json(termine).get("sens_lateralite", 1.0))
            self.segmenteur = oc.Segmenteur()

    # -- préparation et scores ------------------------------------------------------------------------------
    def preparer(self, image: Image.Image):
        """Entrée du réseau (uint8 224 × 224) et informations de localisation (v4)."""
        if self.version_courte == "v3":
            return oc.preparer_v3(image), None
        loc = oc.localiser_poumons(image, self.segmenteur, self.sens_lateralite)
        return oc.preparer_v4(image, loc["boite"]), loc

    def scores_seuls(self, image: Image.Image):
        """Scores des trois graines (sans cartes) et localisation éventuelle : utilisé par la calibration."""
        entree, loc = self.preparer(image)
        x = oc.vers_reseau(entree)
        import torch
        with torch.no_grad():
            scores = [float((lambda s: s[0, 1] - s[0, 0])(m(torch.from_numpy(x)))) for m in self.modeles]
        return np.array(scores), loc

    def analyser(self, image_source) -> dict:
        debut = time.perf_counter()
        image = ouvrir_image(image_source)
        stats = oc.statistiques_alerte(image)
        alertes = oc.alertes_image(stats, self.alertes_parametres)
        entree, loc = self.preparer(image)
        if loc is not None and not loc["valide"]:
            alertes.append("poumons")
        x = oc.vers_reseau(entree)
        resultats = [oc.analyser_modele(self.nom, m, x) for m in self.modeles]
        scores = np.array([r[0] for r in resultats])
        score = float(scores.mean())
        dispersion = float(oc.ecart_type_graines(scores))
        if dispersion > self.systeme["seuil_ecart_graines"]:
            alertes.append("graines")
        zone_brute = oc.zone_de(score, self.seuils["seuil_bas"], self.seuils["seuil_haut"])
        zone = oc.zone_finale(zone_brute, len(alertes) > 0)
        decision = np.mean([r[1] for r in resultats], axis=0)
        attention = np.mean([r[2] for r in resultats], axis=0) if resultats[0][2] is not None else None
        zone_poumons = oc.zone_dans_entree_v4(loc["zone"], loc["boite"]) if (loc is not None and loc["valide"]) else None
        resultat = {"score": score, "scores_graines": scores.tolist(), "ecart_type_graines": dispersion, "zone": zone, "zone_brute": zone_brute,
                    "alertes": alertes, "statistiques_image": stats, "entree": entree, "decision": decision, "attention": attention,
                    "zone_poumons": zone_poumons, "duree_secondes": time.perf_counter() - debut}
        if self.journal:
            self.ecrire_journal(image, resultat)
        return resultat

    def ecrire_journal(self, image, resultat):
        fichier = self.dossier_app / FICHIER_JOURNAL
        nouveau = not fichier.exists()
        with open(fichier, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            if nouveau:
                w.writerow(["horodatage", "empreinte_image", "systeme", "source_seuils", "seuil_bas", "seuil_haut", "score",
                            "ecart_type_graines", "zone_brute", "zone_finale", "alertes"])
            w.writerow([dt.datetime.now().isoformat(timespec="seconds"), empreinte_image(image), self.cle, self.source_seuils,
                        f"{self.seuils['seuil_bas']:.6g}", f"{self.seuils['seuil_haut']:.6g}", f"{resultat['score']:.6g}",
                        f"{resultat['ecart_type_graines']:.6g}", NOMS_ZONES[resultat["zone_brute"]], NOMS_ZONES[resultat["zone"]],
                        ";".join(resultat["alertes"])])


# --------------------------------------------------------------------------------------------------------------
# Affichage : cartes et fiche de résultat
# --------------------------------------------------------------------------------------------------------------
def _superposer(entree_u8, carte, nom_palette, alpha, contour=None, taille=448):
    from matplotlib import colormaps
    fond = np.stack([entree_u8] * 3, axis=-1).astype(np.float64) / 255.0
    couleur = colormaps[nom_palette](np.clip(carte, 0, 1))[..., :3]
    melange = (1 - alpha) * fond + alpha * couleur
    image = Image.fromarray((np.clip(melange, 0, 1) * 255).astype(np.uint8)).resize((taille, taille), Image.Resampling.BICUBIC)
    if contour is not None and contour.any():
        grand = np.asarray(Image.fromarray(contour.astype(np.uint8) * 255).resize((taille, taille), Image.Resampling.NEAREST)) > 127
        from scipy import ndimage
        bord = grand & ~ndimage.binary_erosion(grand, iterations=2)
        tableau = np.asarray(image).copy()
        tableau[bord] = (255, 210, 63)
        image = Image.fromarray(tableau)
    return image


def images_explication(resultat, avec_attention: bool):
    carte_decision = _superposer(resultat["entree"], resultat["decision"], "jet", 0.45, resultat["zone_poumons"])
    if avec_attention and resultat["attention"] is not None:
        a = resultat["attention"]
        etendue = float(a.max() - a.min())
        normalisee = (a - a.min()) / etendue if etendue > 1e-9 else np.zeros_like(a)
        carte_attention = _superposer(resultat["entree"], normalisee, "viridis", 0.55, resultat["zone_poumons"])
    else:
        carte_attention = Image.fromarray(np.stack([resultat["entree"]] * 3, axis=-1)).resize((448, 448), Image.Resampling.BICUBIC)
    return carte_decision, carte_attention


def _barre_score(score, bas, haut):
    marge = max(2.0, 1.5 * (haut - bas))
    debut, fin = bas - marge, haut + marge
    position = 100 * min(1.0, max(0.0, (score - debut) / (fin - debut)))
    p_bas, p_haut = 100 * (bas - debut) / (fin - debut), 100 * (haut - debut) / (fin - debut)
    return (f'<div style="position:relative;height:26px;border-radius:6px;overflow:hidden;background:linear-gradient(to right,'
            f'#d1e7dd 0,#d1e7dd {p_bas:.1f}%,#ffe5b4 {p_bas:.1f}%,#ffe5b4 {p_haut:.1f}%,#f8d7da {p_haut:.1f}%,#f8d7da 100%);border:1px solid #888">'
            f'<div style="position:absolute;left:{position:.1f}%;top:0;width:4px;height:100%;background:#111;margin-left:-2px"></div></div>'
            f'<div style="font-size:12px;color:#444;margin-top:2px">Trait noir : score de cette radio. Zone verte : score &lt; {bas:+.2f} · '
            f'orange : entre {bas:+.2f} et {haut:+.2f} · rouge : score &gt; {haut:+.2f}</div>')


def fiche_html(analyseur: Analyseur, resultat: dict) -> str:
    zone = resultat["zone"]
    encre, fond = COULEURS_ZONES[zone]
    seuils, source = analyseur.seuils, analyseur.source_seuils
    lignes_alertes = "".join(f"<li>{oc.LIBELLES_ALERTES[a]}</li>" for a in resultat["alertes"])
    bloc_alertes = (f'<div style="margin-top:10px;padding:8px 12px;border-left:5px solid #7a4100;background:#fff4e0;color:#3b2100">'
                    f'<b>Alertes de sécurité ({len(resultat["alertes"])})</b><ul style="margin:4px 0 0 18px">{lignes_alertes}</ul>'
                    + ("<i>Règle de sécurité : une alerte empêche la zone verte ; l'image passe en zone orange.</i>"
                       if resultat["zone_brute"] == 0 else "") + "</div>") if resultat["alertes"] else \
        '<div style="margin-top:10px;color:#333">Aucune alerte de sécurité.</div>'
    cibles = seuils
    if zone == 0:
        explication = (f"Le score est sous le seuil bas. Au réglage utilisé, au plus {100 * (1 - cibles['cible_sensibilite_verte']):.0f} % des radios "
                       "PNEUMONIA de l'échantillon de réglage tombaient dans cette zone. <b>Une zone verte n'exclut pas une pneumonie.</b>")
    elif zone == 1:
        explication = "Le score est entre les deux seuils, ou l'image sort du cadre validé. Aucune conclusion automatique."
    else:
        explication = (f"Le score est au-dessus du seuil haut. Au réglage utilisé, au plus {100 * (1 - cibles['cible_specificite_rouge']):.0f} % des radios "
                       "NORMAL de l'échantillon de réglage tombaient dans cette zone. <b>Ce n'est pas un diagnostic.</b>")
    if source == "locale":
        bandeau_seuils = (f'<div style="color:#333;font-size:13px">Seuils calibrés localement ({seuils["n_normal"]} radios NORMAL, '
                          f'{seuils["n_pneumonie"]} radios PNEUMONIA).</div>')
        if seuils["n_pneumonie"] < 100 or seuils["n_normal"] < 100:
            bandeau_seuils += ('<div style="margin-top:6px;padding:6px 10px;background:#fff4e0;color:#3b2100;border-radius:4px;font-size:13px">'
                               'Échantillon de calibration petit (moins de 100 radios dans une classe) : seuils instables.</div>')
    else:
        bandeau_seuils = ('<div style="margin-top:8px;padding:6px 10px;background:#f8d7da;color:#58151c;border-radius:4px;font-size:13px">'
                          '<b>Seuils du développement, non calibrés localement.</b> Le décalage entre centres d\'imagerie fausse ces seuils '
                          '(voir notebooks 14 et 15) : ne pas s\'y fier tant qu\'une calibration locale n\'a pas été faite.</div>')
    if seuils.get("zones_chevauchantes"):
        bandeau_seuils += ('<div style="margin-top:6px;padding:6px 10px;background:#f8d7da;color:#58151c;border-radius:4px;font-size:13px">'
                           '<b>Les deux seuils se croisent</b> : le modèle sépare mal les deux classes sur l\'échantillon de réglage, il n\'y a plus de zone '
                           'orange. Ne pas utiliser ce réglage pour trier.</div>')
    sc = resultat["scores_graines"]
    return f"""
<div style="font-family:system-ui,Segoe UI,Arial,sans-serif;max-width:820px">
  <div style="padding:8px 12px;border:2px solid #842029;border-radius:6px;background:#fff;color:#58151c;font-size:13px;margin-bottom:10px">
    <b>Prototype de recherche.</b> {oc.AVERTISSEMENT.split('. ', 1)[1]}
  </div>
  <div style="padding:14px 16px;border-radius:8px;background:{fond};color:{encre};border:2px solid {encre}">
    <div style="font-size:22px;font-weight:700">{SYMBOLES_ZONES[zone]} {TITRES_ZONES[zone]}</div>
    <div style="margin-top:6px;font-size:15px">{explication}</div>
  </div>
  <div style="margin-top:12px">{_barre_score(resultat['score'], seuils['seuil_bas'], seuils['seuil_haut'])}</div>
  <div style="margin-top:6px;font-size:14px">Score moyen des 3 modèles : <b>{resultat['score']:+.2f}</b>
     (chaque modèle : {', '.join(f'{v:+.2f}' for v in sc)} ; écart-type {resultat['ecart_type_graines']:.2f}).
     Un score plus élevé signifie plus de signes évocateurs de pneumonie pour ce modèle.</div>
  {bloc_alertes}
  {bandeau_seuils}
  <div style="margin-top:8px;font-size:12px;color:#555">Système : {analyseur.cle} · seuils : {source} · durée {resultat['duree_secondes']:.1f} s</div>
</div>"""


def details_json(analyseur: Analyseur, resultat: dict) -> dict:
    return {"systeme": analyseur.cle, "source_seuils": analyseur.source_seuils, "seuils": {k: analyseur.seuils[k] for k in ("seuil_bas", "seuil_haut")},
            "score": round(resultat["score"], 4), "scores_graines": [round(v, 4) for v in resultat["scores_graines"]],
            "ecart_type_graines": round(resultat["ecart_type_graines"], 4), "zone_avant_regle_de_securite": NOMS_ZONES[resultat["zone_brute"]],
            "zone_finale": NOMS_ZONES[resultat["zone"]], "alertes": resultat["alertes"],
            "statistiques_image": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in resultat["statistiques_image"].items()}}


# --------------------------------------------------------------------------------------------------------------
# Interface (Gradio)
# --------------------------------------------------------------------------------------------------------------
def message_erreur(texte: str) -> str:
    return (f'<div style="padding:12px;border:2px solid #842029;border-radius:6px;background:#f8d7da;color:#58151c;'
            f'font-family:system-ui,Arial,sans-serif;max-width:820px"><b>Analyse impossible.</b> {texte}</div>')


def construire_interface(analyseur: Analyseur):
    import gradio as gr
    avec_attention = analyseur.nom == "CNN_CBAM"

    def analyser(image):
        if image is None:
            return message_erreur("Déposer d'abord une radiographie."), None, None, None
        try:
            resultat = analyseur.analyser(image)
        except ErreurApplication as e:
            return message_erreur(str(e)), None, None, None
        carte_decision, carte_attention = images_explication(resultat, avec_attention)
        return fiche_html(analyseur, resultat), carte_decision, carte_attention, details_json(analyseur, resultat)

    with gr.Blocks(title="Aide au tri des radiographies pulmonaires (prototype de recherche)", analytics_enabled=False) as demo:
        gr.Markdown("# Aide au tri des radiographies pulmonaires\n**Prototype de recherche** : " + oc.AVERTISSEMENT.split('. ', 1)[1] +
                    "\n\nLes images restent sur cet ordinateur.")
        with gr.Row():
            with gr.Column(scale=1):
                entree = gr.Image(type="pil", label="Radiographie thoracique de face (JPEG ou PNG)", sources=["upload"])
                bouton = gr.Button("Analyser", variant="primary")
            with gr.Column(scale=1):
                fiche = gr.HTML(label="Résultat")
        with gr.Row():
            carte_a = gr.Image(label="Où le modèle a regardé (Grad-CAM). Contour jaune : poumons (v4).", interactive=False)
            carte_b = gr.Image(label="Attention spatiale du CBAM (si disponible), sinon image préparée", interactive=False)
        gr.Markdown("*Ces cartes indiquent les zones qui ont le plus pesé dans le score ; elles ne prouvent pas la présence d'une lésion. "
                    "Dans les notebooks 11 à 13, elles tombaient souvent hors des poumons.*")
        with gr.Accordion("Détails techniques", open=False):
            details = gr.JSON()
        bouton.click(analyser, inputs=entree, outputs=[fiche, carte_a, carte_b, details])
    return demo


# --------------------------------------------------------------------------------------------------------------
# Calibration locale
# --------------------------------------------------------------------------------------------------------------
def calibrer(analyseur: Analyseur, csv_chemin: Path, dossier_images: Path, sortie: Path, graine: int = 20260929):
    import pandas as pd
    tableau = pd.read_csv(csv_chemin)
    if not {"fichier", "classe"}.issubset(tableau.columns):
        raise ErreurApplication("Le CSV doit avoir deux colonnes : fichier (chemin de l'image dans le dossier) et classe (NORMAL ou PNEUMONIA).")
    if not set(tableau["classe"]).issubset(set(oc.CLASSES)):
        raise ErreurApplication("Les valeurs de la colonne classe doivent être NORMAL ou PNEUMONIA.")
    if tableau["fichier"].duplicated().any():
        raise ErreurApplication("Le CSV contient plusieurs fois le même fichier : chaque radio ne doit apparaître qu'une fois.")
    effectifs = tableau["classe"].value_counts()
    for classe in oc.CLASSES:
        if effectifs.get(classe, 0) < MIN_RADIOS_CALIBRATION:
            raise ErreurApplication(f"Au moins {MIN_RADIOS_CALIBRATION} radios {classe} sont nécessaires pour calibrer (trouvé : "
                                    f"{int(effectifs.get(classe, 0))}). Pour un résultat utilisable : 100 à 200 par classe.")
    scores = []
    for i, ligne in enumerate(tableau.to_dict("records")):
        image = ouvrir_image(Path(dossier_images) / ligne["fichier"])
        s, loc = analyseur.scores_seuls(image)
        scores.append(float(s.mean()))
        if (i + 1) % 20 == 0 or i + 1 == len(tableau):
            print(f"  {i + 1}/{len(tableau)} radios traitées", flush=True)
    tableau["score"] = scores
    normal = tableau.loc[tableau["classe"] == "NORMAL", "score"].to_numpy()
    pneumonie = tableau.loc[tableau["classe"] == "PNEUMONIA", "score"].to_numpy()
    seuils = oc.seuils_depuis_calibration(normal, pneumonie)
    rng = np.random.default_rng(graine)
    bas, haut = [], []
    for _ in range(500):
        b = oc.seuils_depuis_calibration(rng.choice(normal, len(normal)), rng.choice(pneumonie, len(pneumonie)))
        bas.append(b["seuil_bas"]); haut.append(b["seuil_haut"])
    avertissements = []
    if len(pneumonie) < 100:
        avertissements.append(f"seulement {len(pneumonie)} radios PNEUMONIA : le seuil bas est très instable (au moins 100 recommandées, mieux 200).")
    if len(normal) < 100:
        avertissements.append(f"seulement {len(normal)} radios NORMAL : le seuil haut est instable (au moins 100 recommandées).")
    if seuils["zones_chevauchantes"]:
        avertissements.append("les deux seuils se croisent : le modèle ne sépare pas assez bien les deux classes sur cet échantillon.")
    if not avertissements and (max(bas) - min(bas) > 4 or max(haut) - min(haut) > 4):
        avertissements.append("les seuils varient beaucoup d'un tirage à l'autre : la calibration reste fragile.")
    contenu = {"systeme": analyseur.cle, "source": "locale", "date": dt.datetime.now().isoformat(timespec="seconds"), "seuils": seuils,
               "intervalle95_seuil_bas": [float(np.quantile(bas, 0.025)), float(np.quantile(bas, 0.975))],
               "intervalle95_seuil_haut": [float(np.quantile(haut, 0.025)), float(np.quantile(haut, 0.975))],
               "avertissements": avertissements, "csv": str(csv_chemin)}
    sortie.write_text(json.dumps(contenu, indent=2, ensure_ascii=False), encoding="utf-8")
    return contenu


# --------------------------------------------------------------------------------------------------------------
# Ligne de commande
# --------------------------------------------------------------------------------------------------------------
def lire_arguments(argv=None):
    p = argparse.ArgumentParser(description="Application d'aide au tri des radiographies pulmonaires (prototype de recherche).")
    p.add_argument("commande", nargs="?", default="lancer", choices=["lancer", "analyser", "calibrer"])
    p.add_argument("--systeme", default="v3:CNN_CBAM", help="v3:CNN_CBAM (défaut), v3:CNN_classique, v4:CNN_CBAM, v4:CNN_classique")
    p.add_argument("--projet", default=str(PROJET_PAR_DEFAUT), help="dossier du projet (celui qui contient « optimisation »)")
    p.add_argument("--image", help="commande analyser : chemin de l'image")
    p.add_argument("--csv", help="commande calibrer : CSV avec les colonnes fichier et classe")
    p.add_argument("--images", help="commande calibrer : dossier des images")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--sans-journal", action="store_true", help="ne pas écrire journal_analyses.csv")
    p.add_argument("--sans-navigateur", action="store_true", help="ne pas ouvrir le navigateur automatiquement")
    return p.parse_args(argv)


def principal(argv=None):
    for flux in (sys.stdout, sys.stderr):
        if hasattr(flux, "reconfigure"):
            flux.reconfigure(encoding="utf-8", errors="replace")   # accents lisibles dans la console Windows
    args = lire_arguments(argv)
    version_courte, _, nom = args.systeme.partition(":")
    if version_courte not in ("v3", "v4") or nom not in oc.CONFIGURATIONS:
        raise SystemExit("--systeme doit être v3 ou v4, suivi de « : » et de CNN_classique ou CNN_CBAM.")
    try:
        analyseur = Analyseur(Path(args.projet), version_courte, nom, journal=not args.sans_journal)
        if args.commande == "analyser":
            if not args.image:
                raise ErreurApplication("Indiquer --image chemin.")
            resultat = analyseur.analyser(args.image)
            print(json.dumps(details_json(analyseur, resultat), indent=2, ensure_ascii=False))
        elif args.commande == "calibrer":
            if not (args.csv and args.images):
                raise ErreurApplication("Indiquer --csv et --images.")
            sortie = fichier_seuils_locaux(DOSSIER_APPLICATION, version_courte, nom)
            contenu = calibrer(analyseur, Path(args.csv), Path(args.images), sortie)
            print(json.dumps(contenu, indent=2, ensure_ascii=False))
            print(f"\nÉcrit : {sortie}. L'application les utilisera au prochain lancement (même système : {analyseur.cle}).")
        else:
            demo = construire_interface(analyseur)
            print(f"Système : {analyseur.cle} | seuils : {analyseur.source_seuils}")
            demo.launch(server_name="127.0.0.1", server_port=args.port, inbrowser=not args.sans_navigateur, share=False, show_error=True)
    except ErreurApplication as e:
        raise SystemExit(f"Erreur : {e}")


if __name__ == "__main__":
    principal()

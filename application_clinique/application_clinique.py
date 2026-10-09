"""Application d'aide au tri des radiographies pulmonaires (PROTOTYPE DE RECHERCHE), version 2.

Trois commandes (à lancer depuis le dossier `application_clinique`, noyau `memoire_cnn`) :

    python application_clinique.py                      # ouvre l'application dans le navigateur (adresse locale uniquement)
    python application_clinique.py analyser --image radio.jpeg [--enregistrer --reference CODE]   # une image, sans navigateur
    python application_clinique.py calibrer --csv echantillon.csv --images dossier_images   # seuils calibrés localement

Options : --systeme v4:CNN_CBAM (défaut) | v4:CNN_classique | v3:CNN_CBAM | v3:CNN_classique ; --projet DOSSIER ;
--dossier-analyses DOSSIER ; --sans-journal ; --sans-navigateur ; --port 7860.

L'application n'envoie aucune image sur internet. Le journal (journal_analyses.csv) ne contient ni image ni nom de fichier : seulement
une empreinte de l'image, le score, la zone et les alertes. Les analyses que l'utilisateur choisit d'enregistrer (bouton « Enregistrer »)
sont gardées, avec la radio, dans le dossier `analyses` (onglet Historique).

Ne constitue pas un dispositif médical. Ne remplace pas la lecture d'une radiographie par un médecin.
"""
# Pas de « from __future__ import annotations » : Gradio lit l'annotation gr.SelectData de selectionner() pour lui passer
# la ligne cliquée dans l'historique ; une annotation laissée en chaîne de caractères ne serait pas reconnue.
import argparse
import csv
import datetime as dt
import hashlib
import html
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
import rapports_clinique as rc  # noqa: E402

DOSSIER_APPLICATION = Path(__file__).resolve().parent
PROJET_PAR_DEFAUT = DOSSIER_APPLICATION.parent
FICHIER_PARAMETRES = "tri_parametres.json"
FICHIER_JOURNAL = "journal_analyses.csv"
MIN_RADIOS_CALIBRATION = 10     # en dessous, les seuils n'ont aucun sens statistique : refus
SYSTEME_PAR_DEFAUT = "v4:CNN_CBAM"


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
# Analyse d'une radiographie (inchangée depuis la version 1 : mêmes règles que le notebook 15)
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
        self.nom_lisible = rc.nom_systeme_lisible(version_courte, nom)
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
        horodatage = dt.datetime.now().isoformat(timespec="seconds")
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
        poumons = ({"valide": bool(loc["valide"]), "raisons": list(loc["raisons"]), "aire": float(loc["aire_poumons"])}
                   if loc is not None else None)
        resultat = {"score": score, "scores_graines": scores.tolist(), "ecart_type_graines": dispersion, "zone": zone, "zone_brute": zone_brute,
                    "alertes": alertes, "statistiques_image": stats, "entree": entree, "decision": decision, "attention": attention,
                    "zone_poumons": zone_poumons, "poumons": poumons, "image": image, "horodatage": horodatage,
                    "duree_secondes": time.perf_counter() - debut, "journal_erreur": None}
        if self.journal:
            try:
                self.ecrire_journal(image, resultat)
            except OSError as erreur:     # fichier ouvert dans Excel, par exemple : l'analyse reste valable
                resultat["journal_erreur"] = str(erreur)
        return resultat

    def ecrire_journal(self, image, resultat):
        fichier = self.dossier_app / FICHIER_JOURNAL
        nouveau = not fichier.exists()
        with open(fichier, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            if nouveau:
                w.writerow(["horodatage", "empreinte_image", "systeme", "source_seuils", "seuil_bas", "seuil_haut", "score",
                            "ecart_type_graines", "zone_brute", "zone_finale", "alertes"])
            w.writerow([resultat["horodatage"], empreinte_image(image), self.cle, self.source_seuils,
                        f"{self.seuils['seuil_bas']:.6g}", f"{self.seuils['seuil_haut']:.6g}", f"{resultat['score']:.6g}",
                        f"{resultat['ecart_type_graines']:.6g}", rc.NOMS_ZONES[resultat["zone_brute"]], rc.NOMS_ZONES[resultat["zone"]],
                        ";".join(resultat["alertes"])])


def contenu_analyse(analyseur: Analyseur, resultat: dict, reference: str = "") -> dict:
    """Tout ce qu'il faut pour réafficher l'analyse et écrire son rapport, sans les tableaux d'images (sérialisable en JSON)."""
    seuils = {k: analyseur.seuils[k] for k in ("seuil_bas", "seuil_haut", "zones_chevauchantes", "n_normal", "n_pneumonie",
                                               "cible_sensibilite_verte", "cible_specificite_rouge") if k in analyseur.seuils}
    stats = {k: (float(v) if isinstance(v, (float, np.floating)) else int(v) if isinstance(v, (int, np.integer)) else v)
             for k, v in resultat["statistiques_image"].items()}
    return {"format": rc.FORMAT_ANALYSE, "version_application": rc.VERSION_APPLICATION,
            "horodatage_analyse": resultat["horodatage"], "reference": rc.nettoyer_reference(reference),
            "systeme": analyseur.cle, "version_courte": analyseur.version_courte, "modele": analyseur.nom,
            "systeme_lisible": analyseur.nom_lisible, "source_seuils": analyseur.source_seuils, "seuils": seuils,
            "score": float(resultat["score"]), "scores_graines": [float(v) for v in resultat["scores_graines"]],
            "ecart_type_graines": float(resultat["ecart_type_graines"]),
            "zone_avant_regle": rc.NOMS_ZONES[resultat["zone_brute"]], "zone_finale": rc.NOMS_ZONES[resultat["zone"]],
            "alertes": list(resultat["alertes"]), "alertes_libelles": [oc.LIBELLES_ALERTES[a] for a in resultat["alertes"]],
            "statistiques_image": stats, "empreinte_image": empreinte_image(resultat["image"]), "poumons": resultat["poumons"],
            "attention": rc.resume_attention(resultat["attention"], resultat["zone_poumons"]),
            "duree_secondes": float(resultat["duree_secondes"])}


def details_json(contenu: dict) -> dict:
    """Panneau « Détails techniques » et sortie de la commande analyser (mêmes premières clés qu'en version 1)."""
    return {"systeme": contenu["systeme"], "source_seuils": contenu["source_seuils"],
            "seuils": {k: contenu["seuils"][k] for k in ("seuil_bas", "seuil_haut")},
            "score": round(contenu["score"], 4), "scores_graines": [round(v, 4) for v in contenu["scores_graines"]],
            "ecart_type_graines": round(contenu["ecart_type_graines"], 4), "zone_avant_regle_de_securite": contenu["zone_avant_regle"],
            "zone_finale": contenu["zone_finale"], "alertes": contenu["alertes"],
            "statistiques_image": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in contenu["statistiques_image"].items()},
            "horodatage_analyse": contenu["horodatage_analyse"], "reference": contenu["reference"],
            "empreinte_image": contenu["empreinte_image"], "poumons": contenu["poumons"],
            "attention_spatiale": ({k: (round(v, 4) if v is not None else None) for k, v in contenu["attention"].items()}
                                   if contenu["attention"] else None),
            "duree_secondes": round(contenu["duree_secondes"], 2)}


def enregistrer_resultat(dossier_analyses: Path, contenu: dict, resultat: dict, avec_attention: bool) -> Path:
    carte_decision, carte_attention = rc.images_explication(resultat["entree"], resultat["decision"], resultat["attention"],
                                                            resultat["zone_poumons"], avec_attention)
    return rc.enregistrer_analyse(dossier_analyses, contenu, resultat["image"], resultat["entree"], carte_decision, carte_attention)


# --------------------------------------------------------------------------------------------------------------
# Interface (Gradio) : onglets Analyse et Historique
# --------------------------------------------------------------------------------------------------------------
COLONNES_HISTORIQUE = ["Date", "Heure", "Référence", "Zone", "Score", "Alertes", "Système", "Seuils", "Dossier"]


def construire_interface(analyseur: Analyseur, dossier_analyses: Path):
    import gradio as gr
    import pandas as pd

    avec_attention = analyseur.nom == "CNN_CBAM"
    titre_decision = "Où le modèle a regardé (Grad-CAM)" + (" · contour jaune : poumons" if analyseur.version_courte == "v4" else "")
    titre_attention = "Attention spatiale du CBAM (échelle réelle, 0 à 1)" if avec_attention else "Image préparée, telle que vue par le réseau"
    origine = "calibrés localement" if analyseur.source_seuils == "locale" else "du développement, non calibrés localement"
    entete = ("# Aide au tri des radiographies pulmonaires\n"
              "**Prototype de recherche** : cet outil aide à ordonner la lecture des radiographies ; il ne pose aucun diagnostic et ne "
              "remplace pas la lecture par un médecin.\n\n"
              f"**Système :** {analyseur.nom_lisible} · **seuils :** {origine}  \n"
              "L'application n'envoie aucune image sur internet. Seules les analyses que vous enregistrez sont gardées, avec la radio, "
              "dans le dossier `analyses`.")

    def tableau_vide():
        return pd.DataFrame(columns=COLONNES_HISTORIQUE)

    def _enregistrer(etat, reference):
        if not etat:
            return etat, rc.message_html("Analyser d'abord une radiographie, puis l'enregistrer.", "erreur")
        if etat.get("dossier"):
            return etat, rc.message_html(f"Cette analyse est déjà enregistrée : dossier <b>{html.escape(etat['dossier'])}</b>.", "info")
        contenu = {**etat["contenu"], "reference": rc.nettoyer_reference(reference)}
        try:
            dossier = rc.enregistrer_analyse(dossier_analyses, contenu, etat["radio"], etat["entree"], etat["decision"], etat["attention"])
        except OSError as erreur:
            return etat, rc.message_html(f"Enregistrement impossible : {html.escape(str(erreur))}", "erreur")
        etat = {**etat, "dossier": dossier.name, "contenu": contenu}
        return etat, rc.message_html(f"Analyse enregistrée dans <b>analyses/{html.escape(dossier.name)}</b> : rapport PDF, images et "
                                     "données. Elle apparaît dans l'onglet Historique.", "succes")

    def analyser(image, reference, automatique):
        if image is None:
            return rc.message_html("Déposer d'abord une radiographie.", "erreur"), None, None, "", None, None, ""
        try:
            resultat = analyseur.analyser(image)
        except ErreurApplication as erreur:
            return (rc.message_html(f"<b>Analyse impossible.</b> {html.escape(str(erreur))}", "erreur"), None, None, "", None, None, "")
        contenu = contenu_analyse(analyseur, resultat, reference)
        carte_decision, carte_attention = rc.images_explication(resultat["entree"], resultat["decision"], resultat["attention"],
                                                                resultat["zone_poumons"], avec_attention)
        etat = {"contenu": contenu, "radio": resultat["image"], "entree": resultat["entree"], "decision": carte_decision,
                "attention": carte_attention, "dossier": None}
        statut = ""
        if automatique:
            etat, statut = _enregistrer(etat, reference)
        if resultat["journal_erreur"]:
            statut += rc.message_html("Journal anonyme non mis à jour : journal_analyses.csv est peut-être ouvert dans Excel. "
                                      "L'analyse reste valable.", "info")
        return (rc.fiche_html(etat["contenu"]), carte_decision, carte_attention, rc.legende_html(etat["contenu"]),
                details_json(etat["contenu"]), etat, statut)

    def enregistrer(etat, reference):
        etat, statut = _enregistrer(etat, reference)
        fiche = rc.fiche_html(etat["contenu"]) if etat else gr.skip()
        return etat, statut, fiche

    def nouvelle_image():
        # Même ordre que sorties_analyse : fiche, carte_decision, carte_attention, legende, details, etat, statut
        return "", None, None, "", None, None, ""

    def actualiser(zone, recherche):
        analyses = rc.lister_analyses(dossier_analyses)
        choisies = rc.filtrer(analyses, zone, recherche)
        tableau = pd.DataFrame([rc.ligne_tableau(c) for c in choisies], columns=COLONNES_HISTORIQUE) if choisies else tableau_vide()
        compte = rc.resume_zones(analyses)
        texte = (f"<b>{len(analyses)}</b> analyse(s) enregistrée(s) : {compte['VERTE']} en zone verte, {compte['ORANGE']} en orange, "
                 f"{compte['ROUGE']} en rouge")
        texte += f" · {len(choisies)} affichée(s) avec ce filtre." if len(choisies) != len(analyses) else "."
        journal = rc.resume_journal(analyseur.dossier_app / FICHIER_JOURNAL)
        if journal:
            texte += (f"<br>Journal anonyme (toutes les analyses, enregistrées ou non) : {journal['total']} analyse(s), dont "
                      f"{journal['VERTE']} vertes, {journal['ORANGE']} orange et {journal['ROUGE']} rouges.")
        texte += f"<br>Dossier des analyses : <code>{html.escape(str(dossier_analyses))}</code>"
        return tableau, rc.message_html(texte, "info")

    def vider_selection(message=""):
        return "", None, None, None, "", None, None, message, gr.update(visible=False)

    def selectionner(evt: gr.SelectData):
        ligne = evt.row_value
        if not ligne:
            return vider_selection()
        nom = str(ligne[-1])
        try:
            analyse = rc.charger_analyse(dossier_analyses, nom)
        except (rc.ErreurHistorique, OSError, ValueError) as erreur:
            return vider_selection(rc.message_html(html.escape(str(erreur)), "erreur"))
        contenu = analyse["contenu"]
        return (rc.fiche_html(contenu), analyse["radio"], analyse["decision"], analyse["attention"], rc.legende_html(contenu),
                analyse["rapport"], nom, rc.message_html(f"Analyse ouverte : <b>{html.escape(nom)}</b>.", "info"), gr.update(visible=True))

    def exporter():
        try:
            chemin = rc.exporter_historique(dossier_analyses)
        except OSError as erreur:
            return gr.update(value=None, visible=False), rc.message_html(f"Export impossible : {html.escape(str(erreur))}", "erreur")
        return gr.update(value=str(chemin), visible=True), rc.message_html(f"Historique exporté : <b>{html.escape(chemin.name)}</b> (CSV pour Excel, séparateur « ; »).",
                                            "succes")

    def supprimer(nom, confirme, zone, recherche):
        if not nom:
            tableau, resume = actualiser(zone, recherche)
            return (tableau, resume, *vider_selection(rc.message_html("Choisir d'abord une analyse dans le tableau.", "erreur")), False)
        if not confirme:
            return (gr.skip(), gr.skip(), *(gr.skip(),) * 6, nom,
                    rc.message_html("Cocher d'abord la case de confirmation : la suppression est définitive.", "erreur"), gr.skip(), False)
        try:
            rc.supprimer_analyse(dossier_analyses, nom)
        except rc.ErreurHistorique as erreur:
            return (gr.skip(), gr.skip(), *(gr.skip(),) * 6, nom, rc.message_html(html.escape(str(erreur)), "erreur"), gr.skip(), False)
        tableau, resume = actualiser(zone, recherche)
        return (tableau, resume, *vider_selection(rc.message_html(f"Analyse <b>{html.escape(nom)}</b> supprimée.", "succes")), False)

    with gr.Blocks(title="Aide au tri des radiographies pulmonaires (prototype de recherche)", analytics_enabled=False) as demo:
        gr.Markdown(entete)
        etat = gr.State(None)
        with gr.Tabs():
            with gr.Tab("Analyse"):
                with gr.Row():
                    with gr.Column(scale=1):
                        entree = gr.Image(type="pil", label="Radiographie thoracique de face (JPEG ou PNG)", sources=["upload"])
                        reference = gr.Textbox(label="Référence de l'examen (facultatif)", max_lines=1,
                                               placeholder="Numéro d'examen ou code. Jamais le nom du patient.")
                        automatique = gr.Checkbox(value=False, label="Enregistrer automatiquement chaque analyse")
                        bouton = gr.Button("Analyser", variant="primary")
                    with gr.Column(scale=1):
                        fiche = gr.HTML(apply_default_css=False)
                        bouton_enregistrer = gr.Button("Enregistrer l'analyse (rapport PDF, images et données)", variant="secondary")
                        statut = gr.HTML(apply_default_css=False)
                with gr.Row():
                    carte_decision = gr.Image(label=titre_decision, interactive=False)
                    carte_attention = gr.Image(label=titre_attention, interactive=False)
                legende = gr.HTML(apply_default_css=False)
                with gr.Accordion("Détails techniques", open=False):
                    details = gr.JSON()
            with gr.Tab("Historique") as onglet_historique:
                with gr.Row():
                    filtre_zone = gr.Dropdown(["Toutes", "Verte", "Orange", "Rouge"], value="Toutes", label="Zone", scale=1)
                    recherche = gr.Textbox(label="Rechercher une référence", max_lines=1, scale=2)
                    bouton_actualiser = gr.Button("Actualiser", scale=1)
                resume = gr.HTML(apply_default_css=False)
                tableau = gr.Dataframe(value=tableau_vide(), headers=COLONNES_HISTORIQUE, interactive=False, wrap=True, max_height=380,
                                       column_widths=["10%", "8%", "16%", "8%", "7%", "7%", "14%", "12%", "18%"],
                                       label="Analyses enregistrées (cliquer sur une ligne pour la rouvrir)")
                selection = gr.State(None)
                statut_historique = gr.HTML(apply_default_css=False)
                with gr.Column(visible=False) as panneau_selection:
                    with gr.Row():
                        with gr.Column(scale=1):
                            radio_historique = gr.Image(label="Radiographie enregistrée", interactive=False)
                        with gr.Column(scale=1):
                            fiche_historique = gr.HTML(apply_default_css=False)
                            rapport_historique = gr.File(label="Rapport PDF de cette analyse", interactive=False, height=90)
                    with gr.Row():
                        decision_historique = gr.Image(label="Grad-CAM enregistrée", interactive=False)
                        attention_historique = gr.Image(label="Attention spatiale ou image préparée", interactive=False)
                    legende_historique = gr.HTML(apply_default_css=False)
                with gr.Row():
                    bouton_exporter = gr.Button("Exporter l'historique (CSV pour Excel)")
                    fichier_export = gr.File(label="Export de l'historique", interactive=False, height=90, visible=False)
                with gr.Accordion("Supprimer l'analyse ouverte", open=False):
                    confirmation = gr.Checkbox(value=False, label="Je confirme la suppression définitive (radio, rapport et données)")
                    bouton_supprimer = gr.Button("Supprimer", variant="stop")

        sorties_analyse = [fiche, carte_decision, carte_attention, legende, details, etat, statut]
        bouton.click(analyser, [entree, reference, automatique], sorties_analyse)
        entree.change(nouvelle_image, None, sorties_analyse)
        bouton_enregistrer.click(enregistrer, [etat, reference], [etat, statut, fiche])
        sorties_selection = [fiche_historique, radio_historique, decision_historique, attention_historique, legende_historique,
                             rapport_historique, selection, statut_historique, panneau_selection]
        for evenement in (onglet_historique.select, bouton_actualiser.click, filtre_zone.change, recherche.submit):
            evenement(actualiser, [filtre_zone, recherche], [tableau, resume])
        tableau.select(selectionner, None, sorties_selection)
        bouton_exporter.click(exporter, None, [fichier_export, statut_historique])
        bouton_supprimer.click(supprimer, [selection, confirmation, filtre_zone, recherche],
                               [tableau, resume, *sorties_selection, confirmation])
    return demo


# --------------------------------------------------------------------------------------------------------------
# Calibration locale (inchangée)
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
    p.add_argument("--systeme", default=SYSTEME_PAR_DEFAUT, help="v4:CNN_CBAM (défaut), v4:CNN_classique, v3:CNN_CBAM, v3:CNN_classique")
    p.add_argument("--projet", default=str(PROJET_PAR_DEFAUT), help="dossier du projet (celui qui contient « optimisation »)")
    p.add_argument("--dossier-analyses", default=str(DOSSIER_APPLICATION / rc.NOM_DOSSIER_ANALYSES),
                   help="dossier où sont gardées les analyses enregistrées (défaut : application_clinique/analyses)")
    p.add_argument("--image", help="commande analyser : chemin de l'image")
    p.add_argument("--reference", default="", help="commande analyser : référence de l'examen (jamais le nom du patient)")
    p.add_argument("--enregistrer", action="store_true", help="commande analyser : enregistrer l'analyse (rapport PDF, images, données)")
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
    dossier_analyses = Path(args.dossier_analyses).resolve()
    try:
        analyseur = Analyseur(Path(args.projet), version_courte, nom, journal=not args.sans_journal)
        if args.commande == "analyser":
            if not args.image:
                raise ErreurApplication("Indiquer --image chemin.")
            resultat = analyseur.analyser(args.image)
            contenu = contenu_analyse(analyseur, resultat, args.reference)
            print(json.dumps(details_json(contenu), indent=2, ensure_ascii=False))
            if args.enregistrer:
                dossier = enregistrer_resultat(dossier_analyses, contenu, resultat, nom == "CNN_CBAM")
                print(f"\nAnalyse enregistrée : {dossier}")
        elif args.commande == "calibrer":
            if not (args.csv and args.images):
                raise ErreurApplication("Indiquer --csv et --images.")
            sortie = fichier_seuils_locaux(DOSSIER_APPLICATION, version_courte, nom)
            contenu = calibrer(analyseur, Path(args.csv), Path(args.images), sortie)
            print(json.dumps(contenu, indent=2, ensure_ascii=False))
            print(f"\nÉcrit : {sortie}. L'application les utilisera au prochain lancement (même système : {analyseur.cle}).")
        else:
            dossier_analyses.mkdir(parents=True, exist_ok=True)
            demo = construire_interface(analyseur, dossier_analyses)
            print(f"Système : {analyseur.cle} | seuils : {analyseur.source_seuils} | analyses enregistrées : {dossier_analyses}")
            demo.launch(server_name="127.0.0.1", server_port=args.port, inbrowser=not args.sans_navigateur, share=False, show_error=True,
                        allowed_paths=[str(dossier_analyses)])
    except ErreurApplication as e:
        raise SystemExit(f"Erreur : {e}")


if __name__ == "__main__":
    principal()

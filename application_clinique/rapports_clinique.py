"""Présentation, sauvegarde et historique des analyses (application d'aide au tri, version 2).

Ce module ne contient aucune règle de tri : seuils, zones, alertes et règle de sécurité restent dans outils_clinique.py,
partagé avec le notebook 15. Il met en forme un résultat déjà calculé (fiche, cartes avec échelle de couleur, rapport PDF),
enregistre les analyses que l'utilisateur choisit de garder et relit l'historique.

Une analyse enregistrée = un dossier `analyses/AAAA-MM-JJ_HH-MM-SS[_référence]/` contenant :
    radio.png            la radiographie telle qu'analysée
    entree_reseau.png    l'image préparée (224 × 224) vue par les réseaux
    carte_gradcam.png    la carte Grad-CAM, avec son échelle de couleur
    carte_attention.png  l'attention spatiale du CBAM avec son échelle (ou l'image préparée pour le CNN classique)
    rapport.pdf          le rapport d'une page
    analyse.json         les données (écrit en dernier : un dossier sans ce fichier est ignoré par l'historique)
"""
from __future__ import annotations

import csv
import datetime as dt
import html
import json
import os
import re
import shutil
import textwrap
import time
import unicodedata
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

VERSION_APPLICATION = "2.0"
FORMAT_ANALYSE = 1
NOM_DOSSIER_ANALYSES = "analyses"

NOMS_ZONES = {0: "VERTE", 1: "ORANGE", 2: "ROUGE"}
CODES_ZONES = {nom: code for code, nom in NOMS_ZONES.items()}
COULEURS_ZONES = {0: ("#0f5132", "#d1e7dd"), 1: ("#7a4100", "#ffe5b4"), 2: ("#842029", "#f8d7da")}   # (encre, fond)
SYMBOLES_ZONES = {0: "✔", 1: "◆", 2: "▲"}
TITRES_ZONES = {0: "Zone verte : lecture en file standard",
                1: "Zone orange : incertain, lecture médicale sans priorité particulière",
                2: "Zone rouge : lecture prioritaire recommandée"}
NOMS_VERSIONS = {"v3": "v3, image entière", "v4": "v4, recadrage sur les poumons"}
# Écart d'attention (échelle 0 à 1) jugé notable entre les poumons et le reste de l'image. Au notebook 16 (v4), l'écart moyen
# était de 0,03 à 0,13 selon la classe et la graine : en dessous de 0,15, on parle d'un écart faible.
ECART_ATTENTION_NOTABLE = 0.15
NOMS_MODELES = {"CNN_CBAM": "CNN + CBAM", "CNN_classique": "CNN classique"}

# Couleurs fixes : la fiche et le rapport gardent le même aspect en thème clair comme en thème sombre.
ENCRE, PAPIER, GRIS, BORDURE = "#1f2328", "#ffffff", "#57606a", "#d0d7de"
ROUGE_FONCE, ROUGE_TEXTE, ROUGE_PALE = "#842029", "#58151c", "#f8d7da"
ORANGE_FONCE, ORANGE_TEXTE, ORANGE_PALE = "#7a4100", "#3b2100", "#fff4e0"
POLICE = "font-family:system-ui,'Segoe UI',Arial,sans-serif"
ORIGINES_SEUILS = {"développement": "seuils du développement", "locale": "seuils calibrés localement"}

FICHIERS = {"radio": "radio.png", "entree": "entree_reseau.png", "decision": "carte_gradcam.png",
            "attention": "carte_attention.png", "rapport": "rapport.pdf", "donnees": "analyse.json"}


class ErreurHistorique(RuntimeError):
    """Erreur que l'utilisateur peut corriger (fichier ouvert dans un autre programme, analyse introuvable)."""


# --------------------------------------------------------------------------------------------------------------
# Textes et nombres
# --------------------------------------------------------------------------------------------------------------
def nombre_fr(x: float, decimales: int = 2, signe: bool = False) -> str:
    """Nombre à la française : virgule décimale et vrai signe moins (−2,40 ; +4,15)."""
    texte = f"{x:+.{decimales}f}" if signe else f"{x:.{decimales}f}"
    return texte.replace("-", "−").replace(".", ",")


def nom_systeme_lisible(version_courte: str, nom: str) -> str:
    return f"{NOMS_VERSIONS.get(version_courte, version_courte)} · {NOMS_MODELES.get(nom, nom)}"


def date_lisible(horodatage_iso: str) -> str:
    moment = dt.datetime.fromisoformat(horodatage_iso)
    return moment.strftime("%d/%m/%Y à %H:%M:%S")


def explication_zone(zone: int, seuils: dict) -> tuple[str, str]:
    """(texte, phrase à mettre en gras) pour une zone."""
    if zone == 0:
        part = 100 * (1 - seuils.get("cible_sensibilite_verte", 0.98))
        return (f"Le score est sous le seuil bas. Au réglage utilisé, au plus {part:.0f} % des radios PNEUMONIA de l'échantillon "
                "de réglage tombaient dans cette zone.", "Une zone verte n'exclut pas une pneumonie.")
    if zone == 1:
        return ("Le score est entre les deux seuils, ou une alerte empêche la zone verte.", "Aucune conclusion automatique.")
    part = 100 * (1 - seuils.get("cible_specificite_rouge", 0.90))
    return (f"Le score est au-dessus du seuil haut. Au réglage utilisé, au plus {part:.0f} % des radios NORMAL de l'échantillon "
            "de réglage tombaient dans cette zone.", "Ce n'est pas un diagnostic.")


def bandeaux_seuils(contenu: dict) -> list[tuple[str, str, str]]:
    """Messages sur l'origine des seuils : liste de (niveau, titre en gras, texte). niveau : 'info', 'attention', 'danger'."""
    s, sortie = contenu["seuils"], []
    if contenu["source_seuils"] == "locale":
        sortie.append(("info", "Seuils calibrés localement", f"({s.get('n_normal', '?')} radios NORMAL, {s.get('n_pneumonie', '?')} radios PNEUMONIA)."))
        if min(s.get("n_normal", 0), s.get("n_pneumonie", 0)) < 100:
            sortie.append(("attention", "Échantillon de calibration petit", "(moins de 100 radios dans une classe) : seuils instables."))
    else:
        sortie.append(("danger", "Seuils du développement, non calibrés localement.",
                       "Le décalage entre centres d'imagerie fausse ces seuils (notebooks 14 et 15) : ne pas s'y fier tant qu'une "
                       "calibration locale n'a pas été faite."))
    if s.get("zones_chevauchantes"):
        sortie.append(("danger", "Les deux seuils se croisent :",
                       "le modèle sépare mal les deux classes sur l'échantillon de réglage, il n'y a plus de zone orange. "
                       "Ne pas utiliser ce réglage pour trier."))
    return sortie


def legende_cartes(contenu: dict) -> list[str]:
    """Paragraphes d'aide à la lecture des cartes, adaptés à cette radio."""
    sortie = ["Les cartes montrent les zones qui ont le plus pesé dans le score de cette radio ; elles ne prouvent pas la présence "
              "d'une lésion."]
    if contenu["version_courte"] == "v4":
        sortie.append("Au notebook 16 (v4), environ la moitié de la carte Grad-CAM tombait dans les poumons : le modèle peut encore "
                      "s'appuyer sur des repères hors des poumons, comme la lettre «\u00a0R\u00a0».")
    else:
        sortie.append("En v3 (image entière), les cartes tombaient souvent hors des poumons (notebooks 11 à 13).")
    if contenu["score"] <= 0:
        sortie.append("Ici, le modèle penche vers NORMAL : la carte montre ce qui a fait baisser le score. Pour ces radios, la carte "
                      "est souvent vide ou hors des poumons (notebook 16) : ne pas l'interpréter.")
    a = contenu.get("attention")
    if a:
        phrase = f"Attention spatiale du CBAM sur cette radio : de {nombre_fr(a['minimum'])} à {nombre_fr(a['maximum'])}"
        if a.get("moyenne_poumons") is not None:
            dedans, dehors = a["moyenne_poumons"], a["moyenne_hors_poumons"]
            phrase += f" ; en moyenne {nombre_fr(dedans)} dans les poumons et {nombre_fr(dehors)} ailleurs. "
            if abs(dedans - dehors) < ECART_ATTENTION_NOTABLE:
                phrase += ("L'écart moyen entre les poumons et le reste de l'image est faible : le CBAM pondère peu l'espace, "
                           "comme au notebook 16.")
            elif dedans > dehors:
                phrase += "Sur cette radio, l'attention est nettement plus forte dans les poumons qu'ailleurs."
            else:
                phrase += "Sur cette radio, l'attention est plus forte hors des poumons que dans les poumons."
        elif a["maximum"] - a["minimum"] < ECART_ATTENTION_NOTABLE:
            phrase += ". Elle varie peu sur l'image : le CBAM pondère peu l'espace, comme au notebook 16."
        else:
            phrase += "."
        sortie.append(phrase)
    return sortie


def resume_attention(attention, zone_poumons) -> dict | None:
    if attention is None:
        return None
    a = np.asarray(attention, dtype=np.float64)
    resume = {"minimum": float(a.min()), "maximum": float(a.max()), "moyenne": float(a.mean()),
              "moyenne_poumons": None, "moyenne_hors_poumons": None}
    if zone_poumons is not None and zone_poumons.any() and (~zone_poumons).any():
        resume["moyenne_poumons"] = float(a[zone_poumons].mean())
        resume["moyenne_hors_poumons"] = float(a[~zone_poumons].mean())
    return resume


# --------------------------------------------------------------------------------------------------------------
# Cartes : superposition, contour des poumons, échelle de couleur
# --------------------------------------------------------------------------------------------------------------
def _police(taille: int):
    try:
        import matplotlib
        return ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"), taille)
    except Exception:   # noqa: BLE001 - police de secours, l'image reste lisible
        return ImageFont.load_default()


def superposer(entree_u8, carte, nom_palette, alpha, contour=None, taille=448) -> Image.Image:
    from matplotlib import colormaps
    fond = np.stack([entree_u8] * 3, axis=-1).astype(np.float64) / 255.0
    couleur = colormaps[nom_palette](np.clip(carte, 0, 1))[..., :3]
    melange = (1 - alpha) * fond + alpha * couleur
    image = Image.fromarray((np.clip(melange, 0, 1) * 255).astype(np.uint8)).resize((taille, taille), Image.Resampling.BICUBIC)
    if contour is not None and contour.any():
        from scipy import ndimage
        grand = np.asarray(Image.fromarray(contour.astype(np.uint8) * 255).resize((taille, taille), Image.Resampling.NEAREST)) > 127
        bord = grand & ~ndimage.binary_erosion(grand, iterations=2)
        tableau = np.asarray(image).copy()
        tableau[bord] = (255, 210, 63)
        image = Image.fromarray(tableau)
    return image


def ajouter_echelle(image: Image.Image, nom_palette: str, titre: str, gauche: str, droite: str, reperes=()) -> Image.Image:
    """Ajoute sous la carte une barre de couleur (fond blanc, texte foncé). reperes : positions entre 0 et 1 marquées d'un trait."""
    from matplotlib import colormaps
    largeur, hauteur = image.width, 70
    bande = Image.new("RGB", (largeur, hauteur), PAPIER)
    x0, x1, y0, y1 = 14, largeur - 14, 26, 42
    degrade = (colormaps[nom_palette](np.linspace(0, 1, x1 - x0))[:, :3] * 255).astype(np.uint8)
    bande.paste(Image.fromarray(np.repeat(degrade[None], y1 - y0, axis=0)), (x0, y0))
    dessin = ImageDraw.Draw(bande)
    dessin.rectangle([x0 - 1, y0 - 1, x1, y1], outline=GRIS)
    police, petite = _police(14), _police(13)
    dessin.text((x0, 4), titre, fill=ENCRE, font=police)
    for position in reperes:
        x = x0 + float(np.clip(position, 0, 1)) * (x1 - x0 - 1)
        dessin.rectangle([x - 1, y0 - 5, x + 1, y1 + 4], fill=ENCRE)
    dessin.text((x0, y1 + 6), gauche, fill=ENCRE, font=petite)
    dessin.text((x1 - dessin.textlength(droite, font=petite), y1 + 6), droite, fill=ENCRE, font=petite)
    sortie = Image.new("RGB", (largeur, image.height + hauteur), PAPIER)
    sortie.paste(image.convert("RGB"), (0, 0))
    sortie.paste(bande, (0, image.height))
    return sortie


def images_explication(entree_u8, decision, attention, zone_poumons, avec_attention: bool):
    """Carte Grad-CAM et attention spatiale (ou image préparée), avec leur échelle de couleur."""
    carte_decision = ajouter_echelle(superposer(entree_u8, decision, "jet", 0.45, zone_poumons), "jet",
                                     "Grad-CAM : contribution au score, relative à cette radio",
                                     "0 : aucune", "1 : la plus forte")
    if avec_attention and attention is not None:
        a = np.clip(np.asarray(attention, dtype=np.float64), 0, 1)
        carte_attention = ajouter_echelle(superposer(entree_u8, a, "viridis", 0.55, zone_poumons), "viridis",
                                          f"Attention du CBAM, échelle réelle · ici de {nombre_fr(a.min())} à {nombre_fr(a.max())}",
                                          "0", "1", reperes=(a.min(), a.max()))
    else:
        carte_attention = Image.fromarray(np.stack([entree_u8] * 3, axis=-1)).resize((448, 448), Image.Resampling.BICUBIC)
    return carte_decision, carte_attention


# --------------------------------------------------------------------------------------------------------------
# Fiche de résultat (HTML) : couleurs explicites partout, lisible en thème clair comme sombre
# --------------------------------------------------------------------------------------------------------------
def geometrie_barre(score: float, bas: float, haut: float):
    """Échelle affichée autour des seuils. Renvoie début, fin, position (%) des seuils et du score, côté de dépassement (−1, 0, +1)."""
    marge = max(3.0, 1.5 * (haut - bas))
    debut, fin = bas - marge, haut + marge
    position = lambda v: 100.0 * (v - debut) / (fin - debut)   # noqa: E731
    cote = -1 if score < debut else (1 if score > fin else 0)
    return debut, fin, position(bas), position(haut), min(100.0, max(0.0, position(score))), cote


def barre_score_html(score: float, bas: float, haut: float) -> str:
    debut, fin, p_bas, p_haut, p, cote = geometrie_barre(score, bas, haut)
    fond = (f"linear-gradient(to right,#d1e7dd 0,#d1e7dd {p_bas:.2f}%,#ffe5b4 {p_bas:.2f}%,#ffe5b4 {p_haut:.2f}%,"
            f"#f8d7da {p_haut:.2f}%,#f8d7da 100%)")
    valeur = nombre_fr(score, signe=True)
    if cote == 0:
        aligne = "left:0" if p < 10 else ("right:0" if p > 90 else f"left:{p:.2f}%;transform:translateX(-50%)")
        etiquette = f'<div style="position:absolute;{aligne};top:0;font-size:12px;font-weight:700;color:{ENCRE}">{valeur}</div>'
        repere = (f'<div style="position:absolute;left:{p:.2f}%;top:0;width:4px;height:100%;background:{ENCRE};'
                  f'margin-left:-2px"></div>')
        phrase = "Trait noir : score de cette radio."
    else:
        etiquette = ""
        texte = f"{valeur} ▶" if cote > 0 else f"◀ {valeur}"
        place = "right:6px" if cote > 0 else "left:6px"
        repere = (f'<div style="position:absolute;{place};top:3px;font-size:13px;font-weight:700;color:{ENCRE};'
                  f'background:rgba(255,255,255,0.85);padding:0 4px;border-radius:3px">{texte}</div>')
        phrase = f"Le score de cette radio ({valeur}) dépasse l'échelle affichée, vers {'la droite' if cote > 0 else 'la gauche'}."
    if abs(haut - bas) < 1e-9:
        graduations = (f'<div style="position:absolute;left:{p_bas:.2f}%;transform:translateX(-50%);font-size:12px;color:{GRIS}">'
                       f'{nombre_fr(bas, signe=True)}</div>')
        zones = f"Verte : score &lt; {nombre_fr(bas, signe=True)} · rouge : score &gt; {nombre_fr(haut, signe=True)} (pas de zone orange)."
    else:
        graduations = (f'<div style="position:absolute;right:{100 - p_bas:.2f}%;font-size:12px;color:{GRIS};padding-right:3px">'
                       f'{nombre_fr(bas, signe=True)}</div>'
                       f'<div style="position:absolute;left:{p_haut:.2f}%;font-size:12px;color:{GRIS};padding-left:3px">'
                       f'{nombre_fr(haut, signe=True)}</div>')
        zones = (f"Verte : score &lt; {nombre_fr(bas, signe=True)} · orange : de {nombre_fr(bas, signe=True)} à "
                 f"{nombre_fr(haut, signe=True)} · rouge : score &gt; {nombre_fr(haut, signe=True)}.")
    return (f'<div style="position:relative;height:18px;color:{ENCRE}">{etiquette}</div>'
            f'<div style="position:relative;height:26px;border-radius:6px;overflow:hidden;background:{fond};border:1px solid #8c959f">'
            f'{repere}</div>'
            f'<div style="position:relative;height:18px;color:{GRIS}">{graduations}</div>'
            f'<div style="font-size:12px;color:{GRIS};margin-top:2px">{phrase} {zones}</div>')


def _bandeau_html(niveau: str, titre: str, texte: str) -> str:
    fond, encre = {"info": ("#eef6ff", "#0b3a6e"), "attention": (ORANGE_PALE, ORANGE_TEXTE), "danger": (ROUGE_PALE, ROUGE_TEXTE)}[niveau]
    return (f'<div style="margin-top:8px;padding:7px 11px;background:{fond};color:{encre};border-radius:5px;font-size:13px">'
            f'<b style="color:{encre}">{titre}</b> <span style="color:{encre}">{texte}</span></div>')


def fiche_html(contenu: dict) -> str:
    """Fiche de résultat, reconstruite à partir des seules données enregistrées (sert aussi à l'historique)."""
    zone = CODES_ZONES[contenu["zone_finale"]]
    brute = CODES_ZONES[contenu["zone_avant_regle"]]
    encre, fond = COULEURS_ZONES[zone]
    seuils = contenu["seuils"]
    texte, gras = explication_zone(zone, seuils)
    reference = contenu.get("reference") or ""
    ligne_reference = (f'<div style="font-size:13px;color:{GRIS};margin-bottom:8px">Référence : '
                       f'<b style="color:{ENCRE}">{html.escape(reference)}</b></div>') if reference else ""
    if contenu["alertes"]:
        elements = "".join(f'<li style="color:{ORANGE_TEXTE}">{html.escape(l)}</li>' for l in contenu["alertes_libelles"])
        regle = (f'<div style="color:{ORANGE_TEXTE};font-style:italic;margin-top:4px">Règle de sécurité : une alerte empêche la '
                 'zone verte ; la radio passe en zone orange.</div>') if brute == 0 else ""
        bloc_alertes = (f'<div style="margin-top:10px;padding:8px 12px;border-left:5px solid {ORANGE_FONCE};background:{ORANGE_PALE};'
                        f'color:{ORANGE_TEXTE}"><b style="color:{ORANGE_TEXTE}">Alertes de sécurité ({len(contenu["alertes"])})</b>'
                        f'<ul style="margin:4px 0 0 18px;padding:0;color:{ORANGE_TEXTE}">{elements}</ul>{regle}</div>')
    else:
        bloc_alertes = f'<div style="margin-top:10px;font-size:14px;color:{ENCRE}">Aucune alerte de sécurité.</div>'
    scores = " ; ".join(nombre_fr(v, signe=True) for v in contenu["scores_graines"])
    pied = (f'Système : {html.escape(contenu["systeme_lisible"])} · {ORIGINES_SEUILS.get(contenu["source_seuils"], contenu["source_seuils"])} · '
            f'durée {nombre_fr(contenu["duree_secondes"], 1)} s · {date_lisible(contenu["horodatage_analyse"])}')
    bandeaux = "".join(_bandeau_html(*b) for b in bandeaux_seuils(contenu))
    return f"""
<div style="{POLICE};max-width:860px;background:{PAPIER};color:{ENCRE};border:1px solid {BORDURE};border-radius:10px;padding:14px 16px">
  {ligne_reference}
  <div style="padding:7px 11px;border:2px solid {ROUGE_FONCE};border-radius:6px;background:{PAPIER};color:{ROUGE_TEXTE};font-size:13px;margin-bottom:10px">
    <b style="color:{ROUGE_FONCE}">Prototype de recherche.</b> <span style="color:{ROUGE_TEXTE}">Cet outil aide à ordonner la lecture des
    radiographies ; il ne pose aucun diagnostic et ne remplace pas la lecture par un médecin.</span>
  </div>
  <div style="padding:14px 16px;border-radius:8px;background:{fond};color:{encre};border:2px solid {encre}">
    <div style="font-size:22px;font-weight:700;color:{encre}">{SYMBOLES_ZONES[zone]} {TITRES_ZONES[zone]}</div>
    <div style="margin-top:6px;font-size:15px;color:{encre}">{texte} <b style="color:{encre}">{gras}</b></div>
  </div>
  <div style="margin-top:12px">{barre_score_html(contenu['score'], seuils['seuil_bas'], seuils['seuil_haut'])}</div>
  <div style="margin-top:8px;font-size:14px;color:{ENCRE}">Score moyen des 3 modèles : <b style="color:{ENCRE}">{nombre_fr(contenu['score'], signe=True)}</b>
     (chaque modèle : {scores} ; écart-type {nombre_fr(contenu['ecart_type_graines'])}). Un score plus élevé signifie plus de signes
     évocateurs de pneumonie pour ce modèle.</div>
  {bloc_alertes}
  {bandeaux}
  <div style="margin-top:9px;font-size:12px;color:{GRIS}">{pied}</div>
</div>"""


def legende_html(contenu: dict | None) -> str:
    if not contenu:
        return ""
    paragraphes = "".join(f'<p style="margin:0 0 6px 0;color:{ENCRE}">{html.escape(p)}</p>' for p in legende_cartes(contenu))
    return (f'<div style="{POLICE};max-width:1200px;background:{PAPIER};color:{ENCRE};border:1px solid {BORDURE};border-radius:8px;'
            f'padding:10px 14px;font-size:13px;font-style:italic">{paragraphes}</div>')


def message_html(texte: str, niveau: str = "info") -> str:
    """Message court (succès, erreur, information), lisible dans les deux thèmes."""
    if not texte:
        return ""
    fond, encre, bord = {"succes": ("#d1e7dd", "#0f5132", "#0f5132"), "erreur": (ROUGE_PALE, ROUGE_TEXTE, ROUGE_FONCE),
                         "info": ("#eef6ff", "#0b3a6e", "#0b3a6e")}[niveau]
    return (f'<div style="{POLICE};padding:9px 12px;border:1px solid {bord};border-radius:6px;background:{fond};color:{encre};'
            f'font-size:14px;max-width:860px">{texte}</div>')


# --------------------------------------------------------------------------------------------------------------
# Rapport PDF (une page A4, matplotlib sans pyplot : sûr dans le serveur de l'application)
# --------------------------------------------------------------------------------------------------------------
def _replier(texte: str, largeur_pouces: float, taille_police: float) -> str:
    return textwrap.fill(texte, width=max(20, int(largeur_pouces * 72 / (taille_police * 0.56))))


def ecrire_rapport_pdf(chemin: Path, contenu: dict, radio: Image.Image, carte_decision: Image.Image, carte_attention: Image.Image):
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    L, H, M = 8.27, 11.69, 0.6                       # A4 en pouces, marge
    utile = L - 2 * M
    fig = Figure(figsize=(L, H))
    fx = lambda pouces: pouces / L                    # noqa: E731
    fy = lambda depuis_haut: 1 - depuis_haut / H     # noqa: E731

    def texte(x, y, chaine, **options):
        options.setdefault("color", ENCRE)
        options.setdefault("fontsize", 9)
        options.setdefault("va", "top")
        fig.text(fx(x), fy(y), chaine, parse_math=False, **options)

    def cadre(x, y, largeur, hauteur, fond, bord, epaisseur=1.2):
        fig.add_artist(Rectangle((fx(x), fy(y + hauteur)), largeur / L, hauteur / H, transform=fig.transFigure,
                                 facecolor=fond, edgecolor=bord, linewidth=epaisseur))

    zone = CODES_ZONES[contenu["zone_finale"]]
    encre, fond = COULEURS_ZONES[zone]
    seuils = contenu["seuils"]

    texte(M, 0.45, "Aide au tri des radiographies pulmonaires · rapport d'analyse", fontsize=14, fontweight="bold")
    avertissement = _replier("Cet outil aide à ordonner la lecture des radiographies ; il ne pose aucun diagnostic et ne remplace pas "
                             "la lecture par un médecin.", utile - 0.3, 8.5)
    hauteur_avertissement = 0.36 + 0.15 * (avertissement.count("\n") + 1)
    cadre(M, 0.85, utile, hauteur_avertissement, PAPIER, ROUGE_FONCE, 1.5)
    texte(M + 0.12, 0.93, "Prototype de recherche, non validé cliniquement.", fontsize=8.5, fontweight="bold", color=ROUGE_FONCE)
    texte(M + 0.12, 1.1, avertissement, fontsize=8.5, color=ROUGE_TEXTE, linespacing=1.3)

    if contenu["source_seuils"] == "locale":
        origine = f"calibrés localement ({seuils.get('n_normal', '?')} NORMAL, {seuils.get('n_pneumonie', '?')} PNEUMONIA)"
    else:
        origine = "du développement, non calibrés localement"
    stats = contenu.get("statistiques_image", {})
    infos = [("Date et heure de l'analyse", date_lisible(contenu["horodatage_analyse"])),
             ("Référence", contenu.get("reference") or "—"),
             ("Système", contenu["systeme_lisible"]),
             ("Seuils", f"{origine} : verte si score < {nombre_fr(seuils['seuil_bas'], signe=True)}, rouge si score > "
                        f"{nombre_fr(seuils['seuil_haut'], signe=True)}"),
             ("Image", f"{stats.get('largeur', '?')} × {stats.get('hauteur', '?')} pixels · empreinte {contenu.get('empreinte_image', '?')}"),
             ("Durée de l'analyse", f"{nombre_fr(contenu['duree_secondes'], 1)} s")]
    y = 0.85 + hauteur_avertissement + 0.18
    for etiquette, valeur in infos:
        texte(M, y, etiquette, fontsize=8.5, fontweight="bold")
        texte(M + 1.85, y, _replier(valeur, utile - 1.85, 8.5), fontsize=8.5, linespacing=1.3)
        y += 0.2 + 0.15 * _replier(valeur, utile - 1.85, 8.5).count("\n")

    haut_zone = y + 0.12
    explication, gras = explication_zone(zone, seuils)
    lignes = _replier(f"{explication} {gras}", utile - 0.3, 9)
    titre_zone = _replier(f"{SYMBOLES_ZONES[zone]} {TITRES_ZONES[zone]}", utile - 0.45, 13)
    lignes_titre = titre_zone.count("\n") + 1
    hauteur_zone = 0.24 + 0.26 * lignes_titre + 0.16 * (lignes.count("\n") + 1)
    cadre(M, haut_zone, utile, hauteur_zone, fond, encre, 1.8)
    texte(M + 0.15, haut_zone + 0.12, titre_zone, fontsize=13, fontweight="bold", color=encre, linespacing=1.15)
    texte(M + 0.15, haut_zone + 0.16 + 0.26 * lignes_titre, lignes, fontsize=9, color=encre, linespacing=1.35)

    # Barre du score
    y_barre = haut_zone + hauteur_zone + 0.42
    debut, fin, _, _, _, cote = geometrie_barre(contenu["score"], seuils["seuil_bas"], seuils["seuil_haut"])
    ax = fig.add_axes([fx(M), fy(y_barre + 0.26), utile / L, 0.26 / H])
    ax.axvspan(debut, seuils["seuil_bas"], color="#d1e7dd")
    ax.axvspan(seuils["seuil_bas"], seuils["seuil_haut"], color="#ffe5b4")
    ax.axvspan(seuils["seuil_haut"], fin, color="#f8d7da")
    ax.set_xlim(debut, fin)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    graduations = sorted({seuils["seuil_bas"], seuils["seuil_haut"]})
    ax.set_xticks(graduations)
    ax.set_xticklabels([""] * len(graduations))
    sous_axe = ax.get_xaxis_transform()
    if len(graduations) == 1:
        ax.text(graduations[0], -0.35, nombre_fr(graduations[0], signe=True), ha="center", va="top", fontsize=8, color=GRIS,
                transform=sous_axe, parse_math=False)
    else:   # seuil bas à gauche de son trait, seuil haut à droite du sien : jamais superposés
        ax.text(graduations[0], -0.35, nombre_fr(graduations[0], signe=True) + " ", ha="right", va="top", fontsize=8, color=GRIS,
                transform=sous_axe, parse_math=False)
        ax.text(graduations[1], -0.35, " " + nombre_fr(graduations[1], signe=True), ha="left", va="top", fontsize=8, color=GRIS,
                transform=sous_axe, parse_math=False)
    for cote_axe in ax.spines.values():
        cote_axe.set_color("#8c959f")
    valeur = nombre_fr(contenu["score"], signe=True)
    if cote == 0:
        ax.axvline(contenu["score"], color=ENCRE, linewidth=3)
        ax.text(contenu["score"], 1.12, valeur, ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=ENCRE,
                transform=ax.get_xaxis_transform(), parse_math=False)
    else:
        ax.text(fin if cote > 0 else debut, 0.5, f"{valeur} ▶" if cote > 0 else f"◀ {valeur}", ha="right" if cote > 0 else "left",
                va="center", fontsize=9, fontweight="bold", color=ENCRE, parse_math=False)

    y = y_barre + 0.62
    scores = " ; ".join(nombre_fr(v, signe=True) for v in contenu["scores_graines"])
    phrase = (f"Score moyen des 3 modèles : {valeur} (chaque modèle : {scores} ; écart-type "
              f"{nombre_fr(contenu['ecart_type_graines'])}). Un score plus élevé signifie plus de signes évocateurs de pneumonie "
              "pour ce modèle.")
    bloc = _replier(phrase, utile, 9)
    texte(M, y, bloc, fontsize=9, linespacing=1.35)
    y += 0.19 * (bloc.count("\n") + 1) + 0.12
    if contenu["alertes"]:
        texte(M, y, f"Alertes de sécurité ({len(contenu['alertes'])}) :", fontsize=9, fontweight="bold", color=ORANGE_FONCE)
        y += 0.2
        for libelle in contenu["alertes_libelles"]:
            ligne = _replier("• " + libelle, utile - 0.2, 8.5)
            texte(M + 0.15, y, ligne, fontsize=8.5, color=ORANGE_TEXTE, linespacing=1.3)
            y += 0.17 * (ligne.count("\n") + 1) + 0.03
        if CODES_ZONES[contenu["zone_avant_regle"]] == 0:
            texte(M + 0.15, y, "Règle de sécurité : une alerte empêche la zone verte ; la radio passe en zone orange.", fontsize=8.5,
                  style="italic", color=ORANGE_TEXTE)
            y += 0.2
    else:
        texte(M, y, "Aucune alerte de sécurité.", fontsize=9)
        y += 0.22
    for niveau, titre, detail in bandeaux_seuils(contenu):
        couleur = {"info": "#0b3a6e", "attention": ORANGE_TEXTE, "danger": ROUGE_TEXTE}[niveau]
        ligne = _replier(f"{titre} {detail}", utile - 0.24, 8.5)
        hauteur = 0.16 * (ligne.count("\n") + 1) + 0.12
        cadre(M, y, utile, hauteur, {"info": "#eef6ff", "attention": ORANGE_PALE, "danger": ROUGE_PALE}[niveau], couleur, 0.8)
        texte(M + 0.12, y + 0.07, ligne, fontsize=8.5, color=couleur, linespacing=1.3)
        y += hauteur + 0.08

    # Trois images : leur hauteur s'adapte à la place laissée par la légende et le pied de page (jamais de débordement)
    blocs_legende = [_replier(p, utile, 8) for p in legende_cartes(contenu)]
    hauteur_legende = sum(0.15 * (b.count("\n") + 1) + 0.06 for b in blocs_legende)
    limite_basse = H - 0.62                          # le pied de page commence en dessous
    y_images = y + 0.15
    largeur_image, ecart = (utile - 0.3) / 3, 0.15
    hauteur_disponible = max(1.2, limite_basse - hauteur_legende - 0.18 - (y_images + 0.36))
    titres = ["Radiographie analysée", "Où le modèle a regardé (Grad-CAM)" + (" · contour jaune : poumons" if contenu["version_courte"] == "v4" else ""),
              "Attention spatiale du CBAM" if contenu.get("attention") else "Image préparée vue par le réseau"]
    apercu = radio.convert("L").copy()
    apercu.thumbnail((1000, 1000))
    hauteur_max = 0.0
    for i, (image, titre) in enumerate(zip([apercu, carte_decision, carte_attention], titres)):
        x = M + i * (largeur_image + ecart)
        texte(x, y_images, _replier(titre, largeur_image, 8), fontsize=8, fontweight="bold", linespacing=1.2)
        rapport = image.height / image.width
        hauteur = min(largeur_image * rapport, 3.3, hauteur_disponible)
        largeur = hauteur / rapport
        axe = fig.add_axes([fx(x + (largeur_image - largeur) / 2), fy(y_images + 0.36 + hauteur), largeur / L, hauteur / H])
        axe.imshow(np.asarray(image), cmap="gray" if image.mode == "L" else None, interpolation="lanczos")
        axe.set_axis_off()
        hauteur_max = max(hauteur_max, hauteur)

    y = y_images + 0.36 + hauteur_max + 0.18
    for bloc in blocs_legende:
        texte(M, y, bloc, fontsize=8, style="italic", color=GRIS, linespacing=1.3)
        y += 0.15 * (bloc.count("\n") + 1) + 0.06

    texte(M, H - 0.45, _replier(f"Rapport généré le {dt.datetime.now().strftime('%d/%m/%Y à %H:%M')} par l'application d'aide au tri "
                                f"(version {VERSION_APPLICATION}). Prototype de recherche : ne constitue pas un dispositif médical et ne "
                                "remplace pas la lecture de la radiographie par un médecin.", utile, 7),
          fontsize=7, color=GRIS, linespacing=1.3)

    temporaire = chemin.with_name(chemin.name + ".tmp")
    fig.savefig(temporaire, format="pdf", metadata={"Title": "Rapport d'analyse : aide au tri des radiographies pulmonaires",
                                                    "Creator": f"application_clinique.py {VERSION_APPLICATION}"})
    _remplacer(temporaire, chemin)


# --------------------------------------------------------------------------------------------------------------
# Écritures sûres (OneDrive, Excel ou un lecteur PDF peuvent bloquer un fichier quelques instants)
# --------------------------------------------------------------------------------------------------------------
def _remplacer(temporaire: Path, chemin: Path, essais: int = 20):
    for essai in range(essais):
        try:
            os.replace(temporaire, chemin)
            return
        except PermissionError:
            if essai == essais - 1:
                raise
            time.sleep(0.5)


def _ecrire_json(chemin: Path, objet: dict):
    temporaire = chemin.with_name(chemin.name + ".tmp")
    temporaire.write_text(json.dumps(objet, ensure_ascii=False, indent=2), encoding="utf-8")
    _remplacer(temporaire, chemin)


# --------------------------------------------------------------------------------------------------------------
# Sauvegarde et historique
# --------------------------------------------------------------------------------------------------------------
def nettoyer_reference(texte) -> str:
    """Référence saisie par l'utilisateur, sans retour à la ligne, limitée à 60 caractères."""
    return " ".join(str(texte or "").split())[:60]


def _suffixe(reference: str) -> str:
    """Partie du nom de dossier tirée de la référence : lettres sans accent, chiffres, - et _ (« Réf é/12 » → « Ref-e-12 »)."""
    sans_accents = unicodedata.normalize("NFKD", reference).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^A-Za-z0-9_]+", "-", sans_accents).strip("-")[:40]


def _image_a_garder(image: Image.Image) -> Image.Image:
    return image if image.mode in ("L", "RGB", "I;16") else image.convert("RGB")


def enregistrer_analyse(dossier_analyses: Path, contenu: dict, radio: Image.Image, entree_u8, carte_decision: Image.Image,
                        carte_attention: Image.Image) -> Path:
    """Crée le dossier de l'analyse et y écrit images, rapport PDF puis données (en dernier). Renvoie le dossier."""
    dossier_analyses = Path(dossier_analyses)
    dossier_analyses.mkdir(parents=True, exist_ok=True)
    moment = dt.datetime.fromisoformat(contenu["horodatage_analyse"])
    base = moment.strftime("%Y-%m-%d_%H-%M-%S") + (f"_{_suffixe(contenu.get('reference') or '')}" if _suffixe(contenu.get("reference") or "") else "")
    dossier, numero = dossier_analyses / base, 2
    while dossier.exists():
        dossier, numero = dossier_analyses / f"{base}_{numero}", numero + 1
    dossier.mkdir()
    try:
        _image_a_garder(radio).save(dossier / FICHIERS["radio"])
        Image.fromarray(np.asarray(entree_u8, dtype=np.uint8)).save(dossier / FICHIERS["entree"])
        carte_decision.save(dossier / FICHIERS["decision"])
        carte_attention.save(dossier / FICHIERS["attention"])
        contenu = {**contenu, "dossier": dossier.name, "fichiers": FICHIERS,
                   "horodatage_enregistrement": dt.datetime.now().isoformat(timespec="seconds")}
        ecrire_rapport_pdf(dossier / FICHIERS["rapport"], contenu, radio, carte_decision, carte_attention)
        _ecrire_json(dossier / FICHIERS["donnees"], contenu)
    except Exception:
        shutil.rmtree(dossier, ignore_errors=True)
        raise
    return dossier


def _dossier_analyse(dossier_analyses: Path, nom: str) -> Path:
    """Chemin d'une analyse, après contrôle : un nom simple, un dossier direct de `analyses`, avec ses données."""
    if not nom or nom != Path(nom).name or nom.startswith("."):
        raise ErreurHistorique("Analyse inconnue.")
    dossier = Path(dossier_analyses) / nom
    if not (dossier / FICHIERS["donnees"]).is_file():
        raise ErreurHistorique(f"L'analyse {nom} est introuvable (dossier déplacé ou supprimé ?).")
    return dossier


def lister_analyses(dossier_analyses: Path) -> list[dict]:
    """Analyses enregistrées, de la plus récente à la plus ancienne. Les dossiers incomplets ou illisibles sont ignorés."""
    dossier_analyses = Path(dossier_analyses)
    if not dossier_analyses.is_dir():
        return []
    sortie = []
    for dossier in dossier_analyses.iterdir():
        fichier = dossier / FICHIERS["donnees"]
        if dossier.is_dir() and fichier.is_file():
            try:
                contenu = json.loads(fichier.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            contenu["dossier"] = dossier.name
            sortie.append(contenu)
    sortie.sort(key=lambda c: (c.get("horodatage_analyse", ""), c["dossier"]), reverse=True)
    return sortie


def charger_analyse(dossier_analyses: Path, nom: str) -> dict:
    dossier = _dossier_analyse(dossier_analyses, nom)
    contenu = json.loads((dossier / FICHIERS["donnees"]).read_text(encoding="utf-8"))
    contenu["dossier"] = nom

    def ouvrir(cle):
        chemin = dossier / FICHIERS[cle]
        if not chemin.is_file():
            return None
        with Image.open(chemin) as image:
            image.load()
            return image.copy()

    rapport = dossier / FICHIERS["rapport"]
    return {"contenu": contenu, "radio": ouvrir("radio"), "decision": ouvrir("decision"), "attention": ouvrir("attention"),
            "rapport": str(rapport) if rapport.is_file() else None, "dossier": str(dossier)}


def supprimer_analyse(dossier_analyses: Path, nom: str):
    dossier = _dossier_analyse(dossier_analyses, nom)
    for essai in range(10):
        try:
            shutil.rmtree(dossier)
            return
        except PermissionError as erreur:
            if essai == 9:
                raise ErreurHistorique("Suppression impossible : un fichier de cette analyse est ouvert dans un autre programme "
                                       "(lecteur PDF, visionneuse). Le fermer puis réessayer.") from erreur
            time.sleep(0.5)


def ligne_tableau(contenu: dict) -> dict:
    moment = dt.datetime.fromisoformat(contenu["horodatage_analyse"])
    return {"Date": moment.strftime("%d/%m/%Y"), "Heure": moment.strftime("%H:%M:%S"), "Référence": contenu.get("reference") or "",
            "Zone": contenu["zone_finale"].capitalize(), "Score": nombre_fr(contenu["score"], signe=True),
            "Alertes": len(contenu.get("alertes", [])),
            "Système": f"{contenu.get('version_courte', '')} · {NOMS_MODELES.get(contenu.get('modele', ''), contenu.get('modele', ''))}",
            "Seuils": {"locale": "locaux"}.get(contenu.get("source_seuils", ""), contenu.get("source_seuils", "")),
            "Dossier": contenu["dossier"]}


def filtrer(analyses: list[dict], zone: str, recherche: str) -> list[dict]:
    recherche = (recherche or "").strip().lower()
    sortie = analyses
    if zone and zone != "Toutes":
        sortie = [c for c in sortie if c["zone_finale"] == zone.upper()]
    if recherche:
        sortie = [c for c in sortie if recherche in (c.get("reference") or "").lower() or recherche in c["dossier"].lower()]
    return sortie


def resume_zones(analyses: list[dict]) -> dict:
    compte = {"VERTE": 0, "ORANGE": 0, "ROUGE": 0}
    for c in analyses:
        compte[c["zone_finale"]] = compte.get(c["zone_finale"], 0) + 1
    return compte


def resume_journal(fichier_journal: Path) -> dict | None:
    """Nombre d'analyses du journal anonyme (enregistrées ou non), par zone finale. None si absent ou illisible."""
    try:
        with open(fichier_journal, newline="", encoding="utf-8-sig") as f:
            lignes = list(csv.DictReader(f))
    except (OSError, csv.Error):
        return None
    compte = {"total": len(lignes), "VERTE": 0, "ORANGE": 0, "ROUGE": 0}
    for ligne in lignes:
        if ligne.get("zone_finale") in compte:
            compte[ligne["zone_finale"]] += 1
    return compte


def exporter_historique(dossier_analyses: Path) -> Path:
    """Exporte la liste des analyses enregistrées en CSV pour Excel (séparateur ; et virgule décimale)."""
    analyses = lister_analyses(dossier_analyses)
    Path(dossier_analyses).mkdir(parents=True, exist_ok=True)
    chemin = Path(dossier_analyses) / f"export_historique_{dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    colonnes = ["date", "heure", "reference", "zone_finale", "zone_avant_regle_de_securite", "score", "scores_des_3_modeles",
                "ecart_type_graines", "alertes", "systeme", "origine_des_seuils", "seuil_bas", "seuil_haut", "empreinte_image",
                "dossier"]
    temporaire = chemin.with_name(chemin.name + ".tmp")
    with open(temporaire, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(colonnes)
        for c in analyses:
            moment = dt.datetime.fromisoformat(c["horodatage_analyse"])
            w.writerow([moment.strftime("%d/%m/%Y"), moment.strftime("%H:%M:%S"), c.get("reference") or "", c["zone_finale"],
                        c["zone_avant_regle"], nombre_fr(c["score"], 4, signe=True),
                        " ; ".join(nombre_fr(v, 4, signe=True) for v in c["scores_graines"]), nombre_fr(c["ecart_type_graines"], 4),
                        " | ".join(c.get("alertes_libelles", [])), c.get("systeme_lisible", ""), c.get("source_seuils", ""),
                        nombre_fr(c["seuils"]["seuil_bas"], 4, signe=True), nombre_fr(c["seuils"]["seuil_haut"], 4, signe=True),
                        c.get("empreinte_image", ""), c["dossier"]])
    _remplacer(temporaire, chemin)
    return chemin

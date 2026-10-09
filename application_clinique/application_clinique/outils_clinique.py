"""Outils communs à l'application clinique (application_clinique.py) et au notebook 15.

Ce module regroupe, en un seul endroit, tout ce qui doit rester IDENTIQUE entre la simulation du notebook 15 et
l'application : la règle de tri en trois zones, le calcul des seuils, les alertes de sécurité et la préparation des
images (v3 et v4, copie exacte des notebooks 12 et 16).

PROTOTYPE DE RECHERCHE. Ne constitue pas un dispositif médical et ne remplace pas la lecture d'une radiographie par un
médecin.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage

# --------------------------------------------------------------------------------------------------------------
# Constantes identiques aux notebooks 12 à 16
# --------------------------------------------------------------------------------------------------------------
TAILLE = 224
TAILLE_CARTE = 28
CLASSES = {"NORMAL": 0, "PNEUMONIA": 1}
NOMS_CLASSES = {0: "NORMAL", 1: "PNEUMONIA"}
GRAINES = [42, 123, 2026]
CONFIGURATIONS = {
    "CNN_classique": {"indice": 2, "lr": 0.001, "dropout": 0.5},
    "CNN_CBAM": {"indice": 3, "lr": 0.0005, "dropout": 0.3},
}
V3, V4 = "v3 sans raccourci", "v4 recadrage poumons"
PRETRAITEMENTS = {
    V3: "L; resize 224x224 bilineaire sans padding (etirement); ImageOps.equalize; /255; aucune augmentation",
    V4: ("L; recadrage sur la boite des deux poumons (PSPNet ChestX-Det de torchxrayvision, seuil 0.5, plus grande composante "
         "et trous bouches par poumon, marge 10 %, image entiere si masque invalide); resize 224x224 bilineaire (etirement); "
         "ImageOps.equalize; /255; aucune augmentation"),
}
ARCHITECTURES = {
    "CNN_classique": ("CNNClassique_89026_v1", 89026),
    "CNN_CBAM": ("CNNAvecCBAM_89636_apres_bloc3_r16_k7_residuel_v2", 89636),
}

# Segmentation pulmonaire (notebook 16)
RESOLUTION_SEGMENTATION = 512
TAILLE_MASQUE = 256
SEUIL_MASQUE = 0.5
AIRE_MIN_POUMON = 0.02
AIRE_MAX_POUMONS = 0.70
ECART_MIN_CENTRES = 0.10
MARGE_RECADRAGE = 0.10
POIDS_SEGMENTEUR = {"fichier": "pspnet_chestxray_best_model_4.pth", "octets": 272988989,
                    "sha256": "019b167eac6b729fc1bb92bbbc185fc1730aaa65819f4e3fe718186cadc044fc"}
STRUCTURE = np.ones((3, 3), dtype=bool)

# Règle de tri (notebook 15)
CIBLE_SENSIBILITE_VERTE = 0.98     # au plus 2 % des radios PNEUMONIA tombent en zone verte
CIBLE_SPECIFICITE_ROUGE = 0.90     # au plus 10 % des radios NORMAL tombent en zone rouge
ZONES = ("verte", "orange", "rouge")

AVERTISSEMENT = ("Prototype de recherche, non validé cliniquement. Cet outil aide à ordonner la lecture des radiographies ; "
                 "il ne pose aucun diagnostic et ne remplace pas la lecture par un médecin.")


# --------------------------------------------------------------------------------------------------------------
# Modèles (copie exacte de l'architecture des notebooks 05 à 16)
# --------------------------------------------------------------------------------------------------------------
def _definir_modeles():
    import torch
    from torch import nn

    class CNNClassique(nn.Module):
        def __init__(self, dropout=0.3):
            super().__init__()
            self.bloc1 = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
            self.bloc2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
            self.bloc3 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
            self.reduction = nn.AdaptiveAvgPool2d((4, 4))
            self.classificateur = nn.Sequential(nn.Flatten(), nn.Linear(64 * 4 * 4, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 2))

        def forward(self, x):
            x = self.bloc3(self.bloc2(self.bloc1(x)))
            return self.classificateur(self.reduction(x))

    class AttentionCanaux(nn.Module):
        def __init__(self, canaux, reduction=16):
            super().__init__()
            dim = max(1, canaux // reduction)
            self.reseau_partage = nn.Sequential(nn.Conv2d(canaux, dim, 1, bias=False), nn.ReLU(), nn.Conv2d(dim, canaux, 1, bias=False))

        def forward(self, x):
            moyenne = x.mean(dim=(2, 3), keepdim=True)
            maximum = x.amax(dim=(2, 3), keepdim=True)
            return x * torch.sigmoid(self.reseau_partage(moyenne) + self.reseau_partage(maximum))

    class AttentionSpatiale(nn.Module):
        def __init__(self):
            super().__init__()
            self.convolution = nn.Conv2d(2, 1, 7, padding=3, bias=False)

        def forward(self, x):
            resume = torch.cat([x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1)
            return x * torch.sigmoid(self.convolution(resume))

    class CBAM(nn.Module):
        def __init__(self, canaux=64, reduction=16):
            super().__init__()
            self.attention_canaux = AttentionCanaux(canaux, reduction)
            self.attention_spatiale = AttentionSpatiale()

        def forward(self, x):
            return x + self.attention_spatiale(self.attention_canaux(x))   # connexion résiduelle

    class CNNAvecCBAM(CNNClassique):
        def __init__(self, dropout=0.3):
            super().__init__(dropout)
            self.cbam = CBAM(64, 16)

        def forward(self, x):
            x = self.bloc3(self.bloc2(self.bloc1(x)))
            return self.classificateur(self.reduction(self.cbam(x)))

    return {"CNN_classique": CNNClassique, "CNN_CBAM": CNNAvecCBAM}


_CLASSES_MODELES = {}


def classes_modeles():
    if not _CLASSES_MODELES:
        _CLASSES_MODELES.update(_definir_modeles())
    return _CLASSES_MODELES


def dossier_modele(projet: Path, version: str, nom: str, graine: int) -> Path:
    indice = CONFIGURATIONS[nom]["indice"]
    if version == V3:
        base = projet / "optimisation" / "v3_sans_raccourci_format"
    elif version == V4:
        base = projet / "optimisation" / "poumons_v4" / "modeles_v4"
    else:
        raise ValueError(f"Version non prise en charge par l'application : {version}")
    return base / nom / f"graine_{graine}" / f"configuration_{indice:02d}"


def charger_modele(chemin: Path, version: str, nom: str, graine: int):
    """Charge un modèle sauvegardé par les notebooks 12 ou 16, avec les mêmes contrôles que le notebook 16."""
    import torch
    classe = classes_modeles()[nom]
    architecture, nombre_parametres = ARCHITECTURES[nom]
    etat = torch.load(chemin, map_location="cpu", weights_only=True)
    protocole = etat["protocole"]
    if protocole["architecture"] != architecture or protocole["graine"] != graine:
        raise RuntimeError(f"Sauvegarde inattendue (architecture ou graine) : {chemin}")
    if protocole["classes"] != CLASSES or protocole["pretraitement"] != PRETRAITEMENTS[version]:
        raise RuntimeError(f"Classes ou préparation incompatibles avec {version} : {chemin}")
    if etat["configuration"] != CONFIGURATIONS[nom]:
        raise RuntimeError(f"Sauvegarde d'une autre configuration : {chemin}")
    modele = classe(dropout=CONFIGURATIONS[nom]["dropout"])
    modele.load_state_dict(etat["modele"], strict=True)
    if sum(p.numel() for p in modele.parameters()) != nombre_parametres:
        raise RuntimeError("Nombre de paramètres inattendu.")
    modele.eval()
    modele.requires_grad_(False)
    return modele


def empreinte_fichier(chemin) -> str:
    h = hashlib.sha256()
    with open(chemin, "rb") as fichier:
        for bloc in iter(lambda: fichier.read(1024 * 1024), b""):
            h.update(bloc)
    return h.hexdigest()


# --------------------------------------------------------------------------------------------------------------
# Préparation des images (identique aux notebooks 12 et 16)
# --------------------------------------------------------------------------------------------------------------
def etirer(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("L").resize((TAILLE, TAILLE), Image.Resampling.BILINEAR), dtype=np.uint8)


def egaliser(tableau_u8: np.ndarray) -> np.ndarray:
    return np.array(ImageOps.equalize(Image.fromarray(np.ascontiguousarray(tableau_u8, dtype=np.uint8))), dtype=np.uint8)


def preparer_v3(image: Image.Image) -> np.ndarray:
    """Entrée v3 (uint8, 224 × 224) : étirement puis égalisation d'histogramme."""
    return egaliser(etirer(image))


def preparer_v4(image: Image.Image, boite) -> np.ndarray:
    """Entrée v4 (uint8, 224 × 224) : boîte des poumons (parts de l'image) étirée en 224 × 224, puis égalisation."""
    gris = image.convert("L")
    largeur, hauteur = gris.size
    x0, y0, x1, y1 = boite
    recadree = gris.resize((TAILLE, TAILLE), Image.Resampling.BILINEAR, box=(x0 * largeur, y0 * hauteur, x1 * largeur, y1 * hauteur))
    return np.array(ImageOps.equalize(recadree), dtype=np.uint8)


def vers_reseau(entree_u8: np.ndarray) -> np.ndarray:
    """Lot numpy (1, 1, 224, 224) en float32 dans [0, 1]."""
    return (entree_u8.astype(np.float32) / 255.0)[None, None]


# --------------------------------------------------------------------------------------------------------------
# Segmentation pulmonaire (copie exacte des fonctions du notebook 16)
# --------------------------------------------------------------------------------------------------------------
class Segmenteur:
    """PSPNet ChestX-Det de TorchXRayVision. Les poids sont contrôlés par SHA-256 (téléchargés au premier usage)."""

    def __init__(self):
        import torch
        import torchxrayvision as xrv
        warnings.filterwarnings("ignore", message=r".*upsample.*")
        warnings.filterwarnings("ignore", message=r".*align_corners.*")
        chemin_poids = Path(xrv.utils.get_cache_dir()) / POIDS_SEGMENTEUR["fichier"]
        if chemin_poids.is_file() and chemin_poids.stat().st_size != POIDS_SEGMENTEUR["octets"]:
            chemin_poids.unlink()
        with torch.random.fork_rng(devices=[]):
            self.modele = xrv.baseline_models.chestx_det.PSPNet()
        self.modele.eval()
        self.modele.requires_grad_(False)
        if empreinte_fichier(chemin_poids) != POIDS_SEGMENTEUR["sha256"]:
            raise RuntimeError(f"Poids du segmenteur différents de ceux prévus : supprimer {chemin_poids} puis relancer.")
        cibles = list(self.modele.targets)
        self.indices = [cibles.index("Right Lung"), cibles.index("Left Lung")]
        self.torch = torch

    def probabilites(self, x):
        """x : (B, 512, 512) dans [-1024, 1024] → (B, 2, 512, 512) : poumon droit, poumon gauche du patient."""
        with self.torch.inference_mode():
            sortie = self.modele(self.torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[:, None])
            return self.torch.sigmoid(sortie[:, self.indices]).float().numpy()


def boite_contenu(taille, cote):
    blanc = ImageOps.pad(Image.new("L", taille, 255), (cote, cote), method=Image.Resampling.NEAREST, color=0, centering=(0.5, 0.5))
    return blanc.getbbox()


def entree_segmentation(image: Image.Image):
    gris = image.convert("L")
    carree = ImageOps.pad(gris, (RESOLUTION_SEGMENTATION, RESOLUTION_SEGMENTATION), method=Image.Resampling.BILINEAR,
                          color=0, centering=(0.5, 0.5))
    x = np.asarray(carree, dtype=np.float32) / 255.0 * 2048.0 - 1024.0
    return x, boite_contenu(gris.size, RESOLUTION_SEGMENTATION)


def geometrie_relative(probabilite, boite) -> np.ndarray:
    carte = Image.fromarray(np.ascontiguousarray(probabilite, dtype=np.float32))
    reduite = carte.resize((TAILLE_MASQUE, TAILLE_MASQUE), Image.Resampling.BILINEAR, box=tuple(float(v) for v in boite))
    return np.rint(np.clip(np.asarray(reduite, dtype=np.float32), 0.0, 1.0) * 255.0).astype(np.uint8)


def nettoyer(probabilite_u8):
    binaire = probabilite_u8 >= 128
    etiquettes, nombre = ndimage.label(binaire, structure=STRUCTURE)
    if nombre == 0:
        return binaire, 0, np.nan
    tailles = np.bincount(etiquettes.ravel())[1:]
    principale = etiquettes == int(np.argmax(tailles)) + 1
    return ndimage.binary_fill_holes(principale), int(nombre), float(tailles.max() / tailles.sum())


def qualite_masque(droit_u8, gauche_u8):
    droit, n_droit, principale_droit = nettoyer(droit_u8)
    gauche, n_gauche, principale_gauche = nettoyer(gauche_u8)
    zone = droit | gauche
    surface = float(TAILLE_MASQUE * TAILLE_MASQUE)
    centres = (np.arange(TAILLE_MASQUE) + 0.5) / TAILLE_MASQUE

    def centre(masque, axe):
        return float(centres[np.nonzero(masque)[axe]].mean()) if masque.any() else np.nan

    ligne = {"aire_poumon_droit": float(droit.sum()) / surface, "aire_poumon_gauche": float(gauche.sum()) / surface,
             "aire_poumons": float(zone.sum()) / surface, "composantes_droit": n_droit, "composantes_gauche": n_gauche,
             "part_principale_droit": principale_droit, "part_principale_gauche": principale_gauche,
             "centre_x_droit": centre(droit, 1), "centre_y_droit": centre(droit, 0),
             "centre_x_gauche": centre(gauche, 1), "centre_y_gauche": centre(gauche, 0)}
    if zone.any():
        lignes, colonnes = np.flatnonzero(zone.any(axis=1)), np.flatnonzero(zone.any(axis=0))
        ligne.update({"poumons_x0": colonnes[0] / TAILLE_MASQUE, "poumons_x1": (colonnes[-1] + 1) / TAILLE_MASQUE,
                      "poumons_y0": lignes[0] / TAILLE_MASQUE, "poumons_y1": (lignes[-1] + 1) / TAILLE_MASQUE})
    else:
        ligne.update({"poumons_x0": np.nan, "poumons_x1": np.nan, "poumons_y0": np.nan, "poumons_y1": np.nan})
    return ligne, zone


def masque_valide(ligne: dict, sens_lateralite: float = 1.0):
    """Même règle de validité que le notebook 16, pour une seule radio. Renvoie (valide, raisons)."""
    ecart = (ligne["centre_x_gauche"] - ligne["centre_x_droit"]) * sens_lateralite
    presents = ligne["aire_poumon_droit"] >= AIRE_MIN_POUMON and ligne["aire_poumon_gauche"] >= AIRE_MIN_POUMON
    raisons = []
    if ligne["aire_poumon_droit"] < AIRE_MIN_POUMON:
        raisons.append("poumon droit absent ou trop petit")
    if ligne["aire_poumon_gauche"] < AIRE_MIN_POUMON:
        raisons.append("poumon gauche absent ou trop petit")
    if ligne["aire_poumons"] > AIRE_MAX_POUMONS:
        raisons.append("poumons trop étendus")
    if presents and not (ecart >= ECART_MIN_CENTRES):
        raisons.append("centres inversés ou trop proches")
    return (len(raisons) == 0), raisons


def boite_recadrage(ligne: dict, valide: bool):
    """Boîte v4 (x0, y0, x1, y1) en parts de l'image, et indication d'une boîte limitée par le bord de l'image."""
    if not valide:
        return 0.0, 0.0, 1.0, 1.0, False
    x0, x1, y0, y1 = ligne["poumons_x0"], ligne["poumons_x1"], ligne["poumons_y0"], ligne["poumons_y1"]
    mx, my = MARGE_RECADRAGE * (x1 - x0), MARGE_RECADRAGE * (y1 - y0)
    boite = (x0 - mx, y0 - my, x1 + mx, y1 + my)
    touche = boite[0] < 0 or boite[1] < 0 or boite[2] > 1 or boite[3] > 1
    return max(0.0, boite[0]), max(0.0, boite[1]), min(1.0, boite[2]), min(1.0, boite[3]), bool(touche)


def localiser_poumons(image: Image.Image, segmenteur: Segmenteur, sens_lateralite: float = 1.0) -> dict:
    """Segmente une radio. Renvoie masque nettoyé (256 × 256, géométrie de l'image entière), validité, raisons et boîte v4."""
    x, boite_image = entree_segmentation(image)
    p = segmenteur.probabilites(x[None])[0]
    droit, gauche = geometrie_relative(p[0], boite_image), geometrie_relative(p[1], boite_image)
    ligne, zone = qualite_masque(droit, gauche)
    valide, raisons = masque_valide(ligne, sens_lateralite)
    x0, y0, x1, y1, touche = boite_recadrage(ligne, valide)
    return {"zone": zone, "valide": valide, "raisons": raisons, "boite": (x0, y0, x1, y1), "aire_poumons": ligne["aire_poumons"]}


def redimensionner_masque(masque, taille, boite=None):
    """Masque booléen vers une autre géométrie : interpolation bilinéaire de l'image 0/255, puis seuil à 128 (notebook 16)."""
    image = Image.fromarray(masque.astype(np.uint8) * 255)
    return np.asarray(image.resize(taille, Image.Resampling.BILINEAR, box=boite)) >= 128


def zone_dans_entree_v4(zone_relative, boite):
    """Zone pulmonaire (256 × 256, image entière) placée dans l'entrée v4 (224 × 224, boîte de recadrage)."""
    x0, y0, x1, y1 = boite
    return redimensionner_masque(zone_relative, (TAILLE, TAILLE),
                                 (x0 * TAILLE_MASQUE, y0 * TAILLE_MASQUE, x1 * TAILLE_MASQUE, y1 * TAILLE_MASQUE))


# --------------------------------------------------------------------------------------------------------------
# Scores et cartes (Grad-CAM comme aux notebooks 11, 13 et 16)
# --------------------------------------------------------------------------------------------------------------
class _Capture:
    def __init__(self, module):
        self.sortie = None
        self.poignee = module.register_forward_hook(self._enregistrer)

    def _enregistrer(self, module, entree, sortie):
        self.sortie = sortie

    def retirer(self):
        self.poignee.remove()


def analyser_modele(nom: str, modele, x: np.ndarray):
    """x : (1, 1, 224, 224). Renvoie score (logit PNEUMONIA − logit NORMAL), carte de décision 224 × 224 (0 à 1) et attention spatiale
    du CBAM 224 × 224 (ou None)."""
    import torch
    import torch.nn.functional as F
    couche = modele.cbam if nom == "CNN_CBAM" else modele.bloc3
    capture = _Capture(couche)
    capture_spatiale = _Capture(modele.cbam.attention_spatiale.convolution) if nom == "CNN_CBAM" else None
    try:
        entree = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).requires_grad_(True)
        with torch.enable_grad():
            logits = modele(entree)
            activation = capture.sortie
            difference = logits[:, 1] - logits[:, 0]
            (gradient,) = torch.autograd.grad(difference.sum(), activation)
        poids = gradient.mean(dim=(2, 3), keepdim=True)
        carte_signee = (poids * activation.detach()).sum(dim=1)
        spatiale = torch.sigmoid(capture_spatiale.sortie.detach())[:, 0] if capture_spatiale is not None else None
    finally:
        capture.retirer()
        if capture_spatiale is not None:
            capture_spatiale.retirer()
    score = float((logits[0, 1] - logits[0, 0]).detach())
    prediction = score > 0
    choisie = torch.clamp(carte_signee if prediction else -carte_signee, min=0.0)

    def agrandir(t):
        return F.interpolate(t.unsqueeze(1).float(), size=(TAILLE, TAILLE), mode="bilinear", align_corners=False)[0, 0].numpy()

    decision = agrandir(choisie)
    decision = decision / decision.max() if decision.max() > 0 else np.zeros_like(decision)
    return score, decision, (agrandir(spatiale) if spatiale is not None else None)


# --------------------------------------------------------------------------------------------------------------
# Règle de tri en trois zones (notebook 15)
# --------------------------------------------------------------------------------------------------------------
def seuils_depuis_calibration(scores_normal, scores_pneumonie, cible_sensibilite=CIBLE_SENSIBILITE_VERTE,
                              cible_specificite=CIBLE_SPECIFICITE_ROUGE) -> dict:
    """Deux seuils sur le score (logit PNEUMONIA − logit NORMAL), fixés sur un échantillon étiqueté.

    - seuil_bas : zone verte = score < seuil_bas. Au moins `cible_sensibilite` des PNEUMONIA de l'échantillon ont un score ≥ seuil_bas.
    - seuil_haut : zone rouge = score > seuil_haut. Au moins `cible_specificite` des NORMAL de l'échantillon ont un score ≤ seuil_haut.
    Si les deux seuils se croisent, ils sont ramenés à leur milieu (plus de zone orange) et `zones_chevauchantes` vaut True.
    """
    normal = np.asarray(scores_normal, dtype=np.float64)
    pneumonie = np.asarray(scores_pneumonie, dtype=np.float64)
    if len(normal) == 0 or len(pneumonie) == 0:
        raise ValueError("Il faut au moins une radio de chaque classe pour fixer les seuils.")
    bas = float(np.quantile(pneumonie, 1.0 - cible_sensibilite, method="lower"))
    haut = float(np.quantile(normal, cible_specificite, method="higher"))
    croisement = bas > haut
    if croisement:
        bas = haut = 0.5 * (bas + haut)
    return {"seuil_bas": bas, "seuil_haut": haut, "zones_chevauchantes": bool(croisement),
            "n_normal": int(len(normal)), "n_pneumonie": int(len(pneumonie)),
            "cible_sensibilite_verte": cible_sensibilite, "cible_specificite_rouge": cible_specificite}


def zone_de(score, seuil_bas: float, seuil_haut: float):
    """0 = verte (score < seuil_bas), 2 = rouge (score > seuil_haut), 1 = orange sinon. Accepte un scalaire ou un tableau."""
    score = np.asarray(score, dtype=np.float64)
    zone = np.where(score < seuil_bas, 0, np.where(score > seuil_haut, 2, 1))
    return int(zone) if zone.ndim == 0 else zone


def ecart_type_graines(scores):
    """Écart-type (sans correction, ddof = 0) des scores des trois graines, sur le dernier axe."""
    return np.std(np.asarray(scores, dtype=np.float64), axis=-1)


def zone_finale(zone, alerte):
    """Règle de sécurité : une alerte ne permet jamais une zone verte (rassurante). Verte + alerte → orange.
    Accepte des scalaires ou des tableaux (alerte booléenne)."""
    zone = np.asarray(zone)
    alerte = np.asarray(alerte, dtype=bool)
    resultat = np.where((zone == 0) & alerte, 1, zone)
    return int(resultat) if resultat.ndim == 0 else resultat


# --------------------------------------------------------------------------------------------------------------
# Alertes de sécurité (paramètres fixés par le notebook 15 à partir de l'apprentissage et de la validation)
# --------------------------------------------------------------------------------------------------------------
def statistiques_alerte(image: Image.Image) -> dict:
    """Statistiques simples d'une radio, calculées de la même façon pour l'apprentissage (notebook 15) et l'application."""
    largeur, hauteur = image.size
    gris = image.convert("L")
    gris.thumbnail((256, 256))
    a = np.asarray(gris, dtype=np.float64) / 255.0
    colonnes, lignes = a.mean(axis=0), a.mean(axis=1)
    return {"largeur": int(largeur), "hauteur": int(hauteur), "rapport": largeur / hauteur, "cote_min": int(min(largeur, hauteur)),
            "moyenne": float(a.mean()), "ecart_type": float(a.std()),
            "part_bandes_noires": float(max(np.mean(colonnes < 0.03), np.mean(lignes < 0.03)))}


LIBELLES_ALERTES = {
    "format": "Format d'image inhabituel (proportions hors de la plage vue à l'apprentissage) : le modèle peut s'appuyer sur le format.",
    "resolution": "Image de très petite taille par rapport à l'apprentissage.",
    "luminosite": "Luminosité moyenne hors de la plage vue à l'apprentissage.",
    "contraste": "Contraste hors de la plage vue à l'apprentissage.",
    "bandes_noires": "Bandes noires importantes autour de l'image (capture d'écran, cadrage inhabituel).",
    "graines": "Les trois modèles se contredisent nettement : score peu fiable.",
    "poumons": "Poumons non localisés automatiquement : image entière utilisée (version v4).",
}


def alertes_image(stats: dict, parametres: dict) -> list[str]:
    """Liste des alertes déclenchées par les statistiques d'une image (hors désaccord entre graines et localisation)."""
    p, sortie = parametres, []
    if not (p["rapport_min"] <= stats["rapport"] <= p["rapport_max"]):
        sortie.append("format")
    if stats["cote_min"] < p["cote_min_min"]:
        sortie.append("resolution")
    if not (p["moyenne_min"] <= stats["moyenne"] <= p["moyenne_max"]):
        sortie.append("luminosite")
    if not (p["ecart_type_min"] <= stats["ecart_type"] <= p["ecart_type_max"]):
        sortie.append("contraste")
    if stats["part_bandes_noires"] > p["part_bandes_noires_max"]:
        sortie.append("bandes_noires")
    return sortie


def parametres_alertes_depuis_apprentissage(tableau) -> dict:
    """Plages à 1 %–99 % des statistiques de l'apprentissage (tableau pandas issu de statistiques_alerte)."""
    q = lambda colonne, p: float(np.quantile(tableau[colonne].to_numpy(dtype=float), p))
    return {"rapport_min": q("rapport", 0.01), "rapport_max": q("rapport", 0.99), "cote_min_min": q("cote_min", 0.01),
            "moyenne_min": q("moyenne", 0.01), "moyenne_max": q("moyenne", 0.99),
            "ecart_type_min": q("ecart_type", 0.01), "ecart_type_max": q("ecart_type", 0.99),
            "part_bandes_noires_max": q("part_bandes_noires", 0.99), "n_images_apprentissage": int(len(tableau))}


def charger_json(chemin: Path):
    return json.loads(Path(chemin).read_text(encoding="utf-8"))

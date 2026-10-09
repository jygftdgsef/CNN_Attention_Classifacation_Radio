# Application d'aide au tri des radiographies pulmonaires

**Prototype de recherche, non validé cliniquement.** L'outil aide à ordonner la lecture de radiographies thoraciques de face. Il ne pose aucun diagnostic, ne remplace pas la lecture par un médecin et n'est pas un dispositif médical. Tout se passe sur cet ordinateur : aucune image ne part sur internet.

## Ce que l'outil fait

Pour une radiographie, trois modèles (les trois graines 42, 123 et 2026 du même système) calculent chacun un score : logit PNEUMONIA − logit NORMAL. La moyenne des trois scores place la radio dans une zone :

| Zone | Signification | Conduite proposée |
|---|---|---|
| Verte | score sous le seuil bas | lecture en file standard |
| Orange | entre les deux seuils, ou image hors du cadre validé | lecture médicale sans priorité particulière |
| Rouge | score au-dessus du seuil haut | lecture prioritaire |

**Règle de sécurité.** Si une alerte se déclenche, une radio classée verte passe en orange. Alertes : format inhabituel, image de petite taille, luminosité ou contraste hors plage, bandes noires, poumons non localisés (version v4), désaccord net entre les trois modèles. Les plages viennent des radios d'apprentissage (1 % à 99 %).

**Seuils.** Le seuil bas est fixé pour que 98 % des radios PNEUMONIA de l'échantillon de réglage soient au-dessus. Le seuil haut est fixé pour que 90 % des radios NORMAL soient en dessous. Les notebooks 14 et 15 montrent que les seuils réglés sur la validation ne se transportent pas d'un lot à l'autre : **sans calibration locale, l'outil affiche un bandeau rouge « seuils du développement, non calibrés localement ».**

## Installation (une seule fois)

1. Copier ce dossier `application_clinique` dans le dossier du projet (à côté de `optimisation`, `repartitions`, `archive`).
2. Ouvrir et exécuter **le notebook 15**. Il écrit `tri_parametres.json` dans ce dossier (seuils du développement, plages des alertes).
3. Installer l'interface Gradio **une fois le notebook 16 terminé**, dans une cellule d'un notebook ouvert avec le noyau `memoire_cnn`. La commande impose à pip les versions déjà installées : les préparations d'images et les protocoles des notebooks restent identiques.

   ```python
   import subprocess, sys
   from importlib import metadata
   garder = []
   for nom in ["torch", "torchvision", "torchxrayvision", "numpy", "pandas", "pillow", "scipy", "scikit-learn", "matplotlib"]:
       try:
           garder.append(f"{nom}=={metadata.version(nom)}")
       except metadata.PackageNotFoundError:
           pass
   resultat = subprocess.run([sys.executable, "-m", "pip", "install", "gradio", *garder], capture_output=True, text=True)
   print(resultat.stdout[-1500:], resultat.stderr[-1500:])
   ```

4. Pour la version v4 seulement : le notebook 16 doit être terminé (modèles v4 et `termine.json`), et la bibliothèque `torchxrayvision` installée (le notebook 16 l'installe).

Modèles utilisés : v3 = notebook 12, v4 = notebook 16. Le dossier du projet est déduit de l'emplacement du fichier ; sinon, ajouter `--projet "C:\chemin\vers\le\projet"`.

## Lancer

### Depuis un notebook (le plus simple)

Dans une cellule d'un notebook ouvert avec le noyau `memoire_cnn` :

```python
import subprocess, sys
from pathlib import Path
APP = Path(r"C:\Users\David KUNYIMA\OneDrive\Desktop\Memoire_CNN_Attention\application_clinique")
journal = open(APP / "application.log", "w", encoding="utf-8")
application = subprocess.Popen([sys.executable, str(APP / "application_clinique.py"), "--systeme", "v3:CNN_CBAM"],
                               cwd=APP, stdout=journal, stderr=subprocess.STDOUT)
```

Le navigateur s'ouvre sur l'application après quelques secondes (sinon : `http://127.0.0.1:7860`). En cas de problème, lire `application.log` dans ce dossier. Pour arrêter l'application : `application.terminate()` dans une autre cellule.

Les commandes `analyser` et `calibrer` se lancent de la même façon, avec `subprocess.run` :

```python
resultat = subprocess.run([sys.executable, str(APP / "application_clinique.py"), "calibrer", "--csv", r"C:\chemin\echantillon.csv",
                           "--images", r"C:\chemin\dossier_images", "--systeme", "v3:CNN_CBAM"],
                          cwd=APP, capture_output=True, text=True, encoding="utf-8")
print(resultat.stdout[-4000:], resultat.stderr[-4000:])
```

### Depuis un terminal

Dans un terminal où l'environnement du noyau `memoire_cnn` est actif, depuis ce dossier :

```
python application_clinique.py
```

L'application s'ouvre dans le navigateur, à une adresse locale (127.0.0.1), non accessible depuis un autre ordinateur. Gradio affiche « To create a public link, set `share=True` » : **ne pas le faire**, les radios passeraient alors par un serveur externe. Options :

| Option | Effet |
|---|---|
| `--systeme v3:CNN_CBAM` (défaut) | système utilisé : `v3` ou `v4`, suivi de `CNN_CBAM` ou `CNN_classique` |
| `--sans-journal` | ne rien écrire dans le journal |
| `--sans-navigateur` | ne pas ouvrir le navigateur automatiquement |
| `--port 7860` | numéro de port |

Sans navigateur, une seule image :

```
python application_clinique.py analyser --image radio.jpeg --systeme v3:CNN_CBAM
```

## Calibrer localement (indispensable avant tout usage réel)

Préparer un CSV de deux colonnes, `fichier` (chemin de l'image, relatif au dossier d'images) et `classe` (`NORMAL` ou `PNEUMONIA`) :

```
fichier,classe
patient001.jpeg,NORMAL
patient002.jpeg,PNEUMONIA
```

Puis :

```
python application_clinique.py calibrer --csv echantillon.csv --images dossier_images --systeme v3:CNN_CBAM
```

La commande écrit `seuils_locaux_v3_CNN_CBAM.json` (un fichier par système). L'application l'utilise au lancement suivant pour ce système et affiche « Seuils calibrés localement ». Les intervalles à 95 % (500 tirages) et les avertissements sont dans le fichier.

À respecter pour que la calibration ait un sens :
- radios **du centre qui utilisera l'outil**, prises comme en pratique ; **une radio par patient** ; étiquettes fixées par un médecin (jamais par l'outil) ;
- **100 radios par classe au minimum, 200 de préférence** : avec 100 à 200 PNEUMONIA, le seuil bas repose déjà sur les 2 à 4 radios les moins bien notées, et il bouge beaucoup en dessous (refus sous 10 radios par classe) ;
- garder un autre échantillon, jamais utilisé pour le réglage, pour mesurer ensuite la part de PNEUMONIA en zone verte et de NORMAL en zone rouge.

## Journal

`journal_analyses.csv` : date, empreinte de l'image (calculée sur une version 64 × 64 : elle ne permet pas de retrouver l'image), système, origine et valeurs des seuils, score, écart entre modèles, zone avant et après la règle de sécurité, alertes. Ni image, ni nom de fichier, ni identité. Il sert à suivre les alertes et la répartition des zones dans le temps. `--sans-journal` le désactive.

## Limites à garder en tête

- Modèles entraînés sur un seul jeu de données de radiographies pédiatriques ; aucun test dans un autre centre.
- Les notebooks 11 à 13 montrent que les modèles s'appuient en partie sur autre chose que le poumon (coins, marqueurs, format). Les cartes affichées montrent ce qui a pesé dans le score, pas une lésion.
- Une zone verte n'exclut pas une pneumonie ; une zone rouge n'en établit pas une.
- La localisation des poumons (v4) vient d'un segmenteur entraîné sur des adultes : elle peut échouer sur des poumons d'enfant ou très opaques. L'outil signale alors « poumons non localisés ».
- Une validation prospective dans le centre visé, avec avis d'un comité d'éthique, est nécessaire avant tout usage clinique.

## Dépannage

| Message | Action |
|---|---|
| `tri_parametres.json est introuvable` | exécuter le notebook 15 |
| `Modèle introuvable` | notebook 12 (v3) ou 16 (v4) non terminé, ou `--projet` incorrect |
| `Poids du segmenteur différents` | supprimer le fichier de poids indiqué, relancer (téléchargement contrôlé par SHA-256) |
| page qui ne s'ouvre pas | ouvrir à la main `http://127.0.0.1:7860`, ou changer de `--port` |

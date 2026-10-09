# Application d'aide au tri des radiographies pulmonaires (version 2)

**Prototype de recherche, non validé cliniquement.** L'outil aide à ordonner la lecture de radiographies thoraciques de face. Il ne pose aucun diagnostic, ne remplace pas la lecture par un médecin et n'est pas un dispositif médical. L'application n'envoie aucune image sur internet.

## Nouveautés de la version 2

- **Système par défaut : v4, recadrage sur les poumons, CNN + CBAM** (notebook 16).
- **Enregistrer une analyse** : rapport PDF d'une page, radio, cartes et données, dans le dossier `analyses`.
- **Onglet Historique** : liste des analyses enregistrées, filtre par zone, recherche par référence, réouverture d'une analyse, téléchargement de son rapport, export en CSV pour Excel, suppression.
- **Fiche lisible en thème clair comme sombre**, nombres à la française (−2,40), nom du système en clair, flèche et valeur quand le score sort de l'échelle affichée.
- **Cartes avec leur échelle de couleur.** L'attention du CBAM est montrée sur son échelle réelle (0 à 1), sans étirement : une carte presque uniforme veut dire que le CBAM pondère peu l'espace (notebook 16).
- **Légende des cartes adaptée à la radio**, avec un avertissement quand le modèle penche vers NORMAL : la carte est alors peu informative.

Les règles de tri (seuils, zones, alertes, règle de sécurité) **n'ont pas changé** : elles restent dans `outils_clinique.py`, partagé avec le notebook 15. Les résultats du notebook 15 décrivent donc toujours le comportement de l'application.

## Ce que l'outil fait

Pour une radiographie, trois modèles (les trois graines 42, 123 et 2026 du même système) calculent chacun un score : logit PNEUMONIA − logit NORMAL. La moyenne des trois scores place la radio dans une zone :

| Zone | Signification | Conduite proposée |
|---|---|---|
| Verte | score sous le seuil bas | lecture en file standard |
| Orange | entre les deux seuils, ou image hors du cadre validé | lecture médicale sans priorité particulière |
| Rouge | score au-dessus du seuil haut | lecture prioritaire |

**Règle de sécurité.** Si une alerte se déclenche, une radio classée verte passe en orange. Alertes : format inhabituel, image de petite taille, luminosité ou contraste hors plage, bandes noires, poumons non localisés (version v4), désaccord net entre les trois modèles. Les plages viennent des radios d'apprentissage (1 % à 99 %).

**Seuils.** Le seuil bas est fixé pour que 98 % des radios PNEUMONIA de l'échantillon de réglage soient au-dessus. Le seuil haut est fixé pour que 90 % des radios NORMAL soient en dessous. Les notebooks 14 et 15 montrent que les seuils réglés sur la validation ne se transportent pas d'un lot à l'autre : **sans calibration locale, l'outil affiche un bandeau rouge « seuils du développement, non calibrés localement ».**

## Fichiers

| Fichier | Rôle |
|---|---|
| `application_clinique.py` | l'application (interface, commandes `analyser` et `calibrer`) |
| `outils_clinique.py` | modèles, préparation des images, règles de tri : partagé avec le notebook 15, **inchangé** |
| `rapports_clinique.py` | nouveau : fiche, cartes avec échelle, rapport PDF, sauvegarde et historique |
| `lancer_application.ipynb` | notebook de lancement (installation de Gradio, démarrage, arrêt) |
| `tri_parametres.json` | écrit par le notebook 15 (seuils du développement, plages des alertes) |

## Lancer

Le plus simple : ouvrir **`lancer_application.ipynb`** avec le noyau `memoire_cnn` et exécuter les cellules dans l'ordre (installation de Gradio 6.29.0 une seule fois, lancement, arrêt). Le navigateur s'ouvre sur `http://127.0.0.1:7860`, une adresse locale, inaccessible depuis un autre ordinateur.

Depuis un terminal où l'environnement du noyau `memoire_cnn` est actif, dans ce dossier :

```
python application_clinique.py
```

| Option | Effet |
|---|---|
| `--systeme v4:CNN_CBAM` (défaut) | système utilisé : `v4` ou `v3`, suivi de `CNN_CBAM` ou `CNN_classique` |
| `--dossier-analyses DOSSIER` | où garder les analyses enregistrées (défaut : `application_clinique/analyses`) |
| `--sans-journal` | ne rien écrire dans le journal anonyme |
| `--sans-navigateur` | ne pas ouvrir le navigateur automatiquement |
| `--port 7860` | numéro de port |

Gradio affiche « To create a public link, set `share=True` » : **ne pas le faire**, les radios passeraient alors par un serveur externe.

Sans navigateur, une seule image, avec enregistrement facultatif :

```
python application_clinique.py analyser --image radio.jpeg --enregistrer --reference EX-0142
```

## Enregistrer une analyse

1. Déposer la radio. Facultatif : saisir une **référence** (numéro d'examen, code d'étude), **jamais le nom du patient**.
2. Cliquer sur **Analyser**, puis sur **Enregistrer l'analyse**. La case « Enregistrer automatiquement chaque analyse » évite ce second clic.

Chaque analyse enregistrée devient un dossier `analyses/AAAA-MM-JJ_HH-MM-SS_référence/` :

| Fichier | Contenu |
|---|---|
| `rapport.pdf` | rapport d'une page : zone, score, seuils, alertes, radio, cartes et leur lecture |
| `radio.png` | la radiographie telle qu'analysée |
| `entree_reseau.png` | l'image préparée (224 × 224) vue par les réseaux |
| `carte_gradcam.png`, `carte_attention.png` | les cartes, avec leur échelle de couleur |
| `analyse.json` | toutes les données : horodatage, système, seuils, scores des trois modèles, zones, alertes, statistiques de l'image |

Une analyse déjà enregistrée ne l'est pas deux fois. Ces fichiers sont pratiques pour les figures du mémoire.

## Historique

L'onglet **Historique** liste les analyses enregistrées, de la plus récente à la plus ancienne, avec un résumé par zone et le nombre total d'analyses du journal anonyme (enregistrées ou non).

- **Filtrer** par zone, **rechercher** une référence (touche Entrée), **Actualiser** après un changement fait hors de l'application.
- **Cliquer sur une ligne** rouvre l'analyse : fiche, radio, cartes et rapport PDF à télécharger.
- **Exporter l'historique** produit un CSV pour Excel (séparateur « ; », virgule décimale), gardé dans le dossier `analyses`.
- **Supprimer** l'analyse ouverte efface définitivement son dossier, après confirmation.

## Protection des données

- Seules les analyses **enregistrées** gardent la radio. Le journal anonyme (`journal_analyses.csv`) ne contient ni image ni nom de fichier : une empreinte de l'image, le score, la zone et les alertes.
- **Ne jamais saisir de nom de patient** dans la référence.
- **Le dossier du projet est dans OneDrive** : les analyses enregistrées y sont donc synchronisées avec le compte Microsoft. Pour les radios publiques du jeu de données, ce n'est pas un problème. Pour de vraies radios de patients, choisir un dossier local non synchronisé avec `--dossier-analyses`, et suivre les règles de l'établissement.

## Calibrer localement (indispensable avant tout usage réel)

Préparer un CSV de deux colonnes, `fichier` (chemin de l'image, relatif au dossier d'images) et `classe` (`NORMAL` ou `PNEUMONIA`) :

```
fichier,classe
patient001.jpeg,NORMAL
patient002.jpeg,PNEUMONIA
```

Puis :

```
python application_clinique.py calibrer --csv echantillon.csv --images dossier_images --systeme v4:CNN_CBAM
```

La commande écrit `seuils_locaux_v4_CNN_CBAM.json` (un fichier par système). L'application l'utilise au lancement suivant pour ce système et affiche « seuils calibrés localement ». Les intervalles à 95 % (500 tirages) et les avertissements sont dans le fichier.

À respecter pour que la calibration ait un sens :
- radios **du centre qui utilisera l'outil**, prises comme en pratique ; **une radio par patient** ; étiquettes fixées par un médecin (jamais par l'outil) ;
- **100 radios par classe au minimum, 200 de préférence** : avec 100 à 200 PNEUMONIA, le seuil bas repose déjà sur les 2 à 4 radios les moins bien notées (refus sous 10 radios par classe) ;
- garder un autre échantillon, jamais utilisé pour le réglage, pour mesurer ensuite la part de PNEUMONIA en zone verte et de NORMAL en zone rouge.

## Limites à garder en tête

- Modèles entraînés sur un seul jeu de données de radiographies pédiatriques ; aucun test dans un autre centre.
- Au notebook 16 (v4), environ la moitié de la carte Grad-CAM tombe dans les poumons ; le modèle peut encore s'appuyer sur des repères hors des poumons (lettre « R », cadrage). Les cartes montrent ce qui a pesé dans le score, pas une lésion.
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
| « Journal anonyme non mis à jour » | `journal_analyses.csv` est ouvert dans Excel : le fermer (l'analyse, elle, reste valable) |
| « Suppression impossible » | le rapport PDF ou une image de l'analyse est ouvert dans un autre programme : le fermer, réessayer |
| analyse absente de l'historique | cliquer sur **Actualiser** ; un dossier sans `analyse.json` (enregistrement interrompu) est ignoré |

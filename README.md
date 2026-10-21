# Ai_Act_lolz 🔬

Démo live de Computer Vision pour un show sur l'AI Act européen.
La webcam analyse **un seul visage** en **100 % local**.
Rien n'est enregistré, rien n'est envoyé sur le réseau.

## ⚠️ Contexte d'usage (important)

Ce projet est pédagogique : il illustre ce que la computer vision sait
inférer en direct sur un visage, afin de **montrer au public** pourquoi
l'AI Act et le RGPD encadrent ces usages.

- ✅ Utilisation prévue : filmer **uniquement des personnes consentantes**
  (auto-consentement, ou consentement explicite des personnes identifiées
  dans `people/`).
- ❌ À ne pas faire : pointer la webcam vers d'autres personnes sans
  consentement, enregistrer, diffuser, ou déployer sans consentement
  (cf. `AI_ACT_NOTES.md`, Article 5 de l'AI Act, art. 9 du RGPD).

## Ce que la démo affiche en direct

| Caractéristique | Méthode | Fiabilité |
|---|---|---|
| Identité (nom) | DeepFace VGG-Face + dossier `people/` | Bonne (si photos de qualité) |
| Âge estimé | DeepFace | ± 5–8 ans |
| Féminité perçue / Masculinité perçue (en %) | DeepFace (probabilités brutes) | Stéréotype visuel, ≠ identité de genre |
| Émotion dominante | DeepFace | Approximatif, contesté scientifiquement |
| Ahegao (blague finale) | CNN du projet [ahegao-detector](https://github.com/Sir-Prise/ahegao-detector) (crop visage 75×75 → classifieur, seuil 0.6) | Bonne sur vraie grimace |
| Sourire | MediaPipe (largeur bouche / visage) | Bonne |
| Pose de la tête (yaw / pitch / roll) | MediaPipe + solvePnP | Bonne |
| Attention (regard + yeux ouverts) | MediaPipe | Bonne |
| Stabilité du visage | Vitesse de déplacement du cadrage | Bonne |
| Distance estimée à l'écran | Proportion du visage dans le cadre | Approximative |
| Clignements + rythme (/min) + PERCLOS | EAR (Eye Aspect Ratio) | Bonne |
| Rythme cardiaque (rPPG) | FFT sur le canal vert du front | Expérimental |

\* "Apparence perçue" : le modèle devine des stéréotypes d'apparence,
pas le genre réel de la personne. La démo affiche désormais **les deux
pourcentages bruts** (fémininité / masculinité) plutôt qu'un binaire —
montrer les chiffres rend le biais visible : la machine ne "voit" pas un
genre, elle calcule des probabilités sur des stéréotypes.

Une **fenêtre JSON parallèle** (touche `j`) projette en direct toutes les
données que la machine produit sur vous chaque seconde — l'effet est
saisissant : le visage d'un côté, les colonnes de données de l'autre.

Une **troisième fenêtre « Contrôles »** regroupe la commande de la démo :
un **slider de biais de féminité** (0 % = probabilités brutes du modèle,
100 % = tout basculé vers « féminin » — réglage live, défaut 30 %) et un
**interrupteur par info** pour l'afficher ou la masquer. Le clavier reste
fonctionnel et se synchronise avec les interrupteurs.

Le panneau de données s'adapte à la taille de la fenêtre (il ne dépasse
jamais un quart de l'image) et la vidéo **ne se déforme jamais** :
l'agrandissement de la fenêtre ajoute des bandes noires (letterbox).

Non inclus volontairement : inférence d'origine ethnique — interdite par
l'AI Act (art. 5.1.g) et scientifiquement infondée. Voir `AI_ACT_NOTES.md`.

## Installation pas à pas (Windows)

### 1. Installer Python 3.11

- Téléchargez Python 3.11.x sur https://www.python.org/downloads/
- **Important pendant l'installateur** : cochez **"Add python.exe to PATH"**
  avant de cliquer sur Install.
- Vérifiez dans PowerShell :

```powershell
python --version
```

Doit afficher `Python 3.11.x`.

### 2. Cloner le dépôt

```powershell
cd $HOME
git clone https://github.com/Massicotte/Ai_Act_lolz.git
cd Ai_Act_lolz
```

(Git : https://git-scm.com/download/win — options par défaut.)

### 3. Créer et activer le venv

```powershell
python -m venv venv
venv\Scripts\activate
```

Le prompt doit afficher `(venv)`. Si PowerShell bloque les scripts :

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
venv\Scripts\activate
```

### 4. Installer les dépendances

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

⚠️ Ne mettez **pas** à jour `mediapipe` ensuite : le projet utilise
l'API `mp.solutions`, supprimée dans les versions ≥ 0.10.21. C'est pour
cela que `requirements.txt` pinne `mediapipe==0.10.14`.

### 5. Premier lancement (téléchargement des modèles)

```powershell
python main.py
```

Au **premier lancement**, DeepFace télécharge ses modèles (~500 Mo, une
seule fois, connexion internet requise). La webcam s'ouvre : la démo
affiche le panneau d'analyse en direct.

### 6. (Optionnel) Reconnaissance des personnes

**Le plus simple — enregistrement en direct** : lancez la démo, appuyez sur
`n`, tapez le nom de la personne dans le terminal, et regardez la caméra.
La machine capture elle-même ~12 photos en 6 secondes et apprend le
visage en arrière-plan — aucun fichier à préparer.

**Ou manuellement** : créez un sous-dossier par personne dans `people/`,
du **nom de la personne**, rempli de 3–10 photos claires de son visage :

```powershell
mkdir people\nicol
# copiez ensuite quelques photos de nicol dans people\nicol\
```

```
people/
  nicol/     photo1.jpg  photo2.jpg  ...
  alice/     alice.png
```

Au démarrage, la démo apprend ces visages ; à l'écran, au lieu de
"vous" elle affiche **le nom du dossier** ("nicol", "alice").
Le dossier `people/` est ignoré par git : vos photos ne partent jamais
sur GitHub. **Consentement obligatoire** : données biométriques (RGPD
art. 9).

## Commandes

Chaque info du panneau peut être affichée ou masquée individuellement —
pratique pour construire la progression du show : vous ne montrez au
public, une par une, que les capacités de la machine. Deux façons :
par touche clavier (ci-dessous) ou par interrupteur dans la fenêtre
**Contrôles** (les deux restent synchronisés).

| Touche | Info / action |
|---|---|
| `q` | Quitter |
| `n` | Enregistrer une personne en direct (nom au terminal, capture auto) |
| `p` | Plein écran (letterbox : bandes noires, proportions conservées) |
| `j` | Afficher/masquer la fenêtre JSON brut (données produites chaque seconde) |
| `d` | Afficher/masquer la ROI front (debug rPPG) |
| `i` | Identité (nom reconnu via `people/`) |
| `a` | Âge estimé |
| `f` | Fémininité perçue (%) |
| `m` | Masculinité perçue (%) |
| `e` | Émotion dominante |
| `g` | Ahegao (%) |
| `s` | Sourire (%) |
| `t` | Pose de tête (yaw/pitch) |
| `v` | Attention (%) |
| `b` | Stabilité (%) |
| `l` | Distance estimée (cm) |
| `c` | Clignements (total + /min) |
| `y` | Yeux fermés / PERCLOS (%) |
| `h` | Rythme cardiaque rPPG (BPM, expérimental) |

Le terminal confirme chaque bascule (`Info 'age' : masquee`). Toutes les
infos restent dans le JSON brut, même masquées du panneau.

## Détecteur d'ahegao

Le détecteur reprend le CNN du projet
[ahegao-detector](https://github.com/Sir-Prise/ahegao-detector) (merci à
Sir-Prise) : crop carré du visage, resize 75×75, réseau 3 conv + dense,
seuil 0.6. Les poids (2,7 Mo, quantifiés uint8) sont committés dans
`models/ahegao/` — aucun téléchargement au lancement. L'inférence est en
numpy pur (~2 ms/frame), sans TensorFlow. Au-dessus du seuil : cadre
rouge + "AHEGAO DETECTE" — la blague finale. (La partie buttplug du
projet original n'est volontairement pas reprise.)

## En faire un .exe (PyInstaller)

Empaqueter mediapipe + deepface est lourd : pour le show, le plus fiable est
de lancer depuis le venv (déjà installé = zéro surprise). Sinon :

```powershell
pip install pyinstaller
pyinstaller --onefile --name ai-act-lolz main.py
```

Le dossier `dist/` contient le résultat. Attention : les modèles DeepFace se
téléchargent au premier run — prévoir internet sur la machine du show.

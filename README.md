# Ai_Act_lolz 🔬

Démo live de Computer Vision pour un show sur l'AI Act européen.
La webcam analyse **un seul visage : le vôtre**, en **100 % local**.
Rien n'est enregistré, rien n'est envoyé sur le réseau.

## ⚠️ Contexte d'usage (important)

Ce projet est pédagogique : il illustre ce que la computer vision sait
inférer en direct sur un visage, afin de **montrer au public** pourquoi
l'AI Act et le RGPD encadrent ces usages.

- ✅ Utilisation prévue : **vous, seul·e, filmant votre propre visage** (auto-consentement).
- ❌ À ne pas faire : pointer la webcam vers d'autres personnes, enregistrer,
  diffuser, ou déployer sans consentement (cf. `AI_ACT_NOTES.md`, Article 5 de
  l'AI Act, art. 9 du RGPD).

## Ce que la démo affiche en direct

| Caractéristique | Méthode | Fiabilité |
|---|---|---|
| Âge estimé | DeepFace | ± 5–8 ans |
| Apparence de genre perçue* | DeepFace | Stéréotype visuel, ≠ identité de genre |
| Émotion dominante | DeepFace | Approximatif, contesté scientifiquement |
| Pose de la tête (yaw / pitch) | MediaPipe + solvePnP | Bonne |
| Attention (regard + yeux ouverts) | MediaPipe | Bonne |
| Clignements | EAR (Eye Aspect Ratio) | Bonne |
| Rythme cardiaque (rPPG) | FFT sur le canal vert du front | Expérimental |

\* "Apparence perçue" : le modèle devine des stéréotypes d'apparence,
pas le genre réel de la personne. Excellent point de discussion sur scène.

Non inclus volontairement : inférence d'origine ethnique — interdite par
l'AI Act (art. 5.1.g) et scientifiquement infondée. Voir `AI_ACT_NOTES.md`.

## Installation

Python 3.10 ou 3.11 recommandé (compatibilité mediapipe/deepface).

```bash
git clone https://github.com/Massicotte/Ai_Act_lolz.git
cd Ai_Act_lolz
python -m venv venv
venv\Scripts\activate      # Windows (Linux/Mac : source venv/bin/activate)
pip install -r requirements.txt
python main.py
```

Au **premier lancement**, DeepFace télécharge ses modèles (~500 Mo, une seule fois).

## Commandes

| Touche | Action |
|---|---|
| `q` | Quitter |
| `d` | Afficher/masquer la ROI front (debug rPPG) |

## En faire un .exe (PyInstaller)

Empaqueter mediapipe + deepface est lourd : pour le show, le plus fiable est
de lancer depuis le venv (déjà installé = zéro surprise). Sinon :

```bash
pip install pyinstaller
pyinstaller --onefile --name ai-act-lolz main.py
```

Le dossier `dist/` contient le résultat. Attention : les modèles DeepFace se
téléchargent au premier run — prévoir internet sur la machine du show.

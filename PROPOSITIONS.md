# Propositions — que faire des données produites par la démo ?

Ce document **propose sans implémenter** : des idées d'usages, de
contrôles et de démonstrations construites sur les éléments que la
machine produit déjà en direct (identité, âge, apparence, émotion,
sourire, pose, attention, stabilité, distance, clignements, PERCLOS,
rythme cardiaque).

## 1. Automatiser la scène (contrôle du show)

- **Déclencheur d'étapes** : quand `attention` passe sous 50 % pendant
  plus de 5 s, lancer la slide suivante ou un son — la machine "sent"
  que le public (ou vous) décroche.
- **Disquette comique** : `sourire > 80 %` pendant 3 s → punchline ou
  GIF à l'écran. La machine réagit à votre sourire.
- **Alerte "on te regarde"** : si un visage est détecté sans être dans
  `people/`, afficher "VISAGE INCONNU — vérifier le consentement" :
  démonstration parfaite d'un usage à encadrer par l'AI Act.
- **Zombie detector** : PERCLOS > 25 % → "somnolence détectée". Fait
  sourire, mais c'est exactement ce que font les caméras de vigilance
  conducteur (règlementé par l'AI Act : systèmes de sécurité).

## 2. Retournements pédagogiques (le twist éthique)

- **Écran de détection "AI Act"** : réutiliser le biais genre réglable
  pour montrer en direct qu'un simple seuil change le discours du
  modèle — puis demander au public : "qui a codé ce seuil ? qui est
  responsable ?".
- **Vitrine à consentement** : détecter un visage inconnu, puis
  afficher "Cette personne n'a pas consenti : rien n'est enregistré" —
  contraste fort avec ce que font les vraies plateformes.
- **Diffusion en direct du JSON** : projeter en parallèle le flux brut
  de données que la machine produit sur vous chaque seconde. Le public
  voit le visage ET les colonnes de données : l'effet est saisissant.
- **Inverser la caméra** : filmer le public (avec consentement annoncé)
  et afficher des agrégats (nombre de visages, sourire moyen) — jamais
  d'identification — pour montrer la différence entre données agrégées
  et données identifiantes.

## 3. Contrôler des périphériques

- **OSC / MIDI vers un logiciel de lumière ou son** : la pose de tête
  (yaw/pitch/roll) pilote un panoramique de lumière ; l'émotion
  domine un filtre musical (joie → tempo rapide, tristesse → reverb).
- **Distance → zoom d'interface** : s'approcher de la caméra agrandit
  le panneau de données ("plus vous approchez, plus la machine en
  sait sur vous") — métaphore physique de l'intrusion.
- **BPM → tempo** : le rPPG (expérimental) module le BPM d'une piste
  sonore en direct : le public entend le cœur du présentateur.
- **Clignements → compteur de "pensées par minute"** : ironie assumée
  sur les dashboards pseudoscientifiques.

## 4. Journaliser (localement) pour le débrief

- **CSV horodaté** de toutes les métriques pendant la répétition, pour
  repérer : moments de baisse d'attention, nombre de fois où le modèle
  s'est trompé de genre/âge (excellent chiffre à citer sur scène),
  faux noms ("nicol" affiché alors que c'est quelqu'un d'autre) — le
  taux d'erreur réel de la reconnaissance faciale, mesuré sur vous.
- **Graphe post-show** : courbe attention/émotion le long de la
  timeline du show pour ajuster le script.

## 5. Attention : ce qu'il ne faut PAS faire

- Ne pas diffuser les noms des personnes reconnues hors du contexte
  du consentement (reconnaissance biométrique en public : interdite
  par l'AI Act art. 5 sauf exceptions très strictes).
- Ne pas enregistrer la vidéo sans le dire, même localement.
- Ne pas pister le public : la démo actuelle analyse un visage, le
  vôtre ; l'étendre à la salle entière changera la nature juridique
  de l'usage.

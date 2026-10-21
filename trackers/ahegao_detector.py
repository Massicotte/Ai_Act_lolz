"""Detecteur d'ahegao (blague finale de la demo).

L'ahegao est une expression exageree : yeux mi-clos ou roules, bouche
tres ouverte, langue sortie, joues duvrees. On l'approche par des
heuristiques geometriques sur les landmarks MediaPipe :

- yeux : EAR ( Eye Aspect Ratio ) tres bas SANS clignement prolonge
  (roulement / plissement volontaire)
- bouche : ouverture verticale tres grande normalisee par la taille
  du visage ( AHO )
- langue : presence de pixels clairs au centre de la bouche ouverte
- asymetrie : rien ici, on garde simple

Score = moyenne ponderee de ces indices, exprime en %.
C'est une HEURISTIQUE humoristique, pas un modele entraine : attendez
des faux positifs sur les grimaces genereuses. C'est aussi le propos de
la demo : une machine peut vous faire dire n'importe quoi.

Seuil d'affichage : 75 % (le "AHEGAO DETECTE" ne declenche que sur une
vraie grimaceengagee).
"""

import cv2
import numpy as np

from trackers.facemesh_tracker import FaceMeshTracker


class AhegaoDetector:
    # landmarks MediaPipe (refined)
    _LIPS_TOP = 13
    _LIPS_BOTTOM = 14
    _MOUTH_LEFT = 61
    _MOUTH_RIGHT = 291
    _EYE_L_IDS = (33, 160, 158, 133, 153, 144)
    _EYE_R_IDS = (362, 385, 387, 263, 373, 380)

    EAR_MAX = 0.25       # en dessous : yeux plisses / mi-clos
    AHO_TH = 0.16        # ouverture bouche / hauteur visage au-dessus de ce seuil
    AHO_MAX = 0.30       # ouverture correspondant au score plein
    TONGUE_LUM = 150.0  # luminance moyenne min dans la bouche pour "langue"

    def __init__(self, weights=None):
        w = {"ear": 0.40, "mouth": 0.45, "tongue": 0.15}
        if weights:
            w.update(weights)
        total = sum(w.values()) or 1.0
        self.w = {k: v / total for k, v in w.items()}

    def score(self, frame, face):
        """Retourne un score ahegao 0..1 (et un detail optionnel)."""
        if face is None:
            return 0.0
        pts = getattr(face, "landmarks", None)
        if pts is None:
            return 0.0
        h, w = frame.shape[:2]
        face_h = max(face.box[3] - face.box[1], 1)
        face_w = max(face.box[2] - face.box[0], 1)

        ear = face.ear
        ear_score = max(0.0, min(1.0, (self.EAR_MAX - ear) / self.EAR_MAX))

        mouth_open = abs(pts[self._LIPS_BOTTOM][1] - pts[self._LIPS_TOP][1])
        mouth_w = max(abs(pts[self._MOUTH_RIGHT][0] - pts[self._MOUTH_LEFT][0]), 1.0)
        aho = mouth_open / face_h
        mouth_score = max(0.0, min(1.0, (aho - self.AHO_TH) / (self.AHO_MAX - self.AHO_TH)))

        tongue_score = 0.0
        if aho > self.AHO_TH:
            cx = int((pts[self._LIPS_TOP][0] + pts[self._LIPS_BOTTOM][0]) / 2)
            cy = int((pts[self._LIPS_TOP][1] + pts[self._LIPS_BOTTOM][1]) / 2)
            half = max(2, int(mouth_w * 0.25))
            x1, y1 = max(0, cx - half), max(0, cy - half)
            x2, y2 = min(w, cx + half), min(h, cy + half)
            if x2 > x1 and y2 > y1:
                inner = frame[y1:y2, x1:x2]
                lum = float(np.mean(cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)))
                tongue_score = max(0.0, min(1.0, (lum - 90.0) / (self.TONGUE_LUM - 90.0)))

        score = (
            self.w["ear"] * ear_score
            + self.w["mouth"] * mouth_score
            + self.w["tongue"] * tongue_score
        )
        return float(max(0.0, min(1.0, score)))

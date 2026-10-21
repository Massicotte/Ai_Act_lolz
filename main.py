"""
Ai Act Lolz - demo live de Computer Vision pour un show sur l'AI Act europeen.

CONTEXTE D'USAGE : projet pedagogique, uniquement auto-consentement.
Ce programme analyse UN SEUL visage (le votre) en local. Rien n'est
enregistre, rien n'est envoye sur le reseau. Ne le pointez pas vers
d'autres personnes : voir README.md (section contexte legal).

Reconnaissance : creez des sous-dossiers dans people/ (un par personne,
le nom du dossier = le nom affiche) remplis de photos. La demo affiche
alors le nom de la personne reconnue au lieu de "vous". Donnee biometrique :
consentement obligatoire (RGPD art. 9, AI Act art. 5).

Touches : 'q' quitter, 'd' afficher/masquer la ROI front (debug rPPG),
'f' basculer le plein ecran, 'b' basculer le biais genre de demo.
"""

import cv2
import numpy as np

from trackers.deepface_analyzer import DeepFaceAnalyzer
from trackers.face_recognizer import PeopleRecognizer
from trackers.facemesh_tracker import FaceMeshTracker
from trackers.rppg import RPPGEstimator

FONT = cv2.FONT_HERSHEY_SIMPLEX
WINDOW = "Ai Act Lolz"
PANEL_X, PANEL_Y = 10, 24
LINE_H = 26


def letterbox(frame, canvas_w, canvas_h):
    """Redimensionne la frame en conservant les proportions (bandes noires)."""
    fh, fw = frame.shape[:2]
    scale = min(canvas_w / fw, canvas_h / fh)
    new_w, new_h = int(fw * scale), int(fh * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    ox = (canvas_w - new_w) // 2
    oy = (canvas_h - new_h) // 2
    canvas[oy:oy + new_h, ox:ox + new_w] = resized
    return canvas


def draw_panel(frame, lines):
    """Panneau semi-transparent en haut a gauche."""
    x, y = PANEL_X, PANEL_Y
    w = max(340, max((len(l) for l, _ in lines), default=0) * 11 + 180)
    h = LINE_H * len(lines) + 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 8, y - 18), (x + w, y + h), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for i, (label, value) in enumerate(lines):
        ty = y + LINE_H * i
        cv2.putText(frame, label, (x, ty + 10), FONT, 0.55, (190, 190, 190), 1, cv2.LINE_AA)
        cv2.putText(frame, value, (x + 160, ty + 10), FONT, 0.55, (90, 220, 255), 1, cv2.LINE_AA)


def main():
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    fullscreen = False
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL if not fullscreen else cv2.WINDOW_FULLSCREEN)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la webcam (index 0).")

    mesh = FaceMeshTracker()
    analyzer = DeepFaceAnalyzer(interval=2.5)
    recognizer = PeopleRecognizer(interval=1.0)
    rppg = RPPGEstimator(fps=30.0, window_sec=15.0)

    print(f"{WINDOW} en ligne - 'q' quitter, 'f' plein ecran, 'd' debug rPPG, 'b' biais genre.")
    print("Le premier lancement telecharge les modeles DeepFace, patience...")

    show_roi = False

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # effet miroir

        analyzer.update(frame)
        recognizer.update(frame)
        face = mesh.process(frame)

        lines = []
        if face is not None:
            cv2.rectangle(frame, face.box, (90, 220, 255), 2)
            name = recognizer.name()
            label = name if name else "vous"
            cv2.putText(frame, label, (face.box[0], face.box[1] - 10), FONT, 0.6, (90, 220, 255), 1, cv2.LINE_AA)

            if show_roi:
                cv2.rectangle(frame, face.forehead_box, (0, 255, 120), 1)

            rppg.feed(frame, face.forehead_box)
            bpm = rppg.bpm()

            lines.append(("Identite", label))
            lines.append(("Age estime", analyzer.smoothed_age()))
            lines.append(("Apparence*", analyzer.gender()))
            lines.append(("Emotion dom.", analyzer.emotion()))
            lines.append(("Sourire", f"{int(100 * face.smile)} %"))
            lines.append(("Pose tete", f"yaw {face.yaw:+.0f} pitch {face.pitch:+.0f}"))
            lines.append(("Attention", f"{face.attention} %"))
            lines.append(("Stabilite", f"{face.stability} %"))
            lines.append(("Distance", f"~{face.distance_cm:.0f} cm"))
            lines.append(("Clignements", f"{face.blinks} ({face.blink_rate:.0f}/min)"))
            lines.append(("Yeux fermes", f"{face.perclos:.0f} %"))
            if bpm is not None:
                lines.append(("BPM (exp.)", f"{bpm:.0f}"))
        else:
            lines.append(("Statut", "aucun visage detecte"))

        draw_panel(frame, lines)
        if fullscreen:
            try:
                _, _, win_w, win_h = cv2.getWindowImageRect(WINDOW)
                frame = letterbox(frame, int(win_w), int(win_h))
            except cv2.error:
                pass
        cv2.imshow(WINDOW, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("f"):
            fullscreen = not fullscreen
            cv2.setWindowProperty(
                WINDOW,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
            )
        if key == ord("d"):
            show_roi = not show_roi
        if key == ord("b"):
            new_margin = 100.0 - analyzer.gender_margin
            analyzer.gender_margin = new_margin
            print(f"Biais genre : marge masculine = {analyzer.gender_margin:.0f}% (confiance Man requise).")

    analyzer.stop()
    recognizer.stop()
    mesh.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

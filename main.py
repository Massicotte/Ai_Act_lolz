"""
Ai Act Lolz - demo live de Computer Vision pour un show sur l'AI Act europeen.

CONTEXTE D'USAGE : projet pedagogique, uniquement auto-consentement.
Ce programme analyse UN SEUL visage (le votre) en local. Rien n'est
enregistre, rien n'est envoye sur le reseau. Ne le pointez pas vers
d'autres personnes : voir README.md (section contexte legal).

Touches : 'q' quitter, 'd' afficher/masquer la ROI front (debug rPPG).
"""

import cv2

from trackers.deepface_analyzer import DeepFaceAnalyzer
from trackers.facemesh_tracker import FaceMeshTracker
from trackers.rppg import RPPGEstimator

FONT = cv2.FONT_HERSHEY_SIMPLEX
PANEL_X, PANEL_Y = 10, 24
LINE_H = 26


def draw_panel(frame, lines):
    """Panneau semi-transparent en haut a gauche."""
    x, y = PANEL_X, PANEL_Y
    w = 340
    h = LINE_H * len(lines) + 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 8, y - 18), (x + w, y + h), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for i, (label, value) in enumerate(lines):
        ty = y + LINE_H * i
        cv2.putText(frame, label, (x, ty + 10), FONT, 0.55, (190, 190, 190), 1, cv2.LINE_AA)
        cv2.putText(frame, value, (x + 160, ty + 10), FONT, 0.55, (90, 220, 255), 1, cv2.LINE_AA)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la webcam (index 0).")

    mesh = FaceMeshTracker()
    analyzer = DeepFaceAnalyzer(interval=2.5)
    rppg = RPPGEstimator(fps=30.0, window_sec=15.0)

    print("Ai Act Lolz en ligne - 'q' pour quitter, 'd' pour le debug rPPG.")
    print("Le premier lancement telecharge les modeles DeepFace, patience...")

    show_roi = False

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # effet miroir

        analyzer.update(frame)
        face = mesh.process(frame)

        lines = []
        if face is not None:
            cv2.rectangle(frame, face.box, (90, 220, 255), 2)
            cv2.putText(frame, "vous", (face.box[0], face.box[1] - 10), FONT, 0.6, (90, 220, 255), 1, cv2.LINE_AA)

            if show_roi:
                cv2.rectangle(frame, face.forehead_box, (0, 255, 120), 1)

            rppg.feed(frame, face.forehead_box)
            bpm = rppg.bpm()

            lines.append(("Age estime", analyzer.smoothed_age()))
            lines.append(("Apparence*", analyzer.gender()))
            lines.append(("Emotion dom.", analyzer.emotion()))
            lines.append(("Pose tete", f"yaw {face.yaw:+.0f} pitch {face.pitch:+.0f}"))
            lines.append(("Attention", f"{face.attention} %"))
            lines.append(("Clignements", str(face.blinks)))
            if bpm is not None:
                lines.append(("BPM (exp.)", f"{bpm:.0f}"))
        else:
            lines.append(("Statut", "aucun visage detecte"))

        draw_panel(frame, lines)
        cv2.imshow("Ai Act Lolz", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("d"):
            show_roi = not show_roi

    analyzer.stop()
    mesh.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

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
'f' basculer le plein ecran, 'j' afficher/masquer la fenetre JSON brut.
"""

import json
import time

import cv2
import numpy as np

from trackers.ahegao_detector import AhegaoDetector
from trackers.deepface_analyzer import DeepFaceAnalyzer
from trackers.face_recognizer import PeopleRecognizer
from trackers.facemesh_tracker import FaceMeshTracker
from trackers.rppg import RPPGEstimator

FONT = cv2.FONT_HERSHEY_SIMPLEX
WINDOW = "Ai Act Lolz"
JSON_WINDOW = "Ai Act Lolz - JSON brut"
PANEL_X, PANEL_Y = 10, 24
LINE_H = 26
JSON_INTERVAL = 1.0
AHEGAO_ALERT = 0.75


def draw_json_window(payload):
    """Fenetre noire affichant le JSON brut produit chaque seconde."""
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    canvas = np.zeros((900, 640, 3), dtype=np.uint8)
    y = 28
    for line in text.splitlines():
        cv2.putText(canvas, line, (14, y), FONT, 0.5, (120, 220, 120), 1, cv2.LINE_AA)
        y += 22
        if y > 880:
            break
    return canvas


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
    ahegao = AhegaoDetector()

    cv2.namedWindow(JSON_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(JSON_WINDOW, 640, 900)
    show_json = True
    last_json_t = 0.0

    print(f"{WINDOW} en ligne - 'q' quitter, 'f' plein ecran, 'j' JSON, 'd' debug rPPG.")
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

        fem, masc = analyzer.gender_percents()
        ahegao_score = ahegao.score(frame, face) if face is not None else 0.0

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
            if fem is not None:
                lines.append(("Femininite*", f"{fem:.0f} %"))
                lines.append(("Masculinite*", f"{masc:.0f} %"))
            else:
                lines.append(("Femininite*", "--"))
                lines.append(("Masculinite*", "--"))
            lines.append(("Emotion dom.", analyzer.emotion()))
            lines.append(("Ahegao", f"{int(100 * ahegao_score)} %"))
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

        if ahegao_score >= AHEGAO_ALERT:
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 18)
            cv2.putText(frame, "AHEGAO DETECTE", (w // 2 - 300, h // 2), FONT, 1.4, (0, 0, 255), 4, cv2.LINE_AA)

        draw_panel(frame, lines)

        now_t = time.perf_counter()
        if show_json and now_t - last_json_t >= JSON_INTERVAL:
            last_json_t = now_t
            payload = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "visage_detecte": face is not None,
            }
            if face is not None:
                payload.update({
                    "identite": recognizer.name(),
                    "age_estime": analyzer.smoothed_age(),
                    "femininite_pct": round(fem, 1) if fem is not None else None,
                    "masculinite_pct": round(masc, 1) if masc is not None else None,
                    "emotion": analyzer.emotion(),
                    "ahegao_score": round(ahegao_score, 2),
                    "sourire_pct": int(100 * face.smile),
                    "pose": {"yaw": round(face.yaw, 1), "pitch": round(face.pitch, 1), "roll": round(face.roll, 1)},
                    "attention_pct": face.attention,
                    "stabilite_pct": face.stability,
                    "distance_cm": round(face.distance_cm, 0),
                    "clignements": face.blinks,
                    "clignements_par_min": round(face.blink_rate, 1),
                    "perclos_pct": round(face.perclos, 1),
                    "bpm_estime": round(bpm, 0) if bpm is not None else None,
                })
            cv2.imshow(JSON_WINDOW, draw_json_window(payload))

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
        if key == ord("j"):
            show_json = not show_json
            if not show_json:
                cv2.destroyWindow(JSON_WINDOW)

    analyzer.stop()
    recognizer.stop()
    mesh.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

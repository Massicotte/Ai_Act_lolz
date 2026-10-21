"""
Ai Act Lolz - demo live de Computer Vision pour un show sur l'AI Act europeen.

CONTEXTE D'USAGE : projet pedagogique, uniquement auto-consentement.
Ce programme analyse UN SEUL visage (le votre) en local. Rien n'est
enregistre, rien n'est envoye sur le reseau. Ne le pointez pas vers
d'autres personnes : voir README.md (section contexte legal).

Fenetres :
  1. Video      : flux webcam + panneau de donnees (adapte a la taille,
                  max un quart de l'image ; video jamais deformee)
  2. JSON brut  : tout ce que la machine produit sur vous, chaque seconde
  3. Controles  : slider de biais de femininite (live) + un interrupteur
                  par info (afficher/masquer)

Le clavier reste fonctionnel : voir README.md (tableau des touches).
"""

import json
import os
import time

import cv2
import numpy as np

from trackers.ahegao_detector import AhegaoDetector
from trackers.deepface_analyzer import DeepFaceAnalyzer
from trackers.face_recognizer import PeopleRecognizer
from trackers.facemesh_tracker import FaceMeshTracker
from trackers.rppg import RPPGEstimator

FONT = cv2.FONT_HERSHEY_SIMPLEX
PEOPLE_DIR = "people"
WATCH_EVERY = 2.0  # secondes entre deux scans du dossier people/
WINDOW = "Ai Act Lolz"
JSON_WINDOW = "Ai Act Lolz - JSON brut"
CTRL_WINDOW = "Ai Act Lolz - Controles"
JSON_INTERVAL = 1.0
AHEGAO_ALERT = 0.6  # seuil du modele original Sir-Prise
BIAS_MAX = 100       # 100 % = toute la masse "masculine" bascule sur "feminine"
DEFAULT_BIAS = 30    # biais de demo par defaut (%)

INFOS = [
    ("identite", "Identite [i]"),
    ("age", "Age [a]"),
    ("femininite", "Femininite* [f]"),
    ("masculinite", "Masculinite* [m]"),
    ("emotion", "Emotion [e]"),
    ("ahegao", "Ahegao [g]"),
    ("sourire", "Sourire [s]"),
    ("pose", "Pose tete [t]"),
    ("attention", "Attention [v]"),
    ("stabilite", "Stabilite [b]"),
    ("distance", "Distance [l]"),
    ("clignements", "Clignements [c]"),
    ("perclos", "Yeux fermes [y]"),
    ("bpm", "BPM (exp.) [h]"),
]
KEY_TOGGLES = {
    ord("i"): "identite", ord("a"): "age", ord("f"): "femininite",
    ord("m"): "masculinite", ord("e"): "emotion", ord("g"): "ahegao",
    ord("s"): "sourire", ord("t"): "pose", ord("v"): "attention",
    ord("b"): "stabilite", ord("l"): "distance", ord("c"): "clignements",
    ord("y"): "perclos", ord("h"): "bpm",
}


def letterbox(frame, canvas_w, canvas_h):
    """Redimensionne la frame en conservant les proportions (bandes noires)."""
    fh, fw = frame.shape[:2]
    scale = min(canvas_w / fw, canvas_h / fh)
    new_w, new_h = max(1, int(fw * scale)), max(1, int(fh * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    ox = (canvas_w - new_w) // 2
    oy = (canvas_h - new_h) // 2
    canvas[oy:oy + new_h, ox:ox + new_w] = resized
    return canvas


def draw_panel(canvas, lines):
    """Panneau semi-transparent, dimensionne pour occuper au max ~1/4 de
    l'image, quelle que soit la taille de la fenetre."""
    ch, cw = canvas.shape[:2]
    if not lines:
        return
    scale = max(0.6, min(2.0, (cw * ch) / (1280.0 * 720.0)) ** 0.5)
    x0, y0 = int(10 * scale), int(24 * scale)
    line_h = max(18, int(26 * scale))
    font = 0.55 * scale
    thick = max(1, round(scale))
    pad_x = int(8 * scale)

    label_w = max((len(l) for l, _ in lines), default=0) * int(9 * scale)
    value_w = max((len(v) for _, v in lines), default=0) * int(9 * scale)
    w = pad_x + label_w + int(60 * scale) + value_w + pad_x
    h = line_h * len(lines) + int(10 * scale)
    # contrainte : le panneau ne depasse jamais 1/4 de l'image
    max_w = cw // 2
    max_h = ch // 2
    if w * h > (cw * ch) // 4:
        w = min(w, max_w)
        h = min(h, max_h)
    w = min(w, max_w)
    h = min(h, max_h)

    overlay = canvas.copy()
    cv2.rectangle(overlay, (x0 - pad_x, y0 - line_h), (x0 + w, y0 + h), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)
    for i, (label, value) in enumerate(lines):
        ty = y0 + line_h * i
        if ty + line_h > y0 + h:
            break
        cv2.putText(canvas, label, (x0, ty + int(10 * scale)), FONT, font, (190, 190, 190), thick, cv2.LINE_AA)
        cv2.putText(canvas, value, (x0 + label_w + int(60 * scale), ty + int(10 * scale)), FONT, font, (90, 220, 255), thick, cv2.LINE_AA)


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


def apply_gender_bias(fem, masc, bias_pct):
    """Biais de demo : deplace 'bias_pct' % de la masse percue masculine
    vers la femininite. 0 = brut du modele, 100 = tout feminin."""
    if fem is None or masc is None:
        return None, None
    b = max(0.0, min(100.0, bias_pct)) / 100.0
    fem_b = fem + masc * b
    masc_b = masc * (1.0 - b)
    return min(100.0, fem_b), min(100.0, masc_b)


def build_control_window(visible, bias):
    """Fenetre Controles : un slider de biais + un interrupteur par info."""
    cv2.namedWindow(CTRL_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CTRL_WINDOW, 460, 620)
    cv2.createTrackbar("Biais femininite %", CTRL_WINDOW, int(bias), BIAS_MAX, lambda v: None)
    for key, label in INFOS:
        cv2.createTrackbar(label, CTRL_WINDOW, 1 if visible[key] else 0, 1, lambda v: None)


def read_controls(visible):
    """Lit l'etat des interrupteurs et du slider depuis la fenetre Controles."""
    bias = cv2.getTrackbarPos("Biais femininite %", CTRL_WINDOW)
    for key, label in INFOS:
        visible[key] = bool(cv2.getTrackbarPos(label, CTRL_WINDOW))
    return bias


def sync_control_window(visible):
    """Re-synchronise les interrupteurs apres un toggle clavier."""
    for key, label in INFOS:
        try:
            cv2.setTrackbarPos(label, CTRL_WINDOW, 1 if visible[key] else 0)
        except cv2.error:
            pass


def watch_signature(people_dir):
    """Empreinte du dossier people/ (sous-dossiers + fichiers) pour
    detecter un glisser-deposer sans recharger systematiquement."""
    try:
        entries = []
        for person in sorted(os.listdir(people_dir)):
            folder = os.path.join(people_dir, person)
            if os.path.isdir(folder):
                for fname in sorted(os.listdir(folder)):
                    fp = os.path.join(folder, fname)
                    if os.path.isfile(fp):
                        entries.append(f"{person}/{fname}:{os.path.getmtime(fp):.0f}")
        return "|".join(entries)
    except OSError:
        return ""


def main():
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    fullscreen = False

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError(
            "Impossible d'ouvrir la webcam (index 0).\n"
            " - Windows : Parametres > Confidentialite > Camera > autoriser les applications de bureau.\n"
            " - Verifiez qu'aucune autre app (Zoom, Teams...) n'occupe la webcam."
        )

    mesh = FaceMeshTracker()
    analyzer = DeepFaceAnalyzer(interval=2.5)
    recognizer = PeopleRecognizer(interval=1.0)
    rppg = RPPGEstimator(fps=30.0, window_sec=15.0)
    ahegao = AhegaoDetector()

    cv2.namedWindow(JSON_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(JSON_WINDOW, 640, 900)
    show_json = True
    last_json_t = 0.0

    visible = {key: True for key, _ in INFOS}
    bias = DEFAULT_BIAS
    build_control_window(visible, bias)

    print(f"{WINDOW} en ligne - fenetre Controles : slider de biais + interrupteurs.")
    print("Le premier lancement telecharge les modeles DeepFace, patience...")

    show_roi = False
    read_failures = 0
    frame_count = 0
    people_sig = watch_signature(PEOPLE_DIR)
    last_watch = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            read_failures += 1
            if read_failures > 60:
                print("Webcam illisible (60 echecs consecutifs) : arret. Verifiez la confidentialite camera.")
                break
            continue
        read_failures = 0
        frame_count += 1
        cv2.flip(frame, 1, dst=frame)  # effet miroir, en place

        if analyzer.needs_frame() or recognizer.needs_frame():
            shared = frame.copy()
            analyzer.update(shared)
            recognizer.update(shared)
        face = mesh.process(frame)

        bias = read_controls(visible)
        fem, masc = analyzer.gender_percents()
        fem_b, masc_b = apply_gender_bias(fem, masc, bias)
        ahegao_prob = ahegao.probability(frame, face) if (face is not None and (visible["ahegao"] or show_json)) else None
        ahegao_pct = int(100 * ahegao_prob) if ahegao_prob is not None else 0

        lines = []
        if face is not None:
            cv2.rectangle(frame, face.box, (90, 220, 255), 2)
            name = recognizer.name()
            label = name if name else "vous"
            if visible["identite"]:
                cv2.putText(frame, label, (face.box[0], face.box[1] - 10), FONT, 0.6, (90, 220, 255), 1, cv2.LINE_AA)

            if show_roi:
                cv2.rectangle(frame, face.forehead_box, (0, 255, 120), 1)

            rppg.feed(frame, face.forehead_box)
            bpm = rppg.bpm()

            if visible["age"]:
                lines.append(("Age estime", analyzer.smoothed_age()))
            if visible["femininite"]:
                lines.append(("Femininite*", f"{fem_b:.0f} %" if fem_b is not None else "--"))
            if visible["masculinite"]:
                lines.append(("Masculinite*", f"{masc_b:.0f} %" if masc_b is not None else "--"))
            if visible["emotion"]:
                lines.append(("Emotion dom.", analyzer.emotion()))
            if visible["ahegao"]:
                lines.append(("Ahegao", f"{ahegao_pct} %"))
            if visible["sourire"]:
                lines.append(("Sourire", f"{int(100 * face.smile)} %"))
            if visible["pose"]:
                lines.append(("Pose tete", f"yaw {face.yaw:+.0f} pitch {face.pitch:+.0f}"))
            if visible["attention"]:
                lines.append(("Attention", f"{face.attention} %"))
            if visible["stabilite"]:
                lines.append(("Stabilite", f"{face.stability} %"))
            if visible["distance"]:
                lines.append(("Distance", f"~{face.distance_cm:.0f} cm"))
            if visible["clignements"]:
                lines.append(("Clignements", f"{face.blinks} ({face.blink_rate:.0f}/min)"))
            if visible["perclos"]:
                lines.append(("Yeux fermes", f"{face.perclos:.0f} %"))
            if visible["bpm"] and bpm is not None:
                lines.append(("BPM (exp.)", f"{bpm:.0f}"))
        else:
            lines.append(("Statut", "aucun visage detecte"))

        if ahegao_prob is not None and ahegao_prob >= AHEGAO_ALERT and visible["ahegao"]:
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 18)
            cv2.putText(frame, "AHEGAO DETECTE", (w // 2 - 300, h // 2), FONT, 1.4, (0, 0, 255), 4, cv2.LINE_AA)

        # --- surveillance de people/ : glisser-deposer = reapprentissage auto ---
        now_w = time.perf_counter()
        if now_w - last_watch >= WATCH_EVERY:
            last_watch = now_w
            sig = watch_signature(PEOPLE_DIR)
            if sig != people_sig:
                people_sig = sig
                print("people/ a change : reapprentissage des visages...")
                recognizer.reload()

        # --- affichage : la video ne se deforme jamais, le panneau s'adapte ---
        try:
            _, _, win_w, win_h = cv2.getWindowImageRect(WINDOW)
        except cv2.error:
            win_w, win_h = frame.shape[1], frame.shape[0]
        canvas = letterbox(frame, max(1, int(win_w)), max(1, int(win_h)))
        draw_panel(canvas, lines)
        cv2.imshow(WINDOW, canvas)

        now_t = time.perf_counter()
        if show_json and now_t - last_json_t >= JSON_INTERVAL:
            last_json_t = now_t
            payload = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "visage_detecte": face is not None,
                "biais_femininite_pct": bias,
            }
            if face is not None:
                payload.update({
                    "identite": recognizer.name(),
                    "age_estime": analyzer.smoothed_age(),
                    "femininite_pct": round(fem_b, 1) if fem_b is not None else None,
                    "masculinite_pct": round(masc_b, 1) if masc_b is not None else None,
                    "femininite_brute_pct": round(fem, 1) if fem is not None else None,
                    "masculinite_brute_pct": round(masc, 1) if masc is not None else None,
                    "emotion": analyzer.emotion(),
                    "ahegao_probabilite": round(ahegao_prob, 3) if ahegao_prob is not None else None,
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

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("p"):
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
        if key in KEY_TOGGLES:
            info = KEY_TOGGLES[key]
            visible[info] = not visible[info]
            etat = "affichee" if visible[info] else "masquee"
            print(f"Info '{info}' : {etat}")
            sync_control_window(visible)

    analyzer.stop()
    recognizer.stop()
    mesh.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

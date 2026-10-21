"""Tracking du visage via MediaPipe FaceMesh : pose de la tete,
ouverture des yeux, attention, clignements et ROI front (pour rPPG).
"""

import time
from collections import deque
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class FaceResult:
    box: tuple
    forehead_box: tuple
    yaw: float
    pitch: float
    roll: float
    ear: float
    attention: int
    blinks: int
    blink_rate: float
    perclos: float
    smile: float
    distance_cm: float
    stability: int


class FaceMeshTracker:
    # Modele 3D generique d'un visage (repere : bout du nez)
    _MODEL_3D = np.array(
        [
            (0.0, 0.0, 0.0),
            (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0),
        ],
        dtype=np.float64,
    )
    _NOSE, _CHIN, _EYE_L, _EYE_R, _MOUTH_L, _MOUTH_R = 1, 152, 33, 263, 61, 291
    _EAR_L = (33, 160, 158, 133, 153, 144)
    _EAR_R = (362, 385, 387, 263, 373, 380)
    _FOREHEAD_TOP = 10
    _BLINK_THRESHOLD = 0.18
    _ATTENTION_YAW = 13.0
    _ATTENTION_PITCH = 13.0
    _SMILE_L = (61, 39, 37, 91, 267, 269, 291)
    _SMILE_TH = 0.62
    _BLINK_WINDOW = 60.0

    def __init__(self):
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._attention_hist = deque(maxlen=60)
        self._blink_state = False
        self._blinks = 0
        self._blink_times = deque(maxlen=600)
        self._closed_samples = 0
        self._perclos_samples = 0
        self._prev_center = None
        self._move_hist = deque(maxlen=90)
        self._t0 = None

    def process(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self._mesh.process(rgb)

        if not res.multi_face_landmarks:
            self._attention_hist.append(False)
            self._prev_center = None
            return None

        landmarks = res.multi_face_landmarks[0].landmark
        pts = np.array([(lm.x * w, lm.y * h) for lm in landmarks], dtype=np.float64)

        # --- Pose de la tete (solvePnP) ---
        image_pts = np.array(
            [
                pts[self._NOSE],
                pts[self._CHIN],
                pts[self._EYE_L],
                pts[self._EYE_R],
                pts[self._MOUTH_L],
                pts[self._MOUTH_R],
            ]
        )
        camera = np.array(
            [
                [w, 0.0, w / 2.0],
                [0.0, w, h / 2.0],
                [0.0, 0.0, 1.0],
            ]
        )
        dist = np.zeros((4, 1))
        yaw = pitch = roll = 0.0
        ok, rvec, _ = cv2.solvePnP(
            self._MODEL_3D, image_pts, camera, dist, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if ok:
            rot, _ = cv2.Rodrigues(rvec)
            pitch = float(np.degrees(np.arctan2(rot[2, 1], rot[2, 2])))
            yaw = float(
                np.degrees(
                    np.arctan2(-rot[2, 0], np.sqrt(rot[0, 0] ** 2 + rot[1, 0] ** 2))
                )
            )
            roll = float(np.degrees(np.arctan2(rot[1, 0], rot[0, 0])))

        # --- Ouverture des yeux (EAR) + clignements ---
        ear = (self._ear(pts, self._EAR_L) + self._ear(pts, self._EAR_R)) / 2.0
        now = time.perf_counter()
        if self._t0 is None:
            self._t0 = now
        elapsed = min(now - self._t0, self._BLINK_WINDOW)
        if ear < self._BLINK_THRESHOLD:
            self._closed_samples += 1
            if not self._blink_state:
                self._blink_state = True
                self._blinks += 1
                self._blink_times.append(now)
        else:
            self._blink_state = False
        self._perclos_samples += 1
        while self._blink_times and elapsed >= self._BLINK_WINDOW and now - self._blink_times[0] > self._BLINK_WINDOW:
            self._blink_times.popleft()
        blink_rate = 60.0 * len(self._blink_times) / elapsed if elapsed > 0 else 0.0
        perclos = 100.0 * self._closed_samples / self._perclos_samples if self._perclos_samples else 0.0

        # --- Attention : regarde vers l'ecran, yeux ouverts ---
        looking = (
            abs(yaw) < self._ATTENTION_YAW
            and abs(pitch) < self._ATTENTION_PITCH
            and ear > 0.2
        )
        self._attention_hist.append(looking)
        attention = int(100.0 * sum(self._attention_hist) / len(self._attention_hist))

        # --- Sourire (largeur bouche normalisee par la largeur du visage) ---
        mouth_w = abs(pts[291][0] - pts[61][0])

        # --- Boite englobante du visage ---
        bx1, by1 = pts.min(axis=0)
        bx2, by2 = pts.max(axis=0)
        box = (int(bx1), int(by1), int(bx2), int(by2))
        face_w = max(box[2] - box[0], 1.0)
        smile = max(0.0, min(1.0, (mouth_w / face_w - 0.42) / (self._SMILE_TH - 0.42)))

        # --- Distance estimee (proportion empirique du visage dans le cadre) ---
        distance_cm = 2600.0 / face_w

        # --- Stabilite du visage (inversely proportionnelle au mouvement) ---
        center = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
        if self._prev_center is not None:
            move = (abs(center[0] - self._prev_center[0]) + abs(center[1] - self._prev_center[1])) / 2.0
            self._move_hist.append(move)
        self._prev_center = center
        stab = 100 - int(min(100.0, 5.0 * (sum(self._move_hist) / len(self._move_hist) if self._move_hist else 0.0)))

        # --- ROI front pour le rPPG ---
        top = pts[self._FOREHEAD_TOP]
        eye_mid_y = (pts[self._EYE_L][1] + pts[self._EYE_R][1]) / 2.0
        span_l = pts[self._EYE_L][0] + 0.32 * (pts[self._EYE_R][0] - pts[self._EYE_L][0])
        span_r = pts[self._EYE_L][0] + 0.68 * (pts[self._EYE_R][0] - pts[self._EYE_L][0])
        fy1 = top[1] + 0.30 * (eye_mid_y - top[1])
        fy2 = top[1] + 0.58 * (eye_mid_y - top[1])
        forehead_box = (
            max(0, int(span_l)), max(0, int(fy1)),
            min(w, int(span_r)), min(h, int(fy2)),
        )

        return FaceResult(
            box=box,
            forehead_box=forehead_box,
            yaw=yaw, pitch=pitch, roll=roll,
            ear=ear,
            attention=attention,
            blinks=self._blinks,
            blink_rate=blink_rate,
            perclos=perclos,
            smile=smile,
            distance_cm=distance_cm,
            stability=stab,
        )

    @staticmethod
    def _ear(pts, ids):
        p = pts[list(ids)]
        vertical = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
        horizontal = np.linalg.norm(p[0] - p[3])
        return vertical / (2.0 * horizontal) if horizontal > 1e-6 else 0.0

    def close(self):
        self._mesh.close()

"""Analyse periodique (thread separe) : age, genre percu, emotion dominante.
DeepFace tourne dans son propre thread pour ne pas bloquer le flux video.
Le premier appel telecharge les modeles (connexion internet requise une seule fois).
"""

import threading
import time
from collections import deque

from deepface import DeepFace

GENDER_FR = {"Woman": "feminine*", "Man": "masculine*"}
EMOTION_FR = {
    "happy": "joie",
    "sad": "tristesse",
    "angry": "colere",
    "surprise": "surprise",
    "fear": "peur",
    "disgust": "degout",
    "neutral": "neutre",
}


class DeepFaceAnalyzer:
    def __init__(self, interval=2.5, history=8):
        self.interval = interval
        self._age_hist = deque(maxlen=history)
        self._gender = None
        self._emotion = None
        self._last_result = None
        self._latest = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def update(self, frame):
        self._latest = frame.copy()

    def _loop(self):
        while not self._stop.is_set():
            if self._latest is not None:
                try:
                    res = DeepFace.analyze(
                        self._latest,
                        actions=["age", "gender", "emotion"],
                        detector_backend="mediapipe",
                        enforce_detection=False,
                        silent=True,
                    )
                    if isinstance(res, list):
                        res = res[0]
                    self._age_hist.append(float(res.get("age", 0.0) or 0.0))
                    self._gender = res.get("dominant_gender")
                    self._emotion = res.get("dominant_emotion")
                    self._last_result = time.time()
                except Exception:
                    pass  # pas de visage -> on garde les dernieres valeurs
            self._stop.wait(self.interval)

    def smoothed_age(self):
        if not self._age_hist:
            return "--"
        return f"{sum(self._age_hist) / len(self._age_hist):.0f} ans"

    def gender(self):
        if self._gender is None:
            return "--"
        return GENDER_FR.get(self._gender, self._gender)

    def emotion(self):
        if self._emotion is None:
            return "--"
        return EMOTION_FR.get(self._emotion, self._emotion)

    def alive(self):
        return self._last_result is not None

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2.0)

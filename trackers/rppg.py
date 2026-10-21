"""Estimation experimentale du rythme cardiaque par rPPG
(photoplethysmographie a distance).

Principe : a chaque battement, le sang dans les micro-vaisseaux du front
module imperceptiblement la couleur de la peau. On suit la moyenne du canal
vert sur la ROI front, puis on cherche le pic de frequence dans la bande
cardiaque (50 - 170 bpm) via une FFT.

C'est EXPERIMENTAL : eclairage stable requis, aucune valeur medicale.
"""

import time

import numpy as np


class RPPGEstimator:
    def __init__(self, fps=30.0, window_sec=15.0, min_bpm=50.0, max_bpm=170.0):
        self.capacity = max(60, int(fps * window_sec))
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm
        self._sig = []
        self._times = []

    def feed(self, frame, box):
        x1, y1, x2, y2 = box
        if x2 - x1 < 8 or y2 - y1 < 8:
            return
        roi = frame[y1:y2, x1:x2]
        self._sig.append(float(np.mean(roi[:, :, 1])))  # canal vert
        self._times.append(time.perf_counter())
        if len(self._sig) > self.capacity:
            self._sig.pop(0)
            self._times.pop(0)

    def bpm(self):
        if len(self._sig) < self.capacity // 2:
            return None
        duration = self._times[-1] - self._times[0]
        if duration <= 1.0:
            return None
        fps = (len(self._times) - 1) / duration
        if fps < 15.0:
            return None

        sig = np.asarray(self._sig, dtype=np.float64)
        sig = sig - sig.mean()
        windowed = sig * np.hanning(len(sig))
        spectrum = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(len(sig), d=1.0 / fps)

        mask = (freqs >= self.min_bpm / 60.0) & (freqs <= self.max_bpm / 60.0)
        if not mask.any():
            return None
        peak_hz = freqs[mask][np.argmax(spectrum[mask])]
        return float(peak_hz * 60.0)

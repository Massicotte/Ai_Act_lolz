"""Detecteur d'ahegao base sur le CNN du projet ahegao-detector
(https://github.com/Sir-Prise/ahegao-detector, commite ici sous models/ahegao/).

Pipeline (identique au projet original) :
  1. crop carre du visage depuis la frame (box FaceMesh, etendue en carre)
  2. resize 75x75, RGB, normalise /255
  3. CNN : 3 x [Conv2D(relu) + MaxPool] -> Flatten -> Dense(512, relu)
     -> Dense(1, sigmoid)
  4. probabilite d'ahegao ; seuil de detection a 0.6 (comme l'original)

L'inference est implementee en numpy pur (pas de TensorFlow requis) :
les poids du fichier binaire tfjs sont charges au demarrage une seule
fois, puis chaque frame est classee en ~quelques millisecondes.

La partie buttplug du projet original est volontairement omise.
"""

import json
import os

import cv2
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "ahegao")
THRESHOLD = 0.6  # seuil du projet original
INPUT_SIZE = 75


def _conv2d_fast(x, kernel, bias):
    """Conv2D 'same' via im2col (vectorise)."""
    kh, kw, cin, cout = kernel.shape
    h, w, _ = x.shape
    pad_h = (kh - 1) // 2
    pad_w = (kw - 1) // 2
    xp = np.pad(x, ((pad_h, kh - 1 - pad_h), (pad_w, kw - 1 - pad_w), (0, 0)))
    cols = np.empty((h * w, kh * kw * cin), dtype=np.float32)
    idx = 0
    for i in range(kh):
        for j in range(kw):
            patch = xp[i:i + h, j:j + w, :]
            cols[:, idx:idx + cin] = patch.reshape(h * w, cin)
            idx += cin
    weights = kernel.reshape(kh * kw * cin, cout)
    out = cols @ weights + bias
    return np.maximum(out.reshape(h, w, cout), 0.0)


def _maxpool2x2(x):
    h, w, c = x.shape
    return x[: h // 2 * 2, : w // 2 * 2, :].reshape(h // 2, 2, w // 2, 2, c).max(axis=(1, 3))


class AhegaoDetector:
    def __init__(self, model_dir=MODEL_DIR):
        self.threshold = THRESHOLD
        self._weights = self._load(model_dir)

    @staticmethod
    def _load(model_dir):
        with open(os.path.join(model_dir, "model.json"), "r", encoding="utf-8") as f:
            manifest = json.load(f)
        specs = manifest["weightsManifest"][0]["weights"]
        shard_file = manifest["weightsManifest"][0]["paths"][0]
        with open(os.path.join(model_dir, shard_file), "rb") as f:
            shard = f.read()
        weights = {}
        offset = 0
        for spec in specs:
            n = int(np.prod(spec["shape"])) if spec["shape"] else 1
            raw = np.frombuffer(shard, dtype=np.uint8, count=n, offset=offset)
            offset += n
            quant = spec.get("quantization")
            if quant:
                arr = quant["min"] + raw.astype(np.float32) * quant["scale"]
            else:
                arr = raw.astype(spec.get("dtype", "float32"))
            weights[spec["name"]] = arr.reshape(spec["shape"]).astype(np.float32)
        return weights

    def probability(self, frame, face):
        """Probabilite d'ahegao (0..1) sur le visage detecte, ou None."""
        if face is None:
            return None
        crop = self._square_crop(frame, face.box)
        if crop is None:
            return None
        img = cv2.resize(crop, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        w = self._weights
        x = img
        x = _conv2d_fast(x, w["conv2d/kernel"], w["conv2d/bias"])
        x = _maxpool2x2(x)
        x = _conv2d_fast(x, w["conv2d_1/kernel"], w["conv2d_1/bias"])
        x = _maxpool2x2(x)
        x = _conv2d_fast(x, w["conv2d_2/kernel"], w["conv2d_2/bias"])
        x = _maxpool2x2(x)
        x = x.reshape(-1)
        x = np.maximum(x @ w["dense/kernel"] + w["dense/bias"], 0.0)
        logit = float((x @ w["dense_1/kernel"] + w["dense_1/bias"])[0])
        return float(1.0 / (1.0 + np.exp(-logit)))

    def score(self, frame, face):
        """Compatibilite avec l'ancien nom : probabilite 0..1 (0 sans visage)."""
        p = self.probability(frame, face)
        return 0.0 if p is None else p

    @staticmethod
    def _square_crop(frame, box):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box
        side = max(x2 - x1, y2 - y1)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        half = side // 2
        sx1, sy1 = cx - half, cy - half
        sx2, sy2 = sx1 + side, sy1 + side
        if sx1 < 0 or sy1 < 0 or sx2 > w or sy2 > h:
            # crop aux frontieres : on decoupe ce qui est visible (comme drawImage)
            px1, py1 = max(0, sx1), max(0, sy1)
            px2, py2 = min(w, sx2), min(h, sy2)
            if px2 - px1 < 10 or py2 - py1 < 10:
                return None
            return frame[py1:py2, px1:px2]
        return frame[sy1:sy2, sx1:sx2]

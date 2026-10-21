"""Reconnaissance faciale basee sur un dossier de personnes.

Structure attendue (chaque sous-dossier = une personne, son nom = le nom
du dossier) :

    people/
        nicolas/
            photo1.jpg
            photo2.jpg
        alice/
            alice.png

Au demarrage, les embeddings des photos de reference sont calcules une fois.
Ensuite, la frame courante est comparee a ces references par similarite
cosinus : si le meilleur score depasse le seuil, la personne est identifiee.

CONSENTEMENT OBLIGATOIRE : ne photographiez que des personnes consentantes.
La reconnaissance du nom d'une personne est une donnee biometrique
identifiante (RGPD art. 9, AI Act) : voir README.md.
"""

import os
import threading

import numpy as np

from deepface import DeepFace

MODEL_NAME = "VGG-Face"
MATCH_THRESHOLD = 0.60
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
PEOPLE_DIR = "people"


class PeopleRecognizer:
    def __init__(self, people_dir=PEOPLE_DIR, interval=1.0, threshold=MATCH_THRESHOLD):
        self.interval = interval
        self.threshold = threshold
        self._name = None
        self._score = 0.0
        self._latest = None
        self._stop = threading.Event()
        self._names, self._embeddings = self._build_db(people_dir)
        if self._names:
            print(f"Reconnaissance active : {len(self._names)} personne(s) chargee(s).")
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _build_db(self, people_dir):
        names, embeddings = [], []
        if not os.path.isdir(people_dir):
            return names, embeddings
        for person in sorted(os.listdir(people_dir)):
            folder = os.path.join(people_dir, person)
            if not os.path.isdir(folder):
                continue
            vecs = []
            for fname in sorted(os.listdir(folder)):
                if os.path.splitext(fname)[1].lower() not in IMAGE_EXTS:
                    continue
                try:
                    reps = DeepFace.represent(
                        os.path.join(folder, fname),
                        model_name=MODEL_NAME,
                        detector_backend="mediapipe",
                        enforce_detection=False,
                    )
                    for rep in reps:
                        vecs.append(np.asarray(rep["embedding"], dtype=np.float64))
                except Exception as exc:
                    print(f"[people] {person}/{fname} ignore : {exc}")
            if vecs:
                mean = np.mean(vecs, axis=0)
                mean /= np.linalg.norm(mean) + 1e-9
                names.append(person)
                embeddings.append(mean)
                print(f"[people] {person} : {len(vecs)} photo(s) apprise(s).")
        return names, embeddings

    def update(self, frame):
        self._latest = frame.copy()

    def _loop(self):
        while not self._stop.is_set():
            if self._latest is not None and self._names:
                try:
                    reps = DeepFace.represent(
                        self._latest,
                        model_name=MODEL_NAME,
                        detector_backend="mediapipe",
                        enforce_detection=False,
                    )
                    if reps:
                        vec = np.asarray(reps[0]["embedding"], dtype=np.float64)
                        vec /= np.linalg.norm(vec) + 1e-9
                        sims = [float(np.dot(vec, ref)) for ref in self._embeddings]
                        best = int(np.argmax(sims))
                        self._score = sims[best]
                        self._name = self._names[best] if sims[best] >= self.threshold else None
                except Exception:
                    pass
            self._stop.wait(self.interval)

    def name(self):
        return self._name

    def score(self):
        return self._score

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2.0)

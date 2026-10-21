"""Reconnaissance faciale basee sur un dossier de personnes.

Deux facons d'enregistrer une personne :

  1. EN DIRECT (le plus simple) : lancer la demo, appuyer sur 'n',
     entrer le nom dans le terminal, regarder la camera pendant
     ~5 secondes. La machine capture les photos elle-meme.

  2. Manuellement : creer un sous-dossier dans people/ au nom de la
     personne et y copier 3-10 photos claires de son visage :

         people/
             nicol/     photo1.jpg photo2.jpg ...
             alice/     alice.png

Au demarrage (ou apres un enregistrement), les embeddings des photos de
reference sont calcules en arriere-plan. La frame courante est ensuite
comparee a TOUS les embeddings de reference par similarite cosinus : le
meilleur score au-dessus du seuil designe la personne.

Seuil : 0.40 (reference DeepFace pour VGG-Face en distance cosinus).

CONSENTEMENT OBLIGATOIRE : ne photographiez que des personnes consentantes.
La reconnaissance du nom d'une personne est une donnee biometrique
identifiante (RGPD art. 9, AI Act) : voir README.md.
"""

import os
import threading

import numpy as np

from deepface import DeepFace

MODEL_NAME = "VGG-Face"
MATCH_THRESHOLD = 0.40  # seuil officiel VGG-Face (cosinus) de DeepFace
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
PEOPLE_DIR = "people"


class PeopleRecognizer:
    def __init__(self, people_dir=PEOPLE_DIR, interval=1.0, threshold=MATCH_THRESHOLD):
        self.interval = interval
        self.threshold = threshold
        self._people_dir = people_dir
        self._name = None
        self._score = 0.0
        self._latest = None
        self._db_ready = False
        self._refs = []  # liste de (nom, [embeddings normalises])
        self._db_lock = threading.Lock()
        self._stop = threading.Event()
        os.makedirs(people_dir, exist_ok=True)
        self._db_thread = threading.Thread(target=self._build_db_async, daemon=True)
        self._db_thread.start()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _build_db_async(self):
        """Construit la base d'embeddings en arriere-plan (peut telecharger
        les poids VGG-Face au premier lancement : ne doit jamais bloquer l'UI)."""
        try:
            refs = self._build_db(self._people_dir)
            with self._db_lock:
                self._refs = refs
            count = sum(len(v) for _, v in refs)
            if refs:
                print(f"Reconnaissance active : {len(refs)} personne(s), {count} photo(s).")
            else:
                print("Reconnaissance : aucune reference dans people/ (touche 'n' pour enregistrer).")
        except Exception as exc:
            print(f"[people] chargement des references impossible : {exc}")
        finally:
            self._db_ready = True

    def reload(self):
        """Relance l'apprentissage (apres un enregistrement en direct)."""
        if self._db_thread.is_alive():
            return
        self._db_ready = False
        self._db_thread = threading.Thread(target=self._build_db_async, daemon=True)
        self._db_thread.start()

    @staticmethod
    def _embeddings_for_file(path):
        try:
            reps = DeepFace.represent(
                path,
                model_name=MODEL_NAME,
                detector_backend="mediapipe",
                enforce_detection=False,
            )
        except Exception as exc:
            print(f"[people] ignore {os.path.basename(path)} : {exc}")
            return []
        out = []
        for rep in reps:
            vec = np.asarray(rep["embedding"], dtype=np.float64)
            norm = np.linalg.norm(vec)
            if norm > 1e-9:
                out.append(vec / norm)
        return out

    def _build_db(self, people_dir):
        refs = []
        if not os.path.isdir(people_dir):
            return refs
        for person in sorted(os.listdir(people_dir)):
            folder = os.path.join(people_dir, person)
            if not os.path.isdir(folder):
                continue
            vecs = []
            for fname in sorted(os.listdir(folder)):
                if os.path.splitext(fname)[1].lower() not in IMAGE_EXTS:
                    continue
                vecs.extend(self._embeddings_for_file(os.path.join(folder, fname)))
            if vecs:
                refs.append((person, vecs))
                print(f"[people] {person} : {len(vecs)} photo(s) apprise(s).")
        return refs

    def update(self, frame):
        self._latest = frame

    def needs_frame(self):
        """True si le thread de reconnaissance a consomme sa derniere frame."""
        return self._latest is None

    def _loop(self):
        while not self._stop.is_set():
            if self._latest is not None and self._db_ready:
                try:
                    reps = DeepFace.represent(
                        self._latest,
                        model_name=MODEL_NAME,
                        detector_backend="mediapipe",
                        enforce_detection=False,
                    )
                    if reps and self._refs:
                        vec = np.asarray(reps[0]["embedding"], dtype=np.float64)
                        vec /= np.linalg.norm(vec) + 1e-9
                        best_name, best_sim = None, -1.0
                        for name, ref_vecs in self._refs:
                            sim = max(float(np.dot(vec, rv)) for rv in ref_vecs)
                            if sim > best_sim:
                                best_name, best_sim = name, sim
                        self._score = best_sim
                        self._name = best_name if best_sim >= self.threshold else None
                except Exception:
                    pass
                finally:
                    self._latest = None
            self._stop.wait(self.interval)

    def name(self):
        return self._name

    def score(self):
        return self._score

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2.0)

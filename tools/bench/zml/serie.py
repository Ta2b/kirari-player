"""Verrou partagé par les bancs Kirari de l'utilisateur, y compris entre worktrees."""

from contextlib import contextmanager
import os
from pathlib import Path
import tempfile
import time


@contextmanager
def essai_exclusif():
    if os.name == "nt":
        import msvcrt

        chemin = Path(tempfile.gettempdir()) / "kirari-banc-zml.lock"
    else:
        import fcntl

        # Des frontaux Bazel et Nsight ont leur propre TMPDIR. Tous les worktrees de la
        # session doivent néanmoins prendre le même verrou, y compris depuis un shell SSH.
        uid = os.getuid()
        runtime = Path("/run/user") / str(uid)
        if not runtime.is_dir():
            runtime = Path("/tmp")
        chemin = runtime / f"kirari-banc-zml-{uid}.lock"

    # Un bac à sable peut voir le fichier de la session sur un montage en lecture seule.
    # flock peut y prendre le verrou existant. Si le premier essai ne peut pas le créer,
    # laisser remonter l'erreur : choisir un autre fichier autoriserait deux bancs simultanés.
    try:
        fichier = chemin.open("rb") if os.name != "nt" else chemin.open("a+b")
    except FileNotFoundError:
        fichier = chemin.open("a+b")
    with fichier as verrou:
        if os.name == "nt":
            if verrou.tell() == 0:
                verrou.write(b"0")
                verrou.flush()
            verrou.seek(0)
            while True:
                try:
                    msvcrt.locking(verrou.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except PermissionError:
                    time.sleep(1)
        else:
            fcntl.flock(verrou.fileno(), fcntl.LOCK_EX)
        yield


if __name__ == "__main__":
    import subprocess
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("Usage : python serie.py commande [arguments…]")
    with essai_exclusif():
        raise SystemExit(subprocess.run(sys.argv[1:]).returncode)

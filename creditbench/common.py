import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Managed macOS environments may not allow writes to ~/.matplotlib.  Keep the
# cache with the other ignored build artifacts so plotting is deterministic and
# multiprocessing never falls back to a new temporary font cache per process.
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "build" / "matplotlib"))


def dump_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def find_rscript(explicit=None):
    candidate = explicit or os.environ.get("RSCRIPT") or shutil.which("Rscript")
    if candidate:
        if Path(candidate).is_file():
            return str(Path(candidate).resolve())
        raise FileNotFoundError(f"Rscript not found: {candidate}")
    if os.name == "nt":
        import winreg
        for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
            try:
                with winreg.OpenKey(hive, r"SOFTWARE\R-core\R") as key:
                    base, _ = winreg.QueryValueEx(key, "InstallPath")
                    path = Path(base) / "bin" / "Rscript.exe"
                    if path.exists():
                        return str(path)
            except OSError:
                pass
    raise FileNotFoundError("Install R and set RSCRIPT to the full Rscript path, or use --rscript.")


def child_env():
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    env["R_LIBS_USER"] = str(ROOT / ".Rlib")
    env["PYTHONUTF8"] = "1"
    # The host can inject C.UTF-8, which older Windows R does not support.
    if os.name == "nt":
        env["LC_ALL"] = "C"
        env["LANG"] = "C"
    return env


def hardware():
    import importlib.metadata
    import psutil
    cpu = platform.processor()
    if os.name == "nt":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                cpu = winreg.QueryValueEx(k, "ProcessorNameString")[0]
        except OSError:
            pass
    return {"platform": platform.platform(), "cpu": cpu, "python": platform.python_version(),
            "logical_cpus": os.cpu_count(), "physical_cores": psutil.cpu_count(logical=False),
            "ram_gib": psutil.virtual_memory().total / 2**30,
            "versions": {p: importlib.metadata.version(p) for p in
                         ("numpy", "pandas", "scipy", "scikit-learn", "psutil", "matplotlib")}}


def write_rows(path, rows):
    rows = list(rows)
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def allocations(cores, hybrid_outer):
    if cores < 4 or hybrid_outer < 2 or cores % hybrid_outer or cores // hybrid_outer < 2:
        raise ValueError("Four-way comparison needs cores >= 4 and a divisor 1 < hybrid_outer < cores.")
    available = os.cpu_count() or 1
    if cores > available:
        raise ValueError(f"Requested {cores} CPU slots; only {available} logical CPUs available")
    return {"serial": (1, 1), "outer": (cores, 1), "inner": (1, cores),
            "hybrid": (hybrid_outer, cores // hybrid_outer)}

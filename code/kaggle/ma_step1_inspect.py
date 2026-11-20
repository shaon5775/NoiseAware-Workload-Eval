# Paper 8 M-A step 1: download Shin 2018 EEG+NIRS (TU Berlin) and print the file format. No modelling.
import os, subprocess, glob, zipfile, time
D = "/kaggle/tmp/shin"; os.makedirs(D, exist_ok=True)
base = "http://doc.ml.tu-berlin.de/simultaneous_EEG_NIRS"
for kind, f in [("NIRS", "NIRS_01-26_MATLAB.zip"), ("EEG", "EEG_01-26_MATLAB.zip")]:
    t = time.time(); p = f"{D}/{f}"
    if not os.path.exists(p):
        subprocess.run(["wget", "-q", "-O", p, f"{base}/{kind}/{f}"], check=True)
    print(kind, round(os.path.getsize(p) / 1e9, 2), "GB in", int(time.time() - t), "s", flush=True)
    with zipfile.ZipFile(p) as z:
        names = z.namelist(); print(len(names), "files; first 40:"); print("\n".join(names[:40]))
        z.extractall(f"{D}/{kind}")
    os.remove(p)

import scipy.io as sio, numpy as np
def show(path):
    print("\n=====", path, round(os.path.getsize(path) / 1e6, 1), "MB")
    try:
        m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    except NotImplementedError:
        import h5py; h = h5py.File(path); print("HDF5 keys:", list(h.keys())); return
    for k, v in m.items():
        if k.startswith("__"): continue
        print("key", k, type(v))
        objs = v if isinstance(v, np.ndarray) and v.dtype == object else [v]
        for i, o in enumerate(list(objs)[:3]):
            if hasattr(o, "_fieldnames"):
                for fn in o._fieldnames:
                    a = getattr(o, fn); s = getattr(a, "shape", None)
                    print(f"  [{i}].{fn}:", type(a).__name__, s, (str(a)[:200] if s is None or np.size(a) < 40 else ""))
for kind in ("NIRS", "EEG"):
    mats = sorted(glob.glob(f"{D}/{kind}/**/*.mat", recursive=True))
    print("\n#", kind, len(mats), "mat files; sample names:", [os.path.relpath(x, D) for x in mats[:12]])
    for p in [x for x in mats if "VP001" in x or "01" in os.path.basename(x)][:6]:
        show(p)
print("\nDISK:", subprocess.run(["du", "-sh", D], capture_output=True, text=True).stdout)

# Paper 8 — inspect iNCog-EEG (Shafi et al. 2026, Figshare 10.6084/m9.figshare.30003127.v2, CC BY 4.0).
# Question: does it contain task-performance / event logs (needed for RB-ME) and EOG/EMG channels?
import os, subprocess, zipfile, glob, collections
subprocess.run(["pip", "install", "-q", "mne"], check=False)
import mne; mne.set_log_level("ERROR")
W = "/kaggle/tmp/incog"; os.makedirs(W, exist_ok=True); z = f"{W}/incog.zip"
subprocess.run(["wget", "-q", "-O", z, "https://ndownloader.figshare.com/files/57484816"], check=True)
zipfile.ZipFile(z).extractall(W); os.remove(z)
files = [p for p in glob.glob(f"{W}/**/*", recursive=True) if os.path.isfile(p)]
ext = collections.Counter(os.path.splitext(p)[1].lower() for p in files)
print("n files", len(files), "extensions", dict(ext))
non_edf = [p.replace(W, "") for p in files if not p.lower().endswith(".edf")]
print("NON-EDF FILES:", non_edf[:50])
edfs = sorted(p for p in files if p.lower().endswith(".edf"))
print("first EDFs:", [p.replace(W, "") for p in edfs[:6]])
for p in edfs[:4] + [e for e in edfs if "sub35" in e.lower()][:2]:
    raw = mne.io.read_raw_edf(p, preload=False)
    ann = raw.annotations
    print("\n==", p.replace(W, ""), "| sfreq", raw.info["sfreq"], "| dur s", round(raw.times[-1], 1), "| n ch", len(raw.ch_names))
    print("   channels:", raw.ch_names)
    print("   channel types:", dict(collections.Counter(raw.get_channel_types())))
    print("   annotations:", len(ann), collections.Counter(ann.description).most_common(15))
    try:
        ev = mne.find_events(raw, stim_channel=None, verbose=False); print("   stim events", len(ev))
    except Exception as e:
        print("   no stim channel:", repr(e)[:80])
# summary over all files
durs = collections.defaultdict(list); nann = collections.Counter()
for p in edfs:
    r = mne.io.read_raw_edf(p, preload=False); key = os.path.basename(p).split("_")[-1].lower()
    durs[key].append(round(r.times[-1])); nann[key] += len(r.annotations)
print("\nDURATIONS by condition (s):", {k: (min(v), max(v), len(v)) for k, v in durs.items()})
print("ANNOTATIONS total by condition:", dict(nann))

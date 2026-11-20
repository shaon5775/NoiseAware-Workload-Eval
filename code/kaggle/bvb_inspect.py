# Paper 8 "brain vs behaviour": inspect file formats of COG-BCI (1 person), Pavlov ds003838 (1 person, small files),
# and Berlin Shin 2018 vendor markers (1 person). No modelling.
import os, subprocess, glob, zipfile, json, urllib.request, re
W = "/kaggle/tmp/insp"; os.makedirs(W, exist_ok=True)
def sh(c): print(subprocess.run(c, shell=True, capture_output=True, text=True).stdout[-6000:])
def get(url, path):
    if not os.path.exists(path): subprocess.run(["wget", "-q", "-O", path, url], check=True)
    print("got", path, round(os.path.getsize(path) / 1e6, 1), "MB", flush=True)

print("\n########## COG-BCI")
get("https://zenodo.org/records/7413650/files/sub-01.zip?download=1", f"{W}/cog_sub01.zip")
get("https://zenodo.org/records/7413650/files/COG-BCI_info.pdf?download=1", f"{W}/cog_info.pdf")
for f in ("RSME.txt", "KSS.txt"):
    get(f"https://zenodo.org/records/7413650/files/{f}?download=1", f"{W}/{f}"); print(open(f"{W}/{f}", errors="ignore").read()[:600])
zipfile.ZipFile(f"{W}/cog_sub01.zip").extractall(f"{W}/cog"); sh(f"cd {W}/cog && find . | head -80")
for p in sorted(glob.glob(f"{W}/cog/**/*events*.tsv", recursive=True))[:6] + sorted(glob.glob(f"{W}/cog/**/*.json", recursive=True))[:3]:
    print("\n---", os.path.relpath(p, W)); print(open(p, errors="ignore").read()[:1500])
for p in sorted(glob.glob(f"{W}/cog/**/*", recursive=True)):
    if os.path.isfile(p) and not p.endswith((".tsv", ".json")): print("FILE", os.path.relpath(p, W), round(os.path.getsize(p) / 1e6, 1), "MB")
try:
    import pypdf; t = "\n".join(pg.extract_text() or "" for pg in pypdf.PdfReader(f"{W}/cog_info.pdf").pages)
except Exception as e:
    sh(f"pdftotext {W}/cog_info.pdf {W}/cog_info.txt"); t = open(f"{W}/cog_info.txt", errors="ignore").read() if os.path.exists(f"{W}/cog_info.txt") else repr(e)
print("\n--- COG-BCI_info.pdf text (first 6000 chars)\n", t[:6000])

print("\n########## Pavlov ds003838 (small files of one person with EEG)")
base = "https://s3.amazonaws.com/openneuro.org/ds003838"
lst = urllib.request.urlopen("https://s3.amazonaws.com/openneuro.org?list-type=2&prefix=ds003838/sub-032/&max-keys=200").read().decode()
keys = re.findall(r"<Key>([^<]+)</Key>", lst); sizes = re.findall(r"<Size>(\d+)</Size>", lst)
for k, s in zip(keys, sizes): print(k, round(int(s) / 1e6, 1), "MB")
for k, s in zip(keys, sizes):
    if int(s) < 3e6 and k.endswith((".tsv", ".json")):
        print("\n---", k); print(urllib.request.urlopen(f"https://s3.amazonaws.com/openneuro.org/{k}").read().decode(errors="ignore")[:1500])
print(urllib.request.urlopen(f"{base}/participants.tsv").read().decode()[:800])

print("\n########## Berlin Shin 2018 vendor EEG markers (VP001)")
get("http://doc.ml.tu-berlin.de/simultaneous_EEG_NIRS/EEG/VP001.zip", f"{W}/vp001.zip")
zipfile.ZipFile(f"{W}/vp001.zip").extractall(f"{W}/vp001"); sh(f"cd {W}/vp001 && find . | head -40")
for p in glob.glob(f"{W}/vp001/**/*.vmrk", recursive=True):
    txt = open(p, errors="ignore").read(); print("\n---", os.path.relpath(p, W), "lines", txt.count("\n"))
    marks = re.findall(r"^Mk\d+=([^,]*),([^,]*),", txt, re.M); from collections import Counter
    print(Counter(marks).most_common(40)); print(txt[:1200])

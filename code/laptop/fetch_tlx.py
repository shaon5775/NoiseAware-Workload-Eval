"""Download Tufts fNIRS2MW NASA-TLX (per subject folder) and task_accuracy CSVs from the public Box share."""
import re, os, time, urllib.request

ITEM = re.compile(r'"typedID":"([fd])_(\d+)".{0,300}?"name":"([^"]+)","itemSize":(\d+)')
def get(url):
    for k in range(4):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=60).read()
        except Exception as e:
            time.sleep(2 + 3 * k); err = e
    raise err

def listing(share, folder=None):
    out, seen = [], set()
    for p in range(1, 8):
        url = f"https://tufts.app.box.com/s/{share}" + (f"/folder/{folder}" if folder else "") + f"?page={p}"
        new = [m for m in ITEM.findall(get(url).decode("utf-8", "ignore")) if m[1] not in seen]
        if not new: break
        for m in new: seen.add(m[1]); out.append(m)
    return out

def dl(share, fid, path):
    if os.path.exists(path) and os.path.getsize(path) > 0: return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = get(f"https://tufts.app.box.com/index.php?rm=box_download_shared_file&shared_name={share}&file_id=f_{fid}")
    open(path, "wb").write(data); time.sleep(0.15)

ACC, TLX = "gme0gsc6z9vj0v4dhchtqh04b4fpy8tm", "c5y3m59wwwem0j792epjxhnq4uei5swq"
acc = [m for m in listing(ACC) if m[0] == "f"]
for _, fid, name, _ in acc: dl(ACC, fid, f"data/task_accuracy/{name}")
print("task_accuracy files:", len(acc), flush=True)
n = 0
for _, did, sub, _ in [m for m in listing(TLX) if m[0] == "d"]:
    for _, fid, name, _ in listing(TLX, did):
        dl(TLX, fid, f"data/nasa_tlx/{sub}/{name}"); n += 1
    print(sub, n, flush=True)
print("TLX files:", n)

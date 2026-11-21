"""Build artefact2_kaggle.py + paper8-artefact2.ipynb: embed MATB split-half behaviour + unit keys into artefact2_body.py."""
import io, base64, json, numpy as np, os
H = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(H, "..", "data")
A = dict(np.load(os.path.join(D, "ab_halves.npz"), allow_pickle=True)); K = dict(np.load(os.path.join(D, "unit_keys.npz"), allow_pickle=True))
buf = io.BytesIO()
np.savez_compressed(buf, key=K["S2_key"].astype(str), s=A["S2_s"].astype(str), y=A["S2_y"].astype(int),
                    A=A["S2_A"].astype(np.float32), B=A["S2_B"].astype(np.float32))
emb = base64.b64encode(buf.getvalue()).decode()
body = open(os.path.join(H, "artefact2_body.py"), encoding="utf-8").read()
code = body.replace('if __name__ == "__main__" or True:', f'EMB = "{emb}"\nif __name__ == "__main__" or True:', 1)
open(os.path.join(H, "artefact2_kaggle.py"), "w", encoding="utf-8").write(code)
nb = {"cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": ["!pip install -q mne"]},
                {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": code.splitlines(True)}],
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}}, "nbformat": 4, "nbformat_minor": 4}
json.dump(nb, open(os.path.join(H, "paper8-artefact2.ipynb"), "w", encoding="utf-8"))
print("embedded bytes", len(emb), "script chars", len(code))

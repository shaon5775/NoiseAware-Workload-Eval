"""Build Online Resource 1 (ESM_1.pdf): full chronological test history for Paper 8.
Source rows: github_stage/TEST_HISTORY.md (P8-A ... P8-DEEP) + TEST_LOG.md rows P8-ARTEFACT, P8-ARTNULL, P8-ARTEFACT2.
Writes online_resource/ESM_1.html; print to PDF with Edge headless (see bottom)."""
import re, html, os, subprocess
H = os.path.dirname(os.path.abspath(__file__))
rows = [l for l in open(os.path.join(H, "github_stage", "TEST_HISTORY.md"), encoding="utf-8").read().split("\n") if l.startswith("| P8-")]
# (github_stage/TEST_HISTORY.md now also holds the P8-ARTEFACT, P8-ARTNULL, P8-ARTEFACT2 and P8-KEYPRESS rows)

def clean(s):
    s = re.sub(r"Paper 8( idea \d)?( \([^)]*\))? — ", "", s)
    s = s.replace("paper8/pilot_", "preregistrations/pilot_").replace("paper8/", "")
    s = s.replace("(owner decides)", "").replace("; to decide", "").replace("Next: owner decides; ", "Next: ")
    s = s.replace("Next: decide next step; ", "Next: ")
    s = s.replace("Pre-registered", "Pre-specified").replace("pre-registered", "pre-specified").replace("PRE-REGISTERED", "PRE-SPECIFIED")
    s = html.escape(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    return s.replace("✅", "PASS").replace("❌", "FAIL").replace("ℹ️", "info").replace("✓", "yes").replace("✗", "no")

body = []
for k, r in enumerate(rows, 1):
    c = [x.strip() for x in r.replace("|ρ|", "abs(ρ)").strip().strip("|").split("|")]
    if len(c) < 5: continue
    body.append(f"<tr><td>{k}</td><td>{clean(c[0])}</td><td>{clean(c[1])}</td><td>{clean(c[2])}</td><td>{clean(c[3])}</td><td>{clean(c[4])}</td></tr>")

doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>Online Resource 1</title>
<style>@page {{ size: A4 landscape; margin: 14mm; }} body {{ font-family: Arial, Helvetica, sans-serif; font-size: 8.3pt; color: #000; }}
h1 {{ font-size: 13pt; margin: 0 0 4px; }} .meta {{ font-size: 9pt; margin-bottom: 8px; line-height: 1.35; }}
table {{ border-collapse: collapse; width: 100%; }} th, td {{ border: 0.5pt solid #888; padding: 3px 4px; vertical-align: top; }}
th {{ background: #e8e8e8; text-align: left; }} tr {{ page-break-inside: avoid; }} p {{ font-size: 9pt; }}</style></head><body>
<h1>Online Resource 1 &mdash; Complete chronological record of all tests</h1>
<div class="meta"><b>Article:</b> Does the brain know more than performance shows? Two traps in evaluating EEG and fNIRS mental-workload decoders<br>
<b>Journal:</b> Cognitive Neurodynamics<br>
<b>Authors:</b> Shaon Bandyopadhyay, Rajib Kumar Mondal<br>
<b>Corresponding author:</b> Shaon Bandyopadhyay, School of Technology, The Neotia University, Sarisha, West Bengal, India; shaon.bandyopadhyay@gmail.com</div>
<p>Every test run during this study is listed in the order it was run, including failed ideas and exploratory checks. For each pre-specified test the
rules (data, model, statistics and pass/fail criteria) were written in a dated file before the run; these files, the code and the result files are in the
public repository given under Code availability. "Room checks" are short exploratory analyses that were not pre-specified. Decoder accuracies are balanced
accuracies with leave-one-person-out cross-validation unless stated; &Delta; is the RB-ME statistic (brain AUC minus the bound of a perfect performance reader).</p>
<table><thead><tr><th>#</th><th>ID</th><th>Test (rules file; data; compute)</th><th>Outcome</th><th>Key numbers</th><th>Lesson and next step</th></tr></thead>
<tbody>{''.join(body)}</tbody></table></body></html>"""
os.makedirs(os.path.join(H, "online_resource"), exist_ok=True)
out = os.path.join(H, "online_resource", "ESM_1.html"); open(out, "w", encoding="utf-8").write(doc); print(len(body), "rows ->", out)
edge = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
pdf = os.path.join(H, "online_resource", "ESM_1.pdf")
subprocess.run([edge, "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}", "file:///" + out.replace("\\", "/")], check=False, timeout=120)
print("pdf exists", os.path.exists(pdf), os.path.getsize(pdf) if os.path.exists(pdf) else 0)

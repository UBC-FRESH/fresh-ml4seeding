#!/usr/bin/env python3
"""
Active-learning shortlist for microsite annotation.

1. Find flight frames NOT in the v4 dataset (~202 of 750)
2. unet_resnet34_focal + TTA (8 dihedral views) inference
3. Score: mean per-pixel variance across TTA views (disagreement) +
   entropy-band fraction. Rank, enforce min spacing along flight.
4. Embedded-HTML gallery of the top ~50 with prediction overlays.
"""
import base64
import json
import re
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import cv2
import segmentation_models_pytorch as smp

SMALL = Path("/home/gep/projects/ignitebc-flashforest/annotation/a58_small")
V4 = Path("/home/gep/projects/ignitebc-flashforest/annotation/roboflow_v4")
OUT = Path("/home/gep/projects/ignitebc-flashforest/annotation/active_learning")
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda")
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

TOP_N = 50
MIN_GAP = 3  # min frame-index spacing in shortlist

# ---------- 1. unannotated pool ----------
model_names_in_v4 = set()
for split in ["train", "valid", "test"]:
    for p in (V4 / split).glob("*.jpg"):
        if p.name.endswith("_mask.png"):
            continue
        base = re.sub(r"\.rf\.[a-f0-9]+", "", p.name).replace("_JPG.jpg", ".JPG")
        m = re.match(r"a58_\d+_(?:early|mid|late)_(\d+)_jpg", base.replace(".JPG", ""))
        if m:
            base = f"__frame_{int(m.group(1)):04d}"
        model_names_in_v4.add(base)


def frame_key(stem):
    m = re.search(r"_(\d{4})$", stem)
    return f"__frame_{int(m.group(1)):04d}" if m else None


pool = []
for p in sorted(SMALL.glob("*.JPG")):
    if p.name.replace("_JPG", ".JPG") not in model_names_in_v4 and \
       frame_key(p.stem) not in model_names_in_v4:
        pool.append(p)
print(f"unannotated pool: {len(pool)} frames")

# ---------- 2. inference with TTA ----------
model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=2).to(DEVICE)
model.load_state_dict(torch.load(
    "/home/gep/projects/ignitebc-flashforest/annotation/unet_resnet34_focal_best.pt",
    map_location=DEVICE))
model.eval()


def dihedral(x, k, flip):
    if flip:
        x = np.flip(x, axis=3)
    return np.rot90(x, k, axes=(2, 3))


def dihedral_inv(p, k, flip):
    p = np.rot90(p, -k, axes=(0, 1))
    if flip:
        p = np.flip(p, axis=1)
    return p


@torch.no_grad()
def tta_probs(img):
    """img: HxWx3 normalized. Returns list of 8 prob maps."""
    out = []
    x0 = img.transpose(2, 0, 1)[None]
    for k in range(4):
        for f in (False, True):
            x = dihedral(x0, k, f)
            t = torch.from_numpy(np.ascontiguousarray(x).astype(np.float32)).to(DEVICE)
            p = torch.softmax(model(t), 1)[0, 1].cpu().numpy()
            out.append(dihedral_inv(p, k, f))
    return out


records = []
for i, p in enumerate(pool):
    img = np.array(Image.open(p).convert("RGB").resize((1024, 1024))).astype(np.float32) / 255.0
    probs = tta_probs((img - MEAN) / STD)
    stack = np.stack(probs)
    mean_p = stack.mean(0)
    var_p = stack.var(0)
    disagreement = float(var_p.mean())
    band = float((np.abs(mean_p - 0.5) < 0.2).mean())  # fraction of uncertain pixels
    entropy = float(-(mean_p * np.log2(mean_p + 1e-9) + (1 - mean_p) * np.log2(1 - mean_p + 1e-9)).mean())
    ms_frac = float((mean_p > 0.5).mean())
    fnum = int(re.search(r"_(\d{4})$", p.stem).group(1))
    records.append({"frame": p.stem, "frame_num": fnum,
                    "disagreement": disagreement, "uncertain_band": band,
                    "entropy": entropy, "microsite_frac": ms_frac})
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(pool)}")

# combined score: disagreement is primary, band breaks ties
for r in records:
    r["score"] = r["disagreement"] * 1000 + r["uncertain_band"]

records.sort(key=lambda r: -r["score"])

# ---------- 3. diversity-enforced shortlist ----------
shortlist = []
for r in records:
    if all(abs(r["frame_num"] - s["frame_num"]) >= MIN_GAP for s in shortlist):
        shortlist.append(r)
    if len(shortlist) >= TOP_N:
        break
print(f"shortlist: {len(shortlist)} frames, "
          f"score range {shortlist[-1]['score']:.2f}..{shortlist[0]['score']:.2f}")

# ---------- 4. gallery ----------
def b64(img_bgr, q=85):
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


tiles_html = []
for r in shortlist:
    p = SMALL / f"{r['frame']}.JPG"
    img = cv2.imread(str(p))
    img_rgb = np.array(Image.open(p).convert("RGB").resize((1024, 1024))).astype(np.float32) / 255.0
    probs = tta_probs((img_rgb - MEAN) / STD)
    stack = np.stack(probs)
    mean_p = stack.mean(0)
    var_p = stack.var(0)

    vis = cv2.resize(img, (1024, 1024)).astype(np.float32)
    ms = mean_p > 0.5
    band = np.abs(mean_p - 0.5) < 0.2
    # microsite: green; uncertain band: yellow; high disagreement: red overlay
    vis[ms] = 0.6 * vis[ms] + 0.4 * np.array([0, 200, 0])
    vis[band] = 0.5 * vis[band] + 0.5 * np.array([0, 255, 255])
    hot = var_p > np.percentile(var_p, 95)
    vis[hot] = 0.4 * vis[hot] + 0.6 * np.array([0, 0, 255])
    vis = vis.astype(np.uint8)
    vis = cv2.resize(vis, (768, 768))
    cap = (f"#{r['frame_num']} {r['frame']}  |  score {r['score']:.2f}  "
           f"disagree {r['disagreement']:.4f}  band {r['uncertain_band']*100:.0f}%  "
           f"ms-frac {r['microsite_frac']*100:.0f}%")
    cv2.rectangle(vis, (0, 0), (768, 34), (0, 0, 0), -1)
    cv2.putText(vis, cap, (6, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    tiles_html.append(f'<div class="item"><img src="{b64(vis)}"><div class="cap">{cap}</div></div>')

html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Active-learning shortlist</title>
<style>
body {{ font-family: sans-serif; background:#111; color:#ddd; margin:20px }}
img {{ max-width: 100%; border:1px solid #444 }}
.item {{ margin-bottom: 24px }}
.cap {{ color:#999; font-size: 12px }}
</style></head><body>
<h1>Active-learning shortlist: top {len(shortlist)} of {len(pool)} unannotated A58 frames</h1>
<p>Model: unet_resnet34_focal + TTA. Overlay: <b style="color:#4c4">green = predicted microsite</b>,
<b style="color:#cc4">yellow = uncertain band (|p-0.5|&lt;0.2)</b>,
<b style="color:#f44">red = top-5% TTA disagreement</b>.
Ranked by TTA-view disagreement; frames spaced &ge;{MIN_GAP} apart for site coverage.
Full ranking in shortlist.json.</p>
{''.join(tiles_html)}
</body></html>"""
(OUT / "shortlist_gallery.html").write_text(html)

with open(OUT / "shortlist.json", "w") as f:
    json.dump({"model": "unet_resnet34_focal + TTA(8)", "pool_size": len(pool),
               "top_n": TOP_N, "min_gap": MIN_GAP,
               "shortlist": [r["frame"] for r in shortlist],
               "ranking": records}, f, indent=2)

with open(OUT / "shortlist_frames.txt", "w") as f:
    for r in shortlist:
        f.write(f"{r['frame']}.JPG\n")

print(f"\nGallery: {OUT/'shortlist_gallery.html'}")
print(f"Ranking: {OUT/'shortlist.json'}")
print(f"Frame list: {OUT/'shortlist_frames.txt'}")
print("\nTop 10:")
for r in shortlist[:10]:
    print(f"  {r['frame']}  score={r['score']:.2f} disagree={r['disagreement']:.4f} "
          f"band={r['uncertain_band']:.2f} ms={r['microsite_frac']:.2f}")

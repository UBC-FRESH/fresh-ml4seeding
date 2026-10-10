#!/usr/bin/env python3
"""
Build an inspection gallery for per-image nope masks:
- Section A: diverse full-frame overlays (native 1024x682)
- Section B: 3x zoom crops around detected objects (edge inspection)
- Section C: the 6 user-annotated frames with GT polygons over the mask
Output: annotation/nope_map/gallery/*.png + index.html
"""
import json
import re
import numpy as np
from pathlib import Path
import cv2

SMALL = Path("/home/gep/projects/ignitebc-flashforest/annotation/a58_small")
MASKS = Path("/home/gep/projects/ignitebc-flashforest/annotation/nope_map/per_image")
COCO = Path("/home/gep/projects/ignitebc-flashforest/annotation/a58_features_v3/train/_annotations.coco.json")
OUT = Path("/home/gep/projects/ignitebc-flashforest/annotation/nope_map/gallery")
OUT.mkdir(parents=True, exist_ok=True)

from pycocotools import mask as mask_utils


def overlay(rgb, mask, alpha=0.45):
    """rgb: BGR uint8; mask: 0 ok / 1 bump / 2 pit / 255 nodata"""
    vis = rgb.copy()
    bump = mask == 1
    pit = mask == 2
    nodata = mask == 255
    vis[nodata] = (vis[nodata] * 0.35).astype(np.uint8)
    vis[bump] = (vis[bump] * (1 - alpha) + alpha * np.array([0, 0, 255])).astype(np.uint8)
    vis[pit] = (vis[pit] * (1 - alpha) + alpha * np.array([255, 128, 0])).astype(np.uint8)
    # crisp boundaries for edge inspection
    for val, color in [(1, (0, 255, 255)), (2, (255, 255, 0))]:
        m = (mask == val).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, color, 1)
    return vis


def caption(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return out


# ---------- Section A: diverse full-frame overlays ----------
# select by frame NUMBER (timestamps are irregular)
def stem_for_frame(num):
    hits = sorted(SMALL.glob(f"*_{num:04d}.JPG"))
    return hits[0].stem if hits else None

frames_a = [1, 10, 203, 300, 407, 450, 500, 550, 600, 650, 700, 748]
made = 0
for fnum in frames_a:
    stem = stem_for_frame(fnum)
    if stem is None:
        continue
    img_p = SMALL / f"{stem}.JPG"
    mask_p = MASKS / f"{stem}.png"
    if not img_p.exists() or not mask_p.exists():
        # find nearest available frame with that number
        continue
    img = cv2.imread(str(img_p))
    mask = cv2.imread(str(mask_p), cv2.IMREAD_UNCHANGED)
    vis = overlay(img, mask)
    nope_pct = 100 * ((mask == 1) | (mask == 2)).sum() / max((mask != 255).sum(), 1)
    vis = caption(vis, f"{stem}  |  nope {nope_pct:.0f}% of valid  (red=bump, blue=pit, yellow=edge, dim=no-data)")
    cv2.imwrite(str(OUT / f"A_{stem}.png"), vis)
    made += 1
print(f"Section A: {made} full-frame overlays")

# ---------- Section B: 3x zoom crops around detected objects ----------
zoom_frames = [203, 450, 600, 748]
nzoom = 0
for fnum in zoom_frames:
    stem = stem_for_frame(fnum)
    if stem is None:
        continue
    img_p, mask_p = SMALL / f"{stem}.JPG", MASKS / f"{stem}.png"
    img = cv2.imread(str(img_p))
    mask = cv2.imread(str(mask_p), cv2.IMREAD_UNCHANGED)
    # find up to 2 big bump blobs, crop around them
    n, lab = cv2.connectedComponents((mask == 1).astype(np.uint8))
    sizes = [(lab == i).sum() for i in range(1, n)]
    order = np.argsort(sizes)[::-1][:2]
    for j, idx in enumerate(order):
        if sizes[idx] < 40:
            continue
        ys, xs = np.nonzero(lab == idx + 1)
        cy, cx = int(ys.mean()), int(xs.mean())
        r = 110
        y0, y1 = max(0, cy - r), min(img.shape[0], cy + r)
        x0, x1 = max(0, cx - r * 2), min(img.shape[1], cx + r * 2)
        crop_img = img[y0:y1, x0:x1]
        crop_mask = mask[y0:y1, x0:x1]
        vis = overlay(crop_img, crop_mask, alpha=0.4)
        vis = cv2.resize(vis, (vis.shape[1] * 2, vis.shape[0] * 2),
                         interpolation=cv2.INTER_NEAREST)
        vis = caption(vis, f"{stem} crop @2x around object #{j+1} ({sizes[idx]} mask px)")
        cv2.imwrite(str(OUT / f"B_{stem}_z{j}.png"), vis)
        nzoom += 1
print(f"Section B: {nzoom} zoom crops")

# ---------- Section C: annotated frames with GT polygons ----------
coco = json.load(open(COCO))
cat_names = {c["id"]: c["name"] for c in coco["categories"]}
gt_by_file = {}
for im in coco["images"]:
    anns = [a for a in coco["annotations"] if a["image_id"] == im["id"]]
    gt_by_file[im["file_name"]] = anns

ngt = 0
for fname, anns in sorted(gt_by_file.items()):
    m = re.search(r"DJI_\d+_\d+", fname)
    if not m:
        continue
    stem = m.group(0)
    img_p, mask_p = SMALL / f"{stem}.JPG", MASKS / f"{stem}.png"
    if not img_p.exists() or not mask_p.exists():
        continue
    img = cv2.imread(str(img_p))                       # 1024x682
    mask = cv2.imread(str(mask_p), cv2.IMREAD_UNCHANGED)
    vis = overlay(img, mask, alpha=0.35)
    # GT polygons are in the 1024x1024 stretched frame -> squash y by 682/1024
    hit, miss = 0, 0
    for a in anns:
        seg = a["segmentation"]
        gm = mask_utils.decode(seg).astype(np.uint8) if isinstance(seg, dict) else None
        if gm is None:
            pts = np.array(seg[0]).reshape(-1, 2).astype(np.int32)
            gm = np.zeros((1024, 1024), np.uint8)
            cv2.fillPoly(gm, [pts], 1)
        # instance hit test (same rule as validation: >=20% area covered by mask)
        gm_small = cv2.resize(gm, (1024, 682), interpolation=cv2.INTER_NEAREST)
        covered = ((mask == 1) & (gm_small > 0)).sum()
        is_hit = covered >= 0.2 * gm_small.sum()
        hit += is_hit
        miss += (not is_hit)
        # draw polygon outline (approx from mask contours)
        cnts, _ = cv2.findContours(gm_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = (0, 255, 0) if is_hit else (0, 0, 255)
        cv2.drawContours(vis, cnts, -1, color, 2)
    vis = caption(vis, f"{stem}  |  GT polygons: GREEN=hit, RED=missed "
                       f"({hit}/{hit+miss} hit)  |  fill: nope mask")
    cv2.imwrite(str(OUT / f"C_{stem}_gt.png"), vis)
    ngt += 1
print(f"Section C: {ngt} GT comparison frames")

# ---------- index.html ----------
def section(title, imgs, note=""):
    s = f"<h2>{title}</h2>"
    if note:
        s += f"<p>{note}</p>"
    for p, cap in imgs:
        s += f'<div class="item"><img src="{p}"><div class="cap">{cap}</div></div>\n'
    return s


import glob
a_imgs = [(Path(p).name, Path(p).name) for p in sorted(glob.glob(str(OUT / "A_*.png")))]
b_imgs = [(Path(p).name, Path(p).name) for p in sorted(glob.glob(str(OUT / "B_*.png")))]
c_imgs = [(Path(p).name, Path(p).name) for p in sorted(glob.glob(str(OUT / "C_*.png")))]

html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Nope-map per-image overlay inspection</title>
<style>
body {{ font-family: sans-serif; background:#111; color:#ddd; margin:20px }}
img {{ max-width: 100%; border:1px solid #444 }}
.item {{ margin-bottom: 24px }}
.cap {{ color:#999; font-size: 12px; margin-top:4px }}
h2 {{ border-bottom:1px solid #444; padding-bottom:4px }}
</style></head><body>
<h1>Nope-map per-image overlay inspection (A58 segment 2)</h1>
<p>Mask values: <b style="color:#f66">red fill = bump (HAG&gt;0.15 m or rough&gt;0.05 m)</b>,
<b style="color:#8af">blue = pit (HAG&lt;-0.20 m)</b>, yellow contour = mask boundary,
dimmed = no SfM data. Masks are ray-cast from the COLMAP model that reproduces
these exact frames at 0.22 px reprojection error; no georeferencing involved.</p>
{section("A. Diverse full frames (native 1024x682)", a_imgs,
         "Check: do red areas coincide with visible slash/logs/stumps/berms? Is bare ground clean?")}
{section("B. Zoom crops @2x (edge inspection)", b_imgs,
         "Check: do the yellow mask boundaries hug object edges, or bleed/miss?")}
{section("C. Your annotated frames: GT polygons vs mask", c_imgs,
         "GREEN polygon = mask covers >=20% of it (hit); RED = missed. "
         "Fill shows the nope mask underneath.")}
</body></html>"""
(OUT / "index.html").write_text(html)
print(f"Gallery: {OUT}/index.html")

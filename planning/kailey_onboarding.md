# Onboarding: Microsite ML for Aerial Seeding — Welcome Kailey!

Hi Kailey — welcome aboard. This is your orientation to the machine-learning side of the microsite project. You don't need any ML or coding background to start; this document tells you what everything is, why it exists, and exactly what to try first. Everything here assumes the ecology/forestry knowledge you already have — the ML parts are explained from zero.

---

## 1. The big picture (why this project exists)

Flash Forest wants to plant trees by dropping seed pods from drones. On a real cutblock, most of the ground is *not* plantable: slash piles, logs, stumps, rocks, standing water, compacted road. A pod dropped at random mostly lands on debris and fails. The precision-seeding idea is: fly a drone first, build a map of **where the plantable spots are**, and only drop pods there.

Our part of the research program is the ML that produces that map. The core task: given a low-altitude drone image, mark every pixel as **microsite** (plantable) or **not-microsite**. Right now "microsite" operationally means what you already recognize in the field and in imagery: exposed mineral soil pockets with decent planting conditions. One of *your* research threads — whether that definition is stable across sites and target species — is exactly the kind of question this dataset and pipeline were built to poke at, and I'll come back to it below.

## 2. The sites and the data

You know all three blocks in MKRF on the ground. Here's what we have from the air:

- **A58**: one complete low-altitude flight — **750 frames** (Zenmuse P1, ~15 m altitude, ~1.5 cm pixels, GPS in every frame). Of these, **548 are annotated** (microsite vs not, painted pixel-by-pixel by us). This is the only fully developed dataset so far.
- **A10**: 338 frames, essentially unannotated.
- **G15**: imagery exists, essentially unannotated.
- There are also photogrammetric products for A58: a 3D point cloud (715,000 points), a height map ("height above ground" for every 18 cm cell of the block), and a georeferenced obstacle map ("nope map") that flags bumps/stumps/logs/berms. More on these below — they turned out to matter for your ecology questions in an unexpected way.

Everything ML lives in two git repos on the compute server (and GitHub): the **code repo** `/home/gep/projects/ignitebc-flashforest/fresh-ml4seeding/` (package code in `src/`, experiment scripts in `scripts/`, write-ups in `planning/`, roadmap in `ROADMAP.md`) and its **data submodule** `fresh-ml4seeding/external/fresh-ml4seeding-data/` (datasets in `data/datasets/`, model weights in `data/models/`, derived products in `data/derived/`, evaluation records in `data/results/`). Both are mirrored on GitHub under UBC-FRESH, so you can browse everything there too. You never have to guess where something is — ask and I'll point you.

## 3. The current model, in plain terms

**How it works:** the model is a neural network (a "U-Net" — think of it as a very well-trained pattern-recognizer for textures) that was shown ~400 annotated drone images and learned to reproduce the human-painted microsite masks. It reads colour and texture — it has no concept of "tree" or "soil" beyond what those look like.

**How good it is:** on held-out test images (ones it never trained on), it currently scores **IoU ≈ 0.65**. IoU ("intersection over union") is the overlap between the model's microsite map and the human's: 0.65 means roughly two-thirds of the predicted microsite area matches what a person would mark. Not perfect, but good enough to be useful — and honest about its failures.

**What it gets wrong** (and this is where it gets interesting for you):

- It confuses some slash-covered ground with microsites, and vice versa — the boundary between "bare enough to plant" and "too debris-covered" is genuinely fuzzy even for people.
- It has only ever seen A58. We *expect* it to stumble on A10 and G15 (different harvest history, different ground cover, different light). Measuring exactly *how much* it stumbles is one of the first real experiments you can run.
- We tested whether adding **terrain height data** (from the 3D reconstruction) would help. It didn't — tested rigorously, published as a negative result.
- We also tested using the obstacle map as a hard "veto" (erase any predicted microsite that sits on a bump/log/pile). That *failed badly and informatively*: the veto deleted five times more true microsites than false ones, because **microsites live inside the slash matrix** — the plantable pockets are nestled among debris, not in debris-free areas. You knew this from the field; now it's a quantified, documented fact about the data.

That last result is a nice example of why your ecology expertise matters here: the ML results only make sense when someone who knows the ground interprets them.

## 4. Your two sandboxes (safe places to learn by doing)

Nothing you touch in either sandbox can break anything real.

**(A) Roboflow — the annotation and dataset platform.** Roboflow is the web tool where images get annotated and dataset versions get exported. You have (or will get) an invite to the **FRESH Workspace**. Inside it, your playground is the project **`a58-sandbox`**: all 750 raw A58 flight frames, completely unannotated — a clean slate that is yours to wreck and repaint as often as you like.

Things to try, roughly in order:
1. Open a frame, try the **polygon tool** on something obvious (a slash pile).
2. Try the **SAM3 smart-polygon** tool (click a few points on an object and the model proposes the whole outline — this is the "AI-assisted annotation" feature and it's genuinely fun).
3. Annotate one full frame as if it were production data: mark everything you'd call microsite (exposed mineral soil pockets). This will teach you two things: how long annotation actually takes (you'll immediately appreciate why we need ML), and how many judgment calls there are per image (you'll immediately appreciate why "the definition of microsite" is a research question and not a footnote).

The real projects in the same workspace are `a58-microsites` (the 548 annotated images) and `a58-features` (slash/log/rock/stump polygons). Look but don't modify those — the sandbox is for playing.

**(B) The GitHub repo — the code pipeline.** `https://github.com/UBC-FRESH/a58-microsite-sandbox` — a small, self-contained repo with three scripts and a README:

- `train_unet_focal.py` — trains the model (this is the exact recipe behind the current best model)
- `eval_tta_ensemble.py` — evaluates a saved model on test images
- `infer.py` — runs a trained model over new drone frames and writes microsite maps + overlays

The goal is that you learn to *operate* this pipeline — run scripts, read the numbers, look at the overlays — not to write or modify code. The README's quickstart is the path. All of it runs on our Linux compute server (the same JupyterHub you have access to); we'll do the first run together.

## 5. Your research thread, and how the pipeline feeds it

Your focus is the ecology question: **does "seed pod planting microsite" mean the same thing across (site × species) combinations — and what does that imply for building a model that could actually be deployed?**

Here's the honest state of play, framed as the questions we can now attack:

1. **Does the definition vary?** You're the domain expert; you'll feel it the moment you annotate A10 or G15 imagery next to A58. Different block history (harvest system, burn, time-since-disturbance) changes what "plantable" looks like — and different target species change the tolerance (a spruce pod's needs vs a pine's). The right first step is small: annotate a handful of A10 and G15 frames in your sandbox project. Don't aim for volume; aim for *noticing where you hesitate*.

2. **Does the model transfer?** This is the empirical counterpart. Once you have even ~10–20 annotated frames in another block, we run the A58-trained model on them and measure the score drop. That number — the **transfer gap** — tells us whether a "universal" model is plausible or whether we need per-site/per-ecosystem models. This experiment is designed to be run by you, with the pipeline as-is. It's also a genuinely publishable question; almost nobody has measured this for aerial seeding.

3. **The answer determines the deployment story.** If transfer is good, one model might work across clearcuts in a region. If it's bad, the honest product is either (a) a family of site/ecosystem-specific models, or (b) a big diverse training set that covers the variation — which means many more annotated blocks, which means annotation strategy becomes the bottleneck, which loops back to your judgment about where annotation effort buys the most information.

4. **The ground-truth campaign (the fun one).** We have the M300 with the L1 (photo + lidar) available to re-fly any of the three blocks. You suggested being dispatched to MKRF to physically mark microsite "flavours" — flags, paint, reflective pins — and re-fly so the marks show up in the imagery. This is *gold* for the program: it converts "a person said so on a screen" into "a person verified on the ground," which is exactly the calibration data we need to (a) measure how well image-annotation agrees with field-annotation, and (b) test the lidar terrain data for microsite-relevant structure (microtopography, drainage, moisture proxies) that our camera-only reconstruction was too coarse to capture. When you're ready, we'll design the marking protocol together — what flavours to mark (e.g., clean mineral soil, shaded pocket, ash bed, slash-adjacent pocket, compacted vs loose), how many of each, and how to keep them visible to the drone.

5. **Lidar opens the terrain question again — properly.** Our camera-only 3D reconstruction was too sparse (~26 cm between points) to see the centimetre-scale microtopography that plausibly controls moisture and seedling establishment. Lidar will be dense enough. The earlier negative result ("terrain doesn't help") came with that resolution caveat stamped on it, and it deserves a retest with real lidar once you have field-verified ground truth to test against.

## 6. Concepts you'll meet (mini-glossary, no math)

- **Annotation** – a human marking the correct answer on an image (paint or polygons). The model learns by imitation; garbage in, garbage out.
- **Training / weights / checkpoint** – the learning process, and the saved result (a `.pt` file). "Our current model" = one specific checkpoint.
- **Split (train/valid/test)** – we hide some images from training so we can measure honesty: the model is scored only on images it never studied.
- **IoU** – overlap between model output and human answer; our headline number (currently ≈ 0.65 on A58).
- **TTA (test-time augmentation)** – at prediction time, run the image through rotated/flipped and average; small free accuracy boost.
- **TTA disagreement** – how much the model changes its mind under those flips; high disagreement = the model is guessing. We use it to pick which images are most worth annotating ("active learning").
- **Pre-labels** – model-generated annotations uploaded for a human to *fix*, which is much faster than painting from scratch. (None in your sandbox — clean slate on purpose.)
- **Domain shift / transfer gap** – how much a model trained on one site/time degrades on another. Your experiment #2 above.
- **Semantic vs instance segmentation** – "which pixels are microsite" (semantic, what we do) vs "outline each individual object" (instance, what the feature project does).

## 7. First two weeks: a concrete checklist

1. Get your Roboflow invite accepted, open `a58-sandbox`, do the three annotation exercises in §4A. (~2 h)
2. Read the repo README and skim `docs/DATASETS.md`. (30 min)
3. With me (or the docs), log into the compute server and run the eval script on the existing model; then run inference on a few raw frames and eyeball the overlays. (~1 h, mostly watching it run)
4. Annotate ~10 frames in your sandbox with full care, as if production. Note where you hesitate — that hesitation list is research material.
5. We sit down with your hesitation list and design the A10/G15 transfer mini-experiment and (if the season allows) the MKRF marking trip.

## 8. Where things live / who to ask

- Everything is on GitHub under **UBC-FRESH**, also checked out on the compute server at `/home/gep/projects/ignitebc-flashforest/fresh-ml4seeding/`.
- Documentation (decisions, results, negative results — all written up): `planning/` in the main repo — especially `model_alternatives.md` (model comparison), `nope_map.md` (the obstacle-map work, incl. the veto failure), and `stereo_depth_assessment.md` (the terrain story).
- Data (datasets, model weights, derived maps, evaluation records): the DataLad repo `github.com/UBC-FRESH/fresh-ml4seeding-data` — clone it and `datalad get` what you need (e.g., `datalad get data/models data/datasets`).
- Roboflow: FRESH Workspace — `a58-sandbox` (yours), `a58-microsites` + `a58-features` (look-only).
- Code sandbox repo: `github.com/UBC-FRESH/a58-microsite-sandbox`
- Anything confusing, broken, or surprising: tell Gregory. There is no such thing as a stupid question in this project — and "surprising" results are often the publishable ones, so write them down when you hit them.

Welcome to the program. The ML is the plumbing; the interesting science is yours.


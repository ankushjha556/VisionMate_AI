# 👁️ VisionMate AI

**An AI-powered assistive companion for visually impaired users** — built as a flagship data
science portfolio project by a B.S. Computer Science & Data Analytics student at IIT Patna.

> Camera + voice in → AI reasoning → spoken guidance out. Five working modules, one consistent
> architecture, and a zero-cost build from end to end.

[![Notebook](https://img.shields.io/badge/notebook-Google%20Colab-orange)](notebooks/VisionMate_Master.ipynb)
[![Demo](https://img.shields.io/badge/demo-Streamlit-red)](streamlit_app/app.py)
[![License](https://img.shields.io/badge/license-MIT-blue)](#license)
[![Status](https://img.shields.io/badge/status-active%20development-brightgreen)]()

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Architecture](#architecture)
- [Modules](#modules)
- [Comparative Research](#comparative-research)
- [Results & Honest Limitations](#results--honest-limitations)
- [Repository Structure](#repository-structure)
- [Running the Notebook](#running-the-notebook)
- [Running the Live Demo](#running-the-live-demo)
- [Zero-Cost Stack](#zero-cost-stack)
- [Roadmap](#roadmap)
- [Project Log](#project-log)
- [License](#license)

---

## Problem Statement

Visually impaired people navigate daily life — reading a signboard, identifying a currency note,
knowing what's ahead before taking a step — using canes, guide dogs, or human assistance. Strong
assistive apps already exist (Microsoft Seeing AI, Be My Eyes), but they're largely proprietary,
cloud-locked, and not built with Indian-language users or zero-budget contexts in mind.

**VisionMate AI** is an open, modular assistive system built entirely on free and open-source
tools — designed to be a genuinely usable pipeline, not a single-model demo.

## Architecture

Every module follows the same shape, which is what keeps this feeling like one coherent system
rather than five unrelated demos stitched together:

```
                     Camera / Voice Input
                              │
                              ▼
                     ┌────────────────┐
                     │ Frame / Audio  │
                     │   Processor    │
                     └───────┬────────┘
                              │
      ┌───────────┬───────────┼───────────┬────────────┐
      ▼           ▼           ▼           ▼            ▼
 ┌─────────┐ ┌──────────┐ ┌──────────┐ ┌─────────┐ ┌───────────┐
 │  YOLO   │ │ EasyOCR  │ │MobileNet │ │  MiDaS  │ │  Whisper  │
 │ Scene   │ │  Text    │ │ Currency │ │  Depth  │ │Speech-to- │
 │ Detect  │ │ Detect   │ │ Classify │ │Estimate │ │Text+Lang  │
 └────┬────┘ └────┬─────┘ └────┬─────┘ └────┬────┘ └─────┬─────┘
      │           │            │            │            │
      ▼           ▼            ▼            ▼            ▼
 ┌────────────────────────────────────────────────────────────┐
 │                      Reasoning Layer                        │
 │   spatial zones · confidence filtering · spellcheck ·       │
 │   dedup by count · proximity classification ·               │
 │   never speak uncertainty as fact                           │
 └──────────────────────────┬───────────────────────────────────┘
                              ▼
                     ┌─────────────────┐
                     │ gTTS/Translator │
                     │ (English/Hindi) │
                     └────────┬────────┘
                              ▼
                        Spoken Output
```

**The recurring design principle across every module:** uncertain results are spoken as uncertain
("might be X, please verify"), never as fact — because a blind user can't visually double-check
the output the way a sighted user could.

## Modules

| # | Module | What it does | Models used |
|---|--------|---------------|-------------|
| 1 | **Scene Understanding** | "What is around me?" → detects objects, groups them by spatial zone (left / ahead / right), speaks a natural, prioritized summary | YOLOv11n |
| 2 | **OCR / Text Reading** | Reads signboards, labels, and documents aloud; auto-corrects common OCR misreads (e.g. "Room" → "Poom") before speaking | EasyOCR + spellchecker |
| 3 | **Currency Detection** | Identifies Indian currency notes (₹10–₹2000); flags low-confidence guesses instead of stating them as fact | Fine-tuned MobileNetV2 |
| 4 | **Obstacle Warning** | Fuses object detection with depth estimation for proximity-aware safety alerts ("3 cars close right"), deduplicated so repeated objects aren't announced one-by-one | YOLOv11n + MiDaS |
| 5 | **Multilingual Voice** | Auto-detects Hindi or English from the user's spoken command and replies in the same language — no manual toggle | Whisper + Google Translate |

**Coming soon:** Face Memory (Module 6) and Navigation & Emergency Assistance (Module 7) — both
deliberately deferred. Face Memory needs a genuine privacy/consent design (local-only storage,
explicit user control) before it's built, not bolted on as an afterthought.

## Comparative Research

Model choice for the real-time obstacle-warning module isn't arbitrary — it's backed by a
controlled benchmark across three detector architectures: a lightweight CNN (YOLOv11n), a
transformer detector (RT-DETR-L), and an open-vocabulary text-conditioned detector (Grounding
DINO). Methodology: 2 GPU warm-up runs discarded per image, 5 timed trials, mean ± std reported.

| Model | Latency (street scene) | Detections (street scene) |
|---|---|---|
| YOLOv11n | 36.3 ± 5.6 ms | 18 |
| RT-DETR-L | 70.7 ± 2.7 ms | 39 |
| Grounding DINO | 575.7 ± 25.6 ms | 40 |

**Key findings:**
- Speed and recall trade off cleanly: YOLO is ~15–20x faster than Grounding DINO but detects
  roughly half as many objects on a cluttered scene.
- Grounding DINO's recall is **prompt-bounded, not just model-bounded** — on a sparse indoor scene
  it found only 1 object despite matching RT-DETR's recall on a busy street scene, because the text
  prompt didn't describe most of what was actually present. This is a real, citable limitation of
  open-vocabulary detection.
- For real-time obstacle warnings, YOLO is the only viable choice — a 500ms+ delay before a safety
  alert is unacceptable. RT-DETR/Grounding DINO's extra recall would suit a non-real-time "describe
  everything around me" scan mode instead.

Full methodology, charts, and raw CSV are in [`results/`](results/) and the master notebook. The
complete build narrative — including what broke and why — is in [`PROJECT.md`](PROJECT.md).

## Results & Honest Limitations

| Module | Result | Known limitation |
|---|---|---|
| Scene Understanding | Real-time, coherent spoken descriptions | Limited to 80 COCO classes; no scene-composition understanding |
| OCR | Catches common misreads via spellcheck correction | Tested on clean/synthetic text and one real signboard photo; broader real-world signage not yet benchmarked |
| Currency Detection | 69% test accuracy | Trained on only ~22 images/class — validation accuracy fluctuated 52–86% due to a tiny validation set. Fix: a larger, self-captured dataset |
| Obstacle Warning | Deduplicated, proximity-aware alerts | MiDaS gives relative depth only, not metric distance |
| Multilingual Voice | Verified accurate Hindi/English routing | Currently 2 languages; Whisper transcribes Hindi phonetically (language detection still correct) |
| Research | 3-model latency/recall benchmark with proper warm-up methodology | Only 2 test images — a larger benchmark set would strengthen the claims |

**Also worth being upfront about:** this is currently an image-upload / notebook-driven demo, not
the hands-free, live-camera Android app described in the original vision. It proves the AI pipeline
works end-to-end; the native app (continuous camera feed, wake-word voice trigger, on-device
inference) is real, separate future work — see [Roadmap](#roadmap).

## Repository Structure

```
VisionMate_AI/
├── README.md
├── PROJECT.md                     ← full build narrative, decisions, mistakes, findings
├── notebooks/
│   └── VisionMate_Master.ipynb    ← consolidated, linear, GPU notebook — all 5 modules + research
├── streamlit_app/                 ← permanent, free, deployable live demo (Modules 1–3)
│   ├── app.py
│   ├── requirements.txt
│   └── packages.txt
├── hf_space/                      ← full 5-module Gradio app (needs HF PRO — see note below)
│   ├── app.py
│   ├── requirements.txt
│   └── README.md
└── results/
    ├── model_comparison.csv
    └── model_comparison_chart.png
```

## Running the Notebook

1. Open `notebooks/VisionMate_Master.ipynb` in Google Colab (T4 GPU runtime recommended).
2. Run the Setup section — it mounts Google Drive and installs all dependencies.
3. Each module section is self-contained and runnable independently after Setup.
4. Module 3 requires the Kaggle dataset (`gauravsahani/indian-currency-notes-classifier`) and a
   one-time training run if `currency_classifier.keras` isn't already saved in your Drive.

## Running the Live Demo

**Recommended: Streamlit Community Cloud** (free, persistent public link)

```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py          # test locally first
```

To deploy permanently: push this repo to a **public** GitHub repo, then go to
[share.streamlit.io](https://share.streamlit.io) → sign in with GitHub → point a new app at
`streamlit_app/app.py`. Upload your trained `currency_classifier.keras` into the same folder to
enable the Currency Detection tab. This covers Modules 1–3 with an always-on link.

**Alternative: Hugging Face Spaces (`hf_space/`)** — this app covers all 5 modules, but as of
mid-2026, Hugging Face changed free-tier policy: only **Static** Spaces stay free, while Gradio and
Docker SDKs now require a paid PRO plan. If you have PRO access (or this changes again in the
future), `hf_space/app.py` is ready to deploy as-is.

**Quick temporary link (no deployment at all):** from inside the Colab notebook, run
`demo.launch(share=True)` on the Gradio app to get a public `*.gradio.live` link valid for 72 hours
— useful for sharing something *today* while a permanent deployment is pending.

## Zero-Cost Stack

Every component here is free: Google Colab's T4 GPU tier, open-source models (YOLOv11n, MiDaS,
EasyOCR, Whisper, a self-trained MobileNetV2), a free Kaggle dataset, and free-tier Google
Translate/TTS APIs. Nothing behind a paywall was required to build or reproduce this project.

## Roadmap

- [x] Module 1 — Scene Understanding
- [x] Module 2 — OCR / Text Reading
- [x] Module 3 — Currency Detection
- [x] Module 4 — Obstacle Warning (depth-fused)
- [x] Module 5 — Multilingual Voice
- [x] Comparative model research (YOLO vs RT-DETR vs Grounding DINO)
- [x] Deployable live demo (Streamlit)
- [ ] Module 6 — Face Memory (privacy-respecting design in progress)
- [ ] Module 7 — Navigation & Emergency Assistance
- [ ] Larger, self-captured currency dataset (target: 100+ images/class)
- [ ] Native Android app — live camera feed (CameraX), on-device wake-word trigger, TFLite model
      conversion — turning this pipeline into the actual hands-free app described in the original
      vision

## Project Log

For the full, unfiltered story of how this was built — including the Kaggle 403 error, the OCR
misread that led to the spellcheck layer, the duplicated obstacle-warning bug, and the Hugging Face
Spaces policy change that forced a mid-project pivot — see [`PROJECT.md`](PROJECT.md).

## License

MIT — free to use, modify, and build on.

## Author

Built by Ankush, B.S. Computer Science & Data Analytics, IIT Patna — a flagship portfolio project
combining computer vision, edge-AI-aware engineering, and human-centered accessible design.

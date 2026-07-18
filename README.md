# VisionMate AI

**An AI-powered assistive companion for visually impaired users** — built as a flagship data
science portfolio project by a B.S. Computer Science & Data Analytics student at IIT Patna.

> Camera + voice in → AI reasoning → spoken guidance out. Five working modules, one consistent
> architecture, zero cost to build or run.

[![Notebook](https://img.shields.io/badge/notebook-Google%20Colab-orange)](notebooks/VisionMate_Master.ipynb)
[![Demo](https://img.shields.io/badge/demo-Streamlit-red)](streamlit_app/app.py)
[![License](https://img.shields.io/badge/license-MIT-blue)](#license)

---

## Problem Statement

Visually impaired people rely on canes, guide dogs, or human help for tasks sighted people take
for granted — reading a signboard, identifying a currency note, knowing what's ahead before taking
a step. Existing assistive apps (Microsoft Seeing AI, Be My Eyes) are excellent but proprietary,
cloud-dependent, and not built with Indian-language users or low-connectivity contexts in mind.

**VisionMate AI** is an open, modular system built from the ground up on free and open-source
tools, designed to be genuinely usable rather than a single-model demo.

## Architecture

Every module follows the same pipeline, which keeps the system coherent rather than five
disconnected demos bolted together:

```
   Camera / Voice Input
          │
          ▼
   ┌──────────────┐
   │ Frame/Audio  │
   │  Processor   │
   └──────┬───────┘
          │
   ┌──────┴───────────────────────────────┐
   │                                       │
   ▼                                       ▼
┌─────────┐  ┌─────────┐  ┌──────────┐  ┌─────────┐  ┌──────────────┐
│  YOLO   │  │  EasyOCR │  │ MobileNet│  │  MiDaS  │  │   Whisper    │
│ Scene   │  │   Text   │  │ Currency │  │  Depth   │  │ Speech-to-   │
│ Detect  │  │  Detect  │  │ Classify │  │ Estimate │  │ Text + Lang  │
└────┬────┘  └────┬─────┘  └────┬─────┘  └────┬────┘  └──────┬───────┘
     │            │             │             │              │
     ▼            ▼             ▼             ▼              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Reasoning Layer                              │
│  Spatial zones · confidence filtering · spellcheck · dedup ·    │
│  proximity classification · never speak uncertainty as fact     │
└──────────────────────────────┬────────────────────────────────────┘
                                ▼
                     ┌────────────────────┐
                     │  gTTS / Translator  │
                     │  (Hindi / English)  │
                     └──────────┬─────────┘
                                ▼
                          Spoken Output
```

## Modules

| # | Module | What it does | Models used |
|---|--------|---------------|-------------|
| 1 | **Scene Understanding** | "What is around me?" → detects objects, groups by spatial zone (left/ahead/right), speaks a natural prioritized summary | YOLOv11n |
| 2 | **OCR / Text Reading** | Reads signboards, labels, documents aloud; corrects common OCR misreads before speaking | EasyOCR + spellchecker |
| 3 | **Currency Detection** | Identifies Indian currency notes (₹10–₹2000); flags low-confidence guesses instead of stating them as fact | Fine-tuned MobileNetV2 |
| 4 | **Obstacle Warning** | Fuses object detection with depth estimation to warn "obstacle very close ahead" with real urgency-aware phrasing | YOLOv11n + MiDaS |
| 5 | **Multilingual Voice** | Auto-detects Hindi or English from the user's spoken command and replies in the same language | Whisper + Google Translate |

**Coming soon:** Face Memory (Module 6) and Navigation & Emergency Assistance (Module 7) — both
deliberately deferred pending further design work (privacy-respecting face storage; reliable
outdoor GPS integration).

## Comparative Research

Rather than picking a detection model arbitrarily, this project benchmarks three architectures
head-to-head — a lightweight CNN (YOLOv11n), a transformer detector (RT-DETR-L), and an
open-vocabulary text-conditioned detector (Grounding DINO) — on latency and recall, with proper
methodology (GPU warm-up runs discarded, 5 timed trials, mean ± std reported).

| Model | Latency (street scene) | Detections (street scene) |
|---|---|---|
| YOLOv11n | 36.3 ± 5.6 ms | 18 |
| RT-DETR-L | 70.7 ± 2.7 ms | 39 |
| Grounding DINO | 575.7 ± 25.6 ms | 40 |

**Takeaway:** YOLO's speed makes it the only viable choice for real-time obstacle warnings;
RT-DETR/Grounding DINO's higher recall suits non-real-time, on-demand scene scans instead. Full
methodology, charts, and CSV are in [`results/`](results/) and the master notebook.

See [`notebooks/VisionMate_Master.ipynb`](notebooks/VisionMate_Master.ipynb) for the full write-up
of findings, including a genuine limitation discovered in Grounding DINO (its recall is bounded by
prompt design, not just model capability).

## Results & Honest Limitations

| Module | Result | Known limitation |
|---|---|---|
| Scene Understanding | Real-time, coherent spoken descriptions | Limited to 80 COCO classes |
| OCR | Catches common misreads via spellcheck | Tested on clean/synthetic text; real-world signage not yet benchmarked |
| Currency Detection | 69% test accuracy | Trained on only ~22 images/class — see notebook for full breakdown and planned fix |
| Obstacle Warning | Deduplicated, proximity-aware alerts | MiDaS gives relative depth only, not metric distance |
| Multilingual Voice | Verified accurate Hindi/English routing | Currently 2 languages only |

Documenting limitations honestly, with root causes and concrete next steps, was a deliberate
choice — it reflects the actual engineering process better than presenting inflated numbers.

## Repository Structure

```
VisionMate_AI/
├── README.md
├── notebooks/
│   └── VisionMate_Master.ipynb    ← consolidated, linear, runnable end-to-end
├── streamlit_app/
│   ├── app.py                     ← interactive demo (Modules 1, 2, 3 live)
│   └── requirements.txt
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

## Running the Streamlit Demo

```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

The demo covers Scene Understanding, OCR, and Currency Detection with live image upload — a fast
way for anyone to try the core pipeline without touching Colab.

## Zero-Cost Stack

Every model and tool in this project is free and open-source: YOLOv11n, MiDaS, EasyOCR, Whisper,
and a from-scratch fine-tuned MobileNetV2, trained on Google Colab's free T4 GPU tier, using a free
Kaggle dataset and free Google Translate/TTS APIs. Nothing behind a paywall was required to build
or reproduce this project.

## Roadmap

- [x] Module 1 — Scene Understanding
- [x] Module 2 — OCR / Text Reading
- [x] Module 3 — Currency Detection
- [x] Module 4 — Obstacle Warning (depth-fused)
- [x] Module 5 — Multilingual Voice
- [x] Comparative model research (YOLO vs RT-DETR vs Grounding DINO)
- [ ] Module 6 — Face Memory (privacy-respecting design in progress)
- [ ] Module 7 — Navigation & Emergency Assistance
- [ ] Larger self-captured currency dataset
- [ ] Native Android app (Kotlin) wrapping this pipeline

## License

MIT — free to use, modify, and build on.

## Author

Built by Ankush, B.S. Computer Science & Data Analytics, IIT Patna — as a flagship portfolio
project combining computer vision, edge AI considerations, and human-centered accessible design.

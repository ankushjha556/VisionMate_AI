# VisionMate AI — Project Log

This is the full story of how I built VisionMate AI, in the order I actually built it, including
the stuff that broke and what I learned from it. I'm writing this the way I'd explain it to
another student or in an interview, not like a polished spec sheet.

---

## Why I started this

I'm a second-year B.S. Computer Science & Data Analytics student at IIT Patna, and I wanted a
flagship project — something that would actually stand out on a resume next to the usual
fraud-detection / brain-tumor-segmentation / intrusion-detection projects everyone builds off a
Kaggle dataset. Those are fine projects, but they're all "train a model on a CSV" — nobody outside
data science really understands what they do or why they matter.

I wanted something that:
- Solves a real problem for real people, not just a benchmark number
- Forces me to combine multiple areas (CV, edge AI thinking, real-time systems, human-centered
  design) instead of just one model
- I could actually build and demo, not just describe in a slide

Assistive tech for visually impaired people checked every box. Apps like Microsoft's Seeing AI or
Be My Eyes already exist and are genuinely good, but they're closed-source, cloud-locked, and not
really built with Indian languages or zero-budget constraints in mind. That gap is where this
project sits.

**The catch:** I have no money to spend on this. No paid APIs, no paid compute, nothing. Everything
had to run on free tiers — Google Colab's free T4 GPU, free datasets, open-source models. That
constraint ended up being a good thing, honestly — it forced me to actually understand model
efficiency and architecture tradeoffs instead of just calling GPT-4V for everything.

## Scoping the vision down to something buildable

My first instinct (and the one I pitched initially) was way too big: scene understanding, OCR,
currency detection, obstacle warnings, face memory, navigation, emergency alerts, multilingual
support — eight modules, all built to "the most advanced version possible."

Realistically, trying to build all eight at once, to a polished state, as one person on a free GPU
tier, would have meant nothing actually gets finished. So I cut it down to five modules for this
phase, in an order where each one builds on the last and produces something demoable on its own:

1. Scene Understanding
2. OCR / Text Reading
3. Currency Detection
4. Obstacle Warning (depth-based)
5. Multilingual Voice (Hindi/English)

Face Memory and Navigation/Emergency got pushed to "coming soon" — not because they're less
important, but because Face Memory specifically raises real privacy questions (whose face is it,
where's it stored, who consents) that deserve actual design thought rather than being bolted on in
a rush.

## The architecture I settled on

Every module follows the same shape:

```
Camera/Voice input → AI model → Reasoning layer → Speech output
```

I kept coming back to this pattern because the "reasoning layer" part is where most of the actual
engineering work happens, and it's the same layer of thinking across every module: don't just dump
raw model output at the user, translate it into something a blind person can act on safely. Some
concrete examples of what that meant in practice:

- If YOLO detects 8 people in a crowd, don't say "person" eight times — group it into "8 people
  ahead."
- If OCR reads a word with 56% confidence, don't just speak it as fact — a misread room number or
  medicine label is a real safety problem when the person can't visually double-check it.
- If the currency classifier isn't confident, say "this might be X, please verify" instead of
  stating a number that could be wrong.

That "never state uncertainty as fact" rule ended up being the one idea that shows up in every
single module. It wasn't planned from day one — it came out of actually hitting the problem in
Module 2 (see below) and then I deliberately carried it forward everywhere else.

## Module 1 — Scene Understanding

**Goal:** "What is around me?" → a spoken description of what's nearby and where.

I used YOLOv11n (the nano variant — smallest, fastest, good enough for this) for object detection,
running on COCO's 80 classes. The interesting part wasn't the detection itself, it was turning a
list of bounding boxes into something that sounds like a person talking.

I split each frame into three zones based on horizontal position — left, ahead, right (roughly
33%/33%/33% of the image width) — and then grouped detections by (object type, zone) so a cluster
of cars on the right becomes "3 cars right" instead of three separate announcements. I also
prioritized safety-relevant classes (people, vehicles) over static furniture-type objects when
deciding what to mention first if there's a lot going on.

First real test: a busy street photo with 8 persons, 10 cars, and 5 traffic lights detected. The
raw un-grouped version read out something like "5 cars ahead, 5 persons left, 4 cars right, a
person ahead, 2 traffic lights left, and a traffic light ahead" — technically correct but clunky.
After adding the grouping logic, the same scene reads as something much more natural.

TTS is handled by gTTS (Google's free text-to-speech). It needs an internet connection, which is a
real constraint for an offline-first assistive app, but for now it's free and good quality, so it's
the right tradeoff for this stage.

## Module 2 — OCR / Text Reading

**Goal:** point the camera at a sign, label, or document, hear it read aloud.

Used EasyOCR here over Tesseract because it handles real-world angled/messy text noticeably
better. First hiccup: I initially tested with a random photo of an office hallway (no visible text
at all) because I grabbed a bad stock photo URL — got zero text regions detected, which had me
worried the pipeline was broken. It wasn't; there was just nothing to read. Lesson: always
sanity-check your test input by actually looking at it before you assume the model failed. After
that I switched to generating my own synthetic test images with known text (a fake "EXIT / Room
204 / Cafeteria" signboard) so I had ground truth to check against.

That's where the real finding came from: EasyOCR read "EXIT" perfectly (1.00 confidence) but
misread "Room 204" as "Poom 204" and "Cafeteria" as "Caleteria" — both at 0.56 and 0.81 confidence
respectively. 0.81 is not a "low confidence, obviously sketchy" number — it's a fairly high
confidence for a wrong answer. That's what pushed me to add a spellcheck correction layer on top of
the confidence filter, not just rely on the confidence score alone. After adding
`pyspellchecker`-based correction, both misreads got fixed before being spoken. This is the module
where the "never speak uncertainty as fact" principle actually originated — everything after this
module inherited the same instinct.

## Module 3 — Currency Detection

**Goal:** hold up a note, hear "this is 50 rupees."

This one needed transfer learning since currency isn't a COCO class. I used MobileNetV2 pretrained
on ImageNet, froze the convolutional base, and trained a small classification head on top —
standard approach when you don't have much data or compute.

Getting the dataset was its own mini-saga. My first Kaggle dataset slug returned a 403 Forbidden
error (turned out the dataset either didn't exist under that name or needed manual
acceptance first). I searched Kaggle directly for "indian currency" datasets and picked
`gauravsahani/indian-currency-notes-classifier` based on it having the highest download/vote counts
of the options — a small but real judgment call about dataset trustworthiness.

The dataset gave me 7 classes (₹10, ₹20, ₹50, ₹100, ₹200, ₹500, ₹2000) but only **~22 images per
class** in training and 6 in test. That's genuinely small. Training accuracy reached 94.7% by epoch
30, but validation accuracy bounced around wildly between 52% and 86% epoch to epoch — because the
validation split only had about 3 images per class, so misclassifying a single image swings the
percentage by several points. That's not a bug, it's just what happens with this little data.

**Final honest number: 69.05% test accuracy.** The confusion matrix showed Tennote doing great
(91% F1) while smaller/similarly-toned notes like the 1-Hundred and Twenty got confused with each
other more often. I'm not hiding this number — it's a real, current limitation, and the fix is
obvious and known: collect a much bigger, self-captured dataset (100+ images per note, different
lighting, different angles, real phone camera conditions instead of clean dataset photos) rather
than pretending the model is production-ready. I think stating this plainly is actually a stronger
thing to say in an interview than an inflated number would be.

Same safety principle as Module 2: if the model's confidence is below threshold, it says "I'm not
fully sure, but this might be X rupees, please verify" instead of stating a number outright.

## Module 4 — Obstacle Warning

**Goal:** fuse "what's there" with "how close is it" into an actual safety warning.

This is where MiDaS (small variant, for speed) comes in for monocular depth estimation. MiDaS gives
you *relative* inverse depth — a value that's higher for closer things and lower for farther
things — not actual meters. Getting real metric distance needs camera calibration, which felt like
unnecessary complexity for what a blind user actually needs: not "3.2 meters" but "very close,"
"close," or "far." So I used adaptive percentile thresholds (75th/90th percentile of the depth map,
recalculated per image) to bucket each detected object into a proximity zone.

First working version had an embarrassing bug: when I ran it on a street scene, it spoke "car close
right. car close right. car close right." — three separate cars in the same zone at the same
proximity, spoken as three separate sentences. Technically accurate, but it sounds broken and could
genuinely make someone think there are three distinct urgent things happening instead of one small
cluster of cars. Fixed it the same way as Module 1 — grouped identical (object, zone, proximity)
tuples by count, so it now says "3 cars close right" once. This confirmed that the
group-instead-of-repeat pattern isn't a one-off fix, it's a real recurring design principle for
this whole project.

## Module 5 — Multilingual Voice

**Goal:** the user speaks a command in Hindi or English, gets a response back in the same
language, without touching a settings toggle.

I picked auto-detection over a manual language toggle deliberately, since the whole interaction
model is voice-command-driven ("what's in front of me?") — adding a toggle button would break that
hands-free assumption. Whisper (OpenAI's open-source speech-to-text model, "base" size) does both
transcription and language detection in one pass, which made this straightforward.

Small but interesting finding: when I tested with a Hindi voice note, Whisper transcribed it
phonetically in Latin script ("Tumara nam kya hai" instead of Devanagari script), but still
correctly tagged the detected language as `hi`. Since the pipeline routes off the language code, not
the transcribed text itself, this doesn't actually break anything — but it's a good example of why
you test assumptions instead of just trusting a library does exactly what you'd expect.

For the actual response, English text gets translated to Hindi via Google Translate's free API,
then spoken with gTTS in Hindi. Verified both directions by actually listening to the output, not
just checking the code ran without errors.

## The comparative research piece

I didn't want this project to be "I used YOLO because it's popular." I wanted an actual reason
backed by numbers, since the choice of detector directly matters for the safety-critical obstacle
warning module. So I benchmarked three different detector architectures on the same two test
images:

- **YOLOv11n** — lightweight CNN, single-stage detector
- **RT-DETR-L** — transformer-based detector, still fixed-class like YOLO
- **Grounding DINO** — open-vocabulary, text-prompted detector (you describe what to look for in
  natural language instead of picking from fixed classes)

**Methodology matters here** — my first benchmark attempt gave a wildly misleading number (YOLO
showing 673ms on one image and 28ms on another) because I didn't account for CUDA/GPU warm-up cost
on the very first inference call. Once I added 2 discarded warm-up runs before timing 5 real runs
and reported mean ± standard deviation, the numbers became trustworthy and consistent.

**Final numbers** (street scene, the busier of my two test images):

| Model | Latency | Objects detected |
|---|---|---|
| YOLOv11n | 36.3 ± 5.6 ms | 18 |
| RT-DETR-L | 70.7 ± 2.7 ms | 39 |
| Grounding DINO | 575.7 ± 25.6 ms | 40 |

Three real takeaways came out of this, not just the numbers themselves:

1. **Speed and recall trade off against each other pretty cleanly here.** YOLO is ~15-20x faster
   than Grounding DINO but detects roughly half as many objects on a busy scene. That's a real
   architectural tradeoff, not noise.

2. **Grounding DINO's recall is bounded by the text prompt, not just the model.** On my sparse
   indoor room-scene test, it only found 1 object (a chair) — matching YOLO exactly — even though
   it crushed the recall comparison on the cluttered street scene. Why? My text prompt was `"person
   . car . chair . traffic light . bicycle . motorcycle"`, which simply doesn't describe most of
   what's actually in a minimalist living room (a floor lamp, a TV, a picture frame, a side table).
   This is a genuinely useful, citable finding about open-vocabulary detection: you get out what you
   described, not what's actually there.

3. **For obstacle warnings specifically, none of the "better recall" options are usable.** A
   500-600ms delay before a safety warning could mean someone's already walked into whatever the
   warning was about. YOLO is the only real choice for Module 4's real-time use case. RT-DETR and
   Grounding DINO's extra recall would be much better suited to something like an on-demand "give me
   a full, careful description of everything around me" mode where speed matters less than
   completeness — a natural future extension of Module 1, not something I've built yet.

## Deployment — the part that kept changing under me

I originally planned to deploy the demo on Hugging Face Spaces using Gradio, since historically
that's been free. Partway through this project, HF changed their policy — as of a few weeks before
I finished this, free-tier accounts can no longer select Gradio or Docker as a Space SDK at all,
only Static (client-side) Spaces stay free. That's confirmed by multiple threads on HF's own forum
from people in the same boat, including a nonprofit teaching free coding classes that got blindsided
by the same change mid-course. Static Spaces can't run our models (Gradio-Lite runs entirely in the
browser via WebAssembly, and stuff like YOLO/Whisper/TensorFlow is way too heavy for that), so that
path was a dead end for this project specifically.

Fallback plan: `gradio.launch(share=True)` gives a temporary public link (72 hours) straight from a
Colab session with zero deployment — genuinely the fastest way to get something shareable when you
need it *today*. For something permanent, Streamlit Community Cloud is still free for public repos
and gives a stable, non-expiring link, so that became the actual long-term home for the live demo.

Along the way I also hit a real memory-crash risk on free hosting: loading five heavy models
(YOLO + EasyOCR + TensorFlow all at once) on app startup is a common way to blow past a free
tier's RAM limit. Fixed by loading each model lazily, only when its specific tab is actually opened,
with error handling around each load so one failure doesn't take down the whole app.

## Honest limitations, all in one place

- **Currency Detection: 69% test accuracy**, trained on ~22 images/class. Not production-ready.
  Needs a larger, self-captured dataset with varied lighting/angle conditions.
- **Depth estimation is relative, not metric.** "Very close / close / far," not actual distances.
  Good enough for a warning, not good enough for precise navigation.
- **OCR was tested on clean/synthetic text and one real signboard-style photo.** Real-world
  low-light, angled, handwritten, or non-English signage hasn't been stress-tested yet.
- **Only 2 languages supported** (Hindi/English) — a reasonable start for India, not comprehensive.
- **The research benchmark used only 2 test images.** Directionally solid, but a bigger, more
  varied benchmark set would make the latency/recall numbers more statistically convincing.
- **This is a web-based / notebook-based demo, not the real hands-free app.** There's no live
  camera feed, no wake-word trigger, no continuous background operation. You upload a photo, click
  a button, get a result — useful for proving the AI pipeline works, but not yet the actual
  hands-free assistive tool a blind user could use day to day. That's real, separate future work
  (native Android, CameraX, on-device wake-word detection, TFLite model conversion) — not something
  I want to pretend already exists.

## What's next

- **Module 6 — Face Memory:** deliberately paused on purpose until I've thought through
  privacy/consent properly (local-only storage, explicit user control over who gets remembered).
- **Module 7 — Navigation & Emergency Assistance:** outdoor GPS turn-by-turn guidance and
  emergency location-sharing.
- **A real, larger currency dataset** — collecting my own photos across lighting conditions instead
  of relying on a small pre-made Kaggle set.
- **The actual Android app** — this is the big one. Kotlin, CameraX for a live camera feed,
  on-device wake-word detection (something like Picovoice Porcupine), and converting these models
  to TFLite so they can run on a phone instead of a GPU server. Everything built so far proves the
  AI logic works; this is the phase that turns it into something a blind user could actually carry
  around and use hands-free.

## Zero-cost stack, for the record

Every single piece of this — Google Colab's free T4 GPU, YOLOv11n, MiDaS, EasyOCR, Whisper, a
self-trained MobileNetV2, the Kaggle currency dataset, Google Translate and gTTS's free tiers — cost
nothing to build. That constraint shaped a lot of the technical decisions here (lightweight model
variants, frozen transfer learning instead of training from scratch, percentile-based depth
thresholds instead of needing calibrated hardware), and I think that's a feature of this project's
story, not something to gloss over.

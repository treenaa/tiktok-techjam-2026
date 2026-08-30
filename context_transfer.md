PROJECT CONTEXT TRANSFER — TIKTOK TECHJAM ROBUST AIGC IMAGE DETECTION

TEAM
We are a Year 1 NTU Turing AI Scholars Programme team participating in TikTok TechJam.

Team ownership:
- Melvin — Data pipeline / datasets / transforms / split integrity
- Mateo — Model architecture / encoders / forensic branch / fusion
- Trina — Training / losses / checkpoints / consistency learning
- Jamie — Evaluation / robustness benchmarking / error analysis
- Ryan — Inference / demo / README / submission-facing product

The team has limited prior computer vision / deep learning implementation experience, so coding agents are being used heavily. Because of this, code correctness, interfaces, leakage prevention, and independent auditing are extremely important. Do not assume code is correct merely because it runs or unit tests pass.

==================================================
1. HACKATHON PROBLEM
==================================================

The challenge is:

“Robust Detection of AI-Generated Images Under Real-World Transformations.”

We need to build a prototype that distinguishes AI-generated images from authentic images.

The important part is not only clean-image accuracy.

The detector must remain accurate after realistic post-processing and redistribution, including:

JPEG Compression
- quality 90
- quality 70
- quality 50
- quality 30

Gaussian Blur
- sigma 0.5
- sigma 1.0
- sigma 2.0

Resize
- downscale 0.5x then upscale
- downscale 0.25x then upscale

Gaussian Noise
- sigma 0.02
- sigma 0.05
- sigma 0.10

Color Jitter
- brightness ±20%
- contrast ±20%
- saturation ±20%

Center Crop
- crop to 80%

Real-world analogs include:
- social-media recompression
- messaging apps
- thumbnails
- low-light noise
- filter apps
- profile-picture cropping
- reposting

The task is image-level binary classification:

0 = authentic / real
1 = AI-generated / AIGC

The final prediction should be a probability that an image is AI-generated.

==================================================
2. COMPETITION CONSTRAINTS
==================================================

- Model must contain fewer than 2 billion parameters.
- Hackathon-scale compute.
- No internal TikTok production systems.
- Public or properly licensed datasets are allowed.
- Teams may generate their own transformed samples.
- Focus is proof-of-concept, not full-scale production deployment.
- Video/audio are out of scope.

Provided/reference datasets include:
- SID_Set
- CIFAKE
- WildFake

IMPORTANT:
There is a demonstration/validation subset that must NOT be used for training:

Non-AIGC:
COCO val2017
4998 images

AIGC:
DALL·E Advanced
8843 images

This subset is for demonstration/reference benchmarking only.

No code should silently include this subset in training.

==================================================
3. REQUIRED DELIVERABLES
==================================================

A. Written project description / Devpost
Must describe:
- how the solution addresses the challenge
- tools used
- models/APIs
- frameworks/libraries
- datasets/assets
- limitations

B. Public GitHub repository
Must include:
- structured/commented code
- setup instructions
- reproduction steps
- limitations
- team contributions

Most importantly:

A script must take an IMAGE DIRECTORY as input and output JSON containing:

[
  {
    "image_path": "...",
    "pred": 0.9342
  }
]

Where:
pred = probability that the image is AI-generated.

C. Demo video
Must show working end-to-end inference.

D. Robustness evaluation summary
Need a compact table or visualization comparing clean vs transformed performance.

E. Error analysis
Need representative:
- false positives
- false negatives
- trade-offs

==================================================
4. JUDGING PRIORITIES
==================================================

Judging criteria shown by organizers:

Technical Execution — 35%
Innovation & Problem Insight — 20%
Impact & Relevance — 20%
Feasibility — 15%
Communication — 10%

This means this should NOT become primarily an academic research paper.

The submission should be a working, polished solution.

Research-style comparisons and ablations are supporting evidence.

Priority order:

1. Reliable end-to-end system
2. Strong transformed-image performance
3. Cross-generator / cross-source generalization
4. Clean architecture and code
5. Good technical rationale
6. Clear robustness evidence
7. Polished demo
8. Supporting ablation/error analysis

Do not optimize novelty at the expense of performance or reliability.

==================================================
5. CORE TECHNICAL HYPOTHESIS
==================================================

The central challenge is:
 
GENERATION SIGNAL
“Was this image produced synthetically?”

But not respond strongly to:

REDISTRIBUTION SIGNAL
“Was this JPEG-compressed / resized / blurred / cropped?”

Conceptually:

What SHOULD affect classification?
- generator-specific evidence
- synthetic texture patterns
- structural inconsistencies
- production-pipeline clues

What SHOULD NOT affect classification?
- JPEG quality
- moderate blur
- repost resizing
- crop
- ordinary color changes
- light noise

Our system should therefore learn:

AI-ness

rather than:

dataset identity
JPEG quality
image resolution
source website
compression artifacts alone

==================================================
6. CURRENT PROPOSED ARCHITECTURE
==================================================

The leading architecture is a dual-domain robust detector.

                        INPUT IMAGE
                            |
                -------------------------
                |                       |
                v                       v
       Pretrained vision          Forensic branch
           encoder               frequency/texture
                |                       |
         visual embedding         forensic embedding
                |                       |
                ----------+------------
                          |
                          v
                    feature fusion
                          |
                          v
                    small classifier
                          |
                          v
                 raw AIGC logit
                          |
                     sigmoid
                          |
                          v
                 P(AI-generated)

The pretrained backbone candidates are:

1. I-JEPA
2. DINO
3. CLIP

I-JEPA is currently the differentiated hypothesis, but it MUST NOT be selected solely because it sounds novel.

We should benchmark I-JEPA, DINO, and CLIP under the same:
- data split
- classifier head
- training conditions
- robustness transformations
- metrics

If DINO or CLIP clearly outperforms I-JEPA, use the better model.

The final architecture should be decided empirically.

==================================================
7. WHY I-JEPA IS INTERESTING
==================================================

I-JEPA learns image representations by predicting hidden regions in latent representation space rather than reconstructing exact pixels.

Intuitive motivation:

A brittle detector may rely on tiny pixel artifacts.

JPEG/blur/resize can destroy those artifacts.

A representation model such as I-JEPA may preserve more structural information after transformations.

The hypothesis is approximately:

E(x) ≈ E(T(x))

for realistic transformation T.

However:

I-JEPA may also discard low-level forensic clues useful for detecting AI-generated images.

This motivates the second forensic branch.

I-JEPA itself is NOT the detector.

Think of it as:

I-JEPA = visual feature extractor
classifier = decision maker
forensic branch = low-level production evidence
consistency training = robustness objective

==================================================
8. FORENSIC / FREQUENCY BRANCH
==================================================

The initial forensic branch should stay lightweight.

Basic idea:

image
  |
 FFT
  |
frequency representation
  |
small CNN
  |
forensic embedding

Potential representations:
- log FFT magnitude
- phase spectrum
- magnitude + phase
- residual/high-frequency image

Do NOT immediately build:
- FNOs
- neural operators
- PINNs
- giant spectral transformers

Those can be explored only if the core system already works and evidence suggests they help.

The forensic branch exists because generated images and camera photographs can have different:
- frequency statistics
- texture distributions
- sensor/noise behavior
- generator artifacts

Fusion:

z_visual = pretrained encoder features
z_forensic = forensic features

z = concatenate(z_visual, z_forensic)

z -> MLP -> AIGC logit

==================================================
9. ROBUSTNESS-AWARE TRAINING
==================================================

This is one of the most important ideas.

For source image x:

create a transformed version:

T(x)

Examples:
- JPEG(x)
- blur(x)
- resize(x)
- crop(x)

Both have the same class label.

The desired behavior is:

f(x) ≈ f(T(x))

Example:

Original AI image -> 0.94
JPEG30          -> 0.92
Blur2           -> 0.91
Resize0.25      -> 0.90

Good.

Example:

Original -> 0.94
JPEG30   -> 0.49
Blur2    -> 0.21

Bad.

Training objective:

L_total
=
L_classification(clean)
+
L_classification(transformed)
+
lambda * L_consistency

Default classification:
BCEWithLogitsLoss

Default consistency candidate:
MSE between predicted probabilities

Example:

p_clean = sigmoid(model(x))
p_aug   = sigmoid(model(T(x)))

L_consistency = MSE(p_clean, p_aug)

Avoid complicated research losses initially.

==================================================
10. DEVELOPMENT SEQUENCE
==================================================

Do NOT implement the final complicated architecture immediately.

The intended sequence is:

PHASE 1 — BASELINES

CLIP -> classifier
DINO -> classifier
I-JEPA -> classifier

Use:
- identical train/validation/test splits
- identical head where possible
- identical robustness benchmark

Then compare.

PHASE 2 — ROBUST TRAINING

Winning backbone
+
official stochastic training transformations

Then:

winning backbone
+
consistency loss

PHASE 3 — FORENSIC FUSION

winning backbone
+
forensic branch
+
fusion classifier
+
consistency training

Only retain additions that improve held-out robustness/generalization.

PHASE 4 — PRODUCTIZATION

- reliable checkpoint
- predict.py
- required JSON
- evaluation suite
- interactive demo
- README
- video
- Devpost

==================================================
11. CRITICAL DATA RULES
==================================================

This section is NON-NEGOTIABLE.

A. Source-level splitting

If source image x is in training:

JPEG(x)
blur(x)
crop(x)
resize(x)
noise(x)

must remain TRAINING ONLY.

A transformed copy must NEVER cross into validation/test.

Each source image should have a stable source_id.

Split by source_id, not by individual files.

B. No forbidden demonstration subset in training.

C. Avoid dataset shortcut learning.

Example dangerous setup:

Real = mostly COCO
AI = mostly another dataset

The model may learn:

“COCO style => real”

rather than:

“camera image => real”

This can produce fake 98% accuracy and fail on new sources.

Where possible evaluate:
- across datasets
- across generators
- unseen generators
- unseen source domains

D. Validation and test roles

Training data:
optimize parameters

Validation data:
- model selection
- early stopping
- threshold tuning

Test data:
FINAL evaluation only

Do NOT use test metrics to:
- choose architecture
- tune thresholds
- choose checkpoint
- early stop

==================================================
12. CANONICAL LABEL / OUTPUT CONVENTIONS
==================================================

These conventions must be identical throughout repository.

Label:
0 = real
1 = AIGC

Model forward:
logits = model(images)

Do NOT sigmoid inside model.forward().

Training:
BCEWithLogitsLoss(logits, labels)

Inference:
prob_aigc = sigmoid(logits)

Required prediction:
pred = probability of AIGC

Never accidentally invert this convention.

==================================================
13. METRICS
==================================================

Do not rely only on accuracy.

Core metrics:

- AUROC
- accuracy
- F1
- precision
- recall
- specificity
- false positive rate

Important:
AUROC must use continuous probabilities/scores, NOT hard class predictions.

Robustness metrics:

For each transformation:
- transformed AUROC
- clean-to-transformed performance drop

Define:

AUROC_drop
=
AUROC_clean - AUROC_transformed

Aggregate:
- mean transformed AUROC
- worst-case transformed AUROC
- mean degradation
- worst degradation

==================================================
14. PREDICTION STABILITY METRICS
==================================================

This should be a major explanatory feature.

For image x and transformation T(x):

p0 = P(AI | x)
p1 = P(AI | T(x))

Score drift:

|p0 - p1|

Report:
- mean absolute drift
- median drift
- p95 drift
- binary class flip rate

Example:

Original  0.94
JPEG30    0.92
Blur2     0.91
Resize    0.90

This demonstrates robustness intuitively.

==================================================
15. CORE RESULTS TABLE
==================================================

The final report should contain something approximately like:

Model                 Clean   JPEG30   Blur2   Resize.25   Noise.10   Crop80   Robust Mean
------------------------------------------------------------------------------------------------
CLIP baseline
DINO baseline
I-JEPA baseline
+ consistency
+ forensic branch
FINAL MODEL

Numbers must come from actual experiments.

Do not fabricate results.

This table supports the final architecture but should NOT be presented as the product itself.

==================================================
16. FINAL PRODUCT EXPERIENCE
==================================================

Judge/user uploads an image.

System returns:

AI-generated probability: 93.1%
Prediction: AI-generated

Optional robustness diagnostic:

Original       93.1%
JPEG30         91.8%
Blur2          90.7%
Resize0.25     89.9%
Crop80         92.0%

Mean score drift: X
Class stable: YES

This robustness panel directly demonstrates the challenge objective.

Do not present stability as proof that the classification is correct.
It only shows that the model's prediction is stable.

==================================================
17. REQUIRED CLI
==================================================

Final interface should be approximately:

python predict.py \
  --input ./images \
  --output predictions.json \
  --checkpoint ./checkpoints/best.pt

Required JSON:

[
  {
    "image_path": "images/a.jpg",
    "pred": 0.9342
  },
  {
    "image_path": "images/b.jpg",
    "pred": 0.0711
  }
]

Rules:
- pred is probability of AIGC
- deterministic file ordering
- no debug fields in competition JSON
- corrupted input handling must be explicit
- support normal formats such as jpg/jpeg/png/webp

==================================================
18. TARGET REPOSITORY STRUCTURE
==================================================

robust-aigc/
|
├── configs/
|
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   └── inference/
|
├── tests/
|
├── scripts/
|
├── app/
|
├── train.py
├── evaluate.py
├── predict.py
├── requirements.txt or pyproject.toml
└── README.md

Avoid giant monolithic notebooks.

Notebooks can be used for exploration, but production logic should live in reusable modules.

==================================================
19. TEAM OWNERSHIP
==================================================

MELVIN — DATA

Owns:
src/data/
data tests
dataset adapters
manifest generation
train/val/test split integrity
official transformations
paired clean/transformed views
source_id logic

MATEO — MODELS

Owns:
src/models/
model tests
I-JEPA/DINO/CLIP adapters
classification head
forensic branch
fusion model
feature extraction
parameter counts

TRINA — TRAINING

Owns:
src/training/
train.py
training config
BCE loss
consistency loss
optimizer
checkpointing
early stopping
validation threshold tuning
training metrics

JAMIE — EVALUATION

Owns:
src/evaluation/
evaluate.py
robustness benchmark
metrics
score drift
class flips
cross-generator evaluation
error analysis
latency/throughput results

RYAN — PRODUCT

Owns:
src/inference/
predict.py
app/
README
submission UX
checkpoint loading
batch inference
JSON output
interactive demo
submission checklist

Avoid editing files owned by other teammates unless necessary.

Use clean interfaces instead.

==================================================
20. ENGINEERING PRINCIPLES
==================================================

Because much of the repository is agent-written:

1. Never trust code simply because it runs.

2. Prefer explicit interfaces.

3. Prefer simple architecture over unnecessary abstraction.

4. Add tests around:
- labels
- shapes
- transformations
- split leakage
- checkpoint loading
- probability convention
- metrics

5. Do not silently swallow exceptions.

6. Do not silently skip corrupt files unless explicitly configured.

7. Avoid global state.

8. Use config-driven paths/settings.

9. Seed:
- Python random
- NumPy
- PyTorch

10. Keep CPU fallback working.

11. Do not download huge pretrained models during unit tests.

12. Mock heavy dependencies in tests.

13. Avoid unnecessary dependencies.

14. Add type hints where useful.

15. Document non-obvious mathematical decisions.

==================================================
21. BIGGEST FAILURE MODES
==================================================

Treat these as high-priority audit targets:

- train/test leakage
- augmented versions crossing splits
- source duplicates across splits
- label inversion
- sigmoid applied twice
- BCEWithLogitsLoss used on probabilities instead of logits
- wrong pretrained normalization
- AUROC computed from hard labels
- threshold tuned on test set
- stochastic test transforms
- incorrect model.eval() usage
- frozen encoder accidentally updating
- incorrect checkpoint restoration
- forbidden WildFake demonstration data used in training
- dataset identity shortcut learning
- real/AI datasets having systematically different image sizes/encoding
- FFT branch using inconsistent image scaling
- validation being used differently between models
- incomparable backbone benchmarks

==================================================
22. WHAT NOT TO BUILD YET
==================================================

Do NOT prioritize:

- PINNs
- FNOs
- neural operators
- training JEPA from scratch
- giant ensemble of JEPA + DINO + CLIP
- custom CUDA
- production cloud infrastructure
- huge hyperparameter sweeps
- complicated attention fusion
- dozens of exploratory plots

These may be explored later only if:
1. the core detector works
2. evaluation is reliable
3. final product is already strong

==================================================
23. ROLE OF PCA / EXPLORATORY ANALYSIS
==================================================

PCA is NOT the core model.

Possible supporting use:

JEPA/DINO embeddings
-> PCA
-> 2D visualization

Useful questions:
- Are real and AI samples separable?
- Do transformed copies remain near the clean image?
- Does corruption collapse separation?

This can support explainability/presentation.

Do not spend excessive time on it.

==================================================
24. FINAL PROJECT STORY
==================================================

Do NOT pitch:

“We used I-JEPA because it is novel.”

Do NOT pitch:

“We compared three pretrained models.”

Preferred story:

“AI-generated image detectors often degrade after images are reposted, compressed, resized, blurred, or cropped because the evidence they rely on is fragile.

We built a detector designed to preserve AI-generation evidence while suppressing irrelevant redistribution effects.

The system combines a robust learned visual representation with complementary forensic signals and explicitly trains predictions to remain consistent under realistic transformations.

We then evaluate the detector against the exact post-processing conditions relevant to real social-media redistribution.”

==================================================
25. HOW TO THINK ABOUT THE COMPONENTS
==================================================

Simple intuition:

Pretrained backbone:
“What does this image structurally look like?”

Forensic branch:
“How does this image appear to have been produced?”

Classifier:
“Given those clues, how likely is it AI-generated?”

Robustness augmentation:
“See this same image after realistic damage.”

Consistency loss:
“Do not change your answer merely because it was reposted.”

Evaluation:
“Prove the detector still works after transformation.”

Demo:
“Show the robustness live.”

==================================================
26. SUCCESS CONDITION
==================================================

This project is finals-worthy only if execution supports the idea.

A strong final submission should have:

- working end-to-end detector
- strong clean performance
- relatively small degradation under official transformations
- evidence of cross-generator/generalization behavior
- reliable JSON inference
- polished UI/demo
- reproducible repo
- sensible model size/latency
- clear error analysis
- one compact ablation/robustness table
- no methodological leakage

Novel architecture with mediocre robustness is not enough.

A simpler architecture with excellent robustness, generalization, and execution is preferable.

==================================================
27. INSTRUCTIONS TO ANY CODING AGENT
==================================================

Before making changes:

1. Inspect the current repository.
2. Understand existing interfaces.
3. Identify which subsystem you own.
4. Do not blindly rewrite working code.
5. State assumptions if interfaces are unclear.
6. Prefer minimal compatible changes.
7. Run relevant tests after edits.
8. Do not fabricate datasets, results, checkpoints, or metrics.
9. Flag methodological risks explicitly.
10. Optimize for correctness and integration, not code volume.

When uncertain about a design decision:
prefer the simplest implementation that preserves scientific validity and hackathon requirements.
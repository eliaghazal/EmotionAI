# EmotionAI — Complete Project Explanation
### Everything the professor might ask, explained simply

---

## TABLE OF CONTENTS
1. [What Did We Build? (The One-Paragraph Summary)](#1-what-did-we-build)
2. [How We Satisfy the Professor's Requirements](#2-how-we-satisfy-the-professors-requirements)
3. [What Is a Neural Network? (ELI5)](#3-what-is-a-neural-network-eli5)
4. [What Is a CNN? (Convolutional Neural Network)](#4-what-is-a-cnn)
5. [What Is an ANN vs a CNN?](#5-ann-vs-cnn)
6. [The FER2013 Dataset](#6-the-fer2013-dataset)
7. [Our Custom CNN — Architecture Deep Dive](#7-our-custom-cnn)
8. [Transfer Learning and MobileNetV2](#8-transfer-learning-and-mobilenetv2)
9. [How Training Works (Epochs, Loss, Accuracy)](#9-how-training-works)
10. [Why Training Took 9 Hours](#10-why-training-took-9-hours)
11. [Data Augmentation (MixUp, Flips, Brightness)](#11-data-augmentation)
12. [Label Smoothing](#12-label-smoothing)
13. [Ensemble Inference](#13-ensemble-inference)
14. [Temperature Scaling](#14-temperature-scaling)
15. [Grad-CAM — Making AI Explainable](#15-grad-cam)
16. [Face Detection with OpenCV](#16-face-detection)
17. [The Real-Time Dashboard (Flask + SocketIO)](#17-the-dashboard)
18. [The Four Bugs We Fixed](#18-bugs-we-fixed)
19. [Performance and Results](#19-performance-and-results)
20. [Likely Professor Questions and Answers](#20-qa)

---

## 1. What Did We Build?

We built a program that watches a webcam and in real time tells you what emotion is on the face it sees — and draws a colorful heatmap to show *why* it thinks that.

It uses two separately trained AI models (which we call a CNN and a MobileNetV2), combines their answers for better accuracy, and serves everything through a website you can open in a browser while the camera is running.

**Simple analogy:** Imagine hiring two doctors to look at the same patient. Doctor A trained at one school, Doctor B trained at another. They look at the same face and both give a diagnosis. You average their diagnoses — that combined opinion is more reliable than either one alone. That's the ensemble.

---

## 2. How We Satisfy the Professor's Requirements

The assignment says:

> *Create a computer vision AI application with: face detection, emotion classification, real-time webcam support. Suggested tools: OpenCV, ANN/CNN libraries.*

Here is exactly how each requirement is met:

| Requirement | How We Met It | File |
|---|---|---|
| **Face detection** | OpenCV's Haar cascade detector finds all faces in every frame and returns bounding boxes | `core/detector.py` |
| **Emotion classification** | Two trained deep learning models (CNN + MobileNetV2) classify each detected face into 7 emotion categories | `core/predictor.py`, `models/` |
| **Real-time webcam support** | Flask web server streams live annotated video; browser webcam support via SocketIO for devices without a direct camera | `web/app.py` |
| **OpenCV** | Used for: face detection, image preprocessing, CLAHE contrast normalization, drawing bounding boxes and bars | `core/detector.py`, `core/predictor.py` |
| **ANN/CNN libraries** | TensorFlow/Keras: both models are CNNs built with Keras layers. One is a custom-designed CNN, the other is a pre-trained MobileNetV2 adapted with transfer learning | `models/train_custom_cnn.py`, `models/train_transfer.py` |

We went **well beyond** the minimum requirements by also adding:
- Grad-CAM explainability overlay
- Face identity recognition (register and name faces)
- Temporal smoothing across frames
- An ensemble of two models
- Session tracking, timeline chart, PDF/CSV export

---

## 3. What Is a Neural Network? (ELI5)

**Think of it like a chain of guesses.**

Imagine you want to teach a child to recognize a cat. You show them 10,000 pictures and say "cat" or "not cat" each time. At first they're terrible. But gradually they start to notice patterns: pointy ears, whiskers, slanted eyes. You never explicitly told them what a cat looks like — they figured it out from examples.

A neural network does exactly this, mathematically.

It has **layers** of simple math operations chained together. Each layer transforms the input a little bit. At the end, it outputs a prediction. Then you compare that prediction to the correct answer, measure the error (called **loss**), and nudge every number in the network very slightly to make the error smaller. This is called **backpropagation**. You repeat this millions of times across thousands of images. Eventually, the network "learns" to recognize patterns it was never explicitly programmed to find.

The "numbers" inside the network are called **weights** or **parameters**. Our CNN has about 1.35 million of them. Each one is a small decimal number that gets adjusted during training.

---

## 4. What Is a CNN?

A **Convolutional Neural Network** is a type of neural network designed specifically for images.

**The problem with regular neural networks for images:**
A 48×48 grayscale image has 2,304 pixels. If you connected every pixel to every neuron in the first layer — which a plain neural network would do — you'd have millions of connections just in the first layer. That's slow, hard to train, and wasteful, because most pixels only relate to their immediate neighbors.

**The CNN solution — convolution:**
Instead of connecting every pixel to every neuron, a CNN slides a small **filter** (usually 3×3 pixels) across the entire image. This filter is like a little magnifying glass. It scans every 3×3 patch of the image and produces a single number representing how strongly that pattern is present at that location.

One filter might learn to detect horizontal edges. Another might detect vertical edges. Another might detect curves. At first these filters are random; during training, they learn to detect whatever patterns are most useful.

**Pooling:**
After a convolution, we apply **max pooling** — we divide the feature map into 2×2 blocks and keep only the maximum value. This halves the spatial size (48→24→12→6) and makes the network translation-invariant: it stops caring exactly *where* in the image a feature appears, just *whether* it appears.

**Deep stacking:**
As you stack more layers, the features become more abstract. Layer 1 detects edges. Layer 2 detects combinations of edges (shapes). Layer 3 detects shapes that look like eyes or mouths. The final layers classify based on all of those high-level features.

---

## 5. ANN vs CNN

| | ANN (Artificial Neural Network) | CNN (Convolutional Neural Network) |
|---|---|---|
| Input type | Any (numbers, text, tabular data) | Images (spatial data) |
| How it connects | Every neuron connects to every neuron in the next layer (fully connected) | Uses sliding filters — local connections only |
| Spatial awareness | None — doesn't know pixels are neighbors | Yes — exploits the 2D structure of images |
| Parameters | Many (pixels × neurons = millions in layer 1) | Far fewer (filter size × depth) |
| Best for | Classification, regression, tabular data | Image recognition, object detection, video |

Our project uses a **CNN** because we're processing images. However, CNNs still have fully-connected (ANN-style) **Dense layers** at the end — after the convolutional blocks extract features, a standard Dense layer makes the final classification.

So our model is technically a hybrid: CNN for feature extraction + ANN for classification.

---

## 6. The FER2013 Dataset

**What it is:**
FER2013 stands for Facial Expression Recognition 2013. It was published as a Kaggle competition dataset. It contains 35,887 images scraped from the internet — photos of people's faces in various settings.

**Why it's hard:**
- Images are tiny: only **48×48 pixels**. That's barely enough to see facial features.
- They are **grayscale** — no color information at all.
- Labels were assigned by **crowdsourcing** — random people online clicked which emotion they saw. This means labels are noisy. Two annotators often disagree.
- **Severe class imbalance**: Happy has ~9,000 samples. Disgust has only 547. If you trained a model that always predicted "Happy", it would still get roughly 25% accuracy — just by memorizing the most common class.

**The 7 emotion classes:**
| Class | Samples | Color in our app |
|---|---|---|
| Happy | 8,989 | Gold/Yellow |
| Neutral | 6,198 | Silver/Gray |
| Sad | 6,077 | Blue |
| Fear | 4,097 | Purple |
| Angry | 3,995 | Red |
| Surprise | 3,171 | Orange |
| Disgust | 547 | Green |

**The human baseline:**
When researchers asked humans to label FER2013 images, they averaged about **65% accuracy**. This is our target. If we beat 65%, we're doing better than humans on this task. Our ensemble reaches ~74%.

---

## 7. Our Custom CNN

### The Architecture (layer by layer)

```
Input: 48×48×1 grayscale image
│
├─ Block 1: Plain double convolution
│    Conv2D(64, 3×3) → BatchNorm → ReLU
│    Conv2D(64, 3×3) → BatchNorm → ReLU
│    MaxPool(2×2)   → output: 24×24×64
│    Dropout(0.25)
│
├─ Block 2: Residual + SE Attention
│    ┌─ Shortcut: Conv2D(128, 1×1) → BatchNorm ──────────────────────┐
│    └─ Main: Conv2D(128,3×3) → BN → ReLU → Conv2D(128,3×3) → BN    │
│         SE: GAP → Dense(16,relu) → Dense(128,sigmoid) → Reshape    │
│         Multiply (channel scaling)                                  │
│         Add(main, shortcut) → ReLU                                  │
│    MaxPool(2×2)   → output: 12×12×128
│    Dropout(0.25)
│
├─ Block 3: Residual + SE Attention  ← [last_conv here for Grad-CAM]
│    ┌─ Shortcut: Conv2D(256, 1×1) → BatchNorm ──────────────────────┐
│    └─ Main: Conv2D(256,3×3) → BN → ReLU → Conv2D(256,3×3,"last_conv") → BN
│         SE: GAP → Dense(32,relu) → Dense(256,sigmoid) → Reshape    │
│         Multiply (channel scaling)                                  │
│         Add(main, shortcut) → ReLU                                  │
│    MaxPool(2×2)   → output: 6×6×256
│    Dropout(0.25)
│
├─ GlobalAveragePooling2D  → output: 256 (one number per channel)
├─ Dense(512) → BatchNorm → ReLU → Dropout(0.5)
└─ Dense(7, softmax)  → output: 7 probabilities that sum to 1
```

**Total parameters: ~1,350,000**

### What is BatchNormalization?
After a convolution, the numbers can become very large or very small, which makes training unstable. BatchNorm rescales them to a standard range during each batch. It acts like a thermostat — keeps the numbers from going to extremes.

### What is Dropout?
During training, Dropout randomly turns off a fraction of neurons (e.g., 25% or 50%) on each pass. This prevents the network from becoming too dependent on any one path — it's forced to learn redundant representations. At inference time (when making predictions), all neurons are active.

### What is GlobalAveragePooling (GAP)?
After Block 3, we have a 6×6×256 feature map. Instead of flattening that into 6×6×256 = 9,216 numbers (which would then connect to a Dense layer with millions of parameters), GAP takes the average of each of the 256 feature maps across all 36 spatial positions. Output: just 256 numbers. This dramatically reduces parameters and overfitting.

### What is SE (Squeeze-and-Excitation) Attention?
Imagine you're reading a text and some words are more important than others — you naturally pay more attention to "error" and "urgent" than to "the" and "a". SE attention does the same thing for feature channels.

After a convolution produces 128 channels, some channels might represent "edge at this angle" and others might represent "texture pattern". Not all channels are equally important for recognizing a specific emotion.

SE attention:
1. **Squeezes** the spatial information by taking the global average — collapses 12×12×128 into just 128 numbers.
2. Passes those 128 numbers through two Dense layers that learn to predict an importance score (between 0 and 1) for each of the 128 channels.
3. **Excites** by multiplying each channel by its score — important channels get amplified, unimportant ones get suppressed.

```python
# SE Attention block
se = layers.GlobalAveragePooling2D()(x)       # 12×12×128 → 128
se = layers.Dense(16, activation="relu")(se)   # bottleneck
se = layers.Dense(128, activation="sigmoid")(se) # 0–1 score per channel
se = layers.Reshape((1, 1, 128))(se)           # match spatial dims
x  = layers.Multiply()([x, se])               # scale each channel
```

### What is a Residual Connection?
When you stack many layers, a problem called **vanishing gradients** occurs: the training signal (gradient) becomes so small by the time it reaches early layers that those layers stop learning. Residual connections (also called skip connections) solve this.

Instead of only passing forward the transformed output, we add the original input back:
```
output = Convolutions(input) + input
```

If the convolutions aren't helpful, the network can learn to make them output near-zero (effectively turning them off) and just pass the input through unchanged. This means deeper networks can't perform worse than shallower ones.

---

## 8. Transfer Learning and MobileNetV2

### What is Transfer Learning?
**Analogy:** A surgeon who trained for 10 years knows how to make precise cuts, identify tissue types, use tools carefully. If they want to learn a new type of surgery, they don't have to relearn how to hold a scalpel. They just need to learn the specific new techniques.

Transfer learning works the same way. MobileNetV2 was trained on ImageNet — 1.28 million images of 1,000 different categories (cats, cars, keyboards, etc.). Its layers learned powerful general visual features: edges, textures, shapes, objects. We take those learned features and adapt them for our narrower task (emotion recognition).

### MobileNetV2 Architecture
MobileNetV2 is a lightweight CNN designed for mobile devices. It uses **depthwise separable convolutions** — a trick that achieves similar feature extraction to regular convolutions but with far fewer computations.

Key feature: **inverted residuals with linear bottlenecks** — it expands the channel count, applies depthwise convolutions, then compresses back. This is efficient.

After the convolutional base, MobileNetV2 produces a 3×3×1280 feature map from a 96×96×3 input.

### Our Two-Phase Training

**Phase 1 (20 epochs, lr=1e-3):**
We freeze the entire MobileNetV2 base. Only our new Dense layers are trainable. The model quickly learns to interpret MobileNetV2's features in terms of emotions. High learning rate is fine because we're only updating a few layers.

**Phase 2 (40 epochs, lr=1e-5):**
We unfreeze the last 30 layers of MobileNetV2. Now the base can also adjust. But we use a tiny learning rate (1e-5 = 0.00001) so we make only very small adjustments. If we used a large learning rate, we'd overwrite the 1.2-million-image training in a few steps.

We also add L2 regularization to the Dense layer (`kernel_regularizer=l2(1e-4)`) — this penalizes large weights and reduces overfitting.

### The Critical Preprocessing Bug
MobileNetV2 was originally designed with an internal preprocessing layer that expects inputs in the range **[-1, 1]**. Our images are stored as floats in **[0, 1]**. The fix is one line:
```python
X_scaled = X * 2.0 - 1.0   # [0,1] → [-1,1]
```
Without this fix, the model received inputs it was never designed for and was stuck at 11% accuracy regardless of training length.

---

## 9. How Training Works

### The Training Loop (conceptually)
```
For each epoch:
    For each batch of 64 images:
        1. Forward pass: feed images through the model, get predictions
        2. Compute loss: how wrong were the predictions?
        3. Backward pass: compute gradient — how should each weight change?
        4. Update weights: move each weight a tiny amount in the direction that reduces loss
    Report average accuracy and loss for this epoch
```

### What is Loss?
Loss (also called cost) measures how wrong the model's predictions are. We use **Categorical Cross-Entropy loss**:
```
loss = -Σ (true_label × log(predicted_probability))
```
If the model says 'I'm 95% sure this is Happy' and it is Happy, loss is very low. If the model says 'I'm 5% sure this is Happy' and it's actually Happy, loss is very high.

The goal of training is to minimize loss.

### What is Accuracy?
Accuracy = (number of correct predictions) / (total predictions). We report test accuracy — accuracy on images the model has never seen during training.

### What is the Learning Rate?
The learning rate controls how big each weight update step is. Too large: the model overshoots and never converges. Too small: training takes forever. We use **CosineDecay** — the learning rate starts at 1e-3 and gradually decreases to 1e-6 following a cosine curve, like slowing down as you approach your destination.

### What is Overfitting?
When a model memorizes the training data instead of learning general patterns. It gets excellent training accuracy but poor test accuracy. Analogous to a student who memorizes answers for practice exams but fails the real one. We combat this with Dropout, L2 regularization, MixUp, and label smoothing.

---

## 10. Why Training Took 9 Hours

**CNN training time breakdown:**
- Dataset: 28,709 training images
- Batch size: 64 → ~448 steps per epoch
- Epochs: up to 100 (with early stopping, typically ~60-80)
- Per step: forward pass + backward pass + weight update for 1.35 million parameters
- Hardware: CPU (no GPU acceleration), which is ~10-50× slower than a GPU

**MobileNetV2 training time:**
- Phase 1: 20 epochs × ~896 steps (batch size 32)
- Phase 2: up to 40 epochs, more computation because 30 layers are unfrozen
- MobileNetV2 is larger than our custom CNN: ~2.3 million parameters in the unfrozen portion

**Early stopping** monitored validation accuracy with a patience of 15 (CNN) or 12 (MobileNetV2). Training stops automatically when validation accuracy hasn't improved for that many epochs. This saved time but training still ran for several hours per model.

**Why so slow on CPU?**
Deep learning training involves billions of multiply-add operations per batch. GPUs can do thousands of these in parallel. A CPU can only do a few hundred at a time. On a GPU, this training might take 20-30 minutes instead of 9 hours.

---

## 11. Data Augmentation

**What is it?** Creating modified copies of training images to artificially expand the dataset.

**Why?** With only 28,709 training images, the model might overfit — memorize specific images rather than learning general patterns. Augmentation shows the model many variations of each image, forcing it to learn more robust features.

### Augmentations we applied:

**1. Random horizontal flip** — Mirror the image left-to-right. Emotions look the same on either side of the face.

**2. Random brightness** — Randomly darken or brighten the image by ±15%. Handles different lighting conditions.

**3. Random contrast** — Randomly increase or decrease the difference between light and dark areas. Again handles lighting.

**4. Random crop (zoom simulation)** — Randomly crop the image to 90-100% of its size, then resize back to 48×48. This simulates different camera distances.

**5. MixUp** — Our most interesting augmentation:
```python
# Take two random images and their labels
lam = random value between 0.5 and 1.0
mixed_image = lam × image_A + (1 - lam) × image_B
mixed_label = lam × label_A + (1 - lam) × label_B
```
The model sees a semi-transparent blend of two faces, with a blended label. This forces the model to learn smooth decision boundaries rather than sharp cutoffs. It acts as a strong regularizer.

We ensure `lam ≥ 0.5` so the primary image always dominates — we don't want a 50/50 blend that could confuse the model.

---

## 12. Label Smoothing

**The problem:** A standard training label for "Happy" looks like: `[0, 0, 0, 1, 0, 0, 0]` — 100% confident it's class 3 (Happy). But FER2013's labels are noisy — sometimes what's labeled "Happy" looks a bit like "Surprise" to other annotators. Training the model to be 100% confident on noisy labels leads to overconfidence.

**The fix:** Label smoothing with ε = 0.1:
```
smooth_label = (1 - ε) × one_hot + ε / num_classes
            = 0.9 × [0,0,0,1,0,0,0] + 0.1/7
            = [0.014, 0.014, 0.014, 0.9, 0.014, 0.014, 0.014]
```
The model is now trained to be "90% confident" rather than "100% confident". This produces better-calibrated outputs and generalizes better on noisy data.

---

## 13. Ensemble Inference

**Why average two models?**
Each model has its own biases and blind spots. The CNN might be better at certain lighting conditions; MobileNetV2 might be better at certain face angles. When they disagree, averaging prevents either one from dominating.

Mathematically: if each model has an error rate `e`, and their errors are independent, the error of the average is approximately `e²` — which is much smaller.

```python
# Ensemble (core/predictor.py)
cnn_probs     = cnn_model.predict(cnn_input)[0]      # shape: (7,)
mobile_probs  = mobile_model.predict(mobile_input)[0] # shape: (7,)
ensemble_probs = (cnn_probs + mobile_probs) / 2.0     # simple average
```

**Result:** CNN ~71% + MobileNetV2 ~66% → Ensemble ~74%

---

## 14. Temperature Scaling

**The problem:** Well-trained models are often overconfident. After training, the model might output `[0.02, 0.01, 0.01, 0.94, 0.01, 0.005, 0.005]` for a Happy face — 94% on one class. That's not interesting to visualize, and it might not be well-calibrated.

**Temperature scaling:**
```python
logits = log(probs)          # convert softmax back to log space
scaled = logits / T          # divide by temperature T=4.0
probs  = softmax(scaled)     # re-normalize
```
With T=4.0, the output becomes softer: maybe `[0.07, 0.05, 0.05, 0.55, 0.08, 0.12, 0.08]`. Now you can see the full emotional picture — 55% Happy but also 12% Surprised.

**Why T=4.0?** Empirically tuned. T=1 means no change. Higher T makes the distribution flatter. T=4.0 gives visually interesting distributions without completely destroying the signal.

---

## 15. Grad-CAM

### The "Clever Hans" Problem
A famous horse called Clever Hans appeared to solve math problems. Crowds were amazed. But researchers discovered he was actually just responding to subtle physical cues from his trainer — he never learned math at all. He learned the wrong thing.

Neural networks can do the same. A model that appears to classify emotions might actually be responding to image artifacts, background colors, or compression noise rather than actual facial expressions.

**Grad-CAM lets us verify the model is looking at the face.**

### How Grad-CAM Works (step by step)

1. **Pick a target class** — e.g., class 3 = Happy.

2. **Build a modified model** that outputs both the last convolutional feature maps AND the final class predictions simultaneously.

3. **Forward pass** — feed the face through the model, get both outputs.

4. **Compute gradients** — using TensorFlow's `GradientTape`, compute how much each element of the last feature map contributes to the Happy score:
   ```
   ∂(Happy score) / ∂(each feature map pixel)
   ```
   A large gradient means: "if this feature map value increased, the Happy score would increase a lot." That pixel was important.

5. **Pool gradients** — take the average gradient for each feature map channel. This gives one importance score per channel.

6. **Weighted sum** — multiply each feature map by its importance score and sum:
   ```python
   heatmap = feature_maps @ importance_scores
   ```

7. **Apply ReLU** — keep only positive contributions (negative means it worked against the prediction).

8. **Resize and colorize** — resize the heatmap to the face size, apply a color map (red = most important, blue = least).

### The Nested Model Bug (and fix)
MobileNetV2 is a model-within-a-model. We found that when we accessed the conv layer's output through the inner model's graph (`base.get_layer('conv').output`), the tensor was connected to `base.input`, not to our outer model's input. This caused a graph disconnection: at runtime, TensorFlow got confused about which output was which, and swapped the predictions tensor with the feature map tensor. When we tried to index `predictions[:, 4]` (class index 4), we got "index out of bounds" because the 3×3 spatial map only has 3 positions.

**Fix:** Use `base.output` — the layer's output as seen from the outer model's graph. This is always correctly wired:
```python
# BROKEN: inner graph tensor
conv_output = model.get_layer("mobilenetv2").get_layer("last_conv").output

# FIXED: outer graph tensor
conv_output = model.get_layer("mobilenetv2").output
```

---

## 16. Face Detection

**OpenCV** (Open Source Computer Vision Library) is used to find faces in each frame.

**How Haar Cascade works:**
1. Convert the frame to grayscale.
2. Apply a cascade of simple rectangular filters at multiple scales and positions.
3. Each filter checks for the presence of a specific pattern (e.g., "dark region above a light region" = eyebrow structure).
4. Only regions that pass ALL filters in the cascade are considered faces.
5. Returns bounding boxes `(x, y, width, height)` for each detected face.

The cascade is fast because most of the image is rejected early in the cascade — the algorithm doesn't need to carefully examine every pixel.

**CLAHE preprocessing:**
Before feeding the detected face crop to the neural network, we apply CLAHE (Contrast Limited Adaptive Histogram Equalization). This normalizes the local contrast of the image. The reason: FER2013 images have a wide range of contrasts, but webcam images in a specific environment tend to look uniform. CLAHE makes webcam crops look more like the training data.

```python
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
gray  = clahe.apply(gray_frame)  # normalized contrast
```

---

## 17. The Dashboard

**Flask** is a Python web framework. It serves HTML pages and handles HTTP requests.

**SocketIO** is a real-time communication protocol. Unlike regular HTTP (request-response), SocketIO maintains a persistent connection — the server can push data to the browser at any time without the browser asking.

### How the video stream works:
1. Python reads a webcam frame using OpenCV.
2. The frame is processed (face detection + emotion classification + annotation).
3. The annotated frame is JPEG-compressed and sent via HTTP as part of a multipart stream.
4. The browser displays it in an `<img>` tag — which keeps refreshing with new frames.
5. Every 0.5 seconds, emotion statistics are pushed via SocketIO to update charts.

### Browser camera mode:
For laptops where the Python process can't access the webcam directly, the browser captures video via `getUserMedia()`, encodes each frame as base64 JPEG, and sends it to the server via SocketIO. The server processes it and sends back an annotated frame. This is the `on_browser_frame` event handler in `web/app.py`.

### Model caching:
Both models are loaded into RAM at startup (from `.keras` files on disk). Switching between CNN and MobileNetV2 via the dashboard doesn't reload from disk — they're already cached in `_model_cache`. This makes switching instant.

---

## 18. Bugs We Fixed

These are real, non-trivial bugs — the kind you'd encounter in a production system. Understanding them demonstrates deep knowledge.

### Bug 1: Keras 3.x Epoch Boundary Bug

**What happened:** After switching to Keras 3.x (released with TensorFlow 2.16+), every other epoch would show "Your input ran out of data; interrupting training." The model was only learning during odd epochs.

**Root cause:** `ImageDataGenerator.flow()` returns a Python generator. In Keras 3.x, the training loop expects the generator to be exhausted at epoch boundaries to signal the end of the epoch. But the generator was being exhausted mid-epoch, so alternating epochs had no data.

**Fix:** Replace with `tf.data.Dataset`. The key change:
```python
ds = ds.repeat()  # Never runs out of data
```
`tf.data.Dataset.repeat()` creates an infinite loop of the data. We control the epoch length by setting `steps_per_epoch = len(X) // batch_size`.

### Bug 2: MobileNetV2 Stuck at 11%

**What happened:** After training MobileNetV2 for many epochs, accuracy never exceeded ~11% regardless of learning rate, architecture changes, or training duration.

**Root cause:** MobileNetV2's internal layers apply a preprocessing step. They expect inputs in `[-1, 1]`. Our data pipeline provided `[0, 1]`. The internal layers saw near-constant activations of `-1` everywhere, making it impossible to learn meaningful features.

**Fix:**
```python
X = (X * 2.0 - 1.0).astype(np.float32)
```
Applied to both training data (in the tf.data pipeline) and validation/test data (as a numpy operation before passing to `model.evaluate()`).

### Bug 3: Grad-CAM "slice index out of bounds"

**What happened:** When using MobileNetV2, calling `generate_heatmap()` caused `InvalidArgumentError: slice index 4 of dimension 1 out of bounds`.

**Root cause:** The Grad-CAM code built a gradient model with two outputs: `[conv_layer_output, model.output]`. For MobileNetV2, the conv layer lived inside a sub-model. Accessing it via `sub_model.get_layer('name').output` gave a tensor from the sub-model's internal computational graph — connected to `sub_model.input`, not the outer `model.input`. TensorFlow couldn't reconcile the two graphs and swapped the outputs at runtime. `conv_layer_output` became the (1,7) predictions tensor, and `model.output` became the (1,3,3,1280) spatial feature map. When code tried `predictions[:, 4]`, it failed because predictions was actually a 3×3 map with only 3 positions along axis 1.

**Fix:**
```python
# Wrong: inner graph tensor
conv_output = sub_model.get_layer(conv_name).output  # ← wrong graph

# Right: outer model's graph
conv_output = sub_model_layer.output  # ← sub_model itself as a layer
```

### Bug 4: ReduceLROnPlateau Crash at Epoch 6

**What happened:** Training crashed at epoch 6 with `TypeError: This optimizer was created with a LearningRateSchedule and hence the learning rate is not settable. If you need the learning rate to be settable, use a float lr value.`

**Root cause:** We used `CosineDecay` as the learning rate in Adam. `CosineDecay` is a schedule object — it computes the LR as a function of the training step automatically. It's baked into the optimizer. `ReduceLROnPlateau` works by calling `optimizer.learning_rate = new_value` — but you can't set a LearningRateSchedule to a float value.

**Fix:** Remove `ReduceLROnPlateau` entirely. `CosineDecay` already handles LR annealing automatically, starting at `1e-3` and smoothly decaying to `1e-6` — ReduceLROnPlateau is redundant.

---

## 19. Performance and Results

| Model | Test Accuracy | Notes |
|---|---|---|
| Human baseline | ~65% | Average human agreement on FER2013 labels |
| CNN (broken, pre-fix) | ~12% | All three bugs active |
| MobileNetV2 (broken, pre-fix) | ~11% | Wrong input range |
| Custom CNN (fixed + improved) | ~71% | SE attention, residual, MixUp, label smoothing |
| MobileNetV2 (fixed + improved) | ~66% | Two-phase fine-tuning, lower phase2 LR |
| **Ensemble** | **~74%** | CNN + MobileNetV2 averaged |

### Why doesn't accuracy reach 90%+?
FER2013 is genuinely hard. The images are 48×48 pixels — very low resolution. The labels are noisy. State-of-the-art academic models (with much more compute, larger architectures, and tricks we didn't implement) reach about 73–76% on this exact dataset. We're right in that range. Going beyond ~75% on FER2013 typically requires using higher-resolution data, pre-training on face-specific datasets, or combining FER2013 with other emotion datasets.

---

## 20. Q&A

**Q: What is the difference between your CNN and a regular ANN?**
A CNN uses convolutional filters that slide across the image, exploiting spatial locality. A regular ANN connects every input neuron to every hidden neuron, ignoring the 2D structure. For images, CNN is far more efficient and effective.

**Q: Why did you use two models instead of just one?**
Two models trained independently make different kinds of errors. Averaging their predictions (ensemble) tends to cancel errors out. The result is consistently better than either model alone.

**Q: Why is your accuracy not higher? 74% seems low.**
FER2013's human-level accuracy is ~65%. Our 74% is already above that. State-of-the-art is ~76%. The bottleneck is the dataset itself — noisy labels and 48×48 resolution — not our architecture.

**Q: What is Grad-CAM and why did you use it?**
Grad-CAM creates a visual heatmap showing which pixels influenced the model's prediction. We used it to verify the model is actually looking at facial features (eyes, mouth, brow) rather than background noise. It also makes the system explainable — which is important for trust.

**Q: What is transfer learning?**
Using a model pre-trained on one task (ImageNet classification) as the starting point for a different but related task (emotion recognition). The pre-trained features (edges, textures, shapes) are useful for both tasks.

**Q: How does real-time processing work?**
Each webcam frame is: detected with OpenCV (fast, ~1ms) → cropped → preprocessed → passed through both neural networks → probabilities averaged → temperature-scaled → annotated with OpenCV → JPEG-compressed → streamed to the browser. The whole pipeline runs at ~15 frames per second.

**Q: What is CLAHE?**
Contrast Limited Adaptive Histogram Equalization. It normalizes local contrast in an image. We apply it to each face crop so the webcam images match the contrast distribution of FER2013 training images. Without it, the model underperforms on real webcam input.

**Q: Why did training take 9 hours?**
We trained on CPU (no GPU). The CNN ran for ~80 epochs × 448 steps × 1.35M parameters. MobileNetV2 is similar in scale. GPU training would reduce this to 20-30 minutes.

**Q: What is an epoch?**
One complete pass through the entire training dataset. With 28,709 images and batch size 64, one epoch = ~448 gradient updates.

**Q: What is overfitting? How did you prevent it?**
Overfitting is when a model memorizes training examples instead of learning general patterns. We used: Dropout (randomly disables neurons during training), L2 regularization (penalizes large weights), MixUp (forces smooth decision boundaries), label smoothing (prevents overconfidence), and early stopping (stops training when validation accuracy stops improving).

**Q: What is temperature scaling?**
A post-training calibration technique. We divide the model's logits (pre-softmax values) by a temperature constant (T=4.0) before applying softmax. Higher temperature → softer probability distribution. This prevents the model from reporting "99% Happy" for every expression and makes the live probability bars interesting to watch.

**Q: What is the Ensemble and how does it improve accuracy?**
We average the probability outputs of the CNN and MobileNetV2. The models have different strengths and make different errors. When both agree, we get high confidence and they're usually right. When they disagree, averaging prevents either one from dominating on its weak examples.

**Q: How does face recognition work?**
We use the `face_recognition` Python library, which uses a pre-trained deep learning model to extract a 128-dimensional face embedding — a compact numerical representation of a person's face. To register someone, we store their embedding. At runtime, we compare new face embeddings to stored ones using Euclidean distance. Small distance = same person.

**Q: What frameworks did you use?**
- **Python 3.13** — language
- **TensorFlow / Keras** — neural network training and inference
- **OpenCV (cv2)** — image processing, face detection, drawing
- **Flask** — web server
- **Flask-SocketIO** — real-time browser communication
- **NumPy** — numerical array operations
- **Pillow (PIL)** — image loading and resizing
- **scikit-learn** — class weight computation
- **face_recognition** — face embedding and matching
- **tqdm** — progress bars during dataset loading

---

*Document prepared for EmotionAI — Computer Vision Final Project*

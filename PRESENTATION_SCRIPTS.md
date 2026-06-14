# EmotionAI — Presentation Scripts
**Presenters:** Elia · George · Maysaa
**Distribution:** Elia = Slides 1–5 · George = Slides 6–10 · Maysaa = Slides 11–15

---

## ELIA — Slides 1 through 5

---

### Slide 1 — Title

> *Walk up confidently, wait for the slide to appear, then begin.*

"Good [morning/afternoon], everyone. My name is Elia, and with me today are George and Maysaa. Together we built **EmotionAI** — a real-time emotion recognition system that uses your webcam, two trained neural networks, and a live web dashboard to detect and classify human emotions as they happen.

Over the next few minutes, we're going to walk you through the problem we set out to solve, how we built the system, what went wrong along the way — and we'll finish with a live demo.

Let's get started."

> *Advance to Slide 2.*

---

### Slide 2 — Problem Definition

"So — why does this even matter?

Think about it: humans are incredibly good at reading emotions. You can glance at someone's face for a fraction of a second and immediately know if they're happy, frustrated, or surprised. Machines have no ability to do that naturally.

This gap creates real missed opportunities. Imagine a car that detects when you're drowsy and alerts you before an accident. Or an online learning platform that notices students are confused and slows down. Or a mental health tool that can flag distress in a therapy session. All of those require a machine that can read faces.

Now, why is it hard? A few reasons. First, the same person can express the same emotion in dozens of different ways depending on their culture, age, and personality. Second, real-world conditions — bad lighting, a turned head, glasses, a hand in front of the face — all make detection harder. And third, and this is a key point we'll come back to: the main dataset we trained on, called FER2013, has labels that were assigned by crowdsourcing online. Even humans only agree on the correct emotion about 65% of the time. So our goal isn't perfection — it's beating that human baseline.

Our goal for this project: build a live webcam app that detects faces, classifies the emotion on each face into one of seven categories, and — importantly — shows *why* it made that decision, visually."

> *Advance to Slide 3.*

---

### Slide 3 — Dataset: FER2013

"We trained our models on a dataset called FER2013 — Facial Expression Recognition 2013. It contains just under 36,000 images, split into training, validation, and test sets.

Each image is tiny — only 48 by 48 pixels, in grayscale. That's roughly the size of a postage stamp. And each one is labeled with one of seven emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

Now, there's a problem here that we had to specifically address: the classes are not balanced. The 'Happy' class has almost nine thousand samples. But 'Disgust' has only 547. If we just trained naively on this data, the model would learn to predict 'Happy' for almost everything, because that's the safest bet.

The fix is shown in the code snippet here. We use a function called `compute_class_weights` — it calculates how much extra attention the model should pay to the rare classes. Disgust gets a much higher weight than Happy, so the model is penalized more heavily when it gets Disgust wrong. This balances the learning."

> *Advance to Slide 4.*

---

### Slide 4 — System Architecture

"Let me give you the big picture of how the full system fits together.

On the left, you have the webcam. Every frame that comes in gets passed to the Face Detector, which uses OpenCV to find all the faces in the image and returns their bounding boxes.

Those face crops are then passed into what we call the EmotionPredictor. Inside, we have two neural network models running in parallel — our Custom CNN and a MobileNetV2 model we fine-tuned using transfer learning. Both models give us a set of seven probabilities — one per emotion. We average those two sets of probabilities together. That's the ensemble. We then apply temperature scaling to soften the output so it doesn't just scream one emotion at 99% confidence, and finally we apply temporal smoothing so the result doesn't jitter between frames.

All of that feeds into a Flask and SocketIO web server, which streams the annotated video and pushes live statistics to a dashboard in the browser. You can also trigger a Grad-CAM overlay, which we'll explain later.

George is going to walk you through the technical details of each of those components."

> *Advance to Slide 5.*

---

### Slide 5 — AI Techniques Used

"Before I hand over to George, here's a quick map of every AI technique in the system and why we used each one.

The core of the project is our **Custom CNN** — a convolutional neural network we designed and trained from scratch on FER2013. It's the primary model.

The second model is **MobileNetV2**, a pre-trained network that was originally trained on over a million general images. We adapted it to recognize emotions using a technique called transfer learning — which George will explain in detail.

We combine both models through **ensemble inference** — simply averaging their probability outputs. Two imperfect models, combined, outperform either one alone.

**Grad-CAM** is our explainability layer. It generates a heatmap showing which pixels of the face the model was looking at when it made its decision. That's important both for debugging and for showing the professor that the model is actually learning facial features, not just guessing.

We also have **face recognition** using the face_recognition library, which lets you register your face and have the system greet you by name.

And finally, **temperature scaling** — a calibration technique that prevents the model from being overconfident, so we see the full emotional distribution rather than just a single peak.

With that, I'll hand over to George, who's going to take us inside the models."

> *George steps forward.*

---

## GEORGE — Slides 6 through 10

---

### Slide 6 — Face Detection (OpenCV)

"Thanks Elia. I'm George, and I'm going to walk you through how the system actually processes each frame — starting with face detection.

Before we can classify an emotion, we need to find the face. We use OpenCV for this. OpenCV is a massive open-source computer vision library. We give it a camera frame, and it returns the bounding boxes of every face it finds.

But there's a subtle problem here. Our emotion models were trained on FER2013 images, which have very specific contrast characteristics. Webcam images in different lighting conditions look very different. If we just feed a raw webcam crop into the model, it gets confused because the input distribution doesn't match what it was trained on.

The fix is something called **CLAHE** — Contrast Limited Adaptive Histogram Equalization. It's a preprocessing step that normalizes the contrast of the face crop before we feed it to the model. Think of it like adjusting the levels on a photo to make it look more like the training data.

You can see in the code: we create a CLAHE object with a clip limit of 2.0, apply it to the grayscale image, and then resize it to 48 by 48 pixels for the CNN, or 96 by 96 pixels in RGB for MobileNetV2. And critically — MobileNetV2 needs its inputs in the range negative one to positive one, not zero to one. We found this out the hard way: when we forgot that, the model was stuck at 11% accuracy no matter how long we trained it."

> *Advance to Slide 7.*

---

### Slide 7 — Custom CNN Architecture

"Now let me talk about the core of the project: the neural network we designed ourselves.

A CNN — Convolutional Neural Network — is a type of AI model designed specifically for images. Instead of looking at the whole image at once, it scans small patches — usually 3 by 3 pixels — across the entire image using filters. Each filter learns to detect something different: edges, curves, textures, and eventually more complex patterns like eyes or mouths. The deeper you go into the network, the more abstract the patterns become.

Our architecture has three main blocks. Block 1 is simple — two convolution layers with 64 filters each, followed by a max pooling layer that halves the image size from 48 to 24 pixels.

Blocks 2 and 3 are more advanced. We added two improvements: **residual connections** and **SE attention**.

A residual connection is basically a shortcut. Instead of only passing the output of the convolutions forward, we also add the original input back to it. This helps gradients flow backward during training and prevents the network from forgetting what it learned in earlier layers.

SE attention — Squeeze and Excitation — is a channel attention mechanism. After the convolutions, we ask: 'which of these 128 feature channels are actually useful for this image?' We compute a score between 0 and 1 for each channel, and multiply each channel by its score. Channels that aren't helpful get suppressed. Channels that are important get amplified.

You can see the code here. GlobalAveragePooling collapses the spatial dimensions — that's the squeeze. Then two Dense layers learn to predict the importance of each channel — that's the excitation. The final Multiply applies those scores.

The network ends with a GlobalAveragePooling layer — rather than flattening the entire feature map, we take the average of each feature map. This dramatically reduces the number of parameters and makes the model less likely to overfit. Then a Dense layer with 512 neurons, and finally 7 output neurons with softmax — one per emotion."

> *Advance to Slide 8.*

---

### Slide 8 — Transfer Learning (MobileNetV2)

"Our second model uses a completely different approach called **transfer learning**.

Here's the idea. Training a deep neural network from scratch on a small dataset like FER2013 is hard. You don't have enough data for the model to learn every visual concept from zero. But what if someone already trained a massive model on 1.2 million images? That model already knows what edges look like, what textures look like, what shapes look like. Can we borrow that knowledge and adapt it for emotions?

That's exactly what MobileNetV2 is. It was pre-trained on ImageNet — a huge dataset of general images. Its layers already contain powerful general-purpose visual features. We take that entire pre-trained model and add our own small head on top — just a few Dense layers — and train *only that head* while keeping the rest frozen.

This is Phase 1: the base is completely frozen, we train for 20 epochs at a learning rate of 1e-3. The model quickly learns to map MobileNetV2's features to emotion categories.

Then in Phase 2, we 'unfreeze' the last 30 layers of MobileNetV2 and fine-tune them with a very small learning rate — 1e-5, which is ten times smaller than Phase 1. Why so small? Because the pretrained weights are already good. If we use a large learning rate, we'd destroy the knowledge that was trained in for months. This is the 'fine-tuning' step.

The critical detail in the code: MobileNetV2 was designed to receive inputs in the range negative one to positive one. Our images are in zero to one. So we apply the simple transform X times 2 minus 1. Skipping this step gave us 11% accuracy — barely better than random guessing across 7 classes."

> *Advance to Slide 9.*

---

### Slide 9 — Training Pipeline

"Now let me explain how we actually ran the training.

We trained both models ourselves, on this machine. The full training process — both models together — took around 9 hours. The CNN alone ran for up to 100 epochs. An epoch means the model has seen every single training image once. With 28,709 images and a batch size of 64, that's about 448 steps per epoch. Across 100 epochs, that's tens of thousands of gradient updates.

We faced a significant bug early on. The original code used Keras's `ImageDataGenerator.flow()` to feed batches to the model. In Keras 3.x, there's a bug: every other epoch, the generator runs out of data after a single step. The model would appear to be training but was actually doing nothing useful for half of all epochs. The fix was to switch to `tf.data.Dataset` and call `.repeat()` on it, which makes the dataset loop infinitely. That's what you see in the code here.

We also added two regularization techniques to prevent overfitting.

The first is **MixUp augmentation**. Instead of feeding the model clean individual images, we randomly blend two images together and blend their labels proportionally. So the model might see an image that's 70% one face and 30% another, with a label that's 70% 'Happy' and 30% 'Sad'. This forces the model to learn smoother decision boundaries.

The second is **label smoothing**. Instead of a hard label like 'this is definitely Happy — 100%', we soften it to 'this is about 90% Happy and 1% each of the others'. This prevents the model from becoming overconfident, which is important because FER2013's labels aren't perfectly accurate to begin with."

> *Advance to Slide 10.*

---

### Slide 10 — Ensemble Inference

"The final piece I want to cover is how we combine the two models at inference time.

Neither model is perfect on its own. The Custom CNN gets around 71%. MobileNetV2 gets around 66%. But they make *different kinds of mistakes* — the CNN is good at some faces, MobileNetV2 is good at others. So if we average their predictions, the errors tend to cancel out and the correct answer gets reinforced.

In the code, it's literally two lines: run both models, add the probability arrays, divide by two.

Then we apply temperature scaling. Without it, a well-trained model often outputs something like '97% Happy, 1% Neutral, 0.5% Sad'. That's technically accurate but not very interesting to look at. Temperature scaling divides the logits — the raw model outputs before softmax — by a temperature value of 4. The higher the temperature, the softer the distribution. So instead of 97% Happy, you might see 55% Happy, 20% Surprise, 15% Neutral. The full emotional picture becomes visible. That's what you'll see in the bars on the live demo.

I'll now hand over to Maysaa who will cover Grad-CAM, our results, and the challenges we faced."

> *Maysaa steps forward.*

---

## MAYSAA — Slides 11 through 15

---

### Slide 11 — Grad-CAM

"Thank you George. I'm Maysaa, and I'll start with one of my favorite parts of this project — making the AI explain itself.

When a model classifies an emotion, it's easy to just show the result and move on. But how do we know it's actually looking at the *face*? What if it's just picking up on background colors, or image artifacts? This is a real concern in machine learning called the 'Clever Hans' problem — models that appear to work but are learning the wrong thing.

Grad-CAM solves this. The full name is Gradient-weighted Class Activation Mapping. Here's the idea in simple terms: once the model makes a prediction, we ask — 'which neurons in the last convolutional layer were most responsible for this decision?' We compute the gradient of the predicted class score with respect to the feature maps in that last layer. The gradient tells us how much each part of the feature map contributed to the final answer. We then create a weighted sum — a heatmap — that highlights the most important spatial regions.

That heatmap is then resized to the size of the face and overlaid in color. For 'Happy', the model typically highlights the corners of the mouth and the eyes. For 'Angry', it lights up the brow region. For 'Sad', the inner eyebrows and downturned mouth. This gives us confidence that the model has actually learned facial expressions.

In the live demo, you can toggle this on and off in real time."

> *Advance to Slide 12.*

---

### Slide 12 — Performance

"Now let's talk about the numbers.

We started in a bad place. When we first ran training, both models were stuck at around 11 to 12% accuracy. To put that in context — random guessing across 7 classes gives you about 14%. So our trained models were *worse than random*. That told us immediately that something was fundamentally broken in the training pipeline, not just the model design.

After identifying and fixing the three main bugs — which I'll cover in the next slide — and after adding our improvements, here's where we ended up.

The Custom CNN reached approximately 71% accuracy on the test set. MobileNetV2 reached approximately 66%. And the ensemble — combining both — reaches approximately 74%.

The critical benchmark here is the human baseline: approximately 65%. That's the average accuracy when human annotators tried to label FER2013 images. Our ensemble *exceeds that benchmark*. The system is better at classifying emotions from these noisy 48-by-48 images than the humans who labeled them.

We should be transparent: FER2013 is a hard dataset. State-of-the-art models trained with much more compute reach around 73–76%. We're right in that range."

> *Advance to Slide 13.*

---

### Slide 13 — Challenges Solved

"This was not a smooth project. We ran into four significant bugs, and I think being transparent about them actually shows more understanding of the system than if everything had just worked.

The first bug: every other epoch, the model appeared to train but was only seeing one batch of data. This was a known issue in Keras 3.x with `ImageDataGenerator`. The data generator was exhausting itself halfway through and silently continuing with no data. The fix was to switch to `tf.data.Dataset` and add `.repeat()`, which creates an infinite stream.

The second bug: MobileNetV2 was stuck at 11% accuracy no matter how long we trained. The model expected input in the range negative one to positive one. We were feeding it zero to one. This one-line fix — X times 2 minus 1 — brought accuracy from 11% to over 60%.

The third bug: Grad-CAM was crashing with 'slice index out of bounds'. The MobileNetV2 model is a *nested* model — it contains another model inside it. When we tried to access the last conv layer's output through the inner model's graph, the outer model's output and the conv output were from different graphs, so at runtime the two outputs swapped. The model was treating the spatial feature map — shape 3 by 3 by 1280 — as the class prediction, so when we tried to access class index 4, it failed because there are only 3 spatial positions. The fix was to use `layer.output` from the outer model's graph rather than `layer.get_layer().output` from the inner one.

The fourth bug: adding ReduceLROnPlateau caused a crash at epoch 6. In Keras 3, when you use CosineDecay as your learning rate schedule, the learning rate is no longer a simple variable — it's a function of the training step. ReduceLROnPlateau tries to overwrite it, which throws an error. The fix: remove ReduceLROnPlateau entirely. CosineDecay already handles learning rate annealing automatically."

> *Advance to Slide 14.*

---

### Slide 14 — Live Demo

"Now let's actually see it running.

*(Launch the app: `python web/app.py`)*

What you're seeing is the live video feed from the webcam. The colored box around the face — green means a registered face, orange means unknown. Inside the box, you can see seven horizontal bars, one per emotion, showing the live probability distribution updated every frame.

*(Show a neutral face, then smile slowly)*

Watch the bars as I smile. Happy goes up, Neutral goes down. Notice how it's not just snapping to one emotion — you can see the whole distribution shifting.

*(Toggle Grad-CAM on)*

Now I'm turning on Grad-CAM. The colored heatmap overlay shows where the model is looking. You can see it lighting up around the mouth and eye region — those are the facial muscles that change most with expressions. That tells us the model has genuinely learned facial anatomy.

*(Register face)*

I can also register my face live. I type in my name, the system stores my face embedding, and from this point on it will recognize me by name.

*(Switch model)*

And I can switch between the Custom CNN and MobileNetV2 at runtime — no restart needed. Both models are cached in memory after first load."

> *Advance to Slide 15.*

---

### Slide 15 — Conclusion

"To wrap up — here's what we built.

Two independently trained deep learning models. A custom SE-ResNet CNN that we designed from scratch, and a fine-tuned MobileNetV2 using transfer learning. Combined as an ensemble, they reach approximately 74% accuracy — exceeding the human-level baseline on FER2013.

A real-time dashboard with a live video stream, emotion bars, Grad-CAM visualization, face registration, and session export to CSV and PDF.

The techniques we applied: convolutional neural networks, transfer learning, residual connections, SE channel attention, MixUp augmentation, label smoothing, ensemble inference, temperature scaling, and Grad-CAM explainability.

We also debugged four real production-level issues in Keras 3 and TensorFlow that are not in any textbook — which gave us a much deeper understanding of how these systems actually work under the hood.

Thank you. We're happy to take any questions."

> *All three presenters stand at the front for questions.*

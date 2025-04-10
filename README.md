# Image-Captioning-with-Deep-Learning
A deep learning project that automatically generates descriptive captions for images using a CNN-LSTM architecture. Trained on Flickr8k dataset with features extracted from InceptionV3 and evaluated using BLEU score.



# VisionCaptioner: Generating Image Descriptions Using Deep Learning 🧠🖼️

## 📌 Overview

VisionCaptioner is an image captioning project that combines computer vision and natural language processing to generate textual descriptions for images. The project employs a hybrid architecture using **InceptionV3 (CNN)** for image feature extraction and a **LSTM-based RNN** for caption generation.

This notebook-based implementation guides you from **data preprocessing, feature extraction, tokenizer preparation**, through to **training and evaluating** a deep captioning model. BLEU scores are used for performance evaluation.

---

## 🧠 Key Features

- CNN + LSTM architecture for end-to-end image captioning.
- Feature extraction using pre-trained **InceptionV3**.
- Text processing using **tokenizer** and **padding** with **Keras**.
- Caption generation via **greedy decoding**.
- Evaluation using **BLEU scores**.
- Trained on **Flickr8k** dataset.

---

## 🚀 How It Works

1. **Preprocess Captions**  
   Clean and tokenize captions from Flickr8k.

2. **Extract Image Features**  
   Use InceptionV3 (without the final classification layer) to extract high-level visual features.

3. **Create Sequences**  
   Convert descriptions into input-output sequences for training.

4. **Model Architecture**  
   - Image features → Dense Layer  
   - Captions → Embedding → LSTM  
   - Merged and passed through Dense output layer with softmax

5. **Train the Model**  
   Using teacher forcing and categorical crossentropy loss.

6. **Evaluate**  
   Caption generation on test set and evaluation using BLEU score.

---

## 📊 Evaluation

- **Metric:** BLEU Score  
- **Output:** Automatic generation of image captions like:  
  _“a man riding a horse through a field”_

---



## 💾 Requirements

- Python 3.x
- TensorFlow / Keras
- NumPy
- Pandas
- NLTK
- Matplotlib
- tqdm
- PIL

Install dependencies with:

```bash
pip install -r requirements.txt



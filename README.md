# Driver Drowsiness Detection using Vision Transformers

A real-time driver drowsiness detection system built with Vision Transformers (ViT), Swin Transformer, and ResNet-50. Trained on 85K+ images and deployed as a live inference pipeline using OpenCV and PyTorch.

This was a course project for **ECS 271: Machine Learning & Discovery** at UC Davis (Fall 2025).

---

## What it does

The system processes a live webcam feed frame-by-frame, detects eye regions using Haar Cascade classifiers, and classifies each eye as open or closed using a fine-tuned ViT model. A scoring mechanism tracks prolonged eye closure — if the score exceeds a threshold, a **DROWSY** alert is triggered on screen.

The core motivation: existing CNN-based drowsiness systems fail under real-world conditions like shadows, poor lighting, head movement, and partial occlusion. Transformer architectures handle these better because they capture global image context rather than just local features.

---

## Results

### Test Accuracy on MRL Dataset (held-out test set)

| Metric | Score |
|--------|-------|
| Accuracy | 99% |
| Precision | 0.99 |
| Recall | 0.99 |
| F1-Score | 0.99 |

### Cross-Dataset Generalization on CEW Dataset (unseen data)

| Model | Accuracy | Closed Eye Precision | Open Eye Precision |
|-------|----------|----------------------|--------------------|
| ViT | **93.49%** | **0.95** | 0.93 |
| Swin Transformer | 92.53% | 0.88 | 0.94 |
| ResNet-50 | 74.16% | 0.74 | 0.76 |

**ViT outperformed both Swin Transformer and ResNet-50 on unseen real-world data**, confirming that attention-based architectures generalise better for this task. ResNet-50's significantly lower accuracy (74.16%) demonstrates the limitations of CNN-based approaches in real-world conditions.

---

## Architecture

Three models were trained and compared:

**Vision Transformer (ViT)**
- Images split into fixed-size patches → converted to token embeddings
- Standard Transformer encoder with global self-attention across all patches
- Binary classification head: open vs. closed eye
- Pre-trained on ImageNet, fine-tuned on MRL Eye Dataset

**Swin Transformer**
- Hierarchical feature extraction with shifted window attention
- More compute-efficient than pure ViT while retaining global context
- Strong generalisation, second-best overall performance

**ResNet-50 (baseline)**
- Standard CNN with local feature extraction
- Significantly underperforms on unseen real-world data
- Confirms why CNN-based systems fail in production driving conditions

---

## Dataset

**Training: MRL Eye Dataset (Kaggle)**
- 84,898 images (42,952 open / 41,946 closed)
- Balanced classes with real-world variation: lighting changes, head pose, glasses/occlusion
- Split: 70% train / 15% validation / 15% test

**Testing: CEW Dataset**
- Used exclusively for cross-dataset generalisation evaluation
- High variation in lighting and camera angles
- Models were never trained on this data

---

## Real-Time Pipeline

```
Webcam → Capture Frame → Convert to Grayscale
       → Haar Cascade Eye Detection
       → Crop & Preprocess Eye Region
       → ViTImageProcessor (HuggingFace)
       → Fine-tuned ViT → Open / Closed prediction
       → Scoring mechanism → DROWSY alert if score > threshold
```

The pipeline runs frame-by-frame in real time. A cumulative score tracks eye closure duration — a single closed frame increments the score, an open frame decrements it. Alert triggers when score exceeds 8.

---

## How to Run

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/driver-drowsiness-detection-vit.git
cd driver-drowsiness-detection-vit
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the model weights
The trained ViT model weights (`best_vit_model.pth`) are available in this repo. Place them in the root directory or update the path in `real_time_detection.py`:
```python
state_dict = torch.load("best_vit_model.pth", map_location=device)
```

### 4. Run the real-time detection
```bash
python real_time_detection.py
```

Press `q` to quit the webcam window.

**Requirements:** Python 3.8+, webcam access, CUDA GPU (optional but recommended for speed)

---

## Training Setup

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 3e-5 |
| Weight Decay | 0.01 |
| Epochs | 5 |
| Loss Function | Cross-Entropy |
| LR Scheduler | Cosine with 10% warmup |
| Batch Training | Mini-batch via PyTorch DataLoaders |

---

## Project Structure

```
driver-drowsiness-detection-vit/
├── real_time_detection.py     # Real-time inference pipeline
├── requirements.txt           # Dependencies
├── best_vit_model.pth         # Trained ViT model weights
├── report/
│   └── MLD_Project_Report.pdf # Full research paper
├── presentation/
│   └── MLD_Presentation.pdf   # Project slides with demo
└── README.md
```

---

## Team

- **Ketki Kulkarni** — UC Davis MS Computer Science
- Hetvi Bhadani — UC Davis MS Computer Science  
- Varun Singh — UC Davis MS Computer Science

---

## Key Takeaway

This project shows that **Transformer architectures significantly outperform CNNs for real-world eye-state classification**. The 19+ percentage point accuracy gap between ViT (93.49%) and ResNet-50 (74.16%) on unseen data makes a strong case for using attention-based models in safety-critical applications like driver monitoring systems.

# CNNs-for-Image-Segmentation-using-U-Net-or-FCN-
# 🧠 CNNs for Image Segmentation (U-Net / FCN / DeepLabV3)

This project implements **semantic image segmentation** using pretrained **CNN architectures** — U-Net, FCN, and DeepLabV3 — to perform pixel-level object detection.

## 🎯 Aim
Segment objects in images instead of just classifying them, achieving a fine-grained understanding at the pixel level.

## 🧩 Features
- Upload any image for segmentation
- Use pretrained models (U-Net / FCN / DeepLabV3)
- Visualize:
  - Original Image
  - Segmentation Mask
  - Overlay of segmentation + image
- Detect object classes automatically

## 🛠️ Technologies
- Python, PyTorch, TensorFlow/Keras
- Libraries: `torchvision`, `Pillow`, `NumPy`, `Matplotlib`
- Platform: Google Colab

## 🚀 How to Run
1. Clone the repo:
   ```bash
   git clone https://github.com/<your-username>/cnn-image-segmentation.git
   cd cnn-image-segmentation

🧠 Model Information
1️⃣ FCN-ResNet50 (Fully Convolutional Network)

Full Form: Fully Convolutional Network
Base Model: ResNet50

The FCN transforms a traditional image-classification CNN into a pixel-wise prediction network.
Instead of assigning one label to the entire image, it predicts a class label for every pixel.

🔍 How It Works

Encoder (ResNet50 Backbone): Extracts multi-level visual features from the input image.

Decoder (Fully Convolutional Layers): Replaces fully connected layers with convolution + upsampling layers to reconstruct the original resolution.

Output: Produces a segmentation mask where each pixel belongs to a specific class (e.g., background, car, person).

✅ Key Features

Efficient and simple architecture for semantic segmentation.

Reuses pretrained classification networks like ResNet.

Performs well for basic segmentation tasks.

⚠️ Limitations

May produce coarse boundaries.

Limited understanding of large-scale object context.

2️⃣ DeepLabV3-ResNet50

Full Form: DeepLab Version 3
Base Model: ResNet50

DeepLabV3 is an advanced semantic segmentation model that improves upon FCN by capturing multi-scale context and refining pixel-level accuracy.

🔍 How It Works

Encoder (ResNet50 Backbone): Extracts spatial and semantic features.

Atrous (Dilated) Convolutions: Expands the receptive field without reducing spatial resolution, allowing the model to “see” more of the image.

ASPP (Atrous Spatial Pyramid Pooling): Combines multiple atrous convolutions with different dilation rates to detect objects of varying sizes.

Output: Produces detailed segmentation maps with sharper boundaries and richer context.

✅ Key Features

Captures both local and global context using multi-scale analysis.

Handles objects of different sizes effectively.

Generates smoother, more accurate segmentations.

⚠️ Limitations

Slightly higher computational cost compared to FCN.

Requires more memory and processing time.

🧾 Model Comparison
Feature	FCN-ResNet50	DeepLabV3-ResNet50
Backbone	ResNet50	ResNet50
Convolution Type	Standard	Atrous (Dilated)
Context Capture	Local	Multi-scale
Segmentation Quality	Basic	High (sharp boundaries)
Speed	Faster	Slightly slower
Best Use	Simple segmentation tasks	Complex multi-object segmentation

<img width="502" height="252" alt="FCN Deep Learning" src="https://github.com/user-attachments/assets/f672b701-a7b0-4706-ae1f-eb3056a9defc" />

Output :
<img width="1190" height="459" alt="image" src="https://github.com/user-attachments/assets/59b43bf7-40ac-4cc2-9add-a9613686f3bc" />


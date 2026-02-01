

import torch
import torchvision
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from google.colab import files

# Upload your image
uploaded = files.upload()
img_path = list(uploaded.keys())[0]

# Preprocess
input_image = Image.open(img_path).convert("RGB")
preprocess = transforms.Compose([
    transforms.Resize((520, 520)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image).unsqueeze(0)

# Choose Model (set "use_fcn = True" for FCN, False for DeepLabV3)

use_fcn = False

if use_fcn:
    model = torchvision.models.segmentation.fcn_resnet50(pretrained=True).eval()
    print("Using FCN ResNet50 model")
else:
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True).eval()
    print("Using DeepLabV3 ResNet50 model")

# Run Inference
with torch.no_grad():
    output = model(input_tensor)["out"][0]
pred = output.argmax(0).byte().cpu().numpy()


# Define COCO/VOC Classes

VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'dining table', 'dog', 'horse', 'motorbike',
    'person', 'potted plant', 'sheep', 'sofa', 'train',
    'tv/monitor'
]

# Random color map for classes
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(VOC_CLASSES), 3), dtype=np.uint8)


# Overlay Segmentation Mask on Original Image

mask_color = colors[pred]  # apply colors
overlay = (0.5 * np.array(input_image.resize((520,520))) + 0.5 * mask_color).astype(np.uint8)


# Show Results

plt.figure(figsize=(15,8))

# Original
plt.subplot(1,3,1)
plt.imshow(input_image)
plt.title("Original Image")
plt.axis("off")

# Raw Mask
plt.subplot(1,3,2)
plt.imshow(pred, cmap="tab20b")
plt.title("Segmentation Mask")
plt.axis("off")

# Overlay
plt.subplot(1,3,3)
plt.imshow(overlay)
plt.title("Overlay (Segmentation + Image)")
plt.axis("off")

plt.show()


# Show Class Legend (only detected classes)

detected_classes = np.unique(pred)
print("✅ Detected Classes in Image:")
for c in detected_classes:
    print(f" - {VOC_CLASSES[c]}")



# Explain Model Processing

print("\n📝 Model Processing Steps:")

print("1. Input Image Preprocessing:")
print("   - Image resized to 256x256 pixels")
print("   - Converted to tensor and normalized using ImageNet mean and std")
print("   - Added batch dimension for model input")

print("2. Feature Extraction (Encoder):")
print("   - DeepLabV3 uses a ResNet50 backbone to extract features at multiple scales")
print("   - Convolutional layers capture edges, textures, and object shapes")
print("   - Atrous (dilated) convolutions expand receptive field without reducing resolution")

print("3. Segmentation Head (Decoder):")
print("   - Features are processed by a classifier to predict per-pixel class probabilities")
print("   - Each pixel gets a score for each VOC class (21 classes total)")

print("4. Post-processing:")
print("   - Argmax operation selects the class with highest probability per pixel")
print("   - Resulting array is the segmentation mask")
print("   - Mask can be visualized directly or overlaid on original image")

print("5. Output:")
print("   - Segmentation mask (raw class IDs)")
print("   - Overlay image (original + mask)")

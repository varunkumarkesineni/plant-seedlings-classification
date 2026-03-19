# 🌱 Plant Seedlings Classification

Classification of 12 plant species using ResNet-18 deep CNN and PyTorch.

## Project Info
- College: Audisankara College of Engineering and Technology
- Architecture: ResNet-18 (Transfer Learning)
- Dataset: V2 Plant Seedlings Dataset (Kaggle)
- Best Accuracy: 76.58%
- Epochs: 50

## Species Classified
3 Crops: Common wheat, Maize, Sugar beet  
9 Weeds: Black-grass, Charlock, Cleavers, Common Chickweed,
         Fat Hen, Loose Silky-bent, Scentless Mayweed,
         Shepherds Purse, Small-flowered Cranesbill

## How to Run
1. Install dependencies:
   pip install torch torchvision pillow matplotlib scikit-learn seaborn numpy

2. Download dataset from Kaggle:
   https://www.kaggle.com/vbookshelf/v2-plant-seedlings-dataset

3. Train the model:
   python plant_seedlings.py

4. Launch GUI:
   python plant_gui.py

## Results
- Best Validation Accuracy: 76.58%
- Training Time: ~110 minutes (CPU)

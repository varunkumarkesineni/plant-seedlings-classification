"""
==============================================================
  PLANT SEEDLING CLASSIFICATION - Training Script
  With CHECKPOINT & RESUME support
==============================================================

FIRST TIME:
    python plant_seedlings.py
    --> Trains from epoch 0, saves checkpoint every 5 epochs

IF CMD CLOSES (e.g. at epoch 31):
    python plant_seedlings.py
    --> Automatically detects latest checkpoint and RESUMES from there

OUTPUTS:
    plant_model.pth         --> Final best model (use with plant_gui.py)
    checkpoint_epoch_X.pth  --> Auto-saved every 5 epochs
    metrics.png             --> Training graphs
    confusion_matrix.png    --> Confusion matrix heatmap
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os, itertools, time, copy, shutil

seed = 123
np.random.seed(seed)

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION  — change paths here if needed
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR          = './dataset/train/'   # folder with 12 class subfolders
MODEL_NAME        = "resnet"
NUM_CLASSES       = 12
BATCH_SIZE        = 64
NUM_EPOCHS        = 50
FEATURE_EXTRACT   = True
CHECKPOINT_EVERY  = 5                    # save a checkpoint every N epochs
CHECKPOINT_DIR    = './checkpoints'      # folder to store checkpoints
FINAL_MODEL_PATH  = 'plant_model.pth'   # final model saved here

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

CLASS_NAMES = [
    "Black-grass", "Charlock", "Cleavers", "Common Chickweed",
    "Common wheat", "Fat Hen", "Loose Silky-bent", "Maize",
    "Scentless Mayweed", "Shepherds Purse",
    "Small-flowered Cranesbill", "Sugar beet",
]

# ─────────────────────────────────────────────────────────────────────────────
#  DEVICE
# ─────────────────────────────────────────────────────────────────────────────
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ─────────────────────────────────────────────────────────────────────────────
#  DATA
# ─────────────────────────────────────────────────────────────────────────────
data_transforms = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

dataset      = datasets.ImageFolder(DATA_DIR, transform=data_transforms)
train_size   = int(0.73 * len(dataset))
val_size     = len(dataset) - train_size
train_set, val_set = torch.utils.data.random_split(
    dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(seed)
)

train_loader = torch.utils.data.DataLoader(train_set, BATCH_SIZE, shuffle=True,  num_workers=0)
valid_loader = torch.utils.data.DataLoader(val_set,   BATCH_SIZE, shuffle=False, num_workers=0)
dataloaders  = {'train': train_loader, 'val': valid_loader}
print(f"Train: {len(train_set)} images  |  Val: {len(val_set)} images")

# ─────────────────────────────────────────────────────────────────────────────
#  MODEL
# ─────────────────────────────────────────────────────────────────────────────
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(num_classes, feature_extract):
    try:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    except Exception:
        model = models.resnet18(pretrained=True)
    set_parameter_requires_grad(model, feature_extract)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

model_ft = initialize_model(NUM_CLASSES, FEATURE_EXTRACT)
model_ft = model_ft.to(device)

params_to_update = [p for p in model_ft.parameters() if p.requires_grad]
optimizer_ft     = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
criterion        = nn.CrossEntropyLoss()

print("Params to learn:")
for name, param in model_ft.named_parameters():
    if param.requires_grad:
        print(f"\t{name}")

# ─────────────────────────────────────────────────────────────────────────────
#  CHECKPOINT HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def save_checkpoint(epoch, model, optimizer, best_acc, history):
    path = os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{epoch}.pth')
    torch.save({
        'epoch'         : epoch,
        'model_state'   : model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'best_acc'      : best_acc,
        'history'       : history,
    }, path)
    print(f"  [CHECKPOINT] Saved → {path}")

def find_latest_checkpoint():
    """Return (epoch, path) of the most recent checkpoint, or (0, None)."""
    files = [f for f in os.listdir(CHECKPOINT_DIR)
             if f.startswith('checkpoint_epoch_') and f.endswith('.pth')]
    if not files:
        return 0, None
    latest = max(files, key=lambda f: int(f.split('_')[-1].replace('.pth', '')))
    epoch  = int(latest.split('_')[-1].replace('.pth', ''))
    return epoch, os.path.join(CHECKPOINT_DIR, latest)

def load_checkpoint(path, model, optimizer):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    return ckpt['epoch'], ckpt['best_acc'], ckpt['history']

# ─────────────────────────────────────────────────────────────────────────────
#  CHECK FOR EXISTING CHECKPOINT
# ─────────────────────────────────────────────────────────────────────────────
start_epoch, ckpt_path = find_latest_checkpoint()
best_acc        = 0.0
best_model_wts  = copy.deepcopy(model_ft.state_dict())
history = {
    'loss': [], 'acc': [],
    'val_loss': [], 'val_acc': []
}

if ckpt_path:
    print(f"\n{'='*55}")
    print(f"  CHECKPOINT FOUND: {ckpt_path}")
    print(f"  Resuming training from epoch {start_epoch + 1}")
    print(f"{'='*55}\n")
    start_epoch, best_acc, history = load_checkpoint(ckpt_path, model_ft, optimizer_ft)
    best_model_wts = copy.deepcopy(model_ft.state_dict())
    start_epoch += 1   # resume from NEXT epoch
else:
    print(f"\n{'='*55}")
    print(f"  No checkpoint found. Starting from epoch 0.")
    print(f"{'='*55}\n")

# ─────────────────────────────────────────────────────────────────────────────
#  TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
since = time.time()

for epoch in range(start_epoch, NUM_EPOCHS):
    print(f'Epoch {epoch}/{NUM_EPOCHS - 1}')
    print('-' * 10)

    for phase in ['train', 'val']:
        model_ft.train() if phase == 'train' else model_ft.eval()

        running_loss     = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer_ft.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model_ft(inputs)
                loss    = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                if phase == 'train':
                    loss.backward()
                    optimizer_ft.step()

            running_loss     += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss     / len(dataloaders[phase].dataset)
        epoch_acc  = running_corrects.double() / len(dataloaders[phase].dataset)

        print(f'{phase} Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}')

        if phase == 'val' and epoch_acc > best_acc:
            best_acc       = epoch_acc
            best_model_wts = copy.deepcopy(model_ft.state_dict())

        if phase == 'val':
            history['val_loss'].append(epoch_loss)
            history['val_acc'].append(float(epoch_acc))
        else:
            history['loss'].append(epoch_loss)
            history['acc'].append(float(epoch_acc))

    print()

    # ── Save checkpoint every N epochs ───────────────────────────────────────
    if (epoch + 1) % CHECKPOINT_EVERY == 0 or epoch == NUM_EPOCHS - 1:
        save_checkpoint(epoch, model_ft, optimizer_ft, best_acc, history)

# ─────────────────────────────────────────────────────────────────────────────
#  TRAINING COMPLETE
# ─────────────────────────────────────────────────────────────────────────────
elapsed = time.time() - since
print(f'\nTraining complete in {elapsed//60:.0f}m {elapsed%60:.0f}s')
print(f'Best val Acc: {best_acc:.4f}')

model_ft.load_state_dict(best_model_wts)

# Save final model
torch.save(model_ft.state_dict(), FINAL_MODEL_PATH)
print(f'Final model saved → {FINAL_MODEL_PATH}')

# ─────────────────────────────────────────────────────────────────────────────
#  PLOT: Training Loss & Accuracy
# ─────────────────────────────────────────────────────────────────────────────
def show_plots(history):
    epochs = range(1, len(history['loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    ax1.plot(epochs, history['loss'],     color='navy',     marker='o', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], color='firebrick', marker='*', label='Val Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.set_xlabel('Epochs'); ax1.set_ylabel('Loss')
    ax1.legend(); ax1.grid(True)

    ax2.plot(epochs, history['acc'],     color='navy',     marker='o', label='Train Accuracy')
    ax2.plot(epochs, history['val_acc'], color='firebrick', marker='*', label='Val Accuracy')
    ax2.set_title('Training & Validation Accuracy')
    ax2.set_xlabel('Epochs'); ax2.set_ylabel('Accuracy')
    ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    plt.savefig('metrics.png', dpi=120)
    plt.show()
    plt.close()
    print("Saved → metrics.png")

show_plots(history)

# ─────────────────────────────────────────────────────────────────────────────
#  CONFUSION MATRIX
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating confusion matrix on validation set...")
model_ft.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for inputs, labels in valid_loader:
        inputs = inputs.to(device)
        outputs = model_ft(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Greens',
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    ax=ax
)
ax.set_xlabel('Predicted Class', fontsize=11)
ax.set_ylabel('True Class', fontsize=11)
ax.set_title('Confusion Matrix – Validation Set', fontsize=13, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=120)
plt.show()
plt.close()
print("Saved → confusion_matrix.png")

# ─────────────────────────────────────────────────────────────────────────────
#  CLASSIFICATION REPORT
# ─────────────────────────────────────────────────────────────────────────────
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))
print("\nDone! Run plant_gui.py to open the GUI.")

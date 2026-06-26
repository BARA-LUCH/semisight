"""
models/cnn_wafer.py — SemiSight CNN Wafer Defect Classifier
Convolutional Neural Network on raw 26x26 wafer map images.
Grad-level: raw image input, CNN architecture, benchmark comparison vs feature-based XGBoost.
Inspired by WM-811K published benchmarks.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

DEFECT_CLASSES = [
    "Center", "Donut", "Edge-Loc", "Edge-Ring",
    "Loc", "Near-full", "Random", "Scratch", "None"
]
WAFER_SIZE = 26


# ── Dataset ───────────────────────────────────────────────────────────────────
class WaferMapDataset(Dataset):
    """
    PyTorch Dataset for raw wafer map images.
    Input: 26x26 grid with values {-1 (outside), 0 (pass die), 1 (fail die)}
    Normalized to [0, 1] with outside-wafer mask as separate channel.
    """
    def __init__(self, images: np.ndarray, labels: np.ndarray, augment: bool = False):
        self.images  = images   # (N, 26, 26)
        self.labels  = labels   # (N,) int class indices
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].copy().astype(np.float32)

        # Channel 1: defect map (0=pass die, 1=fail die, outside→0)
        defect_channel = np.where(img == 1, 1.0, 0.0)
        # Channel 2: valid die mask (1=inside wafer, 0=outside)
        valid_channel  = np.where(img != -1, 1.0, 0.0)
        # Channel 3: pass die map
        pass_channel   = np.where(img == 0, 1.0, 0.0)

        # Stack to (3, 26, 26)
        tensor = np.stack([defect_channel, valid_channel, pass_channel], axis=0)

        # Data augmentation (rotation, flip)
        if self.augment:
            k = np.random.randint(0, 4)
            tensor = np.rot90(tensor, k, axes=(1, 2)).copy()
            if np.random.random() > 0.5:
                tensor = np.flip(tensor, axis=2).copy()
            if np.random.random() > 0.5:
                tensor = np.flip(tensor, axis=1).copy()

        return (
            torch.FloatTensor(tensor),
            torch.LongTensor([self.labels[idx]])[0]
        )


# ── CNN Architecture ──────────────────────────────────────────────────────────
class WaferCNN(nn.Module):
    """
    CNN for wafer map defect classification.
    Architecture inspired by published WM-811K benchmarks.
    Input: (batch, 3, 26, 26)
    Output: (batch, 9) class logits

    Architecture:
    Conv1(3→32) → BN → ReLU → Conv2(32→64) → BN → ReLU → MaxPool
    → Conv3(64→128) → BN → ReLU → Conv4(128→128) → BN → ReLU → MaxPool
    → Flatten → FC(512) → Dropout(0.5) → FC(256) → FC(9)
    """
    def __init__(self, n_classes: int = 9):
        super(WaferCNN, self).__init__()

        # Block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)  # 26→13

        # Block 2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)  # 13→6

        # Block 3
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5   = nn.BatchNorm2d(256)
        self.pool3 = nn.AdaptiveAvgPool2d((3, 3))  # →3x3

        # Classifier
        self.fc1     = nn.Linear(256 * 3 * 3, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2     = nn.Linear(512, 256)
        self.fc3     = nn.Linear(256, n_classes)

    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)

        # Block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)

        # Block 3
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool3(x)

        # Classifier
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_feature_maps(self, x):
        """Extract intermediate feature maps for visualization."""
        maps = {}
        x = F.relu(self.bn1(self.conv1(x)))
        maps["conv1"] = x.detach()
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        maps["conv2_pool"] = x.detach()
        x = F.relu(self.bn3(self.conv3(x)))
        maps["conv3"] = x.detach()
        return maps


# ── Training ──────────────────────────────────────────────────────────────────
def generate_wafer_images(wafer_df: pd.DataFrame) -> np.ndarray:
    """
    Regenerate raw 26x26 wafer map images from the dataframe.
    Uses the same seed as generation so maps are consistent.
    """
    from data.loader import _generate_wafer_pattern
    np.random.seed(42)
    images = []
    for _, row in wafer_df.iterrows():
        img = _generate_wafer_pattern(row["defect_class"])
        images.append(img)
    return np.array(images)


def train_cnn(wafer_df: pd.DataFrame, epochs: int = 30,
              batch_size: int = 64, lr: float = 0.001) -> dict:
    """
    Train CNN on raw wafer map images.
    Returns results dict compatible with trainer.py format.
    """
    print(f"🧠 Training WaferCNN on raw {WAFER_SIZE}x{WAFER_SIZE} images...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")

    # Generate raw images
    print("   Generating raw wafer map images...")
    images = generate_wafer_images(wafer_df)
    labels = wafer_df["class_idx"].values.astype(int)

    # Train/val/test split
    idx    = np.arange(len(images))
    idx_tv, idx_test, y_tv, y_test = train_test_split(
        idx, labels, test_size=0.15, stratify=labels, random_state=42)
    idx_train, idx_val, y_train, y_val = train_test_split(
        idx_tv, y_tv, test_size=0.15, stratify=y_tv, random_state=42)

    # Datasets
    train_ds = WaferMapDataset(images[idx_train], y_train, augment=True)
    val_ds   = WaferMapDataset(images[idx_val],   y_val,   augment=False)
    test_ds  = WaferMapDataset(images[idx_test],  y_test,  augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    # Model
    n_classes = len(DEFECT_CLASSES)
    model     = WaferCNN(n_classes=n_classes).to(device)

    # Class weights for imbalanced classes
    class_counts = np.bincount(y_train, minlength=n_classes)
    class_weights = torch.FloatTensor(
        len(y_train) / (n_classes * np.maximum(class_counts, 1))
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0
    best_state   = None

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validate
        model.eval()
        val_loss = 0
        val_preds, val_true = [], []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                out  = model(imgs)
                loss = criterion(out, lbls)
                val_loss += loss.item()
                preds = out.argmax(dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_true.extend(lbls.cpu().numpy())

        val_acc = accuracy_score(val_true, val_preds)
        scheduler.step()

        history["train_loss"].append(train_loss / len(train_loader))
        history["val_loss"].append(val_loss / len(val_loader))
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1:02d}/{epochs} | "
                  f"Train Loss: {train_loss/len(train_loader):.4f} | "
                  f"Val Loss: {val_loss/len(val_loader):.4f} | "
                  f"Val Acc: {val_acc:.4f}")

    # Load best model
    model.load_state_dict(best_state)

    # Test evaluation
    model.eval()
    test_preds, test_true = [], []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(device)
            out  = model(imgs)
            preds = out.argmax(dim=1).cpu().numpy()
            test_preds.extend(preds)
            test_true.extend(lbls.numpy())

    test_acc  = accuracy_score(test_true, test_preds)
    test_f1   = f1_score(test_true, test_preds, average="weighted", zero_division=0)
    report    = classification_report(
        test_true, test_preds,
        target_names=DEFECT_CLASSES,
        output_dict=True, zero_division=0
    )
    conf_mat  = confusion_matrix(test_true, test_preds)

    print(f"✅ WaferCNN — Test Accuracy: {test_acc:.4f} | Weighted F1: {test_f1:.4f}")
    print(f"   Best Val Accuracy: {best_val_acc:.4f}")

    return {
        "name":         "WaferCNN (Raw Images)",
        "model":        model,
        "accuracy":     round(test_acc, 4),
        "f1_weighted":  round(test_f1, 4),
        "best_val_acc": round(best_val_acc, 4),
        "report":       report,
        "conf_matrix":  conf_mat.tolist(),
        "history":      history,
        "class_names":  DEFECT_CLASSES,
        "architecture": "3-block CNN (3→32→64→128→256) + FC(512→256→9)",
        "input_shape":  f"(3, {WAFER_SIZE}, {WAFER_SIZE}) — defect/valid/pass channels",
        "params":       sum(p.numel() for p in model.parameters()),
        "device":       str(device),
        "epochs":       epochs,
        "success":      True,
        "images":       images,
        "idx_test":     idx_test,
        "y_test":       np.array(test_true),
        "y_pred":       np.array(test_preds),
    }


def get_cnn_summary(cnn_result: dict) -> str:
    """One-line summary for display."""
    return (
        f"WaferCNN ({cnn_result['params']:,} params) · "
        f"Test Acc: {cnn_result['accuracy']:.4f} · "
        f"F1: {cnn_result['f1_weighted']:.4f} · "
        f"Input: {cnn_result['input_shape']}"
    )

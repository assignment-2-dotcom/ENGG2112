"""
melanoma_train.py
=================
Data preprocessing pipeline for the Melanoma Cancer Dataset.
Goal: prepare clean, augmented, normalised DataLoaders for CNN baseline and
      subsequent transfer-learning experiments (ResNet, EfficientNet, etc.).

Dataset layout (on disk):
    melanoma_cancer_dataset/
        train/
            benign/      5 000 images
            malignant/   4 605 images
        test/
            benign/        500 images
            malignant/     500 images

All images are 300×300 JPEG, RGB.

Preprocessing strategy
----------------------
1. Explore class counts and verify image integrity.
2. Split the provided test set equally into validation + test
   (250 benign + 250 malignant each → 500 val, 500 test).
3. Resize every image to 224×224 — the canonical input size for
   most pretrained CNN backbones (VGG, ResNet, EfficientNet …).
4. Apply data augmentation only on the training split to artificially
   expand diversity and counter the mild class imbalance.
5. Normalise pixel values using ImageNet channel statistics
   (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).
   Using ImageNet stats even for a CNN trained from scratch is a
   reasonable default because the backbone weights used in transfer
   learning were trained with these exact statistics.
6. Wrap everything in PyTorch DataLoaders for efficient batch delivery.
"""

import random
from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

# ---------------------------------------------------------------------------
# 0. Reproducibility
#    Fix random seeds so that the val/test split is deterministic across runs.
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------------------------
BASE_DIR  = Path(__file__).parent          # engg2112/
DATA_DIR  = BASE_DIR / "melanoma_cancer_dataset"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR  = DATA_DIR / "test"

# ---------------------------------------------------------------------------
# 2. Hyper-parameters (data pipeline only)
# ---------------------------------------------------------------------------
IMAGE_SIZE  = 224    # pixels — standard CNN / transfer-learning input
BATCH_SIZE  = 32     # common starting point; halve if GPU OOM
NUM_WORKERS = 0      # 0 = load in main process (required on Windows for scripts
                     # run outside __main__ guard; set to 2-4 on Linux/Mac)

# ---------------------------------------------------------------------------
# 3. Explore the raw dataset
#    Before any transform, count images per class to understand imbalance.
# ---------------------------------------------------------------------------
def count_images(directory: Path) -> dict:
    """Return {class_name: image_count} for a folder of class sub-folders."""
    counts = {}
    for cls_dir in sorted(directory.iterdir()):
        if cls_dir.is_dir():
            n = sum(1 for f in cls_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
            counts[cls_dir.name] = n
    return counts

train_counts = count_images(TRAIN_DIR)
test_counts  = count_images(TEST_DIR)

print("=" * 55)
print("DATASET EXPLORATION")
print("=" * 55)
print(f"Train split  : {train_counts}")
print(f"Test split   : {test_counts}")

# Class imbalance ratio in training data
total_train = sum(train_counts.values())
for cls, n in train_counts.items():
    print(f"  {cls:12s}: {n:5d} images  ({100*n/total_train:.1f}%)")

# ---------------------------------------------------------------------------
# 4. Image property verification
#    Confirm size, channel count, and file integrity on a small sample.
#    A mismatch here would break the model input layer silently.
# ---------------------------------------------------------------------------
from PIL import Image, UnidentifiedImageError

def verify_images(directory: Path, sample_size: int = 50) -> None:
    """
    Randomly sample `sample_size` images from each class sub-folder and
    verify they are:
      • readable (not corrupt)
      • RGB  (3 channels — required by CNN conv layers)
      • at the expected native resolution
    Prints a summary and warns about anomalies.
    """
    print(f"\nVerifying images in: {directory}")
    for cls_dir in sorted(directory.iterdir()):
        if not cls_dir.is_dir():
            continue
        all_files = [f for f in cls_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        sample    = random.sample(all_files, min(sample_size, len(all_files)))

        sizes  = Counter()
        modes  = Counter()
        broken = 0

        for fpath in sample:
            try:
                with Image.open(fpath) as img:
                    img.verify()           # catches truncated/corrupt files
                with Image.open(fpath) as img:
                    sizes[img.size] += 1   # (width, height)
                    modes[img.mode] += 1   # 'RGB', 'L', 'RGBA', …
            except (UnidentifiedImageError, Exception):
                broken += 1

        print(f"  [{cls_dir.name}]  sizes={dict(sizes)}  modes={dict(modes)}  broken={broken}")

verify_images(TRAIN_DIR)
verify_images(TEST_DIR)

# ---------------------------------------------------------------------------
# 5. Define transforms
#
#    Training transforms   — augmentation + normalisation
#    Eval transforms       — only resize/crop + normalisation
#                           (no random ops on val/test: we need stable metrics)
#
#    Why these augmentations?
#    • RandomHorizontalFlip / RandomVerticalFlip: moles have no canonical
#      orientation, so flipped images are equally valid.
#    • RandomRotation(20): same reasoning — slight rotation is realistic.
#    • ColorJitter: mimic varying lighting / camera settings in clinics.
#    • RandomAffine: small shear/scale shifts simulate slight patient movement.
#    • CenterCrop after resize for eval: standard practice that avoids
#      distorting aspect ratio while ensuring a fixed spatial size.
# ---------------------------------------------------------------------------

# ImageNet normalisation constants — required when fine-tuning pretrained backbones
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    # --- Geometric augmentations ---
    transforms.Resize((IMAGE_SIZE + 20, IMAGE_SIZE + 20),   # slightly larger …
                      interpolation=InterpolationMode.BILINEAR),
    transforms.RandomCrop(IMAGE_SIZE),                       # … then random crop
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=20,
                              interpolation=InterpolationMode.BILINEAR),
    transforms.RandomAffine(degrees=0, shear=10, scale=(0.9, 1.1)),

    # --- Colour augmentations ---
    # brightness/contrast/saturation ±20 %, hue ±5 %
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.2, hue=0.05),

    # --- Convert to tensor (HWC uint8 → CHW float32 in [0, 1]) ---
    transforms.ToTensor(),

    # --- Normalise to ImageNet distribution ---
    # Subtracts per-channel mean and divides by std so the network sees
    # zero-centred, unit-variance inputs — essential for fast, stable training.
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

eval_transforms = transforms.Compose([
    # Resize slightly larger, then take the exact centre — no randomness
    transforms.Resize((IMAGE_SIZE + 20, IMAGE_SIZE + 20),
                      interpolation=InterpolationMode.BILINEAR),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ---------------------------------------------------------------------------
# 6. Build datasets
#
#    ImageFolder expects:   root/class_a/img1.jpg
#                           root/class_b/img2.jpg
#    It automatically assigns integer labels: benign=0, malignant=1
#    (alphabetical order).
# ---------------------------------------------------------------------------

# Full training dataset with augmentation
train_dataset = datasets.ImageFolder(root=str(TRAIN_DIR),
                                     transform=train_transforms)

# Full test/val pool — loaded with eval transforms (no augmentation)
test_pool_dataset = datasets.ImageFolder(root=str(TEST_DIR),
                                         transform=eval_transforms)

# Confirm label mapping (alphabetical → benign=0, malignant=1)
print("\n" + "=" * 55)
print("CLASS -> LABEL MAPPING")
print("=" * 55)
print(f"  {train_dataset.class_to_idx}")
# Expected: {'benign': 0, 'malignant': 1}

# ---------------------------------------------------------------------------
# 7. Split test pool equally into validation and test sets
#
#    The provided test folder contains 500 benign + 500 malignant images.
#    We split it 50/50 per class:
#        val  → 250 benign + 250 malignant = 500 images
#        test → 250 benign + 250 malignant = 500 images
#
#    Splitting per class (stratified) keeps the 50/50 class balance in
#    both subsets, preventing an accidentally skewed evaluation set.
# ---------------------------------------------------------------------------

def stratified_split(dataset: datasets.ImageFolder, val_fraction: float = 0.5):
    """
    Split `dataset` into two subsets while preserving the class distribution.

    Returns
    -------
    val_indices  : list[int]
    test_indices : list[int]
    """
    # Group sample indices by class label
    class_indices: dict[int, list[int]] = {}
    for idx, (_, label) in enumerate(dataset.samples):
        class_indices.setdefault(label, []).append(idx)

    val_indices  = []
    test_indices = []

    for label, indices in class_indices.items():
        random.shuffle(indices)                          # in-place, seeded above
        split_at = int(len(indices) * val_fraction)
        val_indices.extend(indices[:split_at])
        test_indices.extend(indices[split_at:])

    return val_indices, test_indices

val_indices, test_indices = stratified_split(test_pool_dataset, val_fraction=0.5)

# Wrap in Subset — no data is copied, only index views are created
val_dataset  = Subset(test_pool_dataset, val_indices)
test_dataset = Subset(test_pool_dataset, test_indices)

print("\n" + "=" * 55)
print("SPLIT SUMMARY")
print("=" * 55)
print(f"  Training   : {len(train_dataset):5d} images")
print(f"  Validation : {len(val_dataset):5d}  images")
print(f"  Test       : {len(test_dataset):5d}  images")

# Verify per-class counts in val and test
def split_class_counts(subset: Subset) -> dict:
    counts = Counter()
    for idx in subset.indices:
        _, label = subset.dataset.samples[idx]
        counts[label] += 1
    return dict(counts)

val_cls  = split_class_counts(val_dataset)
test_cls = split_class_counts(test_dataset)
idx2cls  = {v: k for k, v in test_pool_dataset.class_to_idx.items()}
print(f"  Val  class dist : { {idx2cls[k]: v for k, v in sorted(val_cls.items())} }")
print(f"  Test class dist : { {idx2cls[k]: v for k, v in sorted(test_cls.items())} }")

# ---------------------------------------------------------------------------
# 8. Class weight for the loss function (handles class imbalance in training)
#
#    The training set has ~5 000 benign vs ~4 605 malignant.
#    Weighting the loss inversely proportional to class frequency gives
#    the minority class (malignant) a slightly higher penalty, pushing the
#    model toward equal sensitivity for both classes.
#
#    Usage in training loop:
#        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
# ---------------------------------------------------------------------------

train_labels = [label for _, label in train_dataset.samples]
label_counts = Counter(train_labels)
n_classes    = len(train_dataset.classes)
n_total      = len(train_dataset)

# weight_c = total_samples / (n_classes × count_c)
class_weights = torch.tensor(
    [n_total / (n_classes * label_counts[c]) for c in range(n_classes)],
    dtype=torch.float32,
)
print("\n" + "=" * 55)
print("CLASS WEIGHTS (for weighted loss)")
print("=" * 55)
for c in range(n_classes):
    print(f"  {idx2cls[c]:12s} (label {c}): {class_weights[c]:.4f}")

# ---------------------------------------------------------------------------
# 9. DataLoaders
#
#    shuffle=True  for training — prevents the model from learning the
#                  order of batches instead of the image content.
#    shuffle=False for val/test — order doesn't matter; we just want
#                  reproducible evaluation metrics.
#    drop_last=True for training — avoids a tiny final batch that can
#                  cause instability with BatchNorm layers.
#    pin_memory=True — speeds up CPU→GPU data transfer (no-op on CPU).
# ---------------------------------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=False,   # True only when a CUDA GPU is available
    drop_last=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=False,
)

# ---------------------------------------------------------------------------
# 10. Final sanity check — inspect one batch
#     Confirms shapes, dtype, and that pixel values are normalised correctly.
# ---------------------------------------------------------------------------
print("\n" + "=" * 55)
print("BATCH SANITY CHECK")
print("=" * 55)
images, labels = next(iter(train_loader))
print(f"  Batch tensor shape : {images.shape}")  # [32, 3, 224, 224]
print(f"  Dtype              : {images.dtype}")   # torch.float32
print(f"  Pixel min / max    : {images.min():.3f} / {images.max():.3f}")
print(f"  Label values       : {labels.unique().tolist()}")  # [0, 1]

# After normalisation, values are NOT in [0, 1] — that is expected.
# A clean normalised image typically falls roughly in [-2.1, 2.6].

print("\n" + "=" * 55)
print("PREPROCESSING COMPLETE")
print("Loaders ready: train_loader, val_loader, test_loader")
print("Class weights ready: class_weights  (use with nn.CrossEntropyLoss)")
print("=" * 55)

# ---------------------------------------------------------------------------
# What comes next (CNN baseline, then transfer learning):
#
#   from torch import nn
#   from torchvision import models
#
#   # --- CNN from scratch ---
#   model = nn.Sequential(
#       nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), ...
#   )
#
#   # --- Transfer learning (e.g. ResNet-50) ---
#   backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
#   backbone.fc = nn.Linear(backbone.fc.in_features, 2)
#
#   Both will consume the loaders and class_weights defined above unchanged.
# ---------------------------------------------------------------------------

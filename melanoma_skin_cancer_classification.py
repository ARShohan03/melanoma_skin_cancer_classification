import os
import gc
import json
import yaml
import random
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import time
import subprocess
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import timm
from sklearn.metrics import (
    roc_auc_score, classification_report, accuracy_score,
    roc_curve, f1_score, precision_score, recall_score,
    confusion_matrix, matthews_corrcoef,
    precision_recall_curve, auc
)
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings('ignore')


def _install_package(package_name: str) -> bool:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", package_name],
            capture_output=True, text=True, timeout=120
        )
        return result.returncode == 0
    except Exception:
        return False


print("  [Setup] Attempting to install fvcore for accurate FLOP counting...")
_install_package("fvcore")

FLOP_BACKEND = "none"

try:
    from fvcore.nn import FlopCounterMode as FvcoreFlopCounterMode
    FLOP_BACKEND = "fvcore"
    print("  [FLOPs] Using fvcore FlopCounterMode.")
except ImportError:
    pass

if FLOP_BACKEND == "none":
    try:
        from torch.utils.flop_counter import FlopCounterMode as TorchFlopCounterMode
        FLOP_BACKEND = "torch_builtin"
        print("  [FLOPs] Using PyTorch built-in torch.utils.flop_counter.FlopCounterMode.")
    except (ImportError, AttributeError):
        pass

if FLOP_BACKEND == "none":
    print("  [FLOPs] Neither fvcore nor torch.utils.flop_counter available.")
    print("  [FLOPs] Using comprehensive manual hook-based estimation.")
    print("  [FLOPs] To install fvcore: pip install fvcore")

CONFIG_PATH = "/kaggle/working/melalight_config.yaml"


def load_config(config_path: str = CONFIG_PATH) -> dict:
    if not os.path.exists(config_path):
        print(f"  Config not found at {config_path}. Using defaults.")
        return {}
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


CONFIG = load_config()
VARIANT_NAME = CONFIG.get('publication', {}).get('variant_name', 'MeLALight')
VARIANT_TAG = CONFIG.get('publication', {}).get('variant_tag', 'repvit')

IMG_SIZE   = CONFIG.get('augmentation', {}).get('image_size', 224)
BATCH_SIZE = CONFIG.get('training', {}).get('batch_size', 32)
EPOCHS     = CONFIG.get('training', {}).get('epochs', 60)
LR         = CONFIG.get('training', {}).get('learning_rate', 1e-4)
MIN_LR     = CONFIG.get('training', {}).get('min_learning_rate', 1e-6)
PATIENCE   = CONFIG.get('training', {}).get('patience', 5)
N_FOLDS    = CONFIG.get('training', {}).get('n_folds', 5)
SAVE_DIR   = CONFIG.get('output', {}).get('save_dir', "/kaggle/working/")
os.makedirs(SAVE_DIR, exist_ok=True)

BACKBONE_NAME = 'repvit_m0_9.dist_450e_in1k'
BACKBONE_PRETRAINED = True
BACKBONE_FEATURES_ONLY = True

print(f"{'='*70}")
print(f"BACKBONE CONFIGURATION")
print(f"{'='*70}")
print(f"  Name: {BACKBONE_NAME}")
print(f"  Pretrained: {BACKBONE_PRETRAINED}")
print(f"  Features Only: {BACKBONE_FEATURES_ONLY}")
print(f"  Expected Params: ~5.5M (RepViT M0.9)")
print(f"  FLOP Backend: {FLOP_BACKEND}")
print(f"{'='*70}\n")

BASE_DIR = CONFIG.get('dataset', {}).get('base_dir', "/kaggle/input/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images/melanoma_cancer_dataset")
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR  = os.path.join(BASE_DIR, "test")

assert os.path.exists(TRAIN_DIR) and os.path.exists(TEST_DIR), f"Dataset paths incorrect: {TRAIN_DIR}, {TEST_DIR}"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 300


@dataclass
class EfficiencyMetrics:
    params_total: float
    params_trainable: float
    params_backbone: float
    params_attention: float
    params_head: float
    flops: float
    macs: float
    latency_bs1: float
    latency_bs32: float
    peak_memory: float
    throughput: float

    def to_dict(self) -> Dict:
        return asdict(self)

    def summary_str(self) -> str:
        return f"""
Efficiency Summary ({VARIANT_NAME}):
  Parameters:  {self.params_total:.2f}M (trainable: {self.params_trainable:.2f}M)
  FLOPs:       {self.flops:.4f}B @ 224x224  (backend: {FLOP_BACKEND})
  MACs:        {self.macs:.4f}B @ 224x224
  Latency:     {self.latency_bs1:.2f}ms (BS=1) | {self.latency_bs32:.2f}ms (BS=32)
  Peak Memory: {self.peak_memory:.1f} MB
  Throughput:  {self.throughput:.1f} images/sec
"""


def count_parameters(model: nn.Module) -> Tuple[float, float]:
    total = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    return total, trainable


def compute_flops(model: nn.Module, input_shape: Tuple = (1, 3, 224, 224), device=None) -> float:
    if device is None:
        device = next(model.parameters()).device

    if FLOP_BACKEND == "fvcore":
        return _flops_fvcore(model, input_shape, device)
    elif FLOP_BACKEND == "torch_builtin":
        return _flops_torch_builtin(model, input_shape, device)
    else:
        return _flops_manual(model, input_shape, device)


def _flops_fvcore(model: nn.Module, input_shape: Tuple, device) -> float:
    try:
        dummy = torch.randn(*input_shape).to(device)
        was_training = model.training
        model.eval()
        with torch.no_grad():
            with FvcoreFlopCounterMode(model, display=False) as fcm:
                _ = model(dummy)
        if was_training:
            model.train()
        return fcm.total() / 1e9
    except Exception as e:
        print(f"  [FLOPs] fvcore failed: {e}. Falling back to manual.")
        return _flops_manual(model, input_shape, device)


def _flops_torch_builtin(model: nn.Module, input_shape: Tuple, device) -> float:
    try:
        dummy = torch.randn(*input_shape).to(device)
        was_training = model.training
        model.eval()
        with torch.no_grad():
            with TorchFlopCounterMode(display=False) as fcm:
                _ = model(dummy)
        if was_training:
            model.train()
        return fcm.get_total_flops() / 1e9
    except Exception as e:
        print(f"  [FLOPs] torch.utils.flop_counter failed: {e}. Falling back to manual.")
        return _flops_manual(model, input_shape, device)


def _flops_manual(model: nn.Module, input_shape: Tuple = (1, 3, 224, 224), device=None) -> float:
    """
    Comprehensive manual FLOP estimation with hooks for:
      - Conv2d (grouped, bias)
      - Linear (bias)
      - BatchNorm2d / LayerNorm
      - ReLU / GELU / Sigmoid / SiLU
      - AdaptiveAvgPool2d / AvgPool2d
    Plus 5% correction for residual additions, interpolations, and element-wise ops.
    """
    if device is None:
        device = next(model.parameters()).device

    flops = 0.0
    dummy = torch.randn(*input_shape).to(device)

    def conv2d_hook(module: nn.Conv2d, inp, out):
        nonlocal flops
        batch_size = inp[0].shape[0]
        out_h, out_w = out.shape[2], out.shape[3]
        k_h, k_w = module.kernel_size
        c_in_per_group = module.in_channels // module.groups
        c_out = module.out_channels
        flops += 2 * k_h * k_w * c_in_per_group * c_out * out_h * out_w * batch_size
        if module.bias is not None:
            flops += c_out * out_h * out_w * batch_size

    def linear_hook(module: nn.Linear, inp, out):
        nonlocal flops
        num_vectors = inp[0].numel() // module.in_features
        flops += 2 * module.in_features * module.out_features * num_vectors
        if module.bias is not None:
            flops += module.out_features * num_vectors

    def batchnorm_hook(module, inp, out):
        nonlocal flops
        flops += 2 * inp[0].numel()

    def layernorm_hook(module, inp, out):
        nonlocal flops
        flops += 4 * inp[0].numel()

    def relu_hook(module, inp, out):
        nonlocal flops
        flops += inp[0].numel()

    def gelu_hook(module, inp, out):
        nonlocal flops
        flops += 8 * inp[0].numel()

    def sigmoid_hook(module, inp, out):
        nonlocal flops
        flops += 4 * inp[0].numel()

    def silu_hook(module, inp, out):
        nonlocal flops
        flops += 5 * inp[0].numel()

    def avgpool_hook(module, inp, out):
        nonlocal flops
        in_h, in_w = inp[0].shape[2], inp[0].shape[3]
        out_h, out_w = out.shape[2], out.shape[3]
        channels = inp[0].shape[1]
        batch_size = inp[0].shape[0]
        k_h = max(in_h // max(out_h, 1), 1)
        k_w = max(in_w // max(out_w, 1), 1)
        flops += k_h * k_w * channels * out_h * out_w * batch_size

    hooks = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv2d_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))
        elif isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            hooks.append(module.register_forward_hook(batchnorm_hook))
        elif isinstance(module, nn.LayerNorm):
            hooks.append(module.register_forward_hook(layernorm_hook))
        elif isinstance(module, nn.ReLU):
            hooks.append(module.register_forward_hook(relu_hook))
        elif isinstance(module, nn.GELU):
            hooks.append(module.register_forward_hook(gelu_hook))
        elif isinstance(module, nn.Sigmoid):
            hooks.append(module.register_forward_hook(sigmoid_hook))
        elif isinstance(module, nn.SiLU):
            hooks.append(module.register_forward_hook(silu_hook))
        elif isinstance(module, (nn.AdaptiveAvgPool2d, nn.AvgPool2d)):
            hooks.append(module.register_forward_hook(avgpool_hook))

    was_training = model.training
    model.eval()
    with torch.no_grad():
        _ = model(dummy)

    for h in hooks:
        h.remove()

    if was_training:
        model.train()

    correction_factor = 1.05
    return (flops * correction_factor) / 1e9


def measure_latency(model: nn.Module, batch_size: int = 1, num_iters: int = 100, device=DEVICE) -> float:
    model.eval()
    dummy = torch.randn(batch_size, 3, IMG_SIZE, IMG_SIZE).to(device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(num_iters):
            _ = model(dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.time() - start
    return (elapsed / num_iters) * 1000


def measure_peak_memory(model: nn.Module, batch_size: int = 32, device=DEVICE) -> float:
    if device.type != 'cuda':
        return 0.0
    model.eval()
    dummy = torch.randn(batch_size, 3, IMG_SIZE, IMG_SIZE).to(device)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    with torch.no_grad():
        _ = model(dummy)
    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    torch.cuda.reset_peak_memory_stats()
    return peak_mb


normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    normalize,
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    normalize
])

tta_transforms = [
    val_tf,
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        normalize
    ]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE + 28, IMG_SIZE + 28)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        normalize
    ]),
]

base_train   = datasets.ImageFolder(TRAIN_DIR, transform=None)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=val_tf)

train_labels = np.array([label for _, label in base_train.samples])
test_labels  = np.array([label for _, label in test_dataset.samples])

print(f"Train: {len(base_train):,d}  |  Test: {len(test_dataset):,d}")

class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
print(f"Class weights: {class_weights}")
print(f"Class distribution: {np.bincount(train_labels)}")

pos_weight_value = 1.2 if class_weights[1] / class_weights[0] > 2.0 else class_weights[1] / class_weights[0]
pos_weight = torch.tensor([pos_weight_value]).to(DEVICE)
print(f"Pos weight (mild): {pos_weight.item():.4f}")


class FcaFrequencyAttention(nn.Module):
    def __init__(self, channels: int, dct_size: int = 7, num_freq: int = 8, reduction: int = 8):
        super().__init__()
        self.dct_h = dct_size
        self.dct_w = dct_size
        freq_pairs = [(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (2, 0), (1, 2), (2, 1)][:num_freq]
        self.register_buffer('dct_weights', self._build_dct_filters(dct_size, dct_size, freq_pairs, channels))

        reduced = max(channels // reduction, 4)
        self.excite = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid()
        )

    @staticmethod
    def _dct_basis(h, w, u, v):
        y = torch.arange(h).float().view(-1, 1)
        x = torch.arange(w).float().view(1, -1)
        basis = torch.cos(np.pi * (2 * y + 1) * u / (2 * h)) * torch.cos(np.pi * (2 * x + 1) * v / (2 * w))
        return basis

    def _build_dct_filters(self, h, w, freq_pairs, channels):
        n_groups = len(freq_pairs)
        base = channels // n_groups
        rem = channels % n_groups
        sizes = [base + 1] * rem + [base] * (n_groups - rem)
        filters = torch.zeros(channels, h, w)
        start = 0
        for (u, v), size in zip(freq_pairs, sizes):
            basis = self._dct_basis(h, w, u, v)
            filters[start:start + size] = basis.unsqueeze(0)
            start += size
        return filters

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_resized = x if (h, w) == (self.dct_h, self.dct_w) else F.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
        pooled = (x_resized * self.dct_weights.unsqueeze(0)).sum(dim=(2, 3))
        gate = self.excite(pooled).view(b, c, 1, 1)
        return x * gate


def build_attention_module(channels: int) -> nn.Module:
    return FcaFrequencyAttention(channels)


class MeLALight(nn.Module):
    def __init__(self,
                 num_classes: int = 1,
                 backbone_name: str = BACKBONE_NAME,
                 head_dropout: float = 0.3,
                 l2_weight_decay: float = 1e-5):
        super().__init__()

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=BACKBONE_PRETRAINED,
            features_only=BACKBONE_FEATURES_ONLY,
        )

        for param in self.backbone.parameters():
            param.requires_grad = False

        with torch.no_grad():
            dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
            feat_list = self.backbone(dummy)
            self.ch_second_last = feat_list[-2].shape[1]
            self.ch_last = feat_list[-1].shape[1]

            print(f"[{VARIANT_NAME}] Backbone: {backbone_name}")
            print(f"[{VARIANT_NAME}] Number of stages: {len(feat_list)}")
            print(f"[{VARIANT_NAME}] Applying FcaNet to stages: {len(feat_list)-2} and {len(feat_list)-1}")
            print(f"[{VARIANT_NAME}] Stage {len(feat_list)-2} channels: {self.ch_second_last}")
            print(f"[{VARIANT_NAME}] Stage {len(feat_list)-1} channels: {self.ch_last}")
            print(f"[{VARIANT_NAME}] Head dropout: {head_dropout}")
            print(f"[{VARIANT_NAME}] L2 weight decay: {l2_weight_decay}")

        self.attn_second_last = build_attention_module(self.ch_second_last)
        self.attn_last = build_attention_module(self.ch_last)

        self.gap = nn.AdaptiveAvgPool2d(1)

        fused_channels = self.ch_second_last + self.ch_last
        self.head = nn.Sequential(
            nn.LayerNorm(fused_channels),
            nn.Linear(fused_channels, 256), nn.GELU(), nn.Dropout(head_dropout),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(head_dropout),
            nn.Linear(128, num_classes)
        )

        self._efficiency_metrics = None
        self._efficiency_metrics_computed = False
        self.backbone_frozen = True
        self.l2_weight_decay = l2_weight_decay

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.backbone_frozen = False
        trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.backbone.parameters())
        print(f"[{VARIANT_NAME}] Backbone unfrozen: {trainable:,}/{total:,} params trainable")

    @property
    def efficiency_metrics(self) -> EfficiencyMetrics:
        if not self._efficiency_metrics_computed:
            self._efficiency_metrics = self._compute_efficiency_metrics()
            self._efficiency_metrics_computed = True
        return self._efficiency_metrics

    def _compute_efficiency_metrics(self) -> EfficiencyMetrics:
        total, trainable = count_parameters(self)
        backbone_params = sum(p.numel() for p in self.backbone.parameters()) / 1e6
        attn_params = (sum(p.numel() for p in self.attn_second_last.parameters()) +
                       sum(p.numel() for p in self.attn_last.parameters())) / 1e6
        head_params = sum(p.numel() for p in self.head.parameters()) / 1e6

        device = next(self.parameters()).device
        flops = compute_flops(self, input_shape=(1, 3, IMG_SIZE, IMG_SIZE), device=device)
        macs = flops / 2.0

        print(f"  [FLOPs] Backend used: {FLOP_BACKEND}")
        print(f"  [FLOPs] Total: {flops:.4f} GFLOPs | MACs: {macs:.4f} GMACs")

        latency_bs1 = measure_latency(self, batch_size=1, num_iters=50, device=device)
        latency_bs32 = measure_latency(self, batch_size=32, num_iters=20, device=device)
        peak_memory = measure_peak_memory(self, batch_size=32, device=device)
        throughput = (32.0 / latency_bs32) * 1000.0 if latency_bs32 > 0 else 0.0

        return EfficiencyMetrics(
            params_total=total,
            params_trainable=trainable,
            params_backbone=backbone_params,
            params_attention=attn_params,
            params_head=head_params,
            flops=flops,
            macs=macs,
            latency_bs1=latency_bs1,
            latency_bs32=latency_bs32,
            peak_memory=peak_memory,
            throughput=throughput
        )

    def forward(self, x: torch.Tensor):
        feats_list = self.backbone(x)
        f_second_last = feats_list[-2]
        f_last = feats_list[-1]
        f_second_last = self.attn_second_last(f_second_last)
        f_last = self.attn_last(f_last)
        f_last_up = F.interpolate(f_last, size=f_second_last.shape[2:], mode='bilinear', align_corners=False)
        fused_feats = torch.cat([f_second_last, f_last_up], dim=1)
        pooled = self.gap(fused_feats).flatten(1)
        logits = self.head(pooled)
        return logits, fused_feats


class PlainBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight)


class HistoryTracker:
    def __init__(self):
        self.epochs = []
        self.train_loss = []
        self.train_acc = []
        self.val_loss = []
        self.val_auc = []
        self.val_acc = []
        self.val_f1 = []
        self.val_precision = []
        self.val_recall = []
        self.val_specificity = []
        self.val_mcc = []
        self.lr = []
        self.best_auc = []
        self.patience_counter = []

    def log(self, epoch, train_loss, train_probs, train_targets, val_probs, val_targets,
            val_auc, lr, best_auc, patience, val_loss=None):
        self.epochs.append(epoch)
        self.train_loss.append(train_loss)

        if train_probs is not None and train_targets is not None:
            preds = (train_probs > 0.5).astype(int)
            self.train_acc.append(accuracy_score(train_targets.astype(int), preds))
        else:
            self.train_acc.append(np.nan)

        self.val_auc.append(val_auc)
        self.lr.append(lr)
        self.best_auc.append(best_auc)
        self.patience_counter.append(patience)

        preds = (val_probs > 0.5).astype(int)
        t_int = val_targets.astype(int)
        self.val_acc.append(accuracy_score(t_int, preds))
        self.val_f1.append(f1_score(t_int, preds, zero_division=0))
        self.val_precision.append(precision_score(t_int, preds, zero_division=0))
        self.val_recall.append(recall_score(t_int, preds, zero_division=0))
        self.val_specificity.append(recall_score(1 - t_int, 1 - preds, zero_division=0))
        self.val_mcc.append(matthews_corrcoef(t_int, preds))
        if val_loss is not None:
            self.val_loss.append(val_loss)


def plot_training_curves_separate(history: HistoryTracker, fold_num: int, best_epoch: int, save_dir: str = None):
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(history.epochs, history.train_loss, 'r-', linewidth=2.5, label='Training Loss')
    ax1.plot(history.epochs, history.val_loss, 'g-', linewidth=2.5, label='Validation Loss')
    best_idx = min(best_epoch - 1, len(history.val_loss) - 1)
    ax1.plot(best_epoch, history.val_loss[best_idx], 'bo', markersize=10, label=f'Best epoch = {best_epoch}')
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(f'Fold {fold_num} — Training and Validation Loss', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    all_loss = history.train_loss + history.val_loss
    max_loss = max(all_loss) if all_loss else 1.0
    ax1.set_ylim(0, max(0.5, max_loss * 1.2))
    fig1.suptitle(f'{VARIANT_NAME}', fontsize=11, fontweight='bold', y=0.98, color='#333333')
    fig1.tight_layout()
    if save_dir:
        save_path = os.path.join(save_dir, f"loss_curves_fold{fold_num}_{VARIANT_TAG}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.plot(history.epochs, history.train_acc, 'r-', linewidth=2.5, label='Training Accuracy')
    ax2.plot(history.epochs, history.val_auc, 'g-', linewidth=2.5, label='Validation AUC')
    ax2.plot(best_epoch, history.val_auc[best_idx], 'bo', markersize=10, label=f'Best epoch = {best_epoch}')
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Accuracy / AUC', fontsize=12)
    ax2.set_title(f'Fold {fold_num} — Training and Validation Performance', fontsize=13, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0.5, 1.0])
    fig2.suptitle(f'{VARIANT_NAME}', fontsize=11, fontweight='bold', y=0.98, color='#333333')
    fig2.tight_layout()
    if save_dir:
        save_path = os.path.join(save_dir, f"perf_curves_fold{fold_num}_{VARIANT_TAG}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()


def plot_per_fold_roc_pr_curves(fold_results: List[Dict], save_path: str = None):
    n_folds = len(fold_results)
    fig, axes = plt.subplots(n_folds, 2, figsize=(12, 4.5 * n_folds))

    for fold_idx, result in enumerate(fold_results):
        probs = result['probs']
        targets = result['targets']
        fold_num = result['fold']

        fpr, tpr, _ = roc_curve(targets.astype(int), probs)
        roc_auc = auc(fpr, tpr)
        axes[fold_idx, 0].plot(fpr, tpr, color='darkorange', lw=2.5, label=f'ROC (AUC = {roc_auc:.4f})')
        axes[fold_idx, 0].plot([0, 1], [0, 1], color='navy', lw=1.5, ls='--', label='Random')
        axes[fold_idx, 0].fill_between(fpr, tpr, alpha=0.2, color='darkorange')
        axes[fold_idx, 0].set_xlabel('False Positive Rate')
        axes[fold_idx, 0].set_ylabel('True Positive Rate')
        axes[fold_idx, 0].set_title(f'Fold {fold_num} — ROC Curve', fontweight='bold')
        axes[fold_idx, 0].set_xlim([0, 1]); axes[fold_idx, 0].set_ylim([0, 1.05])
        axes[fold_idx, 0].legend(loc='lower right'); axes[fold_idx, 0].grid(True, alpha=0.3)

        precision, recall, _ = precision_recall_curve(targets.astype(int), probs)
        pr_auc = auc(recall, precision)
        axes[fold_idx, 1].plot(recall, precision, color='purple', lw=2.5, label=f'PR (AUC = {pr_auc:.4f})')
        axes[fold_idx, 1].fill_between(recall, precision, alpha=0.2, color='purple')
        axes[fold_idx, 1].set_xlabel('Recall')
        axes[fold_idx, 1].set_ylabel('Precision')
        axes[fold_idx, 1].set_title(f'Fold {fold_num} — Precision-Recall Curve', fontweight='bold')
        axes[fold_idx, 1].set_xlim([0, 1]); axes[fold_idx, 1].set_ylim([0, 1.05])
        axes[fold_idx, 1].legend(loc='lower left'); axes[fold_idx, 1].grid(True, alpha=0.3)

    fig.suptitle(f'Per-Fold ROC & PR Curves — {VARIANT_NAME}', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    return fig


def plot_fold_statistics_heatmap(fold_results: List[Dict], save_path: str = None):
    metrics_list = ['auc', 'accuracy', 'f1', 'precision', 'recall', 'specificity', 'mcc']
    fold_data = []
    fold_labels = []

    for result in fold_results:
        fold_labels.append(f"Fold {result['fold']}")
        fold_data.append([result.get(m, 0.0) for m in metrics_list])

    fold_data = np.array(fold_data)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(fold_data, annot=True, fmt='.4f', cmap='RdYlGn', vmin=0.8, vmax=1.0,
                xticklabels=metrics_list, yticklabels=fold_labels, cbar_kws={'label': 'Score'},
                ax=ax, linewidths=0.5, linecolor='gray')
    ax.set_title(f'Per-Fold Metrics Heatmap — {VARIANT_NAME}', fontweight='bold', fontsize=13)
    ax.set_ylabel('Fold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    return fig


def plot_confusion_matrix(fold_results: List[Dict], save_path: str = None):
    n_folds = len(fold_results)
    fig, axes = plt.subplots(n_folds + 1, 2, figsize=(12, 5 * (n_folds + 1)))

    all_targets = []
    all_probs = []
    class_names = ['Benign', 'Malignant']

    for fold_idx, result in enumerate(fold_results):
        probs = result['probs']
        targets = result['targets']
        fold_num = result['fold']
        threshold = result.get('threshold', 0.5)

        all_targets.extend(targets)
        all_probs.extend(probs)

        preds = (probs > threshold).astype(int)
        t_int = targets.astype(int)

        cm = confusion_matrix(t_int, preds)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   ax=axes[fold_idx, 0], cbar=False,
                   annot_kws={'size': 14, 'weight': 'bold'})
        axes[fold_idx, 0].set_title(f'Fold {fold_num} — Confusion Matrix (Counts)',
                                   fontweight='bold', fontsize=12)
        axes[fold_idx, 0].set_ylabel('True Label', fontweight='bold')
        axes[fold_idx, 0].set_xlabel('Predicted Label', fontweight='bold')

        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='RdYlGn',
                   vmin=0, vmax=1,
                   xticklabels=class_names, yticklabels=class_names,
                   ax=axes[fold_idx, 1], cbar=True,
                   annot_kws={'size': 14, 'weight': 'bold'})
        axes[fold_idx, 1].set_title(f'Fold {fold_num} — Confusion Matrix (Normalized)',
                                   fontweight='bold', fontsize=12)
        axes[fold_idx, 1].set_ylabel('True Label', fontweight='bold')
        axes[fold_idx, 1].set_xlabel('Predicted Label', fontweight='bold')

    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    global_thr = np.mean([r['threshold'] for r in fold_results])
    all_preds = (all_probs > global_thr).astype(int)

    cm_total = confusion_matrix(all_targets.astype(int), all_preds)
    cm_total_norm = cm_total.astype('float') / cm_total.sum(axis=1)[:, np.newaxis]

    sns.heatmap(cm_total, annot=True, fmt='d', cmap='Blues',
               xticklabels=class_names, yticklabels=class_names,
               ax=axes[-1, 0], cbar=False,
               annot_kws={'size': 16, 'weight': 'bold'},
               linewidths=2, linecolor='white')
    axes[-1, 0].set_title('AGGREGATED — Confusion Matrix (Counts)',
                         fontweight='bold', fontsize=13, color='darkred')
    axes[-1, 0].set_ylabel('True Label', fontweight='bold')
    axes[-1, 0].set_xlabel('Predicted Label', fontweight='bold')

    sns.heatmap(cm_total_norm, annot=True, fmt='.2%', cmap='RdYlGn',
               vmin=0, vmax=1,
               xticklabels=class_names, yticklabels=class_names,
               ax=axes[-1, 1], cbar=True,
               annot_kws={'size': 16, 'weight': 'bold'},
               linewidths=2, linecolor='white')
    axes[-1, 1].set_title('AGGREGATED — Confusion Matrix (Normalized)',
                         fontweight='bold', fontsize=13, color='darkred')
    axes[-1, 1].set_ylabel('True Label', fontweight='bold')
    axes[-1, 1].set_xlabel('Predicted Label', fontweight='bold')

    fig.suptitle(f'Confusion Matrices — {VARIANT_NAME}',
                fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    return fig


def plot_final_classification_matrix(test_probs, test_labels, threshold, save_path=None):
    test_preds = (test_probs > threshold).astype(int)
    test_int = test_labels.astype(int)

    cm = confusion_matrix(test_int, test_preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    class_names = ['Benign', 'Malignant']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=class_names, yticklabels=class_names,
               ax=ax1, cbar=False,
               annot_kws={'size': 18, 'weight': 'bold'},
               linewidths=2, linecolor='white')
    ax1.set_title('Test Set — Confusion Matrix (Counts)',
                 fontweight='bold', fontsize=14)
    ax1.set_ylabel('True Label', fontweight='bold', fontsize=12)
    ax1.set_xlabel('Predicted Label', fontweight='bold', fontsize=12)

    tn, fp, fn, tp = cm.ravel()
    ax1.text(0.5, -0.15, f'TN={tn}  |  FP={fp}  |  FN={fn}  |  TP={tp}',
            transform=ax1.transAxes, ha='center', fontsize=11,
            style='italic', color='darkblue')

    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='RdYlGn',
               vmin=0, vmax=1,
               xticklabels=class_names, yticklabels=class_names,
               ax=ax2, cbar=True,
               annot_kws={'size': 18, 'weight': 'bold'},
               linewidths=2, linecolor='white')
    ax2.set_title('Test Set — Confusion Matrix (Normalized)',
                 fontweight='bold', fontsize=14)
    ax2.set_ylabel('True Label', fontweight='bold', fontsize=12)
    ax2.set_xlabel('Predicted Label', fontweight='bold', fontsize=12)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ax2.text(0.5, -0.15, f'Sensitivity (Recall): {sensitivity:.2%}  |  Specificity: {specificity:.2%}',
            transform=ax2.transAxes, ha='center', fontsize=11,
            style='italic', color='darkgreen')

    fig.suptitle(f'Final Classification Results — {VARIANT_NAME}\nThreshold = {threshold:.3f}',
                fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    return fig


def plot_test_roc_curve(test_probs, test_labels, save_path=None):
    test_int = test_labels.astype(int)
    probs = np.asarray(test_probs).squeeze().reshape(-1)

    fpr, tpr, thresholds = roc_curve(test_int, probs)
    roc_auc = auc(fpr, tpr)

    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_fpr = fpr[best_idx]
    best_tpr = tpr[best_idx]
    best_thr = thresholds[best_idx]

    n_bootstrap = 1000
    rng = np.random.RandomState(42)
    auc_scores = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, len(test_int), len(test_int))
        if len(np.unique(test_int[idx])) < 2:
            continue
        auc_scores.append(roc_auc_score(test_int[idx], probs[idx]))
    ci_lower = np.percentile(auc_scores, 2.5)
    ci_upper = np.percentile(auc_scores, 97.5)

    fig, ax = plt.subplots(figsize=(8, 7))

    ax.plot(fpr, tpr, color='#E74C3C', lw=3,
            label=f'MeLALight (AUC = {roc_auc:.4f})')
    ax.fill_between(fpr, tpr, alpha=0.15, color='#E74C3C')

    ax.plot([0, 1], [0, 1], color='#7F8C8D', lw=2, ls='--',
            label='Random Classifier (AUC = 0.5000)')

    ax.plot(best_fpr, best_tpr, 'o', color='#2C3E50', markersize=12,
            markeredgecolor='white', markeredgewidth=2, zorder=5,
            label=f'Optimal Threshold = {best_thr:.3f}\n'
                  f'(Sensitivity={best_tpr:.3f}, 1−Specificity={best_fpr:.3f})')

    ax.text(0.98, 0.40,
            f'AUC = {roc_auc:.4f}\n95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]',
            transform=ax.transAxes,
            fontsize=12, fontweight='bold',
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8F9FA',
                      edgecolor='#BDC3C7', alpha=0.9))

    ax.set_xlabel('False Positive Rate (1 − Specificity)', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=13, fontweight='bold')
    ax.set_title(f'Test Set ROC Curve — {VARIANT_NAME}', fontsize=15, fontweight='bold')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_aspect('equal')

    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

    print(f"  [ROC] Test AUC = {roc_auc:.4f}  95% CI = [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"  [ROC] Optimal threshold = {best_thr:.3f}  (Youden's J)")
    return fig


def plot_parameter_distribution_pie(params_backbone, params_attention, params_head, save_path=None):
    labels = ['Backbone', 'Attention (FcaNet)', 'Head']
    sizes = [params_backbone, params_attention, params_head]
    total = sum(sizes)
    colors = ['#3498db', '#2ecc71', '#e67e22']
    explode = (0.03, 0.06, 0.06)

    fig, ax = plt.subplots(figsize=(10, 6))

    wedges, _ = ax.pie(
        sizes,
        explode=explode,
        colors=colors,
        startangle=90,
        shadow=False,
        wedgeprops=dict(width=0.7, edgecolor='white', linewidth=2)
    )

    legend_labels = [
        f'{labels[i]}:  {sizes[i]:.3f}M  ({sizes[i]/total*100:.1f}%)'
        for i in range(len(labels))
    ]

    ax.legend(
        wedges, legend_labels,
        title="Component",
        title_fontsize=13,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=12,
        frameon=True,
        fancybox=True,
        shadow=True
    )

    ax.text(0, 0, f'Total\n{total:.2f}M',
            ha='center', va='center',
            fontsize=16, fontweight='bold', color='#2C3E50')

    ax.set_title(f'Parameter Distribution — {VARIANT_NAME}',
                 fontweight='bold', fontsize=14, pad=20)

    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()


def export_metrics_table_csv(fold_results: List[Dict], efficiency: EfficiencyMetrics,
                             test_metrics: Dict, save_path: str = None):
    fold_df = pd.DataFrame(fold_results)
    fold_df = fold_df[['fold', 'auc', 'accuracy', 'f1', 'precision', 'recall', 'specificity', 'mcc', 'threshold']]
    fold_df.columns = ['Fold', 'AUC', 'Accuracy', 'F1', 'Precision', 'Recall', 'Specificity', 'MCC', 'Threshold']

    summary_data = {
        'Metric': ['AUC', 'Accuracy', 'F1', 'Precision', 'Recall', 'Specificity', 'MCC'],
        'Mean': [fold_df['AUC'].mean(), fold_df['Accuracy'].mean(), fold_df['F1'].mean(),
                 fold_df['Precision'].mean(), fold_df['Recall'].mean(), fold_df['Specificity'].mean(),
                 fold_df['MCC'].mean()],
        'Std': [fold_df['AUC'].std(), fold_df['Accuracy'].std(), fold_df['F1'].std(),
                fold_df['Precision'].std(), fold_df['Recall'].std(), fold_df['Specificity'].std(),
                fold_df['MCC'].std()]
    }
    summary_df = pd.DataFrame(summary_data)

    efficiency_data = {
        'Component': ['Backbone', 'Attention', 'Head', 'TOTAL'],
        'Parameters (M)': [f"{efficiency.params_backbone:.2f}", f"{efficiency.params_attention:.2f}",
                          f"{efficiency.params_head:.2f}", f"{efficiency.params_total:.2f}"]
    }
    efficiency_df = pd.DataFrame(efficiency_data)

    hardware_data = {
        'Metric': ['FLOPs (B)', 'MACs (B)', 'FLOP Backend', 'Latency BS=1 (ms)', 'Latency BS=32 (ms)',
                   'Peak Memory (MB)', 'Throughput (img/s)'],
        'Value': [f"{efficiency.flops:.4f}", f"{efficiency.macs:.4f}", FLOP_BACKEND,
                 f"{efficiency.latency_bs1:.2f}",
                 f"{efficiency.latency_bs32:.2f}", f"{efficiency.peak_memory:.1f}", f"{efficiency.throughput:.1f}"]
    }
    hardware_df = pd.DataFrame(hardware_data)

    if save_path:
        with open(save_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write(f"FOLD-WISE RESULTS — {VARIANT_NAME}\n")
            f.write("="*70 + "\n")
            fold_df.to_csv(f, index=False)
            f.write("\n" + "="*70 + "\n")
            f.write("SUMMARY STATISTICS (5-FOLD CV)\n")
            f.write("="*70 + "\n")
            summary_df.to_csv(f, index=False)
            f.write("\n" + "="*70 + "\n")
            f.write("PARAMETER BREAKDOWN\n")
            f.write("="*70 + "\n")
            efficiency_df.to_csv(f, index=False)
            f.write("\n" + "="*70 + "\n")
            f.write(f"EFFICIENCY METRICS (224x224, FP32, backend={FLOP_BACKEND})\n")
            f.write("="*70 + "\n")
            hardware_df.to_csv(f, index=False)

    print(f"\n Metrics exported to {save_path}")
    return fold_df, summary_df, efficiency_df, hardware_df


def plot_accuracy_vs_parameters(accuracy, params_total, save_path=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(params_total, accuracy, s=200, c='royalblue', marker='D', edgecolors='black', linewidth=1.5)
    ax.annotate(f'MeLALight\n{accuracy:.3f}',
                xy=(params_total, accuracy),
                xytext=(params_total*0.8, accuracy-0.02),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=10, ha='center')
    ax.set_xlabel('Total Parameters (M)', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title(f'Accuracy vs Parameters — {VARIANT_NAME}', fontweight='bold', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, params_total * 1.5)
    ax.set_ylim(0.5, 1.0)
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()


def plot_auc_vs_parameters(auc, params_total, save_path=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(params_total, auc, s=200, c='darkorange', marker='s', edgecolors='black', linewidth=1.5)
    ax.annotate(f'MeLALight\n{auc:.3f}',
                xy=(params_total, auc),
                xytext=(params_total*0.8, auc-0.02),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=10, ha='center')
    ax.set_xlabel('Total Parameters (M)', fontsize=12)
    ax.set_ylabel('Test AUC', fontsize=12)
    ax.set_title(f'AUC vs Parameters — {VARIANT_NAME}', fontweight='bold', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, params_total * 1.5)
    ax.set_ylim(0.5, 1.0)
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()


def plot_accuracy_vs_flops(accuracy, flops, save_path=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(flops, accuracy, s=200, c='forestgreen', marker='^', edgecolors='black', linewidth=1.5)
    ax.annotate(f'MeLALight\n{accuracy:.3f}',
                xy=(flops, accuracy),
                xytext=(flops*0.8, accuracy-0.02),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=10, ha='center')
    ax.set_xlabel('FLOPs (B)', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title(f'Accuracy vs FLOPs — {VARIANT_NAME}', fontweight='bold', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, flops * 1.5)
    ax.set_ylim(0.5, 1.0)
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()


def plot_auc_vs_latency(auc, latency_bs1, save_path=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(latency_bs1, auc, s=200, c='crimson', marker='o', edgecolors='black', linewidth=1.5)
    ax.annotate(f'MeLALight\n{auc:.3f}',
                xy=(latency_bs1, auc),
                xytext=(latency_bs1*0.8, auc-0.02),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=10, ha='center')
    ax.set_xlabel('Latency BS=1 (ms)', fontsize=12)
    ax.set_ylabel('Test AUC', fontsize=12)
    ax.set_title(f'AUC vs Latency — {VARIANT_NAME}', fontweight='bold', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, latency_bs1 * 1.5)
    ax.set_ylim(0.5, 1.0)
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()


def train_epoch(model, loader, optimizer, criterion, scaler, pos_weight=None):
    model.train()
    total_loss = 0.0
    all_logits = []
    all_targets = []
    num_batches = 0

    for x, y in tqdm(loader, leave=False, desc="Train"):
        x, y = x.to(DEVICE), y.float().unsqueeze(1).to(DEVICE)

        optimizer.zero_grad()
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits, _ = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        all_logits.extend(logits.reshape(-1).detach().cpu().numpy())
        all_targets.extend(y.reshape(-1).cpu().numpy())
        num_batches += 1

    all_logits = np.array(all_logits)
    all_targets = np.array(all_targets)
    train_probs = torch.sigmoid(torch.tensor(all_logits)).numpy()

    return total_loss / num_batches, train_probs, all_targets


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    logits_list = []
    targets = []
    for x, y in tqdm(loader, leave=False, desc="Eval"):
        x = x.to(DEVICE)
        logits, _ = model(x)
        logits_list.extend(logits.reshape(-1).cpu().numpy())
        targets.extend(y.numpy())

    val_logits = np.array(logits_list)
    val_targets = np.array(targets)
    return val_logits, val_targets


def get_probabilities_from_logits(logits):
    return torch.sigmoid(torch.tensor(logits, dtype=torch.float32)).numpy()


def compute_val_loss(logits, targets, pos_weight=None):
    logits_tensor = torch.tensor(logits, dtype=torch.float32).to(DEVICE).reshape(-1)
    targets_tensor = torch.tensor(targets, dtype=torch.float32).to(DEVICE).reshape(-1)
    return F.binary_cross_entropy_with_logits(logits_tensor, targets_tensor, pos_weight=pos_weight, reduction='mean').item()


@torch.no_grad()
def tta_evaluate(model, transforms_list):
    model.eval()
    all_probs = []
    for tf in transforms_list:
        ds = datasets.ImageFolder(TEST_DIR, transform=tf)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        fold_logits = []
        for x, _ in tqdm(loader, leave=False, desc="TTA"):
            x = x.to(DEVICE)
            logits, _ = model(x)
            fold_logits.extend(logits.reshape(-1).cpu().numpy())
        all_probs.append(torch.sigmoid(torch.tensor(fold_logits)).numpy())
    return np.mean(all_probs, axis=0)


def compute_metrics(probs, targets, thr=0.5) -> Dict:
    probs = np.asarray(probs).squeeze().reshape(-1)
    targets = np.asarray(targets).squeeze().reshape(-1)
    preds = (probs > thr).astype(int)
    t_int = targets.astype(int)
    return {
        'auc': roc_auc_score(t_int, probs),
        'accuracy': accuracy_score(t_int, preds),
        'f1': f1_score(t_int, preds, zero_division=0),
        'precision': precision_score(t_int, preds, zero_division=0),
        'recall': recall_score(t_int, preds, zero_division=0),
        'sensitivity': recall_score(t_int, preds, zero_division=0),
        'specificity': recall_score(1 - t_int, 1 - preds, zero_division=0),
        'mcc': matthews_corrcoef(t_int, preds),
    }


def find_best_threshold(probs, targets, metric='f1'):
    probs = np.asarray(probs).squeeze().reshape(-1)
    targets = np.asarray(targets).squeeze().reshape(-1)
    best_t, best_score = 0.5, 0.0
    for t in np.linspace(0.05, 0.95, 191):
        preds = (probs > t).astype(int)
        score = f1_score(targets, preds, zero_division=0) if metric == 'f1' else accuracy_score(targets, preds)
        if score > best_score:
            best_score, best_t = score, t
    return best_t, best_score


print(f"\n{'='*70}")
print(f"{VARIANT_NAME} — MULTI-STAGE (LAST TWO) WITH PUBLICATION UTILITIES")
print(f"{'='*70}")
print(f"Backbone: {BACKBONE_NAME} (5.5M RepViT M0.9)")
print(f"Head dropout: 0.3 (reduced)")
print(f"Head size: 256→128→1 (smaller)")
print(f"L2 regularization: 1e-5 (added)")
print(f"Patience: {PATIENCE} (reduced from 7)")
print(f"Loss: BCE with pos_weight")
print(f"Epochs: {EPOCHS}")
print(f"FLOP Backend: {FLOP_BACKEND}")
print(f"{'='*70}\n")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
fold_results = []
fold_results_detailed = []
all_test_probs = []

use_amp = torch.cuda.is_available()
scaler = torch.cuda.amp.GradScaler() if use_amp else None

efficiency_metrics = None

for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(np.arange(len(train_labels)), train_labels), 1):
    print(f"\n{'─'*70}\nFOLD {fold_idx}/{N_FOLDS}\n{'─'*70}")

    train_ds = Subset(datasets.ImageFolder(TRAIN_DIR, transform=train_tf), tr_idx)
    val_ds   = Subset(datasets.ImageFolder(TRAIN_DIR, transform=val_tf), va_idx)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = MeLALight(num_classes=1, head_dropout=0.3, l2_weight_decay=1e-5).to(DEVICE)

    if efficiency_metrics is None:
        efficiency_metrics = model.efficiency_metrics
        print(efficiency_metrics.summary_str())

    history = HistoryTracker()
    criterion = PlainBCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=3, min_lr=MIN_LR)

    best_auc = 0.0
    best_val_loss = float('inf')
    patience = 0
    best_path = os.path.join(SAVE_DIR, f"melalight_{VARIANT_TAG}_fold{fold_idx}.pth")
    best_epoch = 0

    print(f"  Phase 1: Training with FROZEN backbone...")
    for epoch in range(1, EPOCHS // 2 + 1):
        train_loss, train_probs, train_targets = train_epoch(
            model, train_loader, optimizer, criterion, scaler, pos_weight=pos_weight
        )
        val_logits, val_targets = evaluate(model, val_loader)
        val_probs = get_probabilities_from_logits(val_logits)
        val_loss = compute_val_loss(val_logits, val_targets, pos_weight=pos_weight)
        val_auc = roc_auc_score(val_targets.astype(int), val_probs)

        current_lr = optimizer.param_groups[0]['lr']
        history.log(epoch, train_loss, train_probs, train_targets,
                   val_probs, val_targets, val_auc, current_lr, best_auc, patience, val_loss=val_loss)
        scheduler.step(val_auc)

        if val_auc > best_auc:
            best_auc = val_auc
            best_val_loss = val_loss
            patience = 0
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)
            print(f"  Epoch {epoch:02d} | TrainLoss:{train_loss:.4f} | ValLoss:{val_loss:.4f} | AUC:{val_auc:.4f} | LR:{current_lr:.2e} | Saved")
        else:
            patience += 1
            print(f"  Epoch {epoch:02d} | TrainLoss:{train_loss:.4f} | ValLoss:{val_loss:.4f} | AUC:{val_auc:.4f} | LR:{current_lr:.2e}")
            if patience >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    patience = 0

    print(f"  Phase 2: Unfreezing backbone for fine-tuning...")
    model.unfreeze_backbone()
    for param_group in optimizer.param_groups:
        param_group['lr'] = LR / 10

    for epoch in range(EPOCHS // 2 + 1, EPOCHS + 1):
        train_loss, train_probs, train_targets = train_epoch(
            model, train_loader, optimizer, criterion, scaler, pos_weight=pos_weight
        )
        val_logits, val_targets = evaluate(model, val_loader)
        val_probs = get_probabilities_from_logits(val_logits)
        val_loss = compute_val_loss(val_logits, val_targets, pos_weight=pos_weight)
        val_auc = roc_auc_score(val_targets.astype(int), val_probs)

        current_lr = optimizer.param_groups[0]['lr']
        history.log(epoch, train_loss, train_probs, train_targets,
                   val_probs, val_targets, val_auc, current_lr, best_auc, patience, val_loss=val_loss)
        scheduler.step(val_auc)

        if val_auc > best_auc:
            best_auc = val_auc
            best_val_loss = val_loss
            patience = 0
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)
            print(f"  Epoch {epoch:02d} | TrainLoss:{train_loss:.4f} | ValLoss:{val_loss:.4f} | AUC:{val_auc:.4f} | LR:{current_lr:.2e} | Saved")
        else:
            patience += 1
            print(f"  Epoch {epoch:02d} | TrainLoss:{train_loss:.4f} | ValLoss:{val_loss:.4f} | AUC:{val_auc:.4f} | LR:{current_lr:.2e}")
            if patience >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(best_path, map_location=DEVICE))

    plot_training_curves_separate(history, fold_idx, best_epoch, save_dir=SAVE_DIR)

    val_logits, val_targets = evaluate(model, val_loader)
    val_probs = get_probabilities_from_logits(val_logits)
    thr, _ = find_best_threshold(val_probs, val_targets.astype(int), metric='f1')
    val_metrics = compute_metrics(val_probs, val_targets, thr=thr)
    val_metrics['fold'] = fold_idx
    val_metrics['threshold'] = thr
    val_metrics['best_epoch'] = best_epoch
    fold_results.append(val_metrics)

    fold_results_detailed.append({
        'fold': fold_idx,
        'probs': val_probs,
        'targets': val_targets,
        **val_metrics
    })

    print(f"\n  {'─'*60}")
    print(f"  CLASSIFICATION REPORT — FOLD {fold_idx} (Validation Set)")
    print(f"  {'─'*60}")
    print(classification_report(val_targets.astype(int), (val_probs > thr).astype(int),
                                target_names=['Benign', 'Malignant'], digits=4))

    print(f"  Fold {fold_idx}  AUC:{val_metrics['auc']:.4f}  Acc:{val_metrics['accuracy']:.4f}  F1:{val_metrics['f1']:.4f}")

    test_probs = tta_evaluate(model, tta_transforms)
    all_test_probs.append(test_probs)

    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()

print(f"\n{'='*70}")
print("GENERATING PUBLICATION ARTIFACTS")
print(f"{'='*70}")

global_thr = np.mean([r['threshold'] for r in fold_results])

test_probs_arr = np.stack(all_test_probs, axis=0)
ensemble_probs = test_probs_arr.mean(axis=0)

test_metrics = compute_metrics(ensemble_probs, test_labels, thr=global_thr)

test_preds = (ensemble_probs > global_thr).astype(int)
print(f"\n{'='*70}")
print(f"DETAILED CLASSIFICATION REPORT — TEST SET")
print(f"{'='*70}")
print(classification_report(test_labels.astype(int), test_preds,
                            target_names=['Benign', 'Malignant'], digits=4))

print(f"\n{'='*70}")
print(f"TEST SET RESULTS — {VARIANT_NAME}")
print(f"{'='*70}")
print(f"  AUC:         {test_metrics['auc']:.4f}")
print(f"  Accuracy:    {test_metrics['accuracy']:.4f}")
print(f"  F1:          {test_metrics['f1']:.4f}")
print(f"  Precision:   {test_metrics['precision']:.4f}")
print(f"  Recall:      {test_metrics['recall']:.4f}")
print(f"  Sensitivity: {test_metrics['sensitivity']:.4f}")
print(f"  Specificity: {test_metrics['specificity']:.4f}")
print(f"  MCC:         {test_metrics['mcc']:.4f}")
print(f"  Threshold:   {global_thr:.3f}")

print(f"\nGenerating publication figures...")

plot_per_fold_roc_pr_curves(
    fold_results_detailed,
    save_path=os.path.join(SAVE_DIR, f"per_fold_roc_pr_{VARIANT_TAG}.png")
)

plot_fold_statistics_heatmap(
    fold_results,
    save_path=os.path.join(SAVE_DIR, f"fold_stats_heatmap_{VARIANT_TAG}.png")
)

plot_confusion_matrix(
    fold_results_detailed,
    save_path=os.path.join(SAVE_DIR, f"confusion_matrices_{VARIANT_TAG}.png")
)

plot_final_classification_matrix(
    ensemble_probs, test_labels, global_thr,
    save_path=os.path.join(SAVE_DIR, f"final_classification_matrix_{VARIANT_TAG}.png")
)

plot_test_roc_curve(
    ensemble_probs, test_labels,
    save_path=os.path.join(SAVE_DIR, f"test_roc_curve_{VARIANT_TAG}.png")
)

print("\nGenerating efficiency trade-off and parameter distribution plots...")

params_total = efficiency_metrics.params_total
params_backbone = efficiency_metrics.params_backbone
params_attention = efficiency_metrics.params_attention
params_head = efficiency_metrics.params_head
flops = efficiency_metrics.flops
latency_bs1 = efficiency_metrics.latency_bs1

test_acc = test_metrics['accuracy']
test_auc = test_metrics['auc']

plot_accuracy_vs_parameters(
    test_acc, params_total,
    save_path=os.path.join(SAVE_DIR, f"accuracy_vs_params_{VARIANT_TAG}.png")
)

plot_auc_vs_parameters(
    test_auc, params_total,
    save_path=os.path.join(SAVE_DIR, f"auc_vs_params_{VARIANT_TAG}.png")
)

plot_accuracy_vs_flops(
    test_acc, flops,
    save_path=os.path.join(SAVE_DIR, f"accuracy_vs_flops_{VARIANT_TAG}.png")
)

plot_auc_vs_latency(
    test_auc, latency_bs1,
    save_path=os.path.join(SAVE_DIR, f"auc_vs_latency_{VARIANT_TAG}.png")
)

plot_parameter_distribution_pie(
    params_backbone, params_attention, params_head,
    save_path=os.path.join(SAVE_DIR, f"param_distribution_pie_{VARIANT_TAG}.png")
)

print(f"\nExporting metrics & documentation...")

fold_df, summary_df, efficiency_df, hardware_df = export_metrics_table_csv(
    fold_results,
    efficiency_metrics,
    test_metrics,
    save_path=os.path.join(SAVE_DIR, f"metrics_summary_{VARIANT_TAG}.csv")
)

print(f"\n{summary_df.to_string(index=False)}")

print(f"\n{'='*70}")
print("ALL PUBLICATION ARTIFACTS GENERATED SUCCESSFULLY")
print(f"{'='*70}")
print(f"\nOutputs in: {SAVE_DIR}")
print(f"\nKey files:")
print(f"  • metrics_summary_{VARIANT_TAG}.csv")
print(f"  • loss_curves_fold{{1..5}}_{VARIANT_TAG}.png")
print(f"  • perf_curves_fold{{1..5}}_{VARIANT_TAG}.png")
print(f"  • per_fold_roc_pr_{VARIANT_TAG}.png")
print(f"  • confusion_matrices_{VARIANT_TAG}.png")
print(f"  • final_classification_matrix_{VARIANT_TAG}.png")
print(f"  • test_roc_curve_{VARIANT_TAG}.png")
print(f"  • accuracy_vs_params_{VARIANT_TAG}.png")
print(f"  • auc_vs_params_{VARIANT_TAG}.png")
print(f"  • accuracy_vs_flops_{VARIANT_TAG}.png")
print(f"  • auc_vs_latency_{VARIANT_TAG}.png")
print(f"  • param_distribution_pie_{VARIANT_TAG}.png")

print(f"\n{efficiency_metrics.summary_str()}")

print(f"\n{'='*70}")
print("SUMMARY OF ENHANCEMENTS APPLIED:")
print(f"{'='*70}")
print(f"1. Same multi-stage architecture (last two backbone stages) – efficient and accurate.")
print(f"2. NEW: 3-tier FLOP counting with comprehensive manual hooks and backend reporting.")
print(f"3. NEW: Test ROC curve with 95% CI and optimal threshold (annotation above legend).")
print(f"4. NEW: Donut-style pie chart with external legend (zero overlap).")
print(f"5. All existing fixes preserved: patience reset, shape consistency, L2, reduced dropout, etc.")
print(f"{'='*70}\n")

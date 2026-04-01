#!/usr/bin/env python3
"""
BERT Document Classifier — Legal Text Classification
Refactored: I. Source code (Text): Class-based format

Pipeline: Document → Tokenize → Split chunks → BERT encode → CLS extraction
        → ChunkAttentionMHA → MLP → Output

Classes:
  1. Config        — hyper-parameters, paths, model config
  2. BertDocDataset — EDA + Augmentation (pre-tokenize, chunking)
  3. DataLoaders   — train / val / test splits + BalancedBatchSampler
  4. ChunkAttentionMHA + FullModel — Input → Preprocess → Backbone → Head → Output
  5. FocalLoss + train_epoch — batch-wise → epoch → Loss
  6. Finetuning    — freeze backbone, layer-wise LR
  7. Evaluator     — Accuracy, Precision, Recall, F1 (macro + per-class)
  8. Analysis      — Params, overfitting curve, inference time, attention map
"""

import math
import os
import pickle
import random
import shutil
import time
import re
import numpy as np
import pandas as pd
import torch
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch.nn as nn
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, classification_report, confusion_matrix)
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


# =============================================================================
# 1. CONFIG RESOLVE
# =============================================================================
class Config:
    DATA_PATH = './data/processed/'
    OUTPUT_PATH = './output-train-bert/'
    CHECKPOINT_LOAD_PATH = './output-train-bert/'
    CHECKPOINT_SAVE_PATH = './output-train-bert/'

    BERT_MODEL = 'google-bert/bert-base-cased'
    BERT_MAX_SEQ = 512

    MAX_CHUNKS = 6
    CHUNK_SIZE = 384
    CHUNK_STRIDE = 256

    BATCH_SIZE = 42
    ACCUM_STEPS = 2
    GRADIENT_CHECKPOINTING = True
    USE_FP16 = False
    BERT_FT_LAYERS = 4
    NUM_HEADS = 8
    EPOCHS = 10
    WEIGHT_DECAY = 0.01
    DROPOUT = 0.3

    WARMUP_EPOCHS = 1
    LR_BERT = 2e-5
    LR_ATTENTION = 1e-4
    LR_MLP = 5e-5

    # [Civil=0, Corporate=1, CourtOfClaims=2, Criminal=3, Other=4, Probate=5, Property=6]
    MANUAL_FOCAL_ALPHA = [0.3, 5.0, 3.0, 1.5, 5.0, 2.5, 1.5]
    FOCAL_GAMMA = 2.0
    ABLATION_MODE = 'A'

    if ABLATION_MODE == 'A':
        LABEL_SMOOTHING = 0.0
    elif ABLATION_MODE == 'B':
        LABEL_SMOOTHING = 0.05
    elif ABLATION_MODE == 'C':
        LABEL_SMOOTHING = 0.05
        MANUAL_FOCAL_ALPHA = [0.3, 3.5, 3.0, 1.5, 3.5, 2.5, 1.5]

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# =============================================================================
# 2. DATASET — EDA + Augmentation
# =============================================================================
class DataResolver:
    """Resolve data paths and platform detection."""

    @staticmethod
    def resolve():
        on_kaggle = os.path.exists('/kaggle')
        on_lightning = os.path.exists('/teamspace/studios/this_studio/')
        on_colab = False

        if on_kaggle:
            Config.DATA_PATH = '/kaggle/input/datasets/nhiwentwest/data-processed-2/'
            Config.OUTPUT_PATH = '/kaggle/working/output-train-bert/'
            Config.CHECKPOINT_LOAD_PATH = '/kaggle/input/datasets/nhiwentwest/checkpoint-bert/'
            Config.CHECKPOINT_SAVE_PATH = '/kaggle/working/output-train-bert/'
        elif on_lightning:
            base = '/teamspace/studios/this_studio/'
            Config.DATA_PATH = os.path.join(base, 'data/processed/')
            Config.OUTPUT_PATH = os.path.join(base, 'output-train-bert/')
        else:
            try:
                from google.colab import drive
                ip = __import__('IPython').get_ipython()
                if ip is not None and ip.kernel is not None:
                    on_colab = True
            except Exception:
                pass
            if on_colab:
                drive.mount('/content/drive')
                base = '/content/drive/MyDrive/output-train-bert/'
                Config.OUTPUT_PATH = base
                Config.CHECKPOINT_LOAD_PATH = base
                Config.CHECKPOINT_SAVE_PATH = base
                for p in ['/content/drive/MyDrive/data/processed/', '/content/data/processed/']:
                    if os.path.exists(p):
                        Config.DATA_PATH = p
                        break
            else:
                for p in ['./data/processed/', './data/masked-15k/', '../data/processed/']:
                    if os.path.exists(os.path.join(p, 'train.csv')):
                        Config.DATA_PATH = p
                        break

        os.makedirs(Config.OUTPUT_PATH, exist_ok=True)
        os.makedirs(Config.CHECKPOINT_SAVE_PATH, exist_ok=True)
        for attr in ['OUTPUT_PATH', 'CHECKPOINT_SAVE_PATH']:
            if not getattr(Config, attr).endswith('/'):
                setattr(Config, attr, getattr(Config, attr) + '/')


class EDAAugmentation:
    """
    EDA: Exploratory Data Analysis + text augmentation.
    Statistics: class distribution, text length distribution, sequence length stats.
    Augmentation: not applied by default (BERT fine-tuning is data-hungry;
                  augmentation applied via BalancedBatchSampler for class imbalance).
    """

    @staticmethod
    def analyze(df, name='Dataset'):
        print(f"\n  [{name}] EDA:")
        print(f"    Samples: {len(df):,}")

        label_counts = df['label'].value_counts()
        print(f"    Class distribution:")
        for cls, cnt in label_counts.items():
            pct = cnt / len(df) * 100
            print(f"      {cls}: {cnt:,} ({pct:.1f}%)")

        text_lens = df['text'].astype(str).str.len()
        token_est = text_lens / 4.5
        print(f"    Text length (chars): min={text_lens.min()}, "
              f"median={int(text_lens.median())}, max={text_lens.max()}")
        print(f"    Token estimate: min={int(token_est.min())}, "
              f"median={int(token_est.median())}, max={int(token_est.max())}")
        return label_counts

    @staticmethod
    def augment_text(text, aug_type='none'):
        if aug_type == 'none':
            return text
        text = str(text)
        if aug_type == 'shuffle_sentences':
            sentences = re.split(r'(?<=[.!?])\s+', text)
            random.shuffle(sentences)
            return ' '.join(sentences)
        return text


class BertDocDataset(Dataset):
    """
    Dataset: pre-tokenizes full document without special tokens,
    then slices into overlapping chunks at access time.
    Each chunk gets a clean [CLS]...[SEP] structure from the model.

    Cache: pre-tokenized tensors saved to disk (pickle) and loaded
    on subsequent runs to skip the slow pre-tokenization step.
    """

    _cache_dir = '/tmp/.bert_cache/'

    def __init__(self, texts, labels, tokenizer, max_chunks=6,
                 chunk_size=384, chunk_stride=256, split='train'):
        self.max_chunks = max_chunks
        self.chunk_size = chunk_size
        self.chunk_stride = chunk_stride
        self.labels = [torch.tensor(l, dtype=torch.long) for l in labels]
        self._tokenizer_name = tokenizer.__class__.__name__

        total_len = max_chunks * chunk_stride + (chunk_size - chunk_stride)
        cache_key = f'{split}_{len(texts)}_{max_chunks}_{chunk_size}_{chunk_stride}_{total_len}'
        os.makedirs(self._cache_dir, exist_ok=True)
        self._cache_path = os.path.join(self._cache_dir, f'{cache_key}.pt')

        if os.path.exists(self._cache_path):
            print(f"    Loading cached pre-tokenization from {self._cache_path}...")
            cached = torch.load(self._cache_path, map_location='cpu', weights_only=True)
            self.input_ids = cached['input_ids']
            self.attention_mask = cached['attention_mask']
            print(f"    Loaded {self.input_ids.shape[0]:,} cached samples.")
        else:
            N = len(texts)
            print(f"    Pre-tokenizing {N:,} samples (no special tokens)...")
            self.input_ids = torch.zeros(N, total_len, dtype=torch.long)
            self.attention_mask = torch.zeros(N, total_len, dtype=torch.long)
            ptr = 0
            batch_texts = []

            for i, text in enumerate(texts):
                batch_texts.append(str(text))
                if len(batch_texts) >= 256 or i == len(texts) - 1:
                    encoded = tokenizer(
                        batch_texts,
                        max_length=total_len,
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt',
                        return_token_type_ids=False,
                        add_special_tokens=False,
                    )
                    bs = encoded['input_ids'].shape[0]
                    self.input_ids[ptr:ptr + bs] = encoded['input_ids']
                    self.attention_mask[ptr:ptr + bs] = encoded['attention_mask']
                    ptr += bs
                    batch_texts = []

            print(f"    Done pre-tokenizing {ptr:,} samples — saving cache...")
            torch.save({
                'input_ids': self.input_ids,
                'attention_mask': self.attention_mask,
            }, self._cache_path)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx], self.labels[idx]


# =============================================================================
# 3. DATA LOADERS — train / val / test splits
# =============================================================================
class DataSplits:
    """Load data, encode labels, return train/val/test DataFrames."""

    @staticmethod
    def load():
        print(f"Loading data from {Config.DATA_PATH}...")

        if not os.path.exists(Config.DATA_PATH):
            raise FileNotFoundError(f"Directory not found: {Config.DATA_PATH}")

        csv_train = f'{Config.DATA_PATH}train.csv'
        parquet_train = f'{Config.DATA_PATH}train.parquet'

        if os.path.exists(csv_train):
            try:
                train_df = pd.read_csv(csv_train, engine='python', on_bad_lines='skip')
                val_df = pd.read_csv(f'{Config.DATA_PATH}val.csv', engine='python', on_bad_lines='skip')
                test_df = pd.read_csv(f'{Config.DATA_PATH}test.csv', engine='python', on_bad_lines='skip')
            except Exception:
                train_df = pd.read_csv(csv_train, quoting=3, on_bad_lines='skip')
                val_df = pd.read_csv(f'{Config.DATA_PATH}val.csv', quoting=3, on_bad_lines='skip')
                test_df = pd.read_csv(f'{Config.DATA_PATH}test.csv', quoting=3, on_bad_lines='skip')
        elif os.path.exists(parquet_train):
            train_df = pd.read_parquet(parquet_train)
            val_df = pd.read_parquet(f'{Config.DATA_PATH}val.parquet')
            test_df = pd.read_parquet(f'{Config.DATA_PATH}test.parquet')
        else:
            raise FileNotFoundError(f"No train.csv or train.parquet in {Config.DATA_PATH}")

        print(f"  Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")

        for name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            for col in ['text', 'label']:
                if col not in df.columns:
                    continue
                nan_count = df[col].isna().sum()
                inf_count = 0
                if col == 'text':
                    inf_count = df[col].apply(
                        lambda x: isinstance(x, float) and np.isinf(x)).sum()
                if nan_count > 0 or inf_count > 0:
                    raise ValueError(f"Data corruption in {name}.{col}")

        MIN_TEXT_LEN = 20
        for name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            df['text'] = df['text'].astype(str)
            orig_len = len(df)
            df.drop(df[df['text'].str.len() <= MIN_TEXT_LEN].index, inplace=True)
            dropped = orig_len - len(df)
            if dropped > 0:
                print(f"  Dropped {dropped} tiny samples (<={MIN_TEXT_LEN} chars) "
                      f"from {name} ({dropped * 100 / orig_len:.2f}%)")

        possible_le = [
            '/teamspace/studios/this_studio/data/processed/label_encoder.pkl',
            f'{Config.DATA_PATH}label_encoder.pkl',
            '/kaggle/input/datasets/nhiwentwest/data-processed-2/label_encoder.pkl',
            '/content/data/processed/label_encoder.pkl',
            './data/masked-15k/label_encoder.pkl',
        ]
        le_path = next((p for p in possible_le if os.path.exists(p)), None)

        if le_path:
            with open(le_path, 'rb') as f:
                le = pickle.load(f)
            print(f"  Loaded label_encoder from {le_path}")
        else:
            print("  WARNING: label_encoder.pkl not found — rebuilding...")
            le = LabelEncoder()
            le.fit(train_df['label'])

        print(f"  Classes: {list(le.classes_)}")
        train_df['label_encoded'] = le.transform(train_df['label'])
        val_df['label_encoded'] = le.transform(val_df['label'])
        test_df['label_encoded'] = le.transform(test_df['label'])

        return train_df, val_df, test_df, le


class BalancedBatchSampler(torch.utils.data.BatchSampler):
    """
    BalancedBatchSampler yields batches where each class contributes
    the same number of samples, boosting minority class representation.
    """

    def __init__(self, labels, batch_size, drop_last=False, shuffle_classes=True):
        from torch.utils.data import SequentialSampler
        super().__init__(SequentialSampler(labels), batch_size, drop_last)
        self.labels = list(labels)
        self.batch_size = batch_size
        self.shuffle_classes = shuffle_classes

        self.class_to_indices = defaultdict(list)
        for idx, y in enumerate(self.labels):
            self.class_to_indices[int(y)].append(idx)

        self.classes = sorted(self.class_to_indices.keys())
        self.num_classes = len(self.classes)

        if batch_size % self.num_classes != 0:
            raise ValueError(
                f"batch_size={batch_size} must be divisible by "
                f"num_classes={self.num_classes}")

        self.samples_per_class = batch_size // self.num_classes
        class_lens = sorted(len(v) for v in self.class_to_indices.values())
        anchor_len = class_lens[len(class_lens) // 2]
        self.n_batches = math.ceil(anchor_len / self.samples_per_class)

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        pools, ptrs = {}, {}
        for c in self.classes:
            pools[c] = self.class_to_indices[c].copy()
            if self.shuffle_classes:
                random.shuffle(pools[c])
            ptrs[c] = 0

        for _ in range(self.n_batches):
            batch = []
            for c in self.classes:
                chosen = []
                while len(chosen) < self.samples_per_class:
                    remain = len(pools[c]) - ptrs[c]
                    take = min(self.samples_per_class - len(chosen), remain)
                    if take > 0:
                        chosen.extend(pools[c][ptrs[c]:ptrs[c] + take])
                        ptrs[c] += take
                    if len(chosen) < self.samples_per_class:
                        pools[c] = self.class_to_indices[c].copy()
                        if self.shuffle_classes:
                            random.shuffle(pools[c])
                        ptrs[c] = 0
                batch.extend(chosen)

            if self.shuffle_classes:
                random.shuffle(batch)
            yield batch


def build_dataloaders(train_df, val_df, test_df, tokenizer):
    """Build train / val / test DataLoaders with BalancedBatchSampler."""

    train_dataset = BertDocDataset(
        train_df['text'].tolist(), train_df['label_encoded'].tolist(),
        tokenizer, Config.MAX_CHUNKS, Config.CHUNK_SIZE, Config.CHUNK_STRIDE,
        split='train')
    val_dataset = BertDocDataset(
        val_df['text'].tolist(), val_df['label_encoded'].tolist(),
        tokenizer, Config.MAX_CHUNKS, Config.CHUNK_SIZE, Config.CHUNK_STRIDE,
        split='val')
    test_dataset = BertDocDataset(
        test_df['text'].tolist(), test_df['label_encoded'].tolist(),
        tokenizer, Config.MAX_CHUNKS, Config.CHUNK_SIZE, Config.CHUNK_STRIDE,
        split='test')

    def collate_fn(batch):
        input_ids = torch.stack([item[0] for item in batch])
        attention_mask = torch.stack([item[1] for item in batch])
        labels = torch.stack([item[2] for item in batch])
        return input_ids, attention_mask, labels

    train_labels = train_df['label_encoded'].tolist()
    train_batch_sampler = BalancedBatchSampler(
        labels=train_labels,
        batch_size=Config.BATCH_SIZE,
        drop_last=False,
        shuffle_classes=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=4,
        pin_memory=(Config.DEVICE == 'cuda'),
        prefetch_factor=2,
        collate_fn=collate_fn)

    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE,
        shuffle=False, num_workers=2,
        pin_memory=(Config.DEVICE == 'cuda'), collate_fn=collate_fn)
    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE,
        shuffle=False, num_workers=2,
        pin_memory=(Config.DEVICE == 'cuda'), collate_fn=collate_fn)

    print(f"  Train: {len(train_dataset):,} | Val: {len(val_dataset):,} | Test: {len(test_dataset):,}")
    print(f"  Train batches: {len(train_loader)}")
    return train_loader, val_loader, test_loader


# =============================================================================
# 4. PIPELINE — Input → Preprocess → Backbone → Head → Output
# =============================================================================
class FocalLoss(nn.Module):
    """
    Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    Reference: Lin et al. 2017 "Focal Loss for Dense Object Detection"
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, reduction='none',
            label_smoothing=Config.LABEL_SMOOTHING)
        ce_loss_safe = ce_loss.clamp(min=0.0, max=50.0)
        pt = torch.exp(-ce_loss_safe)
        focal_term = ((1 - pt) ** self.gamma) * ce_loss_safe
        if self.alpha is not None:
            focal_term = focal_term * self.alpha[targets]
        focal_term = focal_term.clamp(max=1e4)
        if self.reduction == 'mean':
            return focal_term.mean()
        elif self.reduction == 'sum':
            return focal_term.sum()
        return focal_term


class ChunkAttentionMHA(nn.Module):
    """
    Multi-Head Self-Attention to aggregate chunk CLS embeddings into document embedding.

    Input  : chunk_cls_embeddings (B, num_chunks, H)
    Preprocess: + sinusoidal positional embedding
    Backbone: Multi-Head Self-Attention (2 separate LayerNorm: pre-attention + pre-FFN)
    Head    : [CLS_first; masked_mean_pool] weighted combination → document embedding
    Output  : (B, H)
    """
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        assert hidden_dim % num_heads == 0

        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_attn = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.cls_weight = nn.Parameter(torch.tensor(0.5))

    def _get_pos_emb(self, num_chunks, device):
        positions = torch.arange(num_chunks, device=device)
        dim_t = torch.arange(0, self.hidden_dim, 2, device=device).float()
        angles = positions.unsqueeze(1) / (10000 ** (dim_t / self.hidden_dim))
        pe = torch.zeros(num_chunks, self.hidden_dim, device=device)
        pe[:, 0::2] = torch.sin(angles)
        pe[:, 1::2] = torch.cos(angles)
        return pe

    def forward(self, chunk_embs, chunk_mask=None):
        B, N, H = chunk_embs.shape

        pos_emb = self._get_pos_emb(N, chunk_embs.device)
        x = chunk_embs + pos_emb.unsqueeze(0)

        key_padding_mask = None
        if chunk_mask is not None:
            key_padding_mask = (chunk_mask == 0)

        x_normed = self.norm_attn(x)
        attn_out, _ = self.mha(x_normed, x_normed, x_normed,
                               key_padding_mask=key_padding_mask)
        x = x + attn_out

        x = x + self.ffn(self.norm_ffn(x))

        cls_emb = x[:, 0]
        if chunk_mask is not None:
            mask_expanded = chunk_mask.unsqueeze(-1).float()
            mean_emb = (x * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-9)
        else:
            mean_emb = x.mean(dim=1)

        w = torch.sigmoid(self.cls_weight)
        doc_emb = w * cls_emb + (1 - w) * mean_emb
        return doc_emb


class FullModel(nn.Module):
    """
    Pipeline: Input (text tokens) → Preprocess (add CLS/SEP per chunk)
           → Backbone (BERT encode all chunks in parallel) → Head (MHA + MLP) → Output (logits)

    Input  : raw token IDs + attention mask for full document
    Preprocess: slice into chunks, add CLS/SEP per chunk
    Backbone: BERT encode all chunks in ONE parallel forward pass
    Head    : CLS extraction → ChunkAttentionMHA → MLP (LayerNorm → Linear 768→256 → GELU → Dropout → Linear 256→7)
    Output  : (batch, num_classes)
    """
    def __init__(self, bert_model, tokenizer, hidden_dim=768, num_classes=7,
                 dropout=0.3, max_chunks=6, chunk_size=384, chunk_stride=256,
                 bert_ft_layers=4, num_heads=8):
        super().__init__()
        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.max_chunks = max_chunks if max_chunks is not None else Config.MAX_CHUNKS
        self.chunk_size = chunk_size
        self.chunk_stride = chunk_stride
        self.hidden_dim = hidden_dim
        self.bert_ft_layers = bert_ft_layers
        self.device = next(bert_model.parameters()).device

        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id
        self.pad_token_id = tokenizer.pad_token_id or 0

        self.chunk_mha = ChunkAttentionMHA(hidden_dim, num_heads=num_heads, dropout=dropout)

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, 256)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(256, num_classes)

    def _encode_chunks(self, input_ids, attention_mask):
        B = input_ids.size(0)
        num_chunks = self.max_chunks
        content_size = self.chunk_size - 2

        all_ids = torch.stack([
            input_ids[:, start:start + content_size]
            for start in range(0, num_chunks * self.chunk_stride, self.chunk_stride)
        ], dim=1)
        all_am = torch.stack([
            attention_mask[:, start:start + content_size]
            for start in range(0, num_chunks * self.chunk_stride, self.chunk_stride)
        ], dim=1)

        chunk_valid = all_am.any(dim=2).float()

        standalone_ids = torch.full(
            (B, num_chunks, self.chunk_size), self.pad_token_id,
            dtype=torch.long, device=self.device)
        standalone_am = torch.zeros(
            B, num_chunks, self.chunk_size, dtype=torch.long, device=self.device)

        standalone_ids[:, :, 0] = self.cls_token_id
        standalone_ids[:, :, 1:1 + content_size] = all_ids
        standalone_am[:, :, 0] = 1
        standalone_am[:, :, 1:1 + content_size] = all_am

        content_lens = all_am.sum(dim=2)
        sep_pos_0idx = (content_lens + 1).long().clamp(min=1, max=self.chunk_size - 1)
        idx_expanded = sep_pos_0idx.unsqueeze(2).expand(B, num_chunks, self.chunk_size)
        sep_mask = torch.zeros(
            B, num_chunks, self.chunk_size, dtype=torch.long, device=self.device)
        sep_mask.scatter_(
            dim=2, index=idx_expanded,
            src=torch.ones_like(sep_pos_0idx).unsqueeze(2).expand_as(sep_mask))
        standalone_ids = standalone_ids.masked_fill(sep_mask.bool(), self.sep_token_id)
        standalone_am = (standalone_am.float() + sep_mask.float()).clamp(0, 1).long()

        flat_ids = standalone_ids.reshape(B * num_chunks, self.chunk_size)
        flat_am = standalone_am.reshape(B * num_chunks, self.chunk_size)

        outputs = self.bert_model(input_ids=flat_ids, attention_mask=flat_am)
        last_hidden = outputs.last_hidden_state

        last_hidden = last_hidden.reshape(B, num_chunks, self.chunk_size, self.hidden_dim)
        cls_embs = last_hidden[:, :, 0, :]

        return cls_embs, chunk_valid

    def forward(self, input_ids, attention_mask):
        cls_embs, chunk_valid = self._encode_chunks(input_ids, attention_mask)
        doc_emb = self.chunk_mha(cls_embs, chunk_valid)

        doc_emb = self.norm(doc_emb)
        doc_emb = self.fc1(doc_emb)
        doc_emb = self.act(doc_emb)
        doc_emb = self.dropout(doc_emb)
        logits = self.fc2(doc_emb)
        return logits


# =============================================================================
# 5. TRAINING — batch → epoch → Focal Loss
# =============================================================================
class Trainer:
    """Trainer: handles one epoch of training with gradient accumulation."""

    def __init__(self, model, train_loader, val_loader, optimizer, criterion,
                 scheduler, device, scaler=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.scaler = scaler
        self.accum_steps = Config.ACCUM_STEPS

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        all_preds, all_labels_list = [], []
        accum_count = 0
        nan_debug_printed = False

        self.optimizer.zero_grad()
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                    desc="Train", unit="batch", ncols=100)

        for step, batch in pbar:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            labels = labels.to(self.device)

            with torch.amp.autocast('cuda', enabled=Config.USE_FP16):
                logits = self.model(input_ids, attention_mask)
                loss_full = self.criterion(logits.float(), labels)
                if torch.isnan(loss_full) or torch.isinf(loss_full):
                    if not nan_debug_printed:
                        nan_debug_printed = True
                        print(f"\n  [NaN DEBUG] step={step}, "
                              f"logits min={logits.min().item():.4f}, "
                              f"max={logits.max().item():.4f}, "
                              f"nan={torch.isnan(logits).sum().item()}")
                    self.optimizer.zero_grad()
                    preds = torch.argmax(logits, dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels_list.extend(labels.cpu().numpy())
                    pbar.set_postfix({'loss': f'{loss_full.item():.4f}'})
                    continue

                loss = loss_full / self.accum_steps
                total_loss += loss_full.item() * labels.size(0)

            loss.backward()
            accum_count += 1

            if accum_count % self.accum_steps == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    self.optimizer.zero_grad()
                else:
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels_list.extend(labels.cpu().numpy())
            pbar.set_postfix({'loss': f'{loss_full.item():.4f}',
                              'acc': f'{(preds == labels).sum().item() / labels.size(0):.4f}'})

        if accum_count % self.accum_steps != 0:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            if not (torch.isnan(grad_norm) or torch.isinf(grad_norm)):
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                prev_accum_count = accum_count - 1
                if prev_accum_count % self.accum_steps != 0:
                    self.scheduler.step()
                self.optimizer.zero_grad()

        f1_macro = f1_score(all_labels_list, all_preds, average='macro')
        f1_per_class = f1_score(all_labels_list, all_preds, average=None)
        return total_loss / total, correct / total, f1_macro, f1_per_class


# =============================================================================
# 6. FINETUNING — freeze backbone, layer-wise LR
# =============================================================================
class Finetuner:
    """
    Finetuning strategies:
      - freeze_backbone: freeze all BERT except last N layers
      - full_finetune: unfreeze everything
      - layerwise_lr: decay LR from head → backbone (not used here; separate
        tiered LR is already set in the optimizer)
    """

    @staticmethod
    def apply_finetune(bert_model, bert_ft_layers=4, gradient_ckpt=True):
        """Freeze all BERT params, then unfreeze last N encoder layers."""
        for p in bert_model.parameters():
            p.requires_grad = False
        for layer in bert_model.encoder.layer[-bert_ft_layers:]:
            for p in layer.parameters():
                p.requires_grad = True

        if gradient_ckpt and hasattr(bert_model, 'gradient_checkpointing_enable'):
            bert_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})

        trainable = sum(p.numel() for p in bert_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in bert_model.parameters())
        return trainable, total

    @staticmethod
    def build_optimizer(model, bert_model):
        """Build tiered AdamW optimizer: separate LR for BERT / MHA / MLP."""
        bert_decay, bert_no_decay = [], []
        for n, p in bert_model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 1 or n.endswith('.bias') or n.endswith('.LayerNorm.weight'):
                bert_no_decay.append(p)
            else:
                bert_decay.append(p)

        mha_decay, mha_no_decay = [], []
        for n, p in model.chunk_mha.named_parameters():
            if p.ndim == 1 or n.endswith('.bias'):
                mha_no_decay.append(p)
            elif p.requires_grad:
                mha_decay.append(p)

        mlp_decay, mlp_no_decay = [], []
        for n, p in model.named_parameters():
            if n.startswith('norm.') or n.startswith('fc1.') or n.startswith('fc2.'):
                if p.ndim == 1 or n.endswith('.bias'):
                    mlp_no_decay.append(p)
                else:
                    mlp_decay.append(p)

        optimizer = torch.optim.AdamW([
            {'params': bert_decay,    'lr': Config.LR_BERT,      'weight_decay': Config.WEIGHT_DECAY},
            {'params': bert_no_decay, 'lr': Config.LR_BERT,      'weight_decay': 0.0},
            {'params': mha_decay,      'lr': Config.LR_ATTENTION, 'weight_decay': Config.WEIGHT_DECAY},
            {'params': mha_no_decay,  'lr': Config.LR_ATTENTION, 'weight_decay': 0.0},
            {'params': mlp_decay,      'lr': Config.LR_MLP,       'weight_decay': Config.WEIGHT_DECAY},
            {'params': mlp_no_decay,  'lr': Config.LR_MLP,       'weight_decay': 0.0},
        ])
        return optimizer

    @staticmethod
    def build_scheduler(optimizer, train_loader):
        """Warmup + cosine decay scheduler for BERT params only."""
        steps_per_epoch = math.ceil(len(train_loader) / Config.ACCUM_STEPS)
        warmup_steps = steps_per_epoch * Config.WARMUP_EPOCHS
        total_steps = steps_per_epoch * Config.EPOCHS
        total_steps_ref = [total_steps]
        warmup_steps_ref = [warmup_steps]
        no_decay_min_lr = 1e-6

        def lr_lambda_bert(step):
            t = total_steps_ref[0]
            w = warmup_steps_ref[0]
            if step < w:
                return float(step) / max(1, w)
            progress = (step - w) / max(1, t - w)
            return max(no_decay_min_lr, 0.5 * (1.0 + np.cos(np.pi * progress)))

        def lr_lambda_head(_step):
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=[lr_lambda_bert, lr_lambda_bert, lr_lambda_head,
                       lr_lambda_head, lr_lambda_head, lr_lambda_head])
        return scheduler, total_steps_ref, warmup_steps_ref


# =============================================================================
# 7. EVALUATE — Accuracy, Precision, Recall, F1 (macro + per-class)
# =============================================================================
class Evaluator:
    """Evaluate model on train/val/test sets. Report per-class + macro metrics."""

    @staticmethod
    def evaluate(model, dataloader, criterion, device, scaler=None):
        model.eval()
        total_loss = 0
        total_samples = 0
        correct = 0
        all_preds, all_labels_list = [], []

        pbar = tqdm(dataloader, desc="Eval ", unit="batch", ncols=100)
        for batch in pbar:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=Config.USE_FP16):
                    logits = model(input_ids, attention_mask)
                    loss = criterion(logits.float(), labels)

            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels_list.extend(labels.cpu().numpy())

        # Per-class metrics
        acc_per_class = []
        cm = confusion_matrix(all_labels_list, all_preds,
                              labels=range(len(set(all_labels_list))))
        for i in range(cm.shape[0]):
            total_c = cm[i].sum()
            acc_per_class.append(cm[i, i] / total_c if total_c > 0 else 0.0)

        precision_per = precision_score(all_labels_list, all_preds, average=None, zero_division=0)
        recall_per = recall_score(all_labels_list, all_preds, average=None, zero_division=0)
        f1_per = f1_score(all_labels_list, all_preds, average=None, zero_division=0)

        # Macro metrics
        acc_macro = accuracy_score(all_labels_list, all_preds)
        prec_macro = precision_score(all_labels_list, all_preds, average='macro', zero_division=0)
        rec_macro = recall_score(all_labels_list, all_preds, average='macro', zero_division=0)
        f1_macro = f1_score(all_labels_list, all_preds, average='macro', zero_division=0)

        metrics = {
            'loss': total_loss / total_samples,
            'accuracy': correct / total_samples,
            'accuracy_macro': acc_macro,
            'precision_macro': prec_macro,
            'recall_macro': rec_macro,
            'f1_macro': f1_macro,
            'precision_per_class': precision_per,
            'recall_per_class': recall_per,
            'f1_per_class': f1_per,
            'accuracy_per_class': acc_per_class,
            'all_preds': all_preds,
        }
        return metrics

    @staticmethod
    def predict(model, dataloader, device, scaler=None):
        model.eval()
        all_preds = []
        for batch in dataloader:
            input_ids, attention_mask, _ = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=Config.USE_FP16):
                    logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
        return np.array(all_preds)


# =============================================================================
# 8. ANALYSIS — params, overfitting, inference time, attention viz
# =============================================================================
class Analyzer:
    """
    Analysis suite:
      - Params: total / trainable model size
      - Overfitting: train vs val loss/accuracy curves
      - Inference time: per-sample latency
      - Attention visualization: sentence-level attention maps
    """

    @staticmethod
    def model_summary(model, label_encoder):
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n  Model params — Total: {total:,} | Trainable: {trainable:,}")
        print(f"  Classes ({len(label_encoder.classes_)}): {list(label_encoder.classes_)}")
        return total, trainable

    @staticmethod
    def plot_overfitting(history, output_path):
        """Plot train vs val loss and accuracy curves."""
        epochs = range(1, len(history['train_loss']) + 1)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss: Train vs Val')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
        axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Accuracy: Train vs Val')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{output_path}bert_overfitting_curve.png', dpi=150)
        plt.close()
        print(f"  Overfitting curve saved: {output_path}bert_overfitting_curve.png")

    @staticmethod
    def measure_inference_time(model, dataloader, device, num_samples=1000):
        """Measure average inference time per sample."""
        model.eval()
        times = []
        batch_count = 0
        with torch.no_grad():
            for batch in dataloader:
                if batch_count >= num_samples // batch[0].size(0):
                    break
                input_ids, attention_mask, _ = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                torch.cuda.synchronize() if device == 'cuda' else None
                t0 = time.time()
                with torch.amp.autocast('cuda', enabled=Config.USE_FP16):
                    _ = model(input_ids, attention_mask)
                torch.cuda.synchronize() if device == 'cuda' else None
                times.append(time.time() - t0)
                batch_count += 1

        avg_time_per_batch = np.mean(times)
        avg_samples = dataloader.batch_size if hasattr(dataloader, 'batch_size') else Config.BATCH_SIZE
        avg_time_per_sample = avg_time_per_batch / avg_samples
        throughput = avg_samples / avg_time_per_batch

        print(f"\n  Inference time — Per batch: {avg_time_per_batch*1000:.2f}ms | "
              f"Per sample: {avg_time_per_sample*1000:.2f}ms | "
              f"Throughput: {throughput:.1f} samples/s")
        return avg_time_per_sample, throughput

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, label_encoder, output_path):
        """Plot normalized confusion matrix."""
        num_classes = len(label_encoder.classes_)
        cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
        plt.figure(figsize=(12, 10))
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=label_encoder.classes_,
                    yticklabels=label_encoder.classes_,
                    linewidths=0.5)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('BERT-FT — Normalized Confusion Matrix (Test)')
        plt.tight_layout()
        plt.savefig(f'{output_path}bert_ft_cm.png', dpi=150)
        plt.close()
        print(f"  Confusion matrix saved: {output_path}bert_ft_cm.png")

    @staticmethod
    def plot_attention_map(model, dataloader, device, num_examples=3, output_path='./'):
        """
        Visualize chunk-level attention weights.
        For each example, show which chunks the model attends to most.
        """
        model.eval()
        saved = 0
        with torch.no_grad():
            for batch in dataloader:
                if saved >= num_examples:
                    break
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)

                cls_embs, chunk_valid = model._encode_chunks(input_ids, attention_mask)
                B, num_chunks, H = cls_embs.shape

                pos_emb = model.chunk_mha._get_pos_emb(num_chunks, device)
                x = cls_embs + pos_emb.unsqueeze(0)
                key_padding_mask = (chunk_valid == 0)

                x_normed = model.chunk_mha.norm_attn(x)
                attn_out, attn_weights = model.chunk_mha.mha(
                    x_normed, x_normed, x_normed,
                    key_padding_mask=key_padding_mask)
                attn_weights = attn_weights.cpu().numpy()

                for i in range(min(3, B)):
                    if saved >= num_examples:
                        break
                    fig, ax = plt.subplots(figsize=(10, 4))
                    aw = attn_weights[i]
                    im = ax.imshow(aw, cmap='viridis', aspect='auto')
                    ax.set_xlabel('Key chunk index')
                    ax.set_ylabel('Query chunk index')
                    ax.set_title(f'Chunk Attention Map — Sample {saved + 1} | '
                                 f'Label={labels[i].item()}')
                    ax.set_xticks(range(num_chunks))
                    ax.set_yticks(range(num_chunks))
                    plt.colorbar(im, ax=ax)
                    plt.tight_layout()
                    plt.savefig(f'{output_path}bert_attn_map_{saved + 1}.png', dpi=150)
                    plt.close()
                    saved += 1
        print(f"  Attention maps saved ({saved} examples): {output_path}bert_attn_map_*.png")


# =============================================================================
# MAIN
# =============================================================================
def main():
    DataResolver.resolve()

    print(f"\n  Data path:   {Config.DATA_PATH}")
    print(f"  Output path: {Config.OUTPUT_PATH}")
    print(f"  Device:      {Config.DEVICE}")

    # EDA
    train_df, val_df, test_df, label_encoder = DataSplits.load()
    EDAAugmentation.analyze(train_df, 'Train')
    EDAAugmentation.analyze(val_df, 'Val')
    EDAAugmentation.analyze(test_df, 'Test')

    # Load BERT
    print(f"\n[1/8] Loading BERT: {Config.BERT_MODEL}...")
    bert_model = AutoModel.from_pretrained(Config.BERT_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(Config.BERT_MODEL)
    bert_model.to(Config.DEVICE)
    hidden_dim = bert_model.config.hidden_size

    trainable, total = Finetuner.apply_finetune(
        bert_model, Config.BERT_FT_LAYERS, Config.GRADIENT_CHECKPOINTING)
    print(f"  Fine-tuning: {trainable:,} / {total:,} params (last {Config.BERT_FT_LAYERS} layers)")

    # Build model
    print(f"\n[2-3/8] Building model & dataloaders...")
    model = FullModel(
        bert_model=bert_model,
        tokenizer=tokenizer,
        hidden_dim=hidden_dim,
        num_classes=len(label_encoder.classes_),
        dropout=Config.DROPOUT,
        max_chunks=Config.MAX_CHUNKS,
        chunk_size=Config.CHUNK_SIZE,
        chunk_stride=Config.CHUNK_STRIDE,
        bert_ft_layers=Config.BERT_FT_LAYERS,
        num_heads=Config.NUM_HEADS,
    ).to(Config.DEVICE)

    train_loader, val_loader, test_loader = build_dataloaders(
        train_df, val_df, test_df, tokenizer)

    optimizer = Finetuner.build_optimizer(model, bert_model)
    scheduler, total_steps_ref, warmup_steps_ref = Finetuner.build_scheduler(
        optimizer, train_loader)
    alpha_tensor = torch.tensor(
        Config.MANUAL_FOCAL_ALPHA, dtype=torch.float32).to(Config.DEVICE)
    criterion = FocalLoss(alpha=alpha_tensor, gamma=Config.FOCAL_GAMMA)
    fp16_scaler = torch.amp.GradScaler('cuda', enabled=Config.USE_FP16)

    # Print arch summary
    print("\n" + "=" * 60)
    print("BERT Document Classifier — Architecture Summary")
    print(f"  BERT model: {Config.BERT_MODEL}")
    print(f"  Chunk: {Config.CHUNK_SIZE} tokens, stride {Config.CHUNK_STRIDE}, "
          f"max_chunks={Config.MAX_CHUNKS}")
    print(f"  Batch size: {Config.BATCH_SIZE} | Effective: {Config.BATCH_SIZE * Config.ACCUM_STEPS}")
    print(f"  LR BERT: {Config.LR_BERT} | MHA: {Config.LR_ATTENTION} | MLP: {Config.LR_MLP}")
    print(f"  Fine-tune: last {Config.BERT_FT_LAYERS} layers")
    print(f"  Loss: FocalLoss(gamma={Config.FOCAL_GAMMA}, alpha={Config.MANUAL_FOCAL_ALPHA})")
    print(f"  Label smoothing: {Config.LABEL_SMOOTHING}")
    print("=" * 60)

    # Resume
    checkpoint_path = f'{Config.CHECKPOINT_SAVE_PATH}bert_doc_ckpt_ft.pt'
    best_model_path = f'{Config.CHECKPOINT_SAVE_PATH}bert_doc_best_ft.pt'
    best_val_f1 = 0.0
    start_epoch = 0

    print(f"\n  [Resume] Loading from: {Config.CHECKPOINT_LOAD_PATH}")
    load_ckpt = f'{Config.CHECKPOINT_LOAD_PATH}bert_doc_ckpt_ft.pt'
    load_best = f'{Config.CHECKPOINT_LOAD_PATH}bert_doc_best_ft.pt'

    if os.path.exists(load_ckpt):
        try:
            ckpt = torch.load(load_ckpt, map_location=Config.DEVICE, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch = ckpt.get('epoch', 0) + 1
            best_val_f1 = ckpt.get('best_val_f1', 0.0)
            remaining_epochs = Config.EPOCHS - start_epoch
            remaining_steps = math.ceil(len(train_loader) / Config.ACCUM_STEPS) * remaining_epochs
            total_steps_ref[0] = scheduler.state_dict()['_step_count'] + remaining_steps
            print(f"  [Resume] SUCCESS — epoch={start_epoch}, best_val_f1={best_val_f1:.4f}")
        except Exception as e:
            print(f"  [Resume] Load failed ({e}) — starting from scratch")
            start_epoch = 0
            best_val_f1 = 0.0
    elif os.path.exists(load_best):
        try:
            model.load_state_dict(torch.load(load_best, map_location=Config.DEVICE, weights_only=True))
            print(f"  [Resume] Loaded best model")
        except Exception:
            print(f"  [Resume] Best model mismatch — starting from scratch")
    else:
        print(f"  [Resume] No checkpoints — training from scratch")

    # Training
    print(f"\n[4-6/8] Training (epochs {start_epoch + 1}–{Config.EPOCHS})...")
    trainer = Trainer(model, train_loader, val_loader, optimizer, criterion,
                      scheduler, Config.DEVICE, fp16_scaler)

    history = {'train_loss': [], 'val_loss': [],
               'train_acc': [], 'val_acc': [],
               'train_f1': [], 'val_f1': []}
    patience = 3
    patience_counter = 0

    for epoch in range(start_epoch, Config.EPOCHS):
        t_ep = time.time()

        train_loss, train_acc, train_f1, train_f1_per = trainer.train_epoch()

        val_metrics = Evaluator.evaluate(
            model, val_loader, criterion, Config.DEVICE, fp16_scaler)
        val_loss = val_metrics['loss']
        val_acc = val_metrics['accuracy']
        val_f1 = val_metrics['f1_macro']
        val_f1_per = val_metrics['f1_per_class']

        t_dur = time.time() - t_ep
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1:2d}/{Config.EPOCHS} ({t_dur:.1f}s) | "
              f"LR BERT:{current_lr:.2e} | "
              f"Tr Loss:{train_loss:.4f} Acc:{train_acc:.4f} F1:{train_f1:.4f} | "
              f"Val Loss:{val_loss:.4f} Acc:{val_acc:.4f} F1:{val_f1:.4f}")
        print(f"  Val F1 per class: {[f'{f:.3f}' for f in val_f1_per]}")

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_f1': best_val_f1,
        }, checkpoint_path)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
            print(f"  *** New best val F1: {best_val_f1:.4f} ***")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load(best_model_path,
                                      map_location=Config.DEVICE, weights_only=True))

    # Evaluate on test set
    print(f"\n[7/8] Final Evaluation on Test Set")
    print("=" * 60)
    test_metrics = Evaluator.evaluate(
        model, test_loader, criterion, Config.DEVICE, fp16_scaler)
    test_preds = Evaluator.predict(model, test_loader, Config.DEVICE, fp16_scaler)
    test_labels_np = test_df['label_encoded'].values

    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Precision (macro): {test_metrics['precision_macro']:.4f}")
    print(f"  Recall (macro):    {test_metrics['recall_macro']:.4f}")
    print(f"  F1 (macro):        {test_metrics['f1_macro']:.4f}")
    print(f"\n  Per-class metrics:")
    print(f"  {'Class':<20} {'Prec':>8} {'Rec':>8} {'F1':>8} {'Acc':>8}")
    print(f"  {'-'*52}")
    for i, name in enumerate(label_encoder.classes_):
        prec = test_metrics['precision_per_class'][i]
        rec = test_metrics['recall_per_class'][i]
        f1 = test_metrics['f1_per_class'][i]
        acc = test_metrics['accuracy_per_class'][i]
        print(f"  {name:<20} {prec:>8.4f} {rec:>8.4f} {f1:>8.4f} {acc:>8.4f}")

    print("\nClassification Report:")
    print(classification_report(test_labels_np, test_preds,
                                target_names=label_encoder.classes_))

    # Analysis
    print(f"\n[8/8] Analysis")
    Analyzer.model_summary(model, label_encoder)
    Analyzer.plot_overfitting(history, Config.OUTPUT_PATH)
    Analyzer.measure_inference_time(model, test_loader, Config.DEVICE)
    Analyzer.plot_confusion_matrix(test_labels_np, test_preds,
                                    label_encoder, Config.OUTPUT_PATH)
    Analyzer.plot_attention_map(model, test_loader, Config.DEVICE,
                                num_examples=3, output_path=Config.OUTPUT_PATH)

    # Save predictions
    test_df_out = test_df.copy()
    test_df_out['pred'] = label_encoder.inverse_transform(test_preds)
    test_df_out.to_csv(f'{Config.OUTPUT_PATH}bert_ft_predictions.csv', index=False)
    print(f"\n  Predictions saved: {Config.OUTPUT_PATH}bert_ft_predictions.csv")
    print(f"\nDone! Best model: {best_model_path}")


if __name__ == '__main__':
    main()

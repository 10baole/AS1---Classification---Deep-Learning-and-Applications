#!/usr/bin/env python3
"""
BERT Document Classifier — Legal Text Classification
Pipeline: Document → Tokenize → Split chunks → BERT encode → CLS extraction
        → ChunkAttentionMHA → MLP → Output
Classes:
  1. Config        — hyper-parameters, paths
  2. BertDocDataset — pre-tokenize + chunking
  3. DataLoaders   — train / val / test splits + BalancedBatchSampler
  4. ChunkAttentionMHA + FullModel — Backbone + Head
  5. FocalLoss + Trainer
  6. Finetuner     — freeze backbone, tiered LR
  7. Evaluator      — Accuracy, Precision, Recall, F1
  8. Analyzer       — params, overfitting, inference time, attention viz
"""

import math
import os
import pickle
import time
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
# 1. CONFIG
# =============================================================================
class Config:
    DATA_PATH = './data/processed/'
    OUTPUT_PATH = './output-train-bert/'

    BERT_MODEL = 'google-bert/bert-base-cased'

    MAX_CHUNKS = 6
    CHUNK_SIZE = 384
    CHUNK_STRIDE = 256

    BATCH_SIZE = 24
    ACCUM_STEPS = 2
    BERT_FT_LAYERS = 4
    NUM_HEADS = 8
    EPOCHS = 200
    WEIGHT_DECAY = 0.01
    DROPOUT = 0.3

    WARMUP_EPOCHS = 1
    LR_BERT = 2e-5
    LR_ATTENTION = 1e-4
    LR_MLP = 5e-5

    # [Civil=0, Corporate=1, CourtOfClaims=2, Criminal=3, Other=4, Probate=5, Property=6]
    FOCAL_ALPHA = [0.4, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0]
    FOCAL_GAMMA = 2.0

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# =============================================================================
# 2. DATA RESOLVER
# =============================================================================
class DataResolver:
    @staticmethod
    def resolve():
        # Auto-detect data path
        for p in ['./data/processed/', './data/masked-15k/']:
            if os.path.exists(os.path.join(p, 'train.csv')):
                Config.DATA_PATH = p
                break
        os.makedirs(Config.OUTPUT_PATH, exist_ok=True)
        if not Config.OUTPUT_PATH.endswith('/'):
            Config.OUTPUT_PATH += '/'


# =============================================================================
# 3. DATASET
# =============================================================================
class BertDocDataset(Dataset):
    """Pre-tokenize full document, then slice into overlapping chunks at access time."""

    _cache_dir = '/tmp/.bert_cache/'

    def __init__(self, texts, labels, tokenizer, max_chunks=6,
                 chunk_size=384, chunk_stride=256, split='train'):
        self.max_chunks = max_chunks
        self.chunk_size = chunk_size
        self.chunk_stride = chunk_stride
        self.labels = [torch.tensor(l, dtype=torch.long) for l in labels]

        total_len = max_chunks * chunk_stride + (chunk_size - chunk_stride)
        cache_key = f'{split}_{len(texts)}_{max_chunks}_{chunk_size}_{chunk_stride}_{total_len}'
        os.makedirs(self._cache_dir, exist_ok=True)
        self._cache_path = os.path.join(self._cache_dir, f'{cache_key}.pt')

        if os.path.exists(self._cache_path):
            print(f"    Loading cached tokens from {self._cache_path}...")
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
# 4. DATA SPLITS
# =============================================================================
class DataSplits:
    @staticmethod
    def load():
        print(f"Loading data from {Config.DATA_PATH}...")

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
            raise FileNotFoundError(f"No data in {Config.DATA_PATH}")

        print(f"  Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")

        # Keep text as string. No dataset-level dropping by text length.
        for _, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            df['text'] = df['text'].astype(str)

        # Label encoder
        le_path = next((p for p in [
            f'{Config.DATA_PATH}label_encoder.pkl',
            './data/masked-15k/label_encoder.pkl',
        ] if os.path.exists(p)), None)

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


def build_dataloaders(train_df, val_df, test_df, tokenizer):
    train_ds = BertDocDataset(
        train_df['text'].tolist(), train_df['label_encoded'].tolist(),
        tokenizer, Config.MAX_CHUNKS, Config.CHUNK_SIZE, Config.CHUNK_STRIDE, 'train')
    val_ds = BertDocDataset(
        val_df['text'].tolist(), val_df['label_encoded'].tolist(),
        tokenizer, Config.MAX_CHUNKS, Config.CHUNK_SIZE, Config.CHUNK_STRIDE, 'val')
    test_ds = BertDocDataset(
        test_df['text'].tolist(), test_df['label_encoded'].tolist(),
        tokenizer, Config.MAX_CHUNKS, Config.CHUNK_SIZE, Config.CHUNK_STRIDE, 'test')

    def collate_fn(batch):
        input_ids = torch.stack([item[0] for item in batch])
        attention_mask = torch.stack([item[1] for item in batch])
        labels = torch.stack([item[2] for item in batch])
        return input_ids, attention_mask, labels

    # Use a standard shuffled DataLoader for training (balanced sampling disabled).
    train_loader = DataLoader(
        train_ds, batch_size=Config.BATCH_SIZE,
        shuffle=True, num_workers=4,
        pin_memory=(Config.DEVICE == 'cuda'),
        prefetch_factor=2, collate_fn=collate_fn)

    val_loader = DataLoader(
        val_ds, batch_size=Config.BATCH_SIZE,
        shuffle=False, num_workers=2,
        pin_memory=(Config.DEVICE == 'cuda'), collate_fn=collate_fn)
    test_loader = DataLoader(
        test_ds, batch_size=Config.BATCH_SIZE,
        shuffle=False, num_workers=2,
        pin_memory=(Config.DEVICE == 'cuda'), collate_fn=collate_fn)

    print(f"  Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")
    print(f"  Train batches: {len(train_loader)}")
    return train_loader, val_loader, test_loader


# =============================================================================
# 5. MODEL
# =============================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
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
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        assert hidden_dim % num_heads == 0

        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True)
        self.norm_attn = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout))
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
        return w * cls_emb + (1 - w) * mean_emb


class FullModel(nn.Module):
    def __init__(self, bert_model, tokenizer, hidden_dim=768, num_classes=7,
                 dropout=0.3, max_chunks=6, chunk_size=384, chunk_stride=256,
                 num_heads=8):
        super().__init__()
        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.max_chunks = max_chunks
        self.chunk_size = chunk_size
        self.chunk_stride = chunk_stride
        self.hidden_dim = hidden_dim
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
        return last_hidden[:, :, 0, :], chunk_valid

    def forward(self, input_ids, attention_mask):
        cls_embs, chunk_valid = self._encode_chunks(input_ids, attention_mask)
        doc_emb = self.chunk_mha(cls_embs, chunk_valid)
        doc_emb = self.norm(doc_emb)
        doc_emb = self.act(self.fc1(doc_emb))
        doc_emb = self.dropout(doc_emb)
        return self.fc2(doc_emb)


# =============================================================================
# 6. FINETUNING
# =============================================================================
class Finetuner:
    @staticmethod
    def _sanitize_state_dict(maybe_sd):
        sd = maybe_sd
        if isinstance(sd, dict):
            if 'state_dict' in sd and isinstance(sd['state_dict'], dict):
                sd = sd['state_dict']
            elif 'model_state_dict' in sd and isinstance(sd['model_state_dict'], dict):
                sd = sd['model_state_dict']

        if isinstance(sd, dict):
            new_sd = {}
            for k, v in sd.items():
                new_k = k
                if new_k.startswith('model.'):
                    new_k = new_k[len('model.') :]
                if new_k.startswith('state_dict.'):
                    new_k = new_k[len('state_dict.') :]
                new_sd[new_k] = v
            return new_sd
        return sd

    @staticmethod
    def _try_load_model_weights(model, loaded_obj):
        try:
            sd = Finetuner._sanitize_state_dict(loaded_obj)
            if isinstance(sd, dict):
                missing_unexp = model.load_state_dict(sd, strict=False)
                return True, f"loaded permissively (missing_keys={getattr(missing_unexp,'missing_keys', None)}, unexpected_keys={getattr(missing_unexp,'unexpected_keys', None)})"
        except Exception as e:
            msg = f"permissive load failed: {e}"

        try:
            if isinstance(loaded_obj, dict):
                model_keys = set(model.state_dict().keys())
                for key, val in loaded_obj.items():
                    if isinstance(val, dict):
                        cand = Finetuner._sanitize_state_dict(val)
                        if isinstance(cand, dict):
                            overlap = len(model_keys.intersection(set(cand.keys())))
                            if overlap > 0:
                                try:
                                    model.load_state_dict(cand, strict=False)
                                    return True, f"loaded from nested key '{key}' (overlap={overlap})"
                                except Exception:
                                    continue
        except Exception:
            pass

        return False, msg if 'msg' in locals() else 'unknown format'

    @staticmethod
    def apply_finetune(bert_model, bert_ft_layers=4):
        for p in bert_model.parameters():
            p.requires_grad = False
        for layer in bert_model.encoder.layer[-bert_ft_layers:]:
            for p in layer.parameters():
                p.requires_grad = True

        trainable = sum(p.numel() for p in bert_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in bert_model.parameters())
        return trainable, total

    @staticmethod
    def build_optimizer(model, bert_model):
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

        return torch.optim.AdamW([
            {'params': bert_decay,   'lr': Config.LR_BERT,      'weight_decay': Config.WEIGHT_DECAY},
            {'params': bert_no_decay, 'lr': Config.LR_BERT,     'weight_decay': 0.0},
            {'params': mha_decay,    'lr': Config.LR_ATTENTION, 'weight_decay': Config.WEIGHT_DECAY},
            {'params': mha_no_decay, 'lr': Config.LR_ATTENTION, 'weight_decay': 0.0},
            {'params': mlp_decay,    'lr': Config.LR_MLP,      'weight_decay': Config.WEIGHT_DECAY},
            {'params': mlp_no_decay, 'lr': Config.LR_MLP,       'weight_decay': 0.0},
        ])

    @staticmethod
    def build_scheduler(optimizer, train_loader):
        steps_per_epoch = math.ceil(len(train_loader) / Config.ACCUM_STEPS)
        warmup_steps = steps_per_epoch * Config.WARMUP_EPOCHS
        total_steps = steps_per_epoch * Config.EPOCHS
        total_ref = [total_steps]
        warmup_ref = [warmup_steps]

        def lr_bert(step):
            if step < warmup_ref[0]:
                return float(step) / max(1, warmup_ref[0])
            p = (step - warmup_ref[0]) / max(1, total_ref[0] - warmup_ref[0])
            return max(1e-6, 0.5 * (1.0 + np.cos(np.pi * p)))

        def lr_head(_step):
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=[lr_bert, lr_bert, lr_head, lr_head, lr_head, lr_head])
        return scheduler, total_ref, warmup_ref


# =============================================================================
# 7. TRAINER
# =============================================================================
class Trainer:
    def __init__(self, model, train_loader, optimizer, criterion, scheduler, device):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.accum_steps = Config.ACCUM_STEPS

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        all_preds, all_labels = [], []
        accum_count = 0

        self.optimizer.zero_grad()
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                    desc="Train", unit="batch", ncols=100)

        for step, (input_ids, attention_mask, labels) in pbar:
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(input_ids, attention_mask)
            loss_full = self.criterion(logits, labels)
            total_loss += loss_full.item() * labels.size(0)
            loss = loss_full / self.accum_steps

            loss.backward()
            accum_count += 1

            if accum_count % self.accum_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                if not (torch.isnan(grad_norm) or torch.isinf(grad_norm)):
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            pbar.set_postfix({'loss': f'{loss_full.item():.4f}',
                              'acc': f'{(preds == labels).sum().item() / labels.size(0):.4f}'})

        if accum_count % self.accum_steps != 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            if not (torch.isnan(grad_norm) or torch.isinf(grad_norm)):
                self.optimizer.step()
                self.optimizer.zero_grad()

        f1_macro = f1_score(all_labels, all_preds, average='macro')
        f1_per = f1_score(all_labels, all_preds, average=None)
        return total_loss / total, correct / total, f1_macro, f1_per


# =============================================================================
# 8. EVALUATOR
# =============================================================================
class Evaluator:
    @staticmethod
    def evaluate(model, dataloader, criterion, device):
        model.eval()
        total_loss = 0
        total_samples = 0
        correct = 0
        all_preds, all_labels = [], []

        for input_ids, attention_mask, labels in tqdm(dataloader, desc="Eval ", unit="batch", ncols=100):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        cm = confusion_matrix(all_labels, all_preds, labels=range(len(set(all_labels))))
        total_samples = cm.sum()
        ovr_acc_per_class = []
        for i in range(cm.shape[0]):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = total_samples - tp - fn - fp
            ovr_acc_per_class.append((tp + tn) / total_samples if total_samples > 0 else 0.0)

        precision_per = precision_score(all_labels, all_preds, average=None, zero_division=0)
        recall_per = recall_score(all_labels, all_preds, average=None, zero_division=0)
        f1_per = f1_score(all_labels, all_preds, average=None, zero_division=0)

        return {
            'loss': total_loss / total_samples,
            'accuracy': correct / total_samples,
            'accuracy_macro': accuracy_score(all_labels, all_preds),
            'precision_macro': precision_score(all_labels, all_preds, average='macro', zero_division=0),
            'recall_macro': recall_score(all_labels, all_preds, average='macro', zero_division=0),
            'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
            'precision_per_class': precision_per,
            'recall_per_class': recall_per,
            'f1_per_class': f1_per,
            'ovr_accuracy_per_class': ovr_acc_per_class,
            'all_preds': all_preds,
            'all_labels': all_labels,
        }

    @staticmethod
    def predict(model, dataloader, device):
        model.eval()
        all_preds = []
        for input_ids, attention_mask, _ in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            with torch.no_grad():
                logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
        return np.array(all_preds)


# =============================================================================
# 9. ANALYZER
# =============================================================================
class Analyzer:
    @staticmethod
    def model_summary(model, label_encoder):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n  Model params — Total: {total:,} | Trainable: {trainable:,}")
        print(f"  Classes ({len(label_encoder.classes_)}): {list(label_encoder.classes_)}")

    @staticmethod
    def plot_overfitting(history, output_path):
        epochs = range(1, len(history['train_loss']) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss: Train vs Val'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
        axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
        axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
        axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Accuracy: Train vs Val'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_path}bert_overfitting_curve.png', dpi=150)
        plt.close()
        print(f"  Overfitting curve: {output_path}bert_overfitting_curve.png")

    @staticmethod
    def measure_inference_time(model, dataloader, device, num_samples=1000):
        model.eval()
        times = []
        batch_count = 0
        with torch.no_grad():
            for input_ids, attention_mask, _ in dataloader:
                if batch_count >= num_samples // input_ids.size(0):
                    break
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                torch.cuda.synchronize() if device == 'cuda' else None
                t0 = time.time()
                _ = model(input_ids, attention_mask)
                torch.cuda.synchronize() if device == 'cuda' else None
                times.append(time.time() - t0)
                batch_count += 1
        avg_batch = np.mean(times)
        avg_sample = avg_batch / dataloader.batch_size
        print(f"\n  Inference — Per batch: {avg_batch*1000:.2f}ms | "
              f"Per sample: {avg_sample*1000:.2f}ms")

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, label_encoder, output_path):
        num_classes = len(label_encoder.classes_)
        cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
        plt.figure(figsize=(12, 10))
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=label_encoder.classes_,
                    yticklabels=label_encoder.classes_, linewidths=0.5)
        plt.xlabel('Predicted'); plt.ylabel('True')
        plt.title('BERT-FT — Normalized Confusion Matrix (Test)')
        plt.tight_layout()
        plt.savefig(f'{output_path}bert_ft_cm.png', dpi=150)
        plt.close()
        print(f"  Confusion matrix: {output_path}bert_ft_cm.png")

    @staticmethod
    def plot_attention_map(model, dataloader, device, num_examples=3, output_path='./'):
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
                x_normed = model.chunk_mha.norm_attn(x)
                attn_out, attn_weights = model.chunk_mha.mha(
                    x_normed, x_normed, x_normed, key_padding_mask=(chunk_valid == 0))
                attn_weights = attn_weights.cpu().numpy()

                for i in range(min(3, B)):
                    if saved >= num_examples:
                        break
                    fig, ax = plt.subplots(figsize=(10, 4))
                    im = ax.imshow(attn_weights[i], cmap='viridis', aspect='auto')
                    ax.set_xlabel('Key chunk index')
                    ax.set_ylabel('Query chunk index')
                    ax.set_title(f'Chunk Attention — Sample {saved+1} | Label={labels[i].item()}')
                    ax.set_xticks(range(num_chunks))
                    ax.set_yticks(range(num_chunks))
                    plt.colorbar(im, ax=ax)
                    plt.tight_layout()
                    plt.savefig(f'{output_path}bert_attn_map_{saved+1}.png', dpi=150)
                    plt.close()
                    saved += 1
        print(f"  Attention maps ({saved}): {output_path}bert_attn_map_*.png")


# =============================================================================
# MAIN
# =============================================================================
def main():
    DataResolver.resolve()

    print(f"  Data path:   {Config.DATA_PATH}")
    print(f"  Output path: {Config.OUTPUT_PATH}")
    print(f"  Device:      {Config.DEVICE}")

    # Load data
    train_df, val_df, test_df, label_encoder = DataSplits.load()

    # Load BERT
    print(f"\n[1] Loading BERT: {Config.BERT_MODEL}...")
    bert_model = AutoModel.from_pretrained(Config.BERT_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(Config.BERT_MODEL)
    bert_model.to(Config.DEVICE)
    hidden_dim = bert_model.config.hidden_size

    trainable, total = Finetuner.apply_finetune(bert_model, Config.BERT_FT_LAYERS)
    print(f"  Fine-tuning: {trainable:,} / {total:,} params (last {Config.BERT_FT_LAYERS} layers)")

    # Build model
    print(f"\n[2] Building model...")
    model = FullModel(
        bert_model=bert_model, tokenizer=tokenizer,
        hidden_dim=hidden_dim,
        num_classes=len(label_encoder.classes_),
        dropout=Config.DROPOUT,
        max_chunks=Config.MAX_CHUNKS,
        chunk_size=Config.CHUNK_SIZE,
        chunk_stride=Config.CHUNK_STRIDE,
        num_heads=Config.NUM_HEADS,
    ).to(Config.DEVICE)

    train_loader, val_loader, test_loader = build_dataloaders(
        train_df, val_df, test_df, tokenizer)

    optimizer = Finetuner.build_optimizer(model, bert_model)
    scheduler, total_ref, warmup_ref = Finetuner.build_scheduler(optimizer, train_loader)
    alpha_tensor = torch.tensor(Config.FOCAL_ALPHA, dtype=torch.float32).to(Config.DEVICE)
    criterion = FocalLoss(alpha=alpha_tensor, gamma=Config.FOCAL_GAMMA)

    # Architecture summary
    print("\n" + "=" * 60)
    print("BERT Document Classifier")
    print(f"  Model: {Config.BERT_MODEL}")
    print(f"  Chunk: {Config.CHUNK_SIZE} tokens, stride {Config.CHUNK_STRIDE}, max_chunks={Config.MAX_CHUNKS}")
    print(f"  Batch: {Config.BATCH_SIZE} (eff={Config.BATCH_SIZE * Config.ACCUM_STEPS})")
    print(f"  LR: BERT={Config.LR_BERT} | MHA={Config.LR_ATTENTION} | MLP={Config.LR_MLP}")
    print(f"  Loss: FocalLoss(gamma={Config.FOCAL_GAMMA}, alpha={Config.FOCAL_ALPHA})")
    print("=" * 60)

    # Resume
    # Support multiple possible output dirs (local and studio runtime)
    ckpt_path = None
    best_path = None
    candidate_output_dirs = list(dict.fromkeys([
        Config.OUTPUT_PATH,
        './output-train-bert/',
        './output_train_bert/',
        '/teamspace/studios/this_studio/output_train_bert/',
        '/teamspace/studios/this_studio/output-train-bert/',
    ]))

    for d in candidate_output_dirs:
        p_ckpt = os.path.join(d, 'bert_doc_ckpt.pt')
        p_best = os.path.join(d, 'bert_doc_best.pt')
        if ckpt_path is None and os.path.exists(p_ckpt):
            ckpt_path = p_ckpt
        if best_path is None and os.path.exists(p_best):
            best_path = p_best

    if ckpt_path is None:
        ckpt_path = f'{Config.OUTPUT_PATH}bert_doc_ckpt.pt'
    if best_path is None:
        best_path = f'{Config.OUTPUT_PATH}bert_doc_best.pt'

    print(f"  Looking for checkpoint at: {ckpt_path}")
    print(f"  Looking for best model at: {best_path}")

    if os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location=Config.DEVICE)

            # Try direct model_state_dict key first
            loaded_msg = ''
            loaded_obj = ckpt
            if 'model_state_dict' in ckpt:
                loaded_obj = ckpt['model_state_dict']
            success, loaded_msg = Finetuner._try_load_model_weights(model, loaded_obj)
            if success:
                print(f"  [Resume] model weights: {loaded_msg}")
            else:
                # try loading entire checkpoint object
                success, loaded_msg = Finetuner._try_load_model_weights(model, ckpt)
                if success:
                    print(f"  [Resume] model weights from ckpt: {loaded_msg}")
                else:
                    print(f"  [Resume] Could not load model weights: {loaded_msg}")

            # Optimizer
            if 'optimizer_state_dict' in ckpt:
                try:
                    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                    print("  [Resume] optimizer state loaded")
                except Exception as e_opt:
                    print(f"  [Resume] Failed to load optimizer state: {e_opt} — continuing without optimizer state")

            # Scheduler
            if 'scheduler_state_dict' in ckpt:
                try:
                    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                    print("  [Resume] scheduler state loaded")
                except Exception as e_sch:
                    print(f"  [Resume] Failed to load scheduler state: {e_sch} — continuing without scheduler state")

            start_epoch = int(ckpt.get('epoch', 0)) + 1
            best_val_f1 = float(ckpt.get('best_val_f1', 0.0))

            remaining_epochs = max(0, Config.EPOCHS - start_epoch)
            step_count = 0
            try:
                sd = scheduler.state_dict()
                step_count = sd.get('_step_count', sd.get('last_epoch', 0)) if isinstance(sd, dict) else 0
            except Exception:
                step_count = 0
            total_ref[0] = int(step_count) + math.ceil(len(train_loader) / Config.ACCUM_STEPS) * remaining_epochs
            print(f"  [Resume] epoch={start_epoch}, best_val_f1={best_val_f1:.4f}")
        except Exception as e:
            print(f"  [Resume] Failed ({e}) — training from scratch")
            start_epoch = 0
            best_val_f1 = 0.0

    # If the saved checkpoint indicates training already reached or exceeded Config.EPOCHS,
    # decide behavior. By default we skip further training. To force continuation set
    # environment variable FORCE_CONTINUE=1 (optionally EXTRA_EPOCHS=N).
    if start_epoch >= Config.EPOCHS:
        if os.environ.get('FORCE_CONTINUE', '0') == '1':
            extra = int(os.environ.get('EXTRA_EPOCHS', '10'))
            old = Config.EPOCHS
            Config.EPOCHS = start_epoch + extra
            print(f"  FORCE_CONTINUE set: extending Config.EPOCHS from {old} to {Config.EPOCHS} to continue training")
        else:
            print(f"  Checkpoint indicates training already reached epoch {start_epoch} >= Config.EPOCHS ({Config.EPOCHS}).")
            print("  Skipping training loop. To force more training set env var FORCE_CONTINUE=1 or increase Config.EPOCHS.")
            start_epoch = Config.EPOCHS # prevent accidental resume from mid-training

    # Training loop
    history = {'train_loss': [], 'val_loss': [],
               'train_acc': [], 'val_acc': [],
               'train_f1': [], 'val_f1': []}

    for epoch in range(start_epoch, Config.EPOCHS):
        print(f"\n[3] Epoch {epoch + 1} / {Config.EPOCHS}")
        print("=" * 60)

        trainer = Trainer(model, train_loader, optimizer, criterion, scheduler, Config.DEVICE)
        train_loss, train_acc, train_f1, _ = trainer.train_epoch()

        # Validation
        val_metrics = Evaluator.evaluate(model, val_loader, criterion, Config.DEVICE)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_metrics['accuracy'])
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_metrics['f1_macro'])

        print(f"  Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | Train F1: {train_f1:.4f}")
        print(f"  Val. loss: {val_metrics['loss']:.4f} | Val. acc: {val_metrics['accuracy']:.4f} | Val. F1: {val_metrics['f1_macro']:.4f}")

        # Checkpoint
        is_best = val_metrics['f1_macro'] > best_val_f1
        best_val_f1 = max(best_val_f1, val_metrics['f1_macro'])
        ckpt_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_f1': best_val_f1,
        }
        torch.save(ckpt_dict, ckpt_path)
        if is_best:
            torch.save(ckpt_dict, best_path)
            print(f"  [Best Model] saved to {best_path}")

    # After training loop, load best model permissively
    try:
        if os.path.exists(best_path):
            bd = torch.load(best_path, map_location=Config.DEVICE)
            success, msg = Finetuner._try_load_model_weights(model, bd)
            if success:
                print(f"  [Final Load] best model loaded: {msg}")
            else:
                print(f"  [Final Load] best model not loaded permissively: {msg}")
        else:
            print(f"  Best model file not found at {best_path} — skipping final load")
    except Exception as e_best:
        print(f"  Failed to load best model: {e_best}")

    # Evaluate
    print(f"\n[4] Final Evaluation on Test Set")
    print("=" * 60)
    test_metrics = Evaluator.evaluate(model, test_loader, criterion, Config.DEVICE)
    test_preds = Evaluator.predict(model, test_loader, Config.DEVICE)
    test_labels_np = test_df['label_encoded'].values

    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision_macro']:.4f} | Recall: {test_metrics['recall_macro']:.4f} | F1: {test_metrics['f1_macro']:.4f}")
    print(f"\n  {'Class':<20} {'Prec':>8} {'Rec':>8} {'F1':>8} {'OvR-Acc':>8}")
    print(f"  {'-'*52}")
    for i, name in enumerate(label_encoder.classes_):
        print(f"  {name:<20} {test_metrics['precision_per_class'][i]:>8.4f} "
              f"{test_metrics['recall_per_class'][i]:>8.4f} {test_metrics['f1_per_class'][i]:>8.4f} "
              f"{test_metrics['ovr_accuracy_per_class'][i]:>8.4f}")

    print("\nClassification Report:")
    print(classification_report(test_labels_np, test_preds, target_names=label_encoder.classes_))

    # Analysis
    print(f"\n[5] Analysis")
    Analyzer.model_summary(model, label_encoder)
    Analyzer.plot_overfitting(history, Config.OUTPUT_PATH)
    Analyzer.measure_inference_time(model, test_loader, Config.DEVICE)
    Analyzer.plot_confusion_matrix(test_labels_np, test_preds, label_encoder, Config.OUTPUT_PATH)
    Analyzer.plot_attention_map(model, test_loader, Config.DEVICE, num_examples=3,
                                output_path=Config.OUTPUT_PATH)

    test_df_out = test_df.copy()
    test_df_out['pred'] = label_encoder.inverse_transform(test_preds)
    test_df_out.to_csv(f'{Config.OUTPUT_PATH}bert_ft_predictions.csv', index=False)
    print(f"\n  Predictions: {Config.OUTPUT_PATH}bert_ft_predictions.csv")
    print(f"  Best model: {best_path}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
BERT Document Classifier for Legal Text Classification — FIXED

Architecture:
  Document → Tokenize (≤3712 tokens) → Split into overlapping 384-token chunks (stride 256)
          → BERT encode all chunks IN PARALLEL (batched) → (batch*14, 384, 768)
          → Extract CLS token [0, :] from each chunk → (batch, 14, 768)
          → Multi-Head Self-Attention (8 heads) over chunks with sinusoidal positional encoding
          → Combine: sigmoid(w) * [CLS-first] + (1 - sigmoid(w)) * mean-pool
          → MLP: LayerNorm → Linear(768→256) → GELU → Dropout → Linear(256→7)
          → Last 4 BERT layers fine-tuned end-to-end
"""

import math
import os
import pickle
import random
import shutil
import time
import numpy as np
import pandas as pd
import torch
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch.nn as nn
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


# ============================================================
#  CONFIG
# ============================================================
class Config:
    DATA_PATH = './data/processed/'
    OUTPUT_PATH = './output-train-bert/'

    BERT_MODEL = 'google-bert/bert-base-cased'
    BERT_MAX_SEQ = 512

    # ---- CHUNK STRATEGY: parallel batch encoding ----
    # Document length: ~9000 chars avg, max ~15000 chars
    #   → ~2250 tokens avg, ~3750 tokens max (4.5 chars/token for legal text)
    # BERT max = 512 tokens/pass
    # Strategy: chunk=384, stride=256 → 33% overlap, 67% unique tokens/chunk
    #
    # With MAX_CHUNKS=14:
    #   BertDocDataset total_len = 14*256 + 128 = 3584 + 128 = 3712 tokens coverage
    #   Model encodes all 14 chunks in parallel (batch dimension expansion)
    #   BERT passes per doc: ceil(3750/256) = 15, max doc truncated at 3712 tokens
    #   ✓ Covers ~98% of max-length documents
    #
    # Trade-off vs 28 chunks:
    #   28 chunks: covers 7296 tokens but 28 sequential BERT passes = SLOW
    #   14 chunks: covers 3712 tokens but parallelized = 14x FASTER
    #   Legal docs rarely exceed 3750 tokens (4.5k chars avg), 14 chunks sufficient
    #
    # VRAM target: ~12 GB out of 15 GB (T4).
    # MAX_CHUNKS=6: 6 × 256 stride + 128 = 1648 tokens coverage (~44% of max 3750-token doc).
    # BATCH_SIZE=42: 42 × 6 = 252 chunks/BERT call → ~12 GB forward pass (gradient checkpointing).
    # Effective batch = 42 × 1 = 42 (accum_steps=1).
    MAX_CHUNKS = 6
    CHUNK_SIZE = 384            # each chunk = 384 tokens (BERT max = 512, safe)
    CHUNK_STRIDE = 256          # stride=256 → 67% unique tokens/chunk, 33% overlap

    BATCH_SIZE = 42             # divisible by num_classes (7); 42×6=252 chunks per BERT call
    ACCUM_STEPS = 2             # effective batch = 42 × 2 = 84
    GRADIENT_CHECKPOINTING = True
    USE_FP16 = False
    BERT_FT_LAYERS = 4          # fine-tune last N layers
    NUM_HEADS = 8               # number of attention heads for chunk-level MHA
    EPOCHS = 10
    WEIGHT_DECAY = 0.01

    DROPOUT = 0.3

    # Warmup settings
    WARMUP_EPOCHS = 1
    LR_BERT = 2e-5
    LR_ATTENTION = 1e-4
    LR_MLP = 5e-5

    # Class distribution (estimated):
    # Civil=0(46%), Corporate=1(1%), CourtOfClaims=2(4%), Criminal=3(19%),
    # Other=4(2%), Probate=5(8%), Property=6(19%)
    #
    # Focal alpha = balanced weight with soft capping:
    #   raw_alpha[c] = total_samples / (n_classes * count[c])
    #   alpha[c] = 1 + (raw_alpha[c] - 1) * 0.25   # soft cap: blend toward balanced
    #
    # Result: minority classes get moderate boost (1.5x-4x), not extreme (46x).
    # A 46x weight on Corporate would make the model ONLY learn Corporate and ignore others.
    #
    # Alpha values tuned so that effective gradient contribution per batch is
    # roughly balanced across classes (target ~20 each in an effective batch of 96):
    #   Civil=0(46%)  → 0.3   (huge sample count, tiny alpha)
    #   Corporate=1(1%)→ 5.0   (extreme minority, big push)
    #   CourtOfClaims=2(4%)→ 3.0
    #   Criminal=3(19%) → 1.5
    #   Other=4(2%)   → 5.0   (extreme minority, big push)
    #   Probate=5(8%) → 2.5
    #   Property=6(19%)→ 1.5
    # Effective contributions: 13.2, 5.0, 12.0, 27.0, 10.0, 20.0, 27.0
    MANUAL_FOCAL_ALPHA = [0.3, 5.0, 3.0, 1.5, 5.0, 2.5, 1.5]
    FOCAL_GAMMA = 2.0

    # ---- Ablation Mode ----
    # Uncomment exactly ONE of the three modes below. This controls:
    #   A: BalancedBatchSampler + focal loss, NO label smoothing.
    #   B: BalancedBatchSampler + focal loss + mild smoothing (0.05).
    #   C: Like B but with reduced alpha for Other/Corporate if precision drops too far.
    #
    # Rationale: label smoothing + focal loss + balanced batch is a potent triple-regularizer
    # that may over-soften class boundaries for minority classes on a hard dataset.
    # Run A first as baseline; B and C are progressive interventions.
    ABLATION_MODE = 'A'   # CHANGE THIS to 'A', 'B', or 'C' as needed

    if ABLATION_MODE == 'A':
        LABEL_SMOOTHING = 0.0
    elif ABLATION_MODE == 'B':
        LABEL_SMOOTHING = 0.05
    elif ABLATION_MODE == 'C':
        LABEL_SMOOTHING = 0.05
        # Reduce Other/Corporate alphas to prevent oversampling from dominating their signal.
        # Corporate precision often drops when oversampling pushes recall at the cost of precision.
        MANUAL_FOCAL_ALPHA = [0.3, 3.5, 3.0, 1.5, 3.5, 2.5, 1.5]
    else:
            raise ValueError(f"Unknown ABLATION_MODE={ABLATION_MODE}, must be 'A', 'B', or 'C'.")

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'



# ============================================================
#  FOCAL LOSS — FIXED MATH
# ============================================================
class FocalLoss(nn.Module):
    """
    Correct Focal Loss:
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    Reference: Lin et al. 2017 "Focal Loss for Dense Object Detection"
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none',
                                              label_smoothing=Config.LABEL_SMOOTHING)
        # Clamp ce_loss before exp to prevent overflow/underflow
        ce_loss_safe = ce_loss.clamp(min=0.0, max=50.0)
        pt = torch.exp(-ce_loss_safe)
        focal_term = ((1 - pt) ** self.gamma) * ce_loss_safe
        if self.alpha is not None:
            assert targets.max().item() < len(self.alpha), \
                f"Label index {targets.max().item()} out of range for alpha ({len(self.alpha)} classes)"
            focal_term = focal_term * self.alpha[targets]
        focal_term = focal_term.clamp(max=1e4)
        if self.reduction == 'mean':
            return focal_term.mean()
        elif self.reduction == 'sum':
            return focal_term.sum()
        return focal_term


# ============================================================
#  DATA LOADING
# ============================================================
def load_data():
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

    for name, df in [('train', train_df), ('val', val_df)]:
        for col in ['text', 'label']:
            if col not in df.columns:
                continue
            nan_count = df[col].isna().sum()
            inf_count = 0
            if col == 'text':
                inf_count = df[col].apply(lambda x: isinstance(x, float) and np.isinf(x)).sum()
            if nan_count > 0 or inf_count > 0:
                print(f"  ERROR: {name}.{col} has {nan_count} NaN and {inf_count} inf values — fix before training!")
                raise ValueError(f"Data corruption in {name}.{col}")
    print("  Data integrity check passed.")

    # Filter out tiny/corrupted samples (<=20 chars = just headers, no body)
    MIN_TEXT_LEN = 20
    for name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        df['text'] = df['text'].astype(str)
        orig_len = len(df)
        df.drop(df[df['text'].str.len() <= MIN_TEXT_LEN].index, inplace=True)
        dropped = orig_len - len(df)
        if dropped > 0:
            print(f"  Dropped {dropped} tiny samples (<={MIN_TEXT_LEN} chars) from {name} ({dropped*100/orig_len:.2f}%)")
    if 'text' in train_df.columns and train_df['text'].dtype == 'object':
        pass  # already converted

    possible_le = [
        '/teamspace/studios/this_studio/data/processed/label_encoder.pkl',
        f'{Config.DATA_PATH}label_encoder.pkl',
        '/kaggle/input/datasets/phantrntngvyk64cntt/data-processed/label_encoder.pkl',
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


# ============================================================
#  DATASET — pre-tokenize full document then chunk at access time
# ============================================================
class BertDocDataset(Dataset):
    """
    Pre-tokenizes full document WITHOUT special tokens (no CLS/SEP),
    then _encode_chunks slices raw tokens and adds CLS/SEP per chunk.

    This ensures each chunk gets a clean [CLS]...[SEP] structure,
    not corrupted by double special tokens from slicing a pre-tokenized doc.

    Caching: pre-tokenized tensors are saved to disk (pickle) and loaded
    on subsequent runs to skip the slow pre-tokenization step.
    """
    _cache_dir = '/teamspace/studios/this_studio/.bert_cache/'
    _cache_file = None  # class-level, set per split

    def __init__(self, texts, labels, tokenizer, max_chunks=14,
                 chunk_size=384, chunk_stride=384, split='train'):
        self.max_chunks = max_chunks
        self.chunk_size = chunk_size
        self.chunk_stride = chunk_stride
        self.labels = [torch.tensor(l, dtype=torch.long) for l in labels]
        self._tokenizer_name = tokenizer.__class__.__name__

         # Use a stable cache key so we only recompute when params change
        total_len = max_chunks * chunk_stride + (chunk_size - chunk_stride)
        cache_key = f'{split}_{len(texts)}_{max_chunks}_{chunk_size}_{chunk_stride}_{total_len}'
        os.makedirs(self._cache_dir, exist_ok=True)
        self._cache_path = os.path.join(self._cache_dir, f'{cache_key}.pt')

        if os.path.exists(self._cache_path):
            print(f"  Loading cached pre-tokenization from {self._cache_path}...")
            cached = torch.load(self._cache_path, map_location='cpu', weights_only=True)
            self.input_ids = cached['input_ids']        # (N, total_len)
            self.attention_mask = cached['attention_mask']  # (N, total_len)
            print(f"  Loaded {self.input_ids.shape[0]:,} cached samples.")
        else:
            N = len(texts)
            print(f"  Pre-tokenizing {N:,} samples (no special tokens)...")
            # Pre-allocate: avoids list + torch.cat doubling peak RAM
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
            print(f"  Done pre-tokenizing {ptr:,} samples — saving cache...")
            torch.save({
                'input_ids': self.input_ids,
                'attention_mask': self.attention_mask,
            }, self._cache_path)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx], self.labels[idx]


class ChunkAttentionMHA(nn.Module):
    """
    Multi-Head Self-Attention to aggregate chunk CLS embeddings into document embedding.

    Architecture:
      chunk_cls_embeddings (B, num_chunks, H)
        → + chunk sinusoidal positional embedding
        → Multi-Head Self-Attention (2 separate LayerNorm: pre-attention + pre-FFN)
        → [CLS_first; masked_mean_pool] weighted combination
        → document embedding (B, H)
    """
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        assert hidden_dim % num_heads == 0

        # MHA processes (batch, seq, embed) when batch_first=True
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Standard transformer block: 2 separate LayerNorms
        self.norm_attn = nn.LayerNorm(hidden_dim)  # Pre-attention norm
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm_ffn = nn.LayerNorm(hidden_dim)  # Pre-FFN norm

        # Learned weight for combining [CLS-first] with mean-pool
        self.cls_weight = nn.Parameter(torch.tensor(0.5))

    def _get_pos_emb(self, num_chunks, device):
        """Sinusoidal positional encoding — no learned parameters."""
        positions = torch.arange(num_chunks, device=device)
        dim_t = torch.arange(0, self.hidden_dim, 2, device=device).float()
        angles = positions.unsqueeze(1) / (10000 ** (dim_t / self.hidden_dim))
        pe = torch.zeros(num_chunks, self.hidden_dim, device=device)
        pe[:, 0::2] = torch.sin(angles)
        pe[:, 1::2] = torch.cos(angles)
        return pe

    def forward(self, chunk_embs, chunk_mask=None):
        """
        Args:
            chunk_embs: (B, num_chunks, hidden_dim) — CLS embedding per chunk
            chunk_mask: (B, num_chunks) — 1 for valid chunk, 0 for padding
        Returns:
            (B, hidden_dim) — document embedding
        """
        B, N, H = chunk_embs.shape

        # Add sinusoidal positional encoding
        pos_emb = self._get_pos_emb(N, chunk_embs.device)
        x = chunk_embs + pos_emb.unsqueeze(0)

        # Padding mask for MHA: True = masked (ignore)
        key_padding_mask = None
        if chunk_mask is not None:
            key_padding_mask = (chunk_mask == 0)  # True = masked

        # Pre-LN self-attention with residual
        x_normed = self.norm_attn(x)
        attn_out, _ = self.mha(x_normed, x_normed, x_normed,
                               key_padding_mask=key_padding_mask)
        x = x + attn_out

        # Pre-LN FFN with residual
        x = x + self.ffn(self.norm_ffn(x))

        # Combine: w * [CLS-first chunk] + (1-w) * masked_mean_pool
        cls_emb = x[:, 0]  # (B, H) — first (beginning of document)
        if chunk_mask is not None:
            # Masked mean: sum / count (ignores padding chunks)
            mask_expanded = chunk_mask.unsqueeze(-1).float()  # (B, N, 1)
            mean_emb = (x * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-9)
        else:
            mean_emb = x.mean(dim=1)
        w = torch.sigmoid(self.cls_weight)
        doc_emb = w * cls_emb + (1 - w) * mean_emb

        return doc_emb


# ============================================================
#  FULL MODEL — BERT + CLS extraction + MHA + classifier
# ============================================================
class FullModel(nn.Module):
    """
    BERT (last N layers trainable) + CLS extraction + Multi-Head Attention + MLP.

    Processing pipeline:
      text → tokenize → BERT encode all chunks IN PARALLEL (batched)
           → Re-tokenize each chunk with [CLS]...[SEP] to get true CLS embeddings
           → ChunkAttentionMHA: sinusoidal pos + MHA + [CLS*w + masked_mean*(1-w)]
           → MLP: LayerNorm → Linear(768→256) → GELU → Dropout → Linear(256→7)
    """
    def __init__(self, bert_model, tokenizer, hidden_dim=768, num_classes=7, dropout=0.3,
                 max_chunks=None, chunk_size=384, chunk_stride=384,
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

        # Pre-compute special token IDs
        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = 0

        # Multi-Head Self-Attention over CLS embeddings of all chunks
        self.chunk_mha = ChunkAttentionMHA(hidden_dim, num_heads=num_heads, dropout=dropout)

        # MLP: Pre-LN classifier
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, 256)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(256, num_classes)

    def _encode_chunks(self, input_ids, attention_mask):
        """
        Encode chunks with proper BERT [CLS]...[SEP] structure.

        Each chunk is re-tokenized as a standalone BERT input:
          [CLS] + chunk_content[:chunk_size-2] + [SEP]

        PARALLEL: all chunks processed in ONE BERT forward pass via batch flattening.
        """
        B = input_ids.size(0)
        num_chunks = self.max_chunks
        content_size = self.chunk_size - 2  # room for CLS and SEP

        # Slice all chunks: (B, num_chunks, content_size)
        all_ids = torch.stack([
            input_ids[:, start:start + content_size]
            for start in range(0, num_chunks * self.chunk_stride, self.chunk_stride)
        ], dim=1)
        all_am = torch.stack([
            attention_mask[:, start:start + content_size]
            for start in range(0, num_chunks * self.chunk_stride, self.chunk_stride)
        ], dim=1)

        # Mark valid chunks (have at least one real token): (B, num_chunks)
        chunk_valid = all_am.any(dim=2).float()  # (B, num_chunks) — 1.0 if chunk has real tokens

        # Build standalone BERT inputs: [CLS] + content + [SEP]
        # Shape: (B, num_chunks, chunk_size)
        standalone_ids = torch.full((B, num_chunks, self.chunk_size), self.pad_token_id,
                                    dtype=torch.long, device=self.device)
        standalone_am = torch.zeros(B, num_chunks, self.chunk_size, dtype=torch.long, device=self.device)

        standalone_ids[:, :, 0] = self.cls_token_id
        standalone_ids[:, :, 1:1 + content_size] = all_ids
        standalone_am[:, :, 0] = 1
        standalone_am[:, :, 1:1 + content_size] = all_am

        # Place SEP right after the last real token in each chunk, not at a fixed position.
        # Layout: [CLS at 0] + content[1..k] + [SEP at k+1] + PAD[k+2..]
        # For a chunk with 50 real tokens: SEP at pos 51 (after token-50, before 331 PADs).
        # For a full chunk (382 tokens): SEP at pos 383 → clamped to 382.
        # For an all-pad chunk (0 real tokens): SEP at pos 1 (right after CLS).
        content_lens = all_am.sum(dim=2)           # (B, num_chunks) — real token count per chunk
        sep_pos_0idx = (content_lens + 1).long().clamp(min=1, max=self.chunk_size - 1)
        # scatter_ requires index shape == self shape; expand index to (B, num_chunks, chunk_size)
        idx_expanded = sep_pos_0idx.unsqueeze(2).expand(B, num_chunks, self.chunk_size)
        sep_mask = torch.zeros(B, num_chunks, self.chunk_size, dtype=torch.long, device=self.device)
        sep_mask.scatter_(dim=2, index=idx_expanded, src=torch.ones_like(sep_pos_0idx).unsqueeze(2).expand_as(sep_mask))
        standalone_ids = standalone_ids.masked_fill(sep_mask.bool(), self.sep_token_id)
        standalone_am = (standalone_am.float() + sep_mask.float()).clamp(0, 1).long()

        # Flatten to (B*num_chunks, chunk_size) — ONE BERT call
        flat_ids = standalone_ids.reshape(B * num_chunks, self.chunk_size)
        flat_am = standalone_am.reshape(B * num_chunks, self.chunk_size)

        outputs = self.bert_model(input_ids=flat_ids, attention_mask=flat_am)
        last_hidden = outputs.last_hidden_state  # (B*num_chunks, chunk_size, H)

        # Reshape and extract CLS: (B, num_chunks, chunk_size, H) → (B, num_chunks, H)
        last_hidden = last_hidden.reshape(B, num_chunks, self.chunk_size, self.hidden_dim)
        cls_embs = last_hidden[:, :, 0, :]  # CLS token of each chunk

        return cls_embs, chunk_valid

    def forward(self, input_ids, attention_mask):
        cls_embs, chunk_valid = self._encode_chunks(input_ids, attention_mask)
        doc_emb = self.chunk_mha(cls_embs, chunk_valid)

        # MLP: Pre-LN — one dropout only (before final projection)
        doc_emb = self.norm(doc_emb)
        doc_emb = self.fc1(doc_emb)
        doc_emb = self.act(doc_emb)
        doc_emb = self.dropout(doc_emb)
        logits = self.fc2(doc_emb)
        return logits


# ============================================================
#  TRAINING — end-to-end fine-tuning
# ============================================================
def train_epoch(model, dataloader, optimizer, criterion, device, scaler=None,
                 scheduler=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels_list = []
    accum_steps = Config.ACCUM_STEPS
    accum_count = 0

    optimizer.zero_grad()

    pbar = tqdm(enumerate(dataloader), total=len(dataloader),
                desc=f"Train", unit="batch", ncols=100)
    nan_debug_printed = False
    for step, batch in pbar:
        input_ids, attention_mask, labels = batch
        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels         = labels.to(device)

        with torch.amp.autocast('cuda', enabled=Config.USE_FP16):
            logits = model(input_ids, attention_mask)
            loss_full = criterion(logits.float(), labels)
            if torch.isnan(loss_full) or torch.isinf(loss_full):
                if not nan_debug_printed:
                    nan_debug_printed = True
                    print(f"\n  [NaN DEBUG] step={step}")
                    print(f"    logits stats: min={logits.min().item():.4f}, max={logits.max().item():.4f}, mean={logits.mean().item():.4f}, nan={torch.isnan(logits).sum().item()}, inf={torch.isinf(logits).sum().item()}")
                    print(f"    labels: {labels[:4].tolist()}")
                    print(f"    loss_full={loss_full.item():.4f}")
                optimizer.zero_grad()
                # Do NOT increment accum_count — no gradient was accumulated for this batch.
                # Skipping the step keeps the accumulation cycle intact.
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels_list.extend(labels.cpu().numpy())
                pbar.set_postfix({'loss': f'{loss_full.item():.4f}', 'acc': f'{(preds == labels).sum().item() / labels.size(0):.4f}'})
                continue

            loss = loss_full / accum_steps
            total_loss += loss_full.item() * labels.size(0)

        loss.backward()
        accum_count += 1

        # Only step every ACCUM_STEPS batches
        if accum_count % accum_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                print(f"  [NaN/Inf grad] skipping update")
                optimizer.zero_grad()
            else:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels_list.extend(labels.cpu().numpy())

        pbar.set_postfix({
            'loss': f'{loss_full.item():.4f}',
            'acc':  f'{(preds == labels).sum().item() / labels.size(0):.4f}',
        })

    # Handle remaining accumulated gradients if not divisible by accum_steps
    if accum_count % accum_steps != 0:
        if scaler is not None:
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not (torch.isnan(grad_norm) or torch.isinf(grad_norm)):
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            # Only step scheduler here if the trailing micro-batch completed an
            # accumulation cycle. If the last micro-batch already triggered a
            # normal optimizer.step() (accum_count was just bumped to a multiple
            # of accum_steps), skip to avoid double-stepping the LR schedule.
            prev_accum_count = accum_count - 1
            if prev_accum_count % accum_steps != 0:
                scheduler.step()
            optimizer.zero_grad()

    f1_macro = f1_score(all_labels_list, all_preds, average='macro')
    f1_per_class = f1_score(all_labels_list, all_preds, average=None)
    return total_loss / total, correct / total, f1_macro, f1_per_class


def evaluate(model, dataloader, criterion, device, scaler=None):
    model.eval()
    total_loss = 0
    total_samples = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels_list = []

    pbar = tqdm(dataloader, desc="Eval ", unit="batch", ncols=100)
    for batch in pbar:
        input_ids, attention_mask, labels = batch
        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels         = labels.to(device)

        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=Config.USE_FP16):
                logits = model(input_ids, attention_mask)
                loss = criterion(logits.float(), labels)

        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels_list.extend(labels.cpu().numpy())

    f1_macro = f1_score(all_labels_list, all_preds, average='macro')
    f1_per_class = f1_score(all_labels_list, all_preds, average=None)
    return total_loss / total_samples, correct / total, f1_macro, f1_per_class, all_preds


def predict(model, dataloader, device, scaler=None):
    model.eval()
    all_preds = []
    for batch in dataloader:
        input_ids, attention_mask, labels = batch
        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=Config.USE_FP16):
                logits = model(input_ids, attention_mask)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
    return np.array(all_preds)


# ============================================================
#  BALANCED BATCH SAMPLER
# ============================================================
class BalancedBatchSampler(torch.utils.data.BatchSampler):
    """
    BalancedBatchSampler yields batches where each class contributes the same
    number of samples, enabling minority classes to appear in every batch.

    Design:
      - splits batch_size // num_classes samples per class per batch
      - oversamples exhausted classes by cycling through their shuffled indices
      - shuffles class order and sample order within each batch for diversity

    Caveats:
      - batch_size must be divisible by num_classes (BATCH_SIZE=14 for 7 classes)
      - oversampling minority classes means the model sees them more often per epoch;
        this is intentional and matches the goal of boosting their representation
    """

    def __init__(self, labels, batch_size, drop_last=False, shuffle_classes=True):
        # Bug #8: parent BatchSampler expects a Sampler, not a raw list.
        # SequentialSampler is a no-op sampler (we fully override __iter__ anyway)
        # but it satisfies the parent's type expectation.
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
                f"num_classes={self.num_classes} for BalancedBatchSampler."
            )

        self.samples_per_class = batch_size // self.num_classes
        self.max_class_len = max(len(v) for v in self.class_to_indices.values())

        # Anchor to MEDIAN class instead of largest class.
        # With max_class (Civil ~57K): n_batches=28K → 43h/epoch (unusable).
        # With median class (Probate ~10K): n_batches=~5K → ~8h/epoch.
        # Minority classes still cycle (oversample) within each epoch.
        # Majority class (Civil) is partially sampled — Focal Loss alpha=0.3
        # compensates by down-weighting its gradient contribution.
        class_lens = sorted(len(v) for v in self.class_to_indices.values())
        anchor_len = class_lens[len(class_lens) // 2]  # median class size
        self.n_batches = math.ceil(anchor_len / self.samples_per_class)

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        pools = {}
        ptrs = {}

        for c in self.classes:
            pools[c] = self.class_to_indices[c].copy()
            if self.shuffle_classes:
                random.shuffle(pools[c])
            ptrs[c] = 0

        for _ in range(self.n_batches):
            batch = []

            for c in self.classes:
                need = self.samples_per_class
                chosen = []

                while len(chosen) < need:
                    remain = len(pools[c]) - ptrs[c]
                    take = min(need - len(chosen), remain)

                    if take > 0:
                        chosen.extend(pools[c][ptrs[c]:ptrs[c] + take])
                        ptrs[c] += take

                    if len(chosen) < need:
                        pools[c] = self.class_to_indices[c].copy()
                        if self.shuffle_classes:
                            random.shuffle(pools[c])
                        ptrs[c] = 0

                batch.extend(chosen)

            if self.shuffle_classes:
                random.shuffle(batch)

            yield batch


# ============================================================
#  MAIN
# ============================================================
def main():
    on_kaggle = os.path.exists('/kaggle')
    on_lightning = os.path.exists('/teamspace/studios/this_studio/')
    on_colab  = False

    if on_kaggle:
        print("\n[Env] Kaggle detected")
        Config.DATA_PATH   = '/kaggle/input/datasets/phantrntngvyk64cntt/data-processed/'
        Config.OUTPUT_PATH = '/kaggle/working/output-train-bert/'
    elif on_lightning:
        print("\n[Env] Lightning AI detected")
        base = '/teamspace/studios/this_studio/'
        Config.DATA_PATH   = os.path.join(base, 'data/processed/')
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
            print("\n[Env] Google Colab detected — mounting Drive...")
            drive.mount('/content/drive')
            Config.OUTPUT_PATH = '/content/drive/MyDrive/output-train-bert/'
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
    if not Config.OUTPUT_PATH.endswith('/'):
        Config.OUTPUT_PATH += '/'

    # Auto-copy checkpoints from Kaggle dataset
    if on_kaggle:
        ckpt_dataset_path = '/kaggle/input/datasets/phantrntngvyk64cntt/output-train-bert/'
        if os.path.exists(ckpt_dataset_path):
            found_any = False
            for fname in ['bert_doc_ckpt_ft.pt', 'bert_doc_best_ft.pt']:
                src = os.path.join(ckpt_dataset_path, fname)
                dst = os.path.join(Config.OUTPUT_PATH, fname)
                if os.path.exists(src):
                    if not os.path.exists(dst):
                        shutil.copy(src, dst)
                        print(f"  [Checkpoint] COPIED: {fname}")
                    else:
                        print(f"  [Checkpoint] Already exists: {fname}")
                    found_any = True
            if not found_any:
                print(f"  [Checkpoint] No checkpoints found in dataset")
        print()

    print(f"\n  Data path:   {Config.DATA_PATH}")
    print(f"  Output path: {Config.OUTPUT_PATH}")

    device = Config.DEVICE

    # ---- 1. Load BERT ----
    print(f"\n[1/5] Loading BERT: {Config.BERT_MODEL}...")
    bert_model = AutoModel.from_pretrained(Config.BERT_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(Config.BERT_MODEL)
    bert_model.to(device)

    hidden_dim = bert_model.config.hidden_size
    print(f"  BERT hidden: {hidden_dim}")

    # Freeze all, then open last N layers
    for p in bert_model.parameters():
        p.requires_grad = False
    for layer in bert_model.encoder.layer[-Config.BERT_FT_LAYERS:]:
        for p in layer.parameters():
            p.requires_grad = True

    if Config.GRADIENT_CHECKPOINTING and hasattr(bert_model, 'gradient_checkpointing_enable'):
        bert_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        print(f"  Gradient checkpointing enabled")

    trainable_bert = sum(p.numel() for p in bert_model.parameters() if p.requires_grad)
    total_bert = sum(p.numel() for p in bert_model.parameters())
    print(f"  Fine-tuning: {trainable_bert:,} / {total_bert:,} params (last {Config.BERT_FT_LAYERS} layers)")

    # ---- 2. Load data ----
    print("\n[2/5] Loading data...")
    train_df, val_df, test_df, label_encoder = load_data()
    num_classes = len(label_encoder.classes_)

    # ---- 3. Build datasets ----
    print("\n[3/5] Creating datasets...")
    train_dataset = BertDocDataset(
        train_df['text'].tolist(), train_df['label_encoded'].tolist(),
        tokenizer, Config.MAX_CHUNKS, Config.CHUNK_SIZE, Config.CHUNK_STRIDE, split='train')
    val_dataset   = BertDocDataset(
        val_df['text'].tolist(), val_df['label_encoded'].tolist(),
        tokenizer, Config.MAX_CHUNKS, Config.CHUNK_SIZE, Config.CHUNK_STRIDE, split='val')
    test_dataset  = BertDocDataset(
        test_df['text'].tolist(), test_df['label_encoded'].tolist(),
        tokenizer, Config.MAX_CHUNKS, Config.CHUNK_SIZE, Config.CHUNK_STRIDE, split='test')

    def collate_fn(batch):
        input_ids      = torch.stack([item[0] for item in batch])
        attention_mask = torch.stack([item[1] for item in batch])
        labels         = torch.stack([item[2] for item in batch])
        return input_ids, attention_mask, labels

    # BalancedBatchSampler: every batch has equal class representation.
    # This directly addresses the "minority dragged into Civil" problem by ensuring
    # minority classes appear in every batch, not just on average over the epoch.
    train_labels = train_df['label_encoded'].tolist()
    train_batch_sampler = BalancedBatchSampler(
        labels=train_labels,
        batch_size=Config.BATCH_SIZE,
        drop_last=False,
        shuffle_classes=True,
    )
    train_loader = DataLoader(train_dataset,
                              batch_sampler=train_batch_sampler,
                              num_workers=4,
                              pin_memory=(device == 'cuda'),
                              prefetch_factor=2,
                              persistent_workers=(device == 'cuda' and os.environ.get('CUDA_VISIBLE_DEVICES', '0') != ''),
                              collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE,
                              shuffle=False, num_workers=2,
                              pin_memory=(device == 'cuda'), collate_fn=collate_fn)
    test_loader  = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE,
                              shuffle=False, num_workers=2,
                              pin_memory=(device == 'cuda'), collate_fn=collate_fn)

    print(f"  Train: {len(train_dataset):,} | Val: {len(val_dataset):,} | Test: {len(test_dataset):,}")
    print(f"  Train batches: {len(train_loader)}")

    # ---- 4. Build model ----
    print(f"\n[4/5] Building model...")
    model = FullModel(
        bert_model=bert_model,
        tokenizer=tokenizer,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout=Config.DROPOUT,
        max_chunks=Config.MAX_CHUNKS,
        chunk_size=Config.CHUNK_SIZE,
        chunk_stride=Config.CHUNK_STRIDE,
        bert_ft_layers=Config.BERT_FT_LAYERS,
        num_heads=Config.NUM_HEADS,
    ).to(device)

    # ---- Optimizer with separate LR tiers ----
    # no_decay for bias and all LayerNorm weights (.weight/.bias endings)
    bert_decay, bert_no_decay = [], []
    for n, p in bert_model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or n.endswith('.bias') or n.endswith('.LayerNorm.weight'):
            bert_no_decay.append(p)
        else:
            bert_decay.append(p)

    # Separate MHA (chunk_mha) and MLP (norm, fc1, fc2)
    # Only bias (1D) and LayerNorm weights go to no_decay; linear weights get decay
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
        {'params': bert_decay,     'lr': Config.LR_BERT,      'weight_decay': Config.WEIGHT_DECAY},
        {'params': bert_no_decay,  'lr': Config.LR_BERT,      'weight_decay': 0.0},
        {'params': mha_decay,       'lr': Config.LR_ATTENTION, 'weight_decay': Config.WEIGHT_DECAY},
        {'params': mha_no_decay,    'lr': Config.LR_ATTENTION, 'weight_decay': 0.0},
        {'params': mlp_decay,       'lr': Config.LR_MLP,       'weight_decay': Config.WEIGHT_DECAY},
        {'params': mlp_no_decay,   'lr': Config.LR_MLP,       'weight_decay': 0.0},
    ])

    print("=" * 60)
    print("BERT Document Classifier — FIXED (parallel chunk encoding, Pre-LN MLP, chunk positional encoding)")
    print(f"BERT model: {Config.BERT_MODEL}")
    print(f"Chunk strategy: {Config.CHUNK_SIZE} tokens/chunk, stride {Config.CHUNK_STRIDE} ({Config.CHUNK_SIZE - Config.CHUNK_STRIDE} unique / {(Config.CHUNK_SIZE - Config.CHUNK_STRIDE) + Config.CHUNK_STRIDE} = {(Config.CHUNK_SIZE - Config.CHUNK_STRIDE) * 100 // Config.CHUNK_STRIDE}% unique, {Config.CHUNK_STRIDE} overlap)")
    coverage = Config.MAX_CHUNKS * Config.CHUNK_STRIDE + (Config.CHUNK_SIZE - Config.CHUNK_STRIDE)
    print(f"Max chunks: {Config.MAX_CHUNKS} | Token coverage: {coverage} tokens | ALL chunks encoded IN PARALLEL")
    print(f"Batch size: {Config.BATCH_SIZE} | Effective batch: {Config.BATCH_SIZE*Config.ACCUM_STEPS}")
    print(f"LR BERT: {Config.LR_BERT} | LR MHA: {Config.LR_ATTENTION} | LR MLP: {Config.LR_MLP}")
    print(f"Fine-tuning: last {Config.BERT_FT_LAYERS} layers")
    print(f"Chunk aggregation: CLS extraction → MHA ({Config.NUM_HEADS} heads) → [CLS*w + mean*(1-w)]")
    print(f"Classifier: Pre-LN (LayerNorm → Linear 768→256 → GELU → Linear 256→7)")
    print(f"Loss: FocalLoss(gamma={Config.FOCAL_GAMMA}) + BalancedBatchSampler | Ablation mode: {Config.ABLATION_MODE}")
    print(f"       alpha={Config.MANUAL_FOCAL_ALPHA} | label_smoothing={Config.LABEL_SMOOTHING}")
    print("=" * 60)

    # FP16 scaler
    fp16_scaler = torch.amp.GradScaler('cuda', enabled=Config.USE_FP16)

    # ---- LR Scheduler ----
    # scheduler.step() is called once per optimizer update (after ACCUM_STEPS micro-batches).
    # len(train_loader) = number of micro-batches (iterations) per epoch from BalancedBatchSampler.
    # Each optimizer update consumes ACCUM_STEPS micro-batches, so the number of updates (steps)
    # per epoch is: ceil(len(train_loader) / ACCUM_STEPS).
    #
    # NOTE on "oversampled epoch": BalancedBatchSampler generates batches by cycling through the
    # largest class until exhausted. The resulting n_batches covers the minority classes multiple
    # times while the majority class is only seen once per epoch. This means:
    #   - Each epoch = ceil(max_class_count / samples_per_class) micro-batches
    #   - Each epoch = ceil(n_batches / ACCUM_STEPS) optimizer updates (steps)
    # This is intentional: minority classes benefit from repeated exposure, and the scheduler
    # still tracks the correct number of BERT parameter updates across epochs.
    steps_per_epoch = math.ceil(len(train_loader) / Config.ACCUM_STEPS)
    warmup_steps = steps_per_epoch * Config.WARMUP_EPOCHS
    total_steps  = steps_per_epoch * Config.EPOCHS

    # Store total_steps in a mutable container so the closure always reads the
    # current (not original) value. This is needed for resume: if training is
    # interrupted and resumed from a checkpoint, we recompute remaining steps
    # and update total_steps_ref so the cosine schedule starts from the right
    # position in the decay curve.
    total_steps_ref = [total_steps]
    warmup_steps_ref = [warmup_steps]
    no_decay_min_lr = 1e-6  # floor so BERT params never fully anneal to 0
    def lr_lambda_bert(step):
        t = total_steps_ref[0]
        w = warmup_steps_ref[0]
        if step < w:
            return float(step) / max(1, w)
        progress = (step - w) / max(1, t - w)
        return max(no_decay_min_lr, 0.5 * (1.0 + np.cos(np.pi * progress)))

    def lr_lambda_head(_step):
        return 1.0  # head LR stays fixed at its initial group LR throughout training

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[lr_lambda_bert, lr_lambda_bert, lr_lambda_head,
                   lr_lambda_head, lr_lambda_head, lr_lambda_head]
    )

    # Focal loss with alpha
    alpha_tensor = torch.tensor(Config.MANUAL_FOCAL_ALPHA, dtype=torch.float32).to(device)
    criterion = FocalLoss(alpha=alpha_tensor, gamma=Config.FOCAL_GAMMA)

    checkpoint_path = f'{Config.OUTPUT_PATH}bert_doc_ckpt_ft.pt'
    best_model_path = f'{Config.OUTPUT_PATH}bert_doc_best_ft.pt'

    best_val_f1 = 0.0
    start_epoch = 0

    # Resume from checkpoint
    print(f"\n  [Resume] Looking for checkpoints in: {Config.OUTPUT_PATH}")
    print(f"    - Regular checkpoint : {os.path.basename(checkpoint_path)} [{'EXISTS' if os.path.exists(checkpoint_path) else 'NOT FOUND'}]")
    print(f"    - Best checkpoint     : {os.path.basename(best_model_path)} [{'EXISTS' if os.path.exists(best_model_path) else 'NOT FOUND'}]")

    if os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch = ckpt.get('epoch', 0) + 1
            best_val_f1 = ckpt.get('best_val_f1', 0.0)
            # Bug #11: after resuming, update total_steps_ref to the recomputed
            # remaining steps so the cosine schedule starts from the right position.
            # Scheduler's internal _step_count is already restored by load_state_dict.
            remaining_epochs = Config.EPOCHS - start_epoch
            remaining_steps = math.ceil(len(train_loader) / Config.ACCUM_STEPS) * remaining_epochs
            total_steps_ref[0] = scheduler.state_dict()['_step_count'] + remaining_steps
            print(f"  [Resume] SUCCESS — epoch={start_epoch}, best_val_f1={best_val_f1:.4f}, "
                  f"total_steps updated to {total_steps_ref[0]}")
        except Exception as e:
            print(f"  [Resume] Full load FAILED ({e})")
            try:
                ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
                bert_sd = ckpt.get('model_state_dict', ckpt)
                loaded, skipped = 0, 0
                for k, v in bert_sd.items():
                    if k.startswith('bert_model.'):
                        key = k[len('bert_model.'):]
                        if key in model.bert_model.state_dict():
                            model.bert_model.state_dict()[key].copy_(v)
                            loaded += 1
                        else:
                            skipped += 1
                print(f"  [Resume] Loaded {loaded} BERT keys, skipped {skipped}")
            except Exception:
                print(f"  [Resume] No BERT weights found — starting from SCRATCH")
            start_epoch = 0
            best_val_f1 = 0.0
    elif os.path.exists(best_model_path):
        try:
            model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
            print(f"  [Resume] Loaded best model (starting from scratch for training)")
        except Exception as e:
            print(f"  [Resume] Best model mismatch ({e}) — starting from SCRATCH")
    else:
        print(f"  [Resume] NO checkpoints found — starting from SCRATCH")

    # ---- 5. Training ----
    print(f"\n[5/5] Training (end-to-end fine-tune, BERT last {Config.BERT_FT_LAYERS} layers)...")

    patience = 3
    patience_counter = 0

    for epoch in range(start_epoch, Config.EPOCHS):
        t_ep = time.time()

        train_loss, train_acc, train_f1, train_f1_per = train_epoch(
            model, train_loader, optimizer, criterion, device, fp16_scaler,
            scheduler
        )

        val_loss, val_acc, val_f1, val_f1_per, _ = evaluate(
            model, val_loader, criterion, device, fp16_scaler
        )

        t_dur = time.time() - t_ep
        current_lr_bert = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1:2d}/{Config.EPOCHS} ({t_dur:.1f}s) | "
              f"LR BERT:{current_lr_bert:.2e} MHA:{optimizer.param_groups[2]['lr']:.2e} MLP:{optimizer.param_groups[4]['lr']:.2e} | "
              f"Tr Loss:{train_loss:.4f} Acc:{train_acc:.4f} F1:{train_f1:.4f} | "
              f"Val Loss:{val_loss:.4f} Acc:{val_acc:.4f} F1:{val_f1:.4f}")
        print(f"  Val F1 per class: {[f'{f:.3f}' for f in val_f1_per]}")

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
                                      map_location=device, weights_only=True))

    # ---- 6. Final evaluation ----
    print("\n" + "=" * 60)
    print("Final Evaluation on Test Set")
    print("=" * 60)

    test_preds = predict(model, test_loader, device, fp16_scaler)
    test_labels_np = test_df['label_encoded'].values

    test_acc = (test_preds == test_labels_np).mean()
    test_f1 = f1_score(test_labels_np, test_preds, average='macro')
    test_f1_per = f1_score(test_labels_np, test_preds, average=None)

    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Macro F1: {test_f1:.4f}")
    print(f"Test F1 per class: {[f'{f:.3f}' for f in test_f1_per]}")
    print("\nClassification Report:")
    print(classification_report(test_labels_np, test_preds,
                                target_names=label_encoder.classes_))

    cm = confusion_matrix(test_labels_np, test_preds, labels=range(num_classes))
    print("Per-class Accuracy:")
    for i, name in enumerate(label_encoder.classes_):
        total = cm[i].sum()
        correct = cm[i, i]
        print(f"  {name}: {correct}/{total} = {correct/total:.3f}")

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
    plt.savefig(f'{Config.OUTPUT_PATH}bert_ft_cm.png', dpi=150)
    plt.close()
    print(f"\nConfusion matrix saved.")

    test_df_out = test_df.copy()
    test_df_out['pred'] = label_encoder.inverse_transform(test_preds)
    test_df_out.to_csv(f'{Config.OUTPUT_PATH}bert_ft_predictions.csv', index=False)
    print(f"Predictions saved.")

    with open(f'{Config.OUTPUT_PATH}best_model_path.txt', 'w') as f:
        f.write(best_model_path)

    print(f"\nDone! Best model: {best_model_path}")


if __name__ == '__main__':
    main()

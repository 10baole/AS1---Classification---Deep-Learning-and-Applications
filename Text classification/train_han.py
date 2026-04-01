#!/usr/bin/env python3
"""
HAN-inspired Classifier — Legal Text Classification
Refactored: I. Source code (Text): Class-based format

Pipeline: Document → Split sentences → Encode (SentenceTransformer)
        → BiLSTM → Sentence Attention → MLP → Output

Classes:
  1. Config        — hyper-parameters, paths, model config
  2. HANDataset   — EDA + Augmentation (sentence splitting)
  3. DataLoaders  — train / val / test splits
  4. SentenceAttention + HANClassifier — Input → Preprocess → Backbone → Head → Output
  5. FocalLoss + train_epoch — batch-wise → epoch → Loss
  6. Finetuning   — freeze backbone, layer-wise LR
  7. Evaluator    — Accuracy, Precision, Recall, F1 (macro + per-class)
  8. Analysis     — Params, overfitting curve, inference time, attention map
"""

import os
import pickle
import random
import time
import re
import numpy as np
import pandas as pd
import torch
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
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# =============================================================================
# 1. CONFIG RESOLVE
# =============================================================================
class Config:
    DATA_PATH = './data/processed/'
    OUTPUT_PATH = './output-train-han/'

    MODEL_NAME = 'tomaarsen/glove-bilstm-sts'

    SENT_HIDDEN = 128
    MAX_SENTENCES = 50
    MAX_TOKEN = 512
    DROPOUT = 0.3

    BATCH_SIZE = 256
    ENCODE_BATCH_SIZE = 512
    EPOCHS = 10
    LR = 1e-3
    WEIGHT_DECAY = 0.01
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    FOCAL_GAMMA = 2.0
    # [Civil=0, Corporate=1, CourtOfClaims=2, Criminal=3, Other=4, Probate=5, Property=6]
    MANUAL_FOCAL_ALPHA = [0.2, 2.0, 1.0, 0.5, 1.8, 1.2, 1.0]


class DataResolver:
    """Resolve data paths and platform detection."""

    @staticmethod
    def resolve():
        on_kaggle = os.path.exists('/kaggle')
        on_lightning = os.path.exists('/teamspace/studios/this_studio/')
        on_colab = False

        if on_kaggle:
            Config.DATA_PATH = '/kaggle/input/datasets/phantrntngvyk64cntt/data-processed/'
            Config.OUTPUT_PATH = '/kaggle/working/output-train-han/'
        elif on_lightning:
            base = '/teamspace/studios/this_studio/'
            Config.DATA_PATH = os.path.join(base, 'data/processed/')
            Config.OUTPUT_PATH = os.path.join(base, 'output-train-han/')
        else:
            try:
                from google.colab import drive
                import IPython
                ip = IPython.get_ipython()
                if ip is not None and ip.kernel is not None:
                    on_colab = True
            except (ImportError, AttributeError):
                pass

            if on_colab:
                drive.mount('/content/drive')
                Config.OUTPUT_PATH = '/content/drive/MyDrive/output-train-han/'
                if os.path.exists('/content/drive/MyDrive/data/processed/'):
                    Config.DATA_PATH = '/content/drive/MyDrive/data/processed/'
                elif os.path.exists('/content/data/processed/'):
                    Config.DATA_PATH = '/content/data/processed/'
            else:
                for path in ['./data/masked-15k/', './data/processed/', '../data/processed/']:
                    if os.path.exists(os.path.join(path, 'train.csv')):
                        Config.DATA_PATH = path
                        break

        os.makedirs(Config.OUTPUT_PATH, exist_ok=True)
        if not Config.OUTPUT_PATH.endswith('/'):
            Config.OUTPUT_PATH += '/'


# =============================================================================
# 2. DATASET — EDA + Augmentation
# =============================================================================
class EDAAugmentation:
    """
    EDA: Exploratory Data Analysis + text augmentation.
    Statistics: class distribution, text length, sentence count, token count.
    Augmentation: sentence-shuffling (document-level augmentation).
    """

    @staticmethod
    def split_into_sentences(text, max_sentences=50):
        abbreviations = [
            r'\bMr\.\s', r'\bMrs\.\s', r'\bDr\.\s', r'\bProf\.\s',
            r'\bvs\.\s', r'\bv\.\s', r'\bNo\.\s', r'\bU\.S\.\s',
            r'\bU\.S\.C\.\s', r'\bF\.\d+d?\s', r'\bF\.Supp\.\s',
            r'\bS\.Ct\.\s', r'\bL\.\s*Ed\.\s', r'\bIll\.\s',
            r'\bApp\.\s', r'\bCir\.\s', r'\bDist\.\s',
            r'\bCorp\.\s', r'\bInc\.\s', r'\bLtd\.\s', r'\bCo\.\s',
        ]
        text_mod = text
        for abbr in abbreviations:
            text_mod = re.sub(abbr, lambda m: m.group(0).replace('.', '<DOT>'), text_mod)
        sentences = re.split(r'(?<=[.!?])\s+', text_mod)
        cleaned = []
        for sent in sentences:
            sent = sent.replace('<DOT>', '.').strip()
            if len(sent) > 15:
                cleaned.append(sent)
        return cleaned[:max_sentences]

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
        print(f"    Text length (chars): min={text_lens.min()}, "
              f"median={int(text_lens.median())}, max={text_lens.max()}")

        sent_counts = df['text'].astype(str).apply(
            lambda x: len(EDAAugmentation.split_into_sentences(x, 1000)))
        print(f"    Sentence count: min={sent_counts.min()}, "
              f"median={int(sent_counts.median())}, max={sent_counts.max()}")
        return label_counts

    @staticmethod
    def augment_text(text, aug_type='none'):
        """Sentence-level augmentation for legal text."""
        if aug_type == 'none':
            return text
        text = str(text)
        if aug_type == 'shuffle_sentences':
            sentences = EDAAugmentation.split_into_sentences(text, 1000)
            random.shuffle(sentences)
            return ' '.join(sentences)
        return text


class HANDataset(Dataset):
    """
    Dataset: raw text → sentence splitting at load time.
    Each sample: list of sentence strings, padded to max_sentences.
    """

    def __init__(self, texts, labels, max_sentences):
        self.texts = texts
        self.labels = [torch.tensor(l, dtype=torch.long) for l in labels]
        self.max_sentences = max_sentences

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        sentences = EDAAugmentation.split_into_sentences(
            str(self.texts[idx]), self.max_sentences)
        while len(sentences) < self.max_sentences:
            sentences.append('')
        label = self.labels[idx]
        return sentences, label


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


def han_collate(batch, st_model, device, encode_batch_size=32):
    """
    Encode all sentences in the batch with a single encode() call.

    Input : list of (sentences_list, label)
    Preprocess: flatten sentences, batch-encode with SentenceTransformer
    Output: sentence_vectors (batch_size, max_sentences, embed_dim),
            labels (batch_size,), mask (batch_size, max_sentences)
    """
    batch_size = len(batch)
    labels_list = []
    all_sentences = []

    for sentences, label in batch:
        labels_list.append(label)
        all_sentences.append(sentences)

    max_sentences = len(all_sentences[0])
    embed_dim = st_model.get_sentence_embedding_dimension()

    flat_sentences = []
    flat_indices = []
    for doc_id, doc in enumerate(all_sentences):
        for sent_id, s in enumerate(doc):
            s_stripped = s.strip()
            if s_stripped:
                flat_sentences.append(s_stripped)
                flat_indices.append((doc_id, sent_id))

    flat_vectors = torch.zeros(
        len(flat_sentences), embed_dim, dtype=torch.float32, device=device)
    if flat_sentences:
        try:
            with torch.no_grad():
                encoded = st_model.encode(
                    flat_sentences,
                    batch_size=encode_batch_size,
                    convert_to_tensor=True,
                    device=device,
                    show_progress_bar=False,
                )
            if encoded.numel() > 0:
                flat_vectors[:encoded.size(0)] = encoded.to(device)
        except Exception:
            pass

    batch_vectors = torch.zeros(batch_size, max_sentences, embed_dim, device=device)
    batch_mask = torch.zeros(batch_size, max_sentences, dtype=torch.bool, device=device)

    for vec_idx, (doc_id, sent_id) in enumerate(flat_indices):
        batch_vectors[doc_id, sent_id] = flat_vectors[vec_idx]
        batch_mask[doc_id, sent_id] = True

    labels = torch.stack(labels_list).to(device)
    return batch_vectors, labels, batch_mask


def build_dataloaders(train_df, val_df, test_df):
    """Build train / val / test DataLoaders."""

    train_dataset = HANDataset(
        train_df['text'].tolist(),
        train_df['label_encoded'].tolist(),
        Config.MAX_SENTENCES)
    val_dataset = HANDataset(
        val_df['text'].tolist(),
        val_df['label_encoded'].tolist(),
        Config.MAX_SENTENCES)
    test_dataset = HANDataset(
        test_df['text'].tolist(),
        test_df['label_encoded'].tolist(),
        Config.MAX_SENTENCES)

    collate_fn = lambda b: b

    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE,
        shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE,
        shuffle=False, num_workers=0, collate_fn=collate_fn)
    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE,
        shuffle=False, num_workers=0, collate_fn=collate_fn)

    print(f"  Train batches: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")
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
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_term = ((1 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            focal_term = focal_term * self.alpha[targets]
        if self.reduction == 'mean':
            return focal_term.mean()
        elif self.reduction == 'sum':
            return focal_term.sum()
        return focal_term


class SentenceAttention(nn.Module):
    """
    Sentence-level attention to weight LSTM outputs.

    Input  : LSTM output (batch, seq, hidden_dim)
    Preprocess: pass through attention layer → softmax weights
    Head    : weighted sum of LSTM outputs → sentence context vector
    Output  : (batch, hidden_dim), attention weights
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, lstm_out, mask=None):
        w = self.attention(lstm_out)
        if mask is not None:
            w = w.squeeze(-1)
            w = w.masked_fill(~mask, -1e9)
            w = w.unsqueeze(-1)
        w = torch.softmax(w, dim=1)
        context = torch.sum(w * lstm_out, dim=1)
        return context, w.squeeze(-1)


class HANClassifier(nn.Module):
    """
    Pipeline: Sentence embeddings → BiLSTM → Sentence Attention → MLP

    Input  : (batch, max_sentences, embed_dim) — pretrained sentence embeddings
    Preprocess: BiLSTM processes sentence sequence
    Backbone: BiLSTM (1 layer, bidirectional)
    Head    : Sentence Attention → Dropout → MLP (Linear hidden*2→hidden → ReLU → Linear hidden→num_classes)
    Output  : (batch, num_classes)
    """
    def __init__(self, embed_dim, sent_hidden, num_classes, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            embed_dim,
            sent_hidden,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout_lstm = nn.Dropout(dropout)
        self.attention = SentenceAttention(sent_hidden * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(sent_hidden * 2, sent_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(sent_hidden, num_classes),
        )

    def forward(self, sentence_vectors, mask=None):
        lstm_out, _ = self.lstm(sentence_vectors)
        lstm_out = self.dropout_lstm(lstm_out)
        doc_vec, attn_weights = self.attention(lstm_out, mask)
        doc_vec = self.dropout(doc_vec)
        logits = self.fc(doc_vec)
        return logits, attn_weights


# =============================================================================
# 5. TRAINING — batch → epoch → Focal Loss
# =============================================================================
class Trainer:
    """Trainer: handles one epoch of training with gradient accumulation."""

    def __init__(self, model, train_loader, val_loader, st_model,
                 optimizer, criterion, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.st_model = st_model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        all_preds, all_labels = [], []

        pbar = tqdm(self.train_loader, desc="Train", unit="batch", ncols=100)
        for batch in pbar:
            embeddings, labels, mask = han_collate(
                batch, self.st_model, self.device, Config.ENCODE_BATCH_SIZE)
            self.optimizer.zero_grad()

            logits, _ = self.model(embeddings, mask)
            loss = self.criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        f1_macro = f1_score(all_labels, all_preds, average='macro')
        f1_per_class = f1_score(all_labels, all_preds, average=None)
        return total_loss / len(self.train_loader), correct / total, f1_macro, f1_per_class


# =============================================================================
# 6. FINETUNING — freeze backbone, layer-wise LR
# =============================================================================
class Finetuner:
    """
    Finetuning strategies:
      - freeze_backbone: freeze pretrained sentence encoder (default here)
      - full_finetune: unfreeze everything
      - layerwise_lr: decay LR from head → BiLSTM (not used; single LR group for simplicity)
    """

    @staticmethod
    def freeze_backbone(st_model):
        """Freeze pretrained SentenceTransformer encoder."""
        for p in st_model.parameters():
            p.requires_grad = False
        total = sum(p.numel() for p in st_model.parameters())
        print(f"  SentenceTransformer frozen ({total:,} params)")

    @staticmethod
    def build_optimizer(model):
        """Build AdamW optimizer for trainable params (BiLSTM + attention + MLP)."""
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
        return optimizer

    @staticmethod
    def build_scheduler(optimizer, train_loader):
        """Build cosine annealing scheduler with warmup."""
        total_steps = len(train_loader) * Config.EPOCHS
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=1e-6)
        return scheduler


# =============================================================================
# 7. EVALUATE — Accuracy, Precision, Recall, F1 (macro + per-class)
# =============================================================================
class Evaluator:
    """Evaluate model on train/val/test sets. Report per-class + macro metrics."""

    @staticmethod
    def evaluate(model, dataloader, st_model, device, alpha_tensor):
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds, all_labels = [], []

        criterion = FocalLoss(alpha=alpha_tensor, gamma=Config.FOCAL_GAMMA)

        pbar = tqdm(dataloader, desc="Eval ", unit="batch", ncols=100)
        with torch.no_grad():
            for batch in pbar:
                embeddings, labels, mask = han_collate(
                    batch, st_model, device, Config.ENCODE_BATCH_SIZE)
                logits, _ = model(embeddings, mask)
                loss = criterion(logits, labels)

                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        num_classes = len(set(all_labels))
        cm = confusion_matrix(all_labels, all_preds, labels=range(num_classes))
        acc_per_class = []
        for i in range(num_classes):
            total_c = cm[i].sum()
            acc_per_class.append(cm[i, i] / total_c if total_c > 0 else 0.0)

        precision_per = precision_score(all_labels, all_preds, average=None, zero_division=0)
        recall_per = recall_score(all_labels, all_preds, average=None, zero_division=0)
        f1_per = f1_score(all_labels, all_preds, average=None, zero_division=0)

        acc_macro = accuracy_score(all_labels, all_preds)
        prec_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        rec_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        metrics = {
            'loss': total_loss / len(dataloader),
            'accuracy': correct / total,
            'accuracy_macro': acc_macro,
            'precision_macro': prec_macro,
            'recall_macro': rec_macro,
            'f1_macro': f1_macro,
            'precision_per_class': precision_per,
            'recall_per_class': recall_per,
            'f1_per_class': f1_per,
            'accuracy_per_class': acc_per_class,
            'all_preds': all_preds,
            'all_labels': all_labels,
        }
        return metrics


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
    def model_summary(model, st_model, label_encoder):
        """Count total and trainable parameters."""
        total_model = sum(p.numel() for p in model.parameters())
        trainable_model = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_st = sum(p.numel() for p in st_model.parameters() if not p.requires_grad)

        print(f"\n  Model params — Trainable (BiLSTM+Attn+MLP): {trainable_model:,}")
        print(f"  Frozen encoder (SentenceTransformer): {frozen_st:,}")
        print(f"  Classes ({len(label_encoder.classes_)}): {list(label_encoder.classes_)}")
        return total_model, trainable_model

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
        plt.savefig(f'{output_path}han_overfitting_curve.png', dpi=150)
        plt.close()
        print(f"  Overfitting curve saved: {output_path}han_overfitting_curve.png")

    @staticmethod
    def measure_inference_time(model, dataloader, st_model, device, num_samples=1000):
        """Measure average inference time per sample."""
        model.eval()
        times = []
        batch_count = 0
        with torch.no_grad():
            for batch in dataloader:
                if batch_count >= num_samples // batch[0].__len__():
                    break
                embeddings, _, mask = han_collate(
                    batch, st_model, device, Config.ENCODE_BATCH_SIZE)
                torch.cuda.synchronize() if device == 'cuda' else None
                t0 = time.time()
                _, _ = model(embeddings, mask)
                torch.cuda.synchronize() if device == 'cuda' else None
                times.append(time.time() - t0)
                batch_count += 1

        avg_time_per_batch = np.mean(times)
        avg_samples = Config.BATCH_SIZE
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
        cm_norm = np.nan_to_num(cm_norm)
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=label_encoder.classes_,
                    yticklabels=label_encoder.classes_,
                    linewidths=0.5)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('HAN — Normalized Confusion Matrix')
        plt.tight_layout()
        plt.savefig(f'{output_path}han_cm.png', dpi=150)
        plt.close()
        print(f"  Confusion matrix saved: {output_path}han_cm.png")

    @staticmethod
    def plot_attention_map(model, dataloader, st_model, device,
                           num_examples=3, output_path='./'):
        """
        Visualize sentence-level attention weights.
        For each example, show which sentences the model attends to most.
        """
        model.eval()
        saved = 0
        with torch.no_grad():
            for batch in dataloader:
                if saved >= num_examples:
                    break
                embeddings, labels, mask = han_collate(
                    batch, st_model, device, Config.ENCODE_BATCH_SIZE)
                _, attn_weights = model(embeddings, mask)
                attn_weights = attn_weights.cpu().numpy()

                for i in range(min(3, embeddings.size(0))):
                    if saved >= num_examples:
                        break
                    aw = attn_weights[i]
                    valid_sents = mask[i].cpu().numpy()
                    sent_indices = np.where(valid_sents)[0]

                    fig, ax = plt.subplots(figsize=(12, 4))
                    x = np.arange(len(aw))
                    ax.bar(x, aw, color='steelblue', alpha=0.8)
                    ax.set_xlabel('Sentence index')
                    ax.set_ylabel('Attention weight')
                    ax.set_title(f'Sentence Attention — Sample {saved + 1} | Label={labels[i].item()}')
                    ax.set_xticks(x)
                    ax.set_xticklabels([str(int(idx)) for idx in x], fontsize=7)
                    ax.grid(True, alpha=0.3, axis='y')

                    # annotate valid sentence peaks
                    top_indices = sent_indices[np.argsort(aw[sent_indices])[-3:]]
                    for idx in top_indices:
                        ax.annotate(f'sent {idx}', xy=(idx, aw[idx]),
                                    xytext=(idx, aw[idx] + 0.02),
                                    fontsize=7, ha='center',
                                    arrowprops=dict(arrowstyle='->', color='red', lw=0.8))

                    plt.tight_layout()
                    plt.savefig(f'{output_path}han_attn_map_{saved + 1}.png', dpi=150)
                    plt.close()
                    saved += 1

        print(f"  Attention maps saved ({saved} examples): {output_path}han_attn_map_*.png")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 60)
    print("HAN-Inspired Classifier — Architecture Summary")
    print(f"  Pretrained model: {Config.MODEL_NAME}")
    print(f"  Max sentences: {Config.MAX_SENTENCES} | Max tokens/sentence: {Config.MAX_TOKEN}")
    print(f"  BiLSTM hidden: {Config.SENT_HIDDEN} | Dropout: {Config.DROPOUT}")
    print(f"  LR: {Config.LR} | Batch size: {Config.BATCH_SIZE}")
    print(f"  Loss: FocalLoss(gamma={Config.FOCAL_GAMMA}, alpha={Config.MANUAL_FOCAL_ALPHA})")
    print("=" * 60)

    DataResolver.resolve()

    print(f"\n  Data path:   {Config.DATA_PATH}")
    print(f"  Output path: {Config.OUTPUT_PATH}")

    # EDA
    train_df, val_df, test_df, label_encoder = DataSplits.load()
    EDAAugmentation.analyze(train_df, 'Train')
    EDAAugmentation.analyze(val_df, 'Val')
    EDAAugmentation.analyze(test_df, 'Test')

    # Load pretrained SentenceTransformer
    print(f"\n[1/8] Loading SentenceTransformer: {Config.MODEL_NAME}...")
    st_model = SentenceTransformer(Config.MODEL_NAME)
    st_model.to(Config.DEVICE)
    st_model.eval()
    st_model.max_seq_length = Config.MAX_TOKEN
    embed_dim = st_model.get_sentence_embedding_dimension()
    print(f"  Embedding dim: {embed_dim}")
    print(f"  Max seq length: {st_model.max_seq_length}")

    # Build model
    print(f"\n[2-4/8] Building HAN model & dataloaders...")
    model = HANClassifier(
        embed_dim=embed_dim,
        sent_hidden=Config.SENT_HIDDEN,
        num_classes=len(label_encoder.classes_),
        dropout=Config.DROPOUT,
    ).to(Config.DEVICE)

    train_loader, val_loader, test_loader = build_dataloaders(
        train_df, val_df, test_df)

    Finetuner.freeze_backbone(st_model)

    optimizer = Finetuner.build_optimizer(model)
    scheduler = Finetuner.build_scheduler(optimizer, train_loader)
    alpha_tensor = torch.tensor(
        Config.MANUAL_FOCAL_ALPHA, dtype=torch.float32).to(Config.DEVICE)
    criterion = FocalLoss(alpha=alpha_tensor, gamma=Config.FOCAL_GAMMA)

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Total params (HAN): {total_params:,} | Trainable: {trainable:,}")
    print(f"  Focal Alpha: {Config.MANUAL_FOCAL_ALPHA}")

    # Resume
    checkpoint_path = f'{Config.OUTPUT_PATH}han_checkpoint.pt'
    best_model_path = f'{Config.OUTPUT_PATH}han_best.pt'
    start_epoch = 0
    best_val_f1 = 0

    if os.path.exists(checkpoint_path):
        print(f"\n  [Resume] Found checkpoint: {checkpoint_path}")
        try:
            ckpt = torch.load(checkpoint_path, map_location=Config.DEVICE, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch = ckpt.get('epoch', 0) + 1
            best_val_f1 = ckpt.get('best_val_f1', 0)
            print(f"  Resumed from epoch {start_epoch}, best_val_f1={best_val_f1:.4f}")
        except Exception:
            print(f"  [Resume] Load failed — starting from scratch")
            start_epoch = 0
            best_val_f1 = 0
    elif os.path.exists(best_model_path):
        print(f"\n  [Resume] Found {best_model_path} — loading weights only.")
        model.load_state_dict(torch.load(best_model_path,
                                         map_location=Config.DEVICE, weights_only=True))
        print(f"  Loaded best model weights.")
    else:
        print(f"\n  [Start] No checkpoint or best model found — training from scratch.")

    # Training
    print(f"\n[5-6/8] Training (epochs {start_epoch + 1}–{Config.EPOCHS})...")
    trainer = Trainer(model, train_loader, val_loader, st_model,
                      optimizer, criterion, Config.DEVICE)

    history = {'train_loss': [], 'val_loss': [],
               'train_acc': [], 'val_acc': [],
               'train_f1': [], 'val_f1': []}
    patience = 3
    patience_counter = 0

    for epoch in range(start_epoch, Config.EPOCHS):
        t_ep = time.time()

        train_loss, train_acc, train_f1, train_f1_per = trainer.train_epoch()
        scheduler.step()

        val_metrics = Evaluator.evaluate(
            model, val_loader, st_model, Config.DEVICE, alpha_tensor)
        val_loss = val_metrics['loss']
        val_acc = val_metrics['accuracy']
        val_f1 = val_metrics['f1_macro']
        val_f1_per = val_metrics['f1_per_class']

        t_dur = time.time() - t_ep
        print(f"Epoch {epoch+1:2d}/{Config.EPOCHS} ({t_dur:.1f}s) | "
              f"LR:{optimizer.param_groups[0]['lr']:.2e} | "
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
        model, test_loader, st_model, Config.DEVICE, alpha_tensor)
    test_preds = np.array(test_metrics['all_preds'])
    test_labels_np = np.array(test_metrics['all_labels'])

    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Precision (macro): {test_metrics['precision_macro']:.4f}")
    print(f"  Recall (macro):    {test_metrics['recall_macro']:.4f}")
    print(f"  F1 (macro):       {test_metrics['f1_macro']:.4f}")
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
    Analyzer.model_summary(model, st_model, label_encoder)
    Analyzer.plot_overfitting(history, Config.OUTPUT_PATH)
    Analyzer.measure_inference_time(model, test_loader, st_model, Config.DEVICE)
    Analyzer.plot_confusion_matrix(test_labels_np, test_preds,
                                   label_encoder, Config.OUTPUT_PATH)
    Analyzer.plot_attention_map(model, test_loader, st_model, Config.DEVICE,
                                num_examples=3, output_path=Config.OUTPUT_PATH)

    # Save predictions
    test_df['pred'] = label_encoder.inverse_transform(test_preds)
    test_df.to_csv(f'{Config.OUTPUT_PATH}han_predictions.csv', index=False)
    print(f"\n  Predictions saved: {Config.OUTPUT_PATH}han_predictions.csv")
    print(f"\nDone! Best model: {best_model_path}")


if __name__ == '__main__':
    main()

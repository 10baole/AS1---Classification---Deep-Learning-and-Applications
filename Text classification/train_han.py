#!/usr/bin/env python3
"""
HAN-inspired Classifier — Legal Text Classification
Pipeline: Document → Split sentences → Encode (SentenceTransformer)
        → BiLSTM → Sentence Attention → MLP → Output
Classes:
  1. Config        — hyper-parameters, paths
  2. Dataset       — sentence splitting
  3. DataLoaders   — train / val / test splits
  4. SentenceAttention + HANClassifier — Backbone + Head
  5. FocalLoss + Trainer
  6. Finetuner     — freeze backbone
  7. Evaluator     — Accuracy, Precision, Recall, F1
  8. Analyzer      — params, overfitting, inference, attention viz
"""

import os
import pickle
import time
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
# 1. CONFIG
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
    FOCAL_ALPHA = [0.4, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0]


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
_ABBRS = [
    r'\bMr\.\s', r'\bMrs\.\s', r'\bDr\.\s', r'\bProf\.\s',
    r'\bvs\.\s', r'\bv\.\s', r'\bNo\.\s', r'\bU\.S\.\s',
    r'\bU\.S\.C\.\s', r'\bF\.\d+d?\s', r'\bF\.Supp\.\s',
    r'\bS\.Ct\.\s', r'\bL\.\s*Ed\.\s', r'\bIll\.\s',
    r'\bApp\.\s', r'\bCir\.\s', r'\bDist\.\s',
    r'\bCorp\.\s', r'\bInc\.\s', r'\bLtd\.\s', r'\bCo\.\s',
]


def split_into_sentences(text, max_sentences=50):
    text_mod = text
    for abbr in _ABBRS:
        text_mod = re.sub(abbr, lambda m: m.group(0).replace('.', '<DOT>'), text_mod)
    sentences = re.split(r'(?<=[.!?])\s+', text_mod)
    cleaned = []
    for sent in sentences:
        sent = sent.replace('<DOT>', '.').strip()
        if len(sent) > 15:
            cleaned.append(sent)
    return cleaned[:max_sentences]


class HANDataset(Dataset):
    def __init__(self, texts, labels, max_sentences):
        self.texts = texts
        self.labels = [torch.tensor(l, dtype=torch.long) for l in labels]
        self.max_sentences = max_sentences

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        sentences = split_into_sentences(str(self.texts[idx]), self.max_sentences)
        while len(sentences) < self.max_sentences:
            sentences.append('')
        return sentences, self.labels[idx]


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

        for name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            # Keep text as string. Removed dataset-level drop of samples <=20 chars
            # to avoid redundant filtering — sentence-level filter (len>15) remains
            # in split_into_sentences() which is the intended behavior.
            df['text'] = df['text'].astype(str)

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


def st_collate(batch, st_model, device, encode_batch_size=512, log_prefix=''):
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
            if s.strip():
                flat_sentences.append(s.strip())
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
        except Exception as e:
            print(f"[WARN] st_collate encode failed {log_prefix}— using zero embeddings. Error: {e}")

    batch_vectors = torch.zeros(batch_size, max_sentences, embed_dim, device=device)
    batch_mask = torch.zeros(batch_size, max_sentences, dtype=torch.bool, device=device)

    for vec_idx, (doc_id, sent_id) in enumerate(flat_indices):
        batch_vectors[doc_id, sent_id] = flat_vectors[vec_idx]
        batch_mask[doc_id, sent_id] = True

    labels = torch.stack(labels_list).to(device)
    return batch_vectors, labels, batch_mask


def build_dataloaders(train_df, val_df, test_df):
    train_ds = HANDataset(train_df['text'].tolist(), train_df['label_encoded'].tolist(), Config.MAX_SENTENCES)
    val_ds = HANDataset(val_df['text'].tolist(), val_df['label_encoded'].tolist(), Config.MAX_SENTENCES)
    test_ds = HANDataset(test_df['text'].tolist(), test_df['label_encoded'].tolist(), Config.MAX_SENTENCES)

    collate_fn = lambda b: b
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn)

    print(f"  Train batches: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")
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
    def __init__(self, embed_dim, sent_hidden, num_classes, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            embed_dim, sent_hidden, num_layers=1,
            bidirectional=True, batch_first=True)
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
        return self.fc(doc_vec), attn_weights


# =============================================================================
# 6. FINETUNING
# =============================================================================
class Finetuner:
    @staticmethod
    def freeze_backbone(st_model):
        for p in st_model.parameters():
            p.requires_grad = False
        total = sum(p.numel() for p in st_model.parameters())
        print(f"  SentenceTransformer frozen ({total:,} params)")

    @staticmethod
    def build_optimizer(model):
        return torch.optim.AdamW(
            model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)

    @staticmethod
    def build_scheduler(optimizer, train_loader):
        # Scheduler steps once per epoch in main loop, so T_max is in epochs.
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=Config.EPOCHS, eta_min=1e-6)


# =============================================================================
# 7. TRAINER
# =============================================================================
class Trainer:
    def __init__(self, model, train_loader, st_model, optimizer, criterion, device):
        self.model = model
        self.train_loader = train_loader
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
            embeddings, labels, mask = st_collate(
                batch, self.st_model, self.device, Config.ENCODE_BATCH_SIZE, log_prefix='[train] ')
            self.optimizer.zero_grad()

            logits, _ = self.model(embeddings, mask)
            loss = self.criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            batch_n = labels.size(0)
            total_loss += loss.item() * batch_n
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += batch_n
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.4f}'})

        f1_macro = f1_score(all_labels, all_preds, average='macro')
        f1_per = f1_score(all_labels, all_preds, average=None)
        return total_loss / max(total, 1), correct / total, f1_macro, f1_per


# =============================================================================
# 8. EVALUATOR
# =============================================================================
class Evaluator:
    @staticmethod
    def evaluate(model, dataloader, st_model, device, alpha_tensor, num_classes):
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds, all_labels = [], []
        criterion = FocalLoss(alpha=alpha_tensor, gamma=Config.FOCAL_GAMMA)

        pbar = tqdm(dataloader, desc="Eval ", unit="batch", ncols=100)
        with torch.no_grad():
            for batch in pbar:
                embeddings, labels, mask = st_collate(
                    batch, st_model, device, Config.ENCODE_BATCH_SIZE, log_prefix='[eval] ')
                logits, _ = model(embeddings, mask)
                loss = criterion(logits, labels)

                batch_n = labels.size(0)
                total_loss += loss.item() * batch_n
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += batch_n
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        cm = confusion_matrix(all_labels, all_preds, labels=range(num_classes))
        total_samples = cm.sum()
        ovr_acc_per_class = []
        for i in range(num_classes):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = total_samples - tp - fn - fp
            ovr_acc_per_class.append((tp + tn) / total_samples if total_samples > 0 else 0.0)

        precision_per = precision_score(all_labels, all_preds, average=None, zero_division=0)
        recall_per = recall_score(all_labels, all_preds, average=None, zero_division=0)
        f1_per = f1_score(all_labels, all_preds, average=None, zero_division=0)

        return {
            'loss': total_loss / max(total, 1),
            'accuracy': correct / total,
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


# =============================================================================
# 9. ANALYZER
# =============================================================================
class Analyzer:
    @staticmethod
    def model_summary(model, st_model, label_encoder):
        total_model = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_st = sum(p.numel() for p in st_model.parameters() if not p.requires_grad)
        print(f"\n  Trainable (BiLSTM+Attn+MLP): {trainable:,} | Frozen ST: {frozen_st:,}")
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
        plt.savefig(f'{output_path}han_overfitting_curve.png', dpi=150)
        plt.close()
        print(f"  Overfitting curve: {output_path}han_overfitting_curve.png")

    @staticmethod
    def measure_inference_time(model, dataloader, st_model, device, num_samples=1000):
        model.eval()
        times = []
        batch_count = 0
        with torch.no_grad():
            for batch in dataloader:
                if batch_count >= num_samples // Config.BATCH_SIZE:
                    break
                embeddings, _, mask = st_collate(
                    batch, st_model, device, Config.ENCODE_BATCH_SIZE)
                torch.cuda.synchronize() if device == 'cuda' else None
                t0 = time.time()
                _, _ = model(embeddings, mask)
                torch.cuda.synchronize() if device == 'cuda' else None
                times.append(time.time() - t0)
                batch_count += 1
        avg_batch = np.mean(times)
        avg_sample = avg_batch / Config.BATCH_SIZE
        print(f"\n  Inference — Per batch: {avg_batch*1000:.2f}ms | Per sample: {avg_sample*1000:.2f}ms")

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, label_encoder, output_path):
        num_classes = len(label_encoder.classes_)
        cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
        plt.figure(figsize=(12, 10))
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
        cm_norm = np.nan_to_num(cm_norm)
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=label_encoder.classes_,
                    yticklabels=label_encoder.classes_, linewidths=0.5)
        plt.xlabel('Predicted'); plt.ylabel('True')
        plt.title('HAN — Normalized Confusion Matrix')
        plt.tight_layout()
        plt.savefig(f'{output_path}han_cm.png', dpi=150)
        plt.close()
        print(f"  Confusion matrix: {output_path}han_cm.png")

    @staticmethod
    def plot_attention_map(model, dataloader, st_model, device, num_examples=3, output_path='./'):
        model.eval()
        saved = 0
        with torch.no_grad():
            for batch in dataloader:
                if saved >= num_examples:
                    break
                embeddings, labels, mask = st_collate(
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
                    ax.set_title(f'Sentence Attention — Sample {saved+1} | Label={labels[i].item()}')
                    ax.set_xticks(x)
                    ax.set_xticklabels([str(int(idx)) for idx in x], fontsize=7)
                    ax.grid(True, alpha=0.3, axis='y')

                    top_indices = sent_indices[np.argsort(aw[sent_indices])[-3:]]
                    for idx in top_indices:
                        ax.annotate(f'sent {idx}', xy=(idx, aw[idx]),
                                    xytext=(idx, aw[idx] + 0.02),
                                    fontsize=7, ha='center',
                                    arrowprops=dict(arrowstyle='->', color='red', lw=0.8))

                    plt.tight_layout()
                    plt.savefig(f'{output_path}han_attn_map_{saved+1}.png', dpi=150)
                    plt.close()
                    saved += 1
        print(f"  Attention maps ({saved}): {output_path}han_attn_map_*.png")


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

    # Load SentenceTransformer
    print(f"\n[1] Loading SentenceTransformer: {Config.MODEL_NAME}...")
    st_model = SentenceTransformer(Config.MODEL_NAME)
    st_model.to(Config.DEVICE)
    st_model.eval()
    st_model.max_seq_length = Config.MAX_TOKEN
    embed_dim = st_model.get_sentence_embedding_dimension()
    print(f"  Embedding dim: {embed_dim} | Max seq: {st_model.max_seq_length}")

    Finetuner.freeze_backbone(st_model)

    # Build model
    print(f"\n[2] Building HAN model...")
    model = HANClassifier(
        embed_dim=embed_dim,
        sent_hidden=Config.SENT_HIDDEN,
        num_classes=len(label_encoder.classes_),
        dropout=Config.DROPOUT,
    ).to(Config.DEVICE)

    train_loader, val_loader, test_loader = build_dataloaders(train_df, val_df, test_df)

    optimizer = Finetuner.build_optimizer(model)
    scheduler = Finetuner.build_scheduler(optimizer, train_loader)
    num_classes = len(label_encoder.classes_)
    if len(Config.FOCAL_ALPHA) != num_classes:
        raise ValueError(
            f"FOCAL_ALPHA length ({len(Config.FOCAL_ALPHA)}) must equal num_classes ({num_classes}).")
    alpha_tensor = torch.tensor(Config.FOCAL_ALPHA, dtype=torch.float32).to(Config.DEVICE)
    criterion = FocalLoss(alpha=alpha_tensor, gamma=Config.FOCAL_GAMMA)

    # Architecture summary
    print("\n" + "=" * 60)
    print("HAN-Inspired Classifier")
    print(f"  Encoder: {Config.MODEL_NAME} (frozen)")
    print(f"  Max sentences: {Config.MAX_SENTENCES} | BiLSTM hidden: {Config.SENT_HIDDEN}")
    print(f"  Batch: {Config.BATCH_SIZE} | LR: {Config.LR}")
    print(f"  Loss: FocalLoss(gamma={Config.FOCAL_GAMMA}, alpha={Config.FOCAL_ALPHA})")
    print("=" * 60)

    # Resume
    ckpt_path = f'{Config.OUTPUT_PATH}han_checkpoint.pt'
    best_path = f'{Config.OUTPUT_PATH}han_best.pt'
    start_epoch = 0
    best_val_f1 = 0.0

    if os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location=Config.DEVICE, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch = ckpt.get('epoch', 0) + 1
            best_val_f1 = ckpt.get('best_val_f1', 0.0)
            print(f"  [Resume] epoch={start_epoch}, best_val_f1={best_val_f1:.4f}")
        except Exception:
            print(f"  [Resume] Failed — training from scratch")
            start_epoch = 0
            best_val_f1 = 0.0

    # Training
    print(f"\n[3] Training (epochs {start_epoch+1}–{Config.EPOCHS})...")
    trainer = Trainer(model, train_loader, st_model, optimizer, criterion, Config.DEVICE)

    history = {'train_loss': [], 'val_loss': [],
               'train_acc': [], 'val_acc': [],
               'train_f1': [], 'val_f1': []}
    patience = 3
    patience_counter = 0

    for epoch in range(start_epoch, Config.EPOCHS):
        t_ep = time.time()
        train_loss, train_acc, train_f1, _ = trainer.train_epoch()
        scheduler.step()

        val_metrics = Evaluator.evaluate(
            model, val_loader, st_model, Config.DEVICE, alpha_tensor, num_classes)
        val_loss = val_metrics['loss']
        val_acc = val_metrics['accuracy']
        val_f1 = val_metrics['f1_macro']
        val_f1_per = val_metrics['f1_per_class']
        t_dur = time.time() - t_ep

        print(f"Epoch {epoch+1:2d}/{Config.EPOCHS} ({t_dur:.1f}s) | "
              f"LR:{optimizer.param_groups[0]['lr']:.2e} | "
              f"Tr L:{train_loss:.4f} A:{train_acc:.4f} F:{train_f1:.4f} | "
              f"Val L:{val_loss:.4f} A:{val_acc:.4f} F:{val_f1:.4f}")
        print(f"  Val F1/class: {[f'{f:.3f}' for f in val_f1_per]}")

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)

        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_f1': best_val_f1}, ckpt_path)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_path)
            patience_counter = 0
            print(f"  *** New best val F1: {best_val_f1:.4f} ***")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load(best_path, map_location=Config.DEVICE, weights_only=True))

    # Evaluate
    print(f"\n[4] Final Evaluation on Test Set")
    print("=" * 60)
    test_metrics = Evaluator.evaluate(
        model, test_loader, st_model, Config.DEVICE, alpha_tensor, num_classes)
    test_preds = np.array(test_metrics['all_preds'])
    test_labels_np = np.array(test_metrics['all_labels'])

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
    Analyzer.model_summary(model, st_model, label_encoder)
    Analyzer.plot_overfitting(history, Config.OUTPUT_PATH)
    Analyzer.measure_inference_time(model, test_loader, st_model, Config.DEVICE)
    Analyzer.plot_confusion_matrix(test_labels_np, test_preds, label_encoder, Config.OUTPUT_PATH)
    Analyzer.plot_attention_map(model, test_loader, st_model, Config.DEVICE,
                                 num_examples=3, output_path=Config.OUTPUT_PATH)

    test_df_out = test_df.copy()
    test_df_out['pred'] = label_encoder.inverse_transform(test_preds)
    test_df_out.to_csv(f'{Config.OUTPUT_PATH}han_predictions.csv', index=False)
    print(f"\n  Predictions: {Config.OUTPUT_PATH}han_predictions.csv")
    print(f"  Best model: {best_path}")


if __name__ == '__main__':
    main()

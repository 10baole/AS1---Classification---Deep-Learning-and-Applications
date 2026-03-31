#!/usr/bin/env python3
"""
HAN-inspired for Legal Text Classification

- Sentence encoder: tomaarsen/glove-bilstm-sts (frozen, pretrained) → 2048-d embeddings
- Sentence-level BiLSTM + Attention pooling → document vector
- MLP classifier
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer


# ============================================================
#  CONFIG
# ============================================================
class Config:
    # Mặc định là thư mục local, script sẽ tự động chuyển sang Drive nếu chạy trên Colab
    DATA_PATH = './data/processed/'
    OUTPUT_PATH = './output-train-han/'

    # Pretrained model
    MODEL_NAME = 'tomaarsen/glove-bilstm-sts'

    SENT_HIDDEN = 128
    MAX_SENTENCES = 50
    MAX_TOKEN = 512        
    DROPOUT = 0.3

    BATCH_SIZE = 256
    ENCODE_BATCH_SIZE = 512
    EPOCHS = 10
    LR = 1e-3
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    FOCAL_GAMMA = 2.0
    # [Civil=0, Corporate=1, CourtOfClaims=2, Criminal=3, Other=4, Probate=5, Property=6]
    MANUAL_FOCAL_ALPHA = [0.2, 2.0, 1.0, 0.5, 1.8, 1.2, 1.0]


# ============================================================
#  FOCAL LOSS
# ============================================================
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


# ============================================================
#  SENTENCE SPLITTING
# ============================================================
def split_into_sentences(text, max_sentences=50):
    import re

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


# ============================================================
#  DATA LOADING
# ============================================================
def load_data():
    print(f"Loading data from {Config.DATA_PATH}...")

    if not os.path.exists(Config.DATA_PATH):
        raise FileNotFoundError(f"Directory not found: {Config.DATA_PATH}")

    files = os.listdir(Config.DATA_PATH)
    print(f"  Available files: {files}")

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

    # Tìm label encoder — also check Kaggle input dir
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
#  DATASET  — lưu sentences thô, encode trong __getitem__
# ============================================================
class HANDataset(Dataset):
    """Mỗi sample: list các câu (string). Encode khi load."""

    def __init__(self, texts, labels, max_sentences):
        self.texts = texts
        self.labels = labels
        self.max_sentences = max_sentences

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        sentences = split_into_sentences(str(self.texts[idx]), self.max_sentences)
        # Đảm bảo luôn có đủ max_sentences (pad bằng chuỗi rỗng)
        while len(sentences) < self.max_sentences:
            sentences.append('')
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return sentences, label


# ============================================================
#  COLLATE — batch-encode all sentences at once
# ============================================================
def han_collate(batch, st_model, device, encode_batch_size=32):
    """Encode all sentences in the batch with a single encode() call.

    Args:
        batch: list of (sentences_list, label)
        st_model: frozen SentenceTransformer
        device: torch device
        encode_batch_size: sub-batch size inside encode(), tuned to avoid OOM
    Returns:
        sentence_vectors: (batch_size, max_sentences, embed_dim)
        labels: (batch_size,)
        mask: (batch_size, max_sentences) — True = câu hợp lệ
    """
    batch_size = len(batch)
    labels_list = []
    all_sentences = []

    for sentences, label in batch:
        labels_list.append(label)
        all_sentences.append(sentences)

    max_sentences = len(all_sentences[0])
    embed_dim = st_model.get_sentence_embedding_dimension()

    # Build flat sentence list + index map
    flat_sentences = []
    flat_indices = []  # (doc_id, sent_id)
    for doc_id, doc in enumerate(all_sentences):
        for sent_id, s in enumerate(doc):
            s_stripped = s.strip()
            if s_stripped:
                flat_sentences.append(s_stripped)
                flat_indices.append((doc_id, sent_id))

    # Encode all sentences in one call
    flat_vectors = torch.zeros(
        len(flat_sentences), embed_dim, dtype=torch.float32, device=device
    )
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
            pass  # fall back to zeros for failed indices

    # Scatter into per-document tensors
    batch_vectors = torch.zeros(batch_size, max_sentences, embed_dim, device=device)
    batch_mask = torch.zeros(batch_size, max_sentences, dtype=torch.bool, device=device)

    for vec_idx, (doc_id, sent_id) in enumerate(flat_indices):
        batch_vectors[doc_id, sent_id] = flat_vectors[vec_idx]
        batch_mask[doc_id, sent_id] = True

    labels = torch.stack(labels_list).to(device)
    return batch_vectors, labels, batch_mask


# ============================================================
#  MODEL — Sentence Embeddings + BiLSTM + Attention + Classifier
#  (HAN-inspired: frozen sentence encoder + BiLSTM + sentence attention)
#  Tầng dưới: sentence representation có sẵn từ pretrained encoder
#  Tầng trên: sentence-level BiLSTM + sentence attention giống nửa trên của HAN
# ============================================================
class SentenceAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, lstm_out, mask=None):
        # lstm_out: (batch, seq, hidden_dim)  — output của BiLSTM
        # mask: (batch, seq) — True = câu hợp lệ
        w = self.attention(lstm_out)  # (batch, seq, 1)
        if mask is not None:
            w = w.squeeze(-1)
            w = w.masked_fill(~mask, -1e9)
            w = w.unsqueeze(-1)
        w = torch.softmax(w, dim=1)                   # (batch, seq, 1)
        context = torch.sum(w * lstm_out, dim=1)        # (batch, hidden_dim)
        return context, w.squeeze(-1)


class HANClassifier(nn.Module):
    """Sentence embeddings → BiLSTM → Sentence Attention → Classification."""

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
        # sentence_vectors: (batch, max_sentences, embed_dim)
        lstm_out, _ = self.lstm(sentence_vectors)          # (batch, seq, hidden*2)
        lstm_out = self.dropout_lstm(lstm_out)             # explicit dropout (num_layers=1 → LSTM ignores dropout param)
        doc_vec, _ = self.attention(lstm_out, mask)       # (batch, hidden*2)
        doc_vec = self.dropout(doc_vec)
        logits = self.fc(doc_vec)
        return logits


# ============================================================
#  TRAINING
# ============================================================
def train_epoch(model, dataloader, optimizer, st_model, device, alpha_tensor, encode_batch_size=32):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    criterion = FocalLoss(
        alpha=alpha_tensor,
        gamma=Config.FOCAL_GAMMA,
    )

    for batch in dataloader:
        embeddings, labels, mask = han_collate(batch, st_model, device, encode_batch_size)
        optimizer.zero_grad()
        logits = model(embeddings, mask)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_per_class = f1_score(all_labels, all_preds, average=None)
    return total_loss / len(dataloader), correct / total, f1_macro, f1_per_class, all_preds, all_labels


def evaluate(model, dataloader, st_model, device, alpha_tensor, encode_batch_size=32):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    criterion = FocalLoss(
        alpha=alpha_tensor,
        gamma=Config.FOCAL_GAMMA,
    )

    with torch.no_grad():
        for batch in dataloader:
            embeddings, labels, mask = han_collate(batch, st_model, device, encode_batch_size)
            logits = model(embeddings, mask)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_per_class = f1_score(all_labels, all_preds, average=None)
    return total_loss / len(dataloader), correct / total, f1_macro, f1_per_class, all_preds, all_labels


# ============================================================
#  MAIN
# ============================================================
def main():
    print("=" * 60)
    print("HAN-Inspired Training (SentenceEncoder + BiLSTM + Attention)")
    print(f"Pretrained model: {Config.MODEL_NAME}")
    print("=" * 60)

    # Auto-detect platform — check Kaggle FIRST to avoid Colab check false-positive on Kaggle
    on_colab_notebook = False
    on_kaggle = os.path.exists('/kaggle')
    on_lightning = os.path.exists('/teamspace/studios/this_studio/')

    if on_kaggle:
        print("\n[!] Running on Kaggle.")
        Config.DATA_PATH   = '/kaggle/input/datasets/phantrntngvyk64cntt/data-processed/'
        Config.OUTPUT_PATH = '/kaggle/working/output-train-han/'
    elif on_lightning:
        print("\n[!] Running on Lightning AI.")
        base = '/teamspace/studios/this_studio/'
        Config.DATA_PATH   = os.path.join(base, 'data/processed/')
        Config.OUTPUT_PATH = os.path.join(base, 'output-train-han/')
    else:
        # Only try Colab if definitely NOT on Kaggle
        try:
            from google.colab import drive
            import IPython
            ip = IPython.get_ipython()
            if ip is not None and ip.kernel is not None:
                on_colab_notebook = True
        except (ImportError, AttributeError):
            pass

        if on_colab_notebook:
            print("\n[!] Running on Google Colab (notebook). Mounting Drive...")
            drive.mount('/content/drive')
            Config.OUTPUT_PATH = '/content/drive/MyDrive/output-train-han/'
            if os.path.exists('/content/drive/MyDrive/data/processed/'):
                Config.DATA_PATH = '/content/drive/MyDrive/data/processed/'
            elif os.path.exists('/content/data/processed/'):
                Config.DATA_PATH = '/content/data/processed/'
        else:
            # Local paths
            possible_paths = [
                './data/masked-15k/',
                './data/processed/',
                '../data/processed/'
            ]
            for path in possible_paths:
                if os.path.exists(os.path.join(path, 'train.csv')):
                    Config.DATA_PATH = path
                    break
    
    os.makedirs(Config.OUTPUT_PATH, exist_ok=True)
    if not Config.OUTPUT_PATH.endswith('/'):
        Config.OUTPUT_PATH += '/'
    
    print(f"  Data path: {Config.DATA_PATH}")
    print(f"  Output path: {Config.OUTPUT_PATH}")

    # ---- 1. Load pretrained SentenceTransformer ----
    print("\n[1/6] Loading pretrained SentenceTransformer...")
    st_model = SentenceTransformer(Config.MODEL_NAME)
    st_model.to(Config.DEVICE)
    st_model.eval()
    st_model.max_seq_length = Config.MAX_TOKEN   # truncate mỗi câu ở 512 tokens
    embed_dim = st_model.get_sentence_embedding_dimension()
    print(f"  Embedding dim: {embed_dim}")
    print(f"  Max seq length: {st_model.max_seq_length}")

    # ---- 2. Load data ----
    print("\n[2/6] Loading data...")
    train_df, val_df, test_df, label_encoder = load_data()
    num_classes = len(label_encoder.classes_)

    # ---- 3. Create datasets ----
    print("\n[3/6] Creating datasets...")
    train_dataset = HANDataset(train_df['text'].tolist(),
                                train_df['label_encoded'].tolist(),
                                Config.MAX_SENTENCES)
    val_dataset = HANDataset(val_df['text'].tolist(),
                              val_df['label_encoded'].tolist(),
                              Config.MAX_SENTENCES)
    test_dataset = HANDataset(test_df['text'].tolist(),
                               test_df['label_encoded'].tolist(),
                               Config.MAX_SENTENCES)

    # Dùng collate_fn để encode sentences
    collate_fn = lambda b: b  # trả về list of (sentences, label)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE,
                              shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE,
                             shuffle=False, num_workers=0, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE,
                              shuffle=False, num_workers=0, collate_fn=collate_fn)

    print(f"  Train batches: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")

    # ---- 4. Create model ----
    print("\n[4/6] Creating HAN-inspired model (BiLSTM + Attention)...")
    model = HANClassifier(
        embed_dim=embed_dim,
        sent_hidden=Config.SENT_HIDDEN,
        num_classes=num_classes,
        dropout=Config.DROPOUT,
    ).to(Config.DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,} | Trainable: {trainable:,}")
    print(f"  (Pretrained encoder is frozen — BiLSTM + sentence attention train)")

    # Class weights → dùng MANUAL_FOCAL_ALPHA
    alpha_tensor = torch.tensor(Config.MANUAL_FOCAL_ALPHA,
                                dtype=torch.float32).to(Config.DEVICE)
    print(f"  Focal Alpha: {Config.MANUAL_FOCAL_ALPHA}")
    print(f"  Encode batch size: {Config.ENCODE_BATCH_SIZE}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=0.01)

    # ---- 4. Resume from checkpoint ----
    start_epoch = 0
    best_val_f1 = 0
    checkpoint_path = f'{Config.OUTPUT_PATH}han_checkpoint.pt'
    best_model_path = f'{Config.OUTPUT_PATH}han_best.pt'

    if os.path.exists(checkpoint_path):
        print(f"\n[Resume] Found checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=Config.DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_f1 = ckpt.get('best_val_f1', 0)
        print(f"  Resumed from epoch {start_epoch}, best_val_f1={best_val_f1:.4f}")
    elif os.path.exists(best_model_path):
        print(f"\n[Resume] Found {best_model_path} — loading weights only.")
        model.load_state_dict(torch.load(best_model_path, map_location=Config.DEVICE, weights_only=True))
        print(f"  Loaded best model weights. Training will overwrite with new runs.")
    else:
        print("\n[Start] No checkpoint or best model found — training from scratch.")

    # ---- 5. Training ----
    print("\n[5/6] Training...")
    patience = 3
    patience_counter = 0

    for epoch in range(start_epoch, Config.EPOCHS):
        train_loss, train_acc, train_f1, train_f1_per, _, _ = train_epoch(
            model, train_loader, optimizer, st_model, Config.DEVICE, alpha_tensor,
            Config.ENCODE_BATCH_SIZE
        )
        val_loss, val_acc, val_f1, val_f1_per, val_preds, val_labels = evaluate(
            model, val_loader, st_model, Config.DEVICE, alpha_tensor,
            Config.ENCODE_BATCH_SIZE
        )

        print(f"Epoch {epoch+1:2d}/{Config.EPOCHS} | "
              f"Train Loss:{train_loss:.4f} Acc:{train_acc:.4f} F1:{train_f1:.4f} | "
              f"Val Loss:{val_loss:.4f} Acc:{val_acc:.4f} F1:{val_f1:.4f}")
        print(f"  Val F1 per class: {[f'{f:.3f}' for f in val_f1_per]}")

        # Save checkpoint after each epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_f1': best_val_f1,
        }, checkpoint_path)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), f'{Config.OUTPUT_PATH}han_best.pt')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Load best
    model.load_state_dict(torch.load(f'{Config.OUTPUT_PATH}han_best.pt', weights_only=True))

    # ---- 6. Final evaluation ----
    print("\n" + "=" * 60)
    print("Final Evaluation on Test Set")
    print("=" * 60)

    test_loss, test_acc, test_f1, test_f1_per, test_preds, test_labels = evaluate(
        model, test_loader, st_model, Config.DEVICE, alpha_tensor,
        Config.ENCODE_BATCH_SIZE
    )

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Macro F1: {test_f1:.4f}")
    print(f"Test F1 per class: {[f'{f:.3f}' for f in test_f1_per]}")
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds,
                                target_names=label_encoder.classes_))

    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds,
                          labels=range(len(label_encoder.classes_)))
    print("Per-class Accuracy:")
    for i, name in enumerate(label_encoder.classes_):
        total = cm[i].sum()
        correct = cm[i, i]
        print(f"  {name}: {correct}/{total} = {correct/total:.3f}")

    # Save confusion matrix
    plt.figure(figsize=(12, 10))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_,
                linewidths=0.5)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('HAN — Normalized Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f'{Config.OUTPUT_PATH}han_cm.png', dpi=150)
    plt.close()
    print(f"\nConfusion matrix saved to {Config.OUTPUT_PATH}han_cm.png")

    # Save predictions
    test_df['pred'] = label_encoder.inverse_transform(test_preds)
    test_df.to_csv(f'{Config.OUTPUT_PATH}han_predictions.csv', index=False)
    print(f"Predictions saved to {Config.OUTPUT_PATH}han_predictions.csv")


if __name__ == '__main__':
    main()

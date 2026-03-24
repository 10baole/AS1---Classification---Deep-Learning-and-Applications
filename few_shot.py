import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from transformers import RobertaModel, RobertaTokenizerFast, ViTImageProcessor, ViTModel
from transformers.utils import logging as hf_logging

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING",   "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS",      "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
hf_logging.set_verbosity_error()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path):
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML mapping.")
    return config


def resolve_cfg(config):
    paths_cfg   = config.get("paths")
    model_cfg   = config.get("model")
    runtime_cfg = config.get("runtime")
    train_cfg   = config.get("training")
    labels      = config.get("labels")

    train_path = Path(paths_cfg.get("train_path", "")) 
    val_path   = Path(paths_cfg.get("val_path",   ""))
    test_path  = Path(paths_cfg.get("test_path",  ""))
    images_dir = Path(paths_cfg.get("images_dir"))
    save_path  = Path(train_cfg.get("save_path", ""))

    seed = int(runtime_cfg.get("seed"))
    training_batch = int(train_cfg.get("batch_size"))
    infer_batch    = int(runtime_cfg.get("batch_size"))
    epochs         = int(train_cfg.get("epochs"))
    shots          = int(runtime_cfg.get("shots_per_class"))

    text_model      = model_cfg.get("text_model")
    freeze_text     = bool(model_cfg.get("freeze_text", False))
    max_text_length = int(train_cfg.get("max_text_length"))
    
    vision_model    = model_cfg.get("vision_model")
    freeze_image    = bool(model_cfg.get("freeze_image", False))
    image_size      = int(train_cfg.get("image_size"))

    learning_rate     = float(train_cfg.get("lr"))
    weight_decay      = float(train_cfg.get("weight_decay"))
    mlp_hidden_dim    = int(model_cfg.get("mlp_hidden_dim"))
    fusion_hidden_dim = int(model_cfg.get("fusion_hidden_dim"))
    dropout           = float(model_cfg.get("dropout"))

    return {
        "images_dir": images_dir,
        "train_path": train_path,
        "val_path": val_path,
        "test_path": test_path,
        "save_path": save_path,
        "labels": labels,
        "seed": seed,
        "train_batch_size": training_batch,
        "infer_batch_size": infer_batch,
        "epochs": epochs,
        "lr": learning_rate,
        "weight_decay": weight_decay,
        "max_text_length": max_text_length,
        "image_size": image_size,
        "shots_per_class": shots,
        "vision_model": vision_model,
        "text_model": text_model,
        "freeze_image": freeze_image,
        "freeze_text": freeze_text,
        "mlp_hidden_dim": mlp_hidden_dim,
        "fusion_hidden_dim": fusion_hidden_dim,
        "dropout": dropout,
    }


def resolve_image_path(images_dir: Path, image_id: str):
    if not image_id:
        return None

    image_id = str(image_id)
    direct = images_dir / image_id
    if direct.exists():
        return direct

    with_jpg = images_dir / f"{image_id}.jpg"
    if with_jpg.exists():
        return with_jpg

    for ext in (".jpeg", ".png", ".webp"):
        candidate = images_dir / f"{image_id}{ext}"
        if candidate.exists():
            return candidate

    matches = list(images_dir.glob(f"{image_id}.*"))
    if matches:
        return matches[0]
    return None
    

def load_items(json_path, images_dir, labels):
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected list in {json_path}, got {type(data)}")

    items = []
    for idx, row in enumerate(data):
        if not isinstance(row, dict):
            continue

        label = row.get("section")
        if label not in labels:
            continue

        image_id = row.get("image_id")
        image_path = resolve_image_path(images_dir, image_id)
        if image_path is None:
            continue

        text_fields = [
            row.get("headline", ""),
            row.get("abstract", ""),
            row.get("caption",  ""),
        ]
        text = " ".join(part.strip() for part in text_fields if isinstance(part, str) and part.strip())

        items.append(
            {
                "id": row.get("id", idx),
                "image": image_path,
                "text": text if text else " ",
                "label": label,
            }
        )

    return items


def sample_few_shot(items, shots_per_class, seed):
    grouped = defaultdict(list)
    for item in items:
        grouped[item["label"]].append(item)

    rng = random.Random(seed)
    sampled = []
    for _, group in grouped.items():
        local = group[:]
        rng.shuffle(local)
        sampled.extend(local[:shots_per_class])

    rng.shuffle(sampled)
    return sampled    


class N24NewsFewShotDataset(Dataset):
    def __init__(self, items, tokenizer, image_processor, label2idx, image_size, max_text_length):
        self.items = items
        self.tokenizer = tokenizer
        self.label2idx = label2idx
        self.max_text_length = max_text_length

        mean = image_processor.image_mean
        std = image_processor.image_std
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        try:
            image = Image.open(item["image"]).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), (128, 128, 128))

        pixel_values = self.transform(image)
        encoder = self.tokenizer(
            item["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_text_length,
            return_tensors="pt",
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": encoder["input_ids"].squeeze(0),
            "attention_mask": encoder["attention_mask"].squeeze(0),
            "label": torch.tensor(self.label2idx[item["label"]], dtype=torch.long),
        }


class MultimodalFewShotNet(nn.Module):
    def __init__(
        self,
        num_classes,
        vision_model,
        text_model,
        hidden_dim,
        fusion_hidden_dim,
        dropout,
        freeze_image,
        freeze_text,
    ):
        super().__init__()
        self.vit = ViTModel.from_pretrained(vision_model)
        self.roberta = RobertaModel.from_pretrained(text_model)

        if freeze_image:
            for p in self.vit.parameters():
                p.requires_grad = False
        if freeze_text:
            for p in self.roberta.parameters():
                p.requires_grad = False

        img_dim = self.vit.config.hidden_size
        txt_dim = self.roberta.config.hidden_size

        self.image_head = nn.Sequential(
            nn.Linear(img_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.text_head = nn.Sequential(
            nn.Linear(txt_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(img_dim + txt_dim, fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, num_classes),
        )

    def forward(self, pixel_values, input_ids, attention_mask):
        img_out = self.vit(pixel_values=pixel_values, return_dict=True)
        txt_out = self.roberta(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

        img_cls = img_out.last_hidden_state[:, 0, :]
        txt_cls = txt_out.last_hidden_state[:, 0, :]
        fused = torch.cat([img_cls, txt_cls], dim=1)

        img_logits = self.image_head(img_cls)
        txt_logits = self.text_head(txt_cls)
        fusion_logits = self.fusion_head(fused)

        return img_logits, txt_logits, fusion_logits


def Dataloaders(cfg):
    labels = cfg["labels"]

    train_items = load_items(cfg["train_path"], cfg["images_dir"], labels)
    val_items   = load_items(cfg["val_path"],   cfg["images_dir"], labels)
    test_items  = load_items(cfg["test_path"],  cfg["images_dir"], labels)

    for p in (train_items, val_items, test_items):
        if p is None:
            raise RuntimeError(f"Failed to load items: {p}")
    
    train_samples = sample_few_shot(train_items, cfg["shots_per_class"], cfg["seed"])
    label_counts  = Counter([item["label"] for item in train_samples])
    print(f"Few-shot train samples: {len(train_samples)}")
    print("Few-shot label counts:")
    for label in labels:
        print(f"- {label}: {label_counts.get(label, 0)}")
    label2idx = {label: idx for idx, label in enumerate(labels)}

    tokenizer = RobertaTokenizerFast.from_pretrained(cfg["text_model"])
    image_processor = ViTImageProcessor.from_pretrained(cfg["vision_model"])
    image_size = int(cfg["image_size"])
    
    train_ds = N24NewsFewShotDataset(
        train_samples,
        tokenizer,
        image_processor,
        label2idx,
        image_size=image_size,
        max_text_length=cfg["max_text_length"],
    )
    val_ds = N24NewsFewShotDataset(
        val_items,
        tokenizer,
        image_processor,
        label2idx,
        image_size=image_size,
        max_text_length=cfg["max_text_length"],
    )
    test_ds = N24NewsFewShotDataset(
        test_items,
        tokenizer,
        image_processor,
        label2idx,
        image_size=image_size,
        max_text_length=cfg["max_text_length"],
    )

    train_loader = DataLoader(train_ds, batch_size=cfg["train_batch_size"], shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["infer_batch_size"], shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=cfg["infer_batch_size"], shuffle=False)
    
    return train_loader, val_loader, test_loader


def evaluate(model, loader, criterion):
    model.eval()
    losses = []
    trues = []
    preds = []

    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(DEVICE)
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            img_logits, txt_logits, fusion_logits = model(pixel_values, input_ids, attention_mask)

            loss_img = criterion(img_logits, labels)
            loss_txt = criterion(txt_logits, labels)
            loss_fuse = criterion(fusion_logits, labels)
            total_loss = loss_img + loss_txt + loss_fuse
            losses.append(total_loss.item())

            pred = fusion_logits.argmax(dim=1)
            trues.extend(labels.cpu().tolist())
            preds.extend(pred.cpu().tolist())

    avg_loss = float(np.mean(losses)) if losses else 0.0
    acc = accuracy_score(trues, preds) if trues else 0.0
    f1 = f1_score(trues, preds, average="macro") if trues else 0.0
    return avg_loss, acc, f1


def main():
    config  = load_config("config.yaml")
    runtime = resolve_cfg(config)
    set_seed(runtime["seed"])

    if not runtime["images_dir"].exists():
        raise FileNotFoundError(f"Images directory not found: {runtime['images_dir']}")
    
    for p in (runtime["train_path"], runtime["val_path"], runtime["test_path"]):
        if p is None or not p.exists():
            raise FileNotFoundError(f"Missing split JSON path: {p}")

    print(f"Device: {DEVICE}")
    print(f"Vision model: {runtime['vision_model']}")
    print(f"Text model: {runtime['text_model']}")

    train_loader, val_loader, test_loader = Dataloaders(runtime)

    model = MultimodalFewShotNet(
        num_classes=len(runtime["labels"]),
        vision_model=runtime["vision_model"],
        text_model=runtime["text_model"],
        hidden_dim=runtime["mlp_hidden_dim"],
        fusion_hidden_dim=runtime["fusion_hidden_dim"],
        dropout=runtime["dropout"],
        freeze_image=runtime["freeze_image"],
        freeze_text=runtime["freeze_text"],
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=runtime["lr"], weight_decay=runtime["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    runtime["save_path"].parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, runtime["epochs"] + 1):
        model.train()
        losses = []

        for batch in tqdm(train_loader, desc=f"Train epoch {epoch}/{runtime['epochs']}"):
            pixel_values = batch["pixel_values"].to(DEVICE)
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            img_logits, txt_logits, fusion_logits = model(pixel_values, input_ids, attention_mask)

            loss_img = criterion(img_logits, labels)
            loss_txt = criterion(txt_logits, labels)
            loss_fuse = criterion(fusion_logits, labels)
            total_loss = loss_img + loss_txt + loss_fuse

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            losses.append(total_loss.item())

        train_loss = float(np.mean(losses)) if losses else 0.0
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion)
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | val_f1_macro={val_f1:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), runtime["save_path"])
            print(f"Saved best model to {runtime['save_path']}")

    print(f"Loading best checkpoint: {runtime['save_path']}")
    model.load_state_dict(torch.load(runtime["save_path"], map_location=DEVICE))
    test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion)
    print(f"Test: loss={test_loss:.4f} | acc={test_acc:.4f} | f1_macro={test_f1:.4f}")


if __name__ == "__main__":
    main()
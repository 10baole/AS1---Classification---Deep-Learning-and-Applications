import csv
import json
from pathlib import Path
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm
import yaml


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def to_feature_tensor(features):
    if torch.is_tensor(features):
        return features

    if hasattr(features, "pooler_output") and features.pooler_output is not None:
        return features.pooler_output

    if hasattr(features, "last_hidden_state") and features.last_hidden_state is not None:
        return features.last_hidden_state[:, 0, :]

    raise TypeError(f"Unsupported feature output type: {type(features)}")


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

    data_path  = Path(paths_cfg.get("data_path"))
    images_dir = Path(paths_cfg.get("images_dir"))

    prompt_template = "a news article about {label_lower}"
    labels          = config.get("labels")
    label_prompts   = [
        prompt_template.format(label=label, label_lower=label.lower())
        for label in labels
    ]
    
    clip_model  = model_cfg.get("clip_model", "openai/clip-vit-base-patch32")
    batch_size  = int(runtime_cfg.get("batch_size"))
    max_samples = runtime_cfg.get("max_samples", None)

    print_cls_report = bool(runtime_cfg.get("cls_report", False))

    return {
        "data_path": data_path,
        "images_dir": images_dir,
        "batch_size": batch_size,
        "max_samples": max_samples,
        "print_cls_report": print_cls_report,
        "clip_model": clip_model,
        "labels": labels,
        "label_prompts": label_prompts,
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


def load_items(data_path: Path, images_dir: Path, labels):
    with data_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected nytimes_dataset.json to be a JSON list.")

    items = []
    unknown_labels = 0
    missing_images = 0
    
    for idx, row in enumerate(data):
        if not isinstance(row, dict):
            continue

        label = row.get("section")
        if label not in labels:
            unknown_labels += 1
            continue

        image_id = row.get("image_id")
        image_path = resolve_image_path(images_dir, image_id)
        if image_path is None:
            missing_images += 1
            continue

        items.append(
            {
                "id": row.get("id", idx),
                "image": image_path,
                "label": label,
            }
        )

    return items, missing_images, unknown_labels


def batchify(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def write_confusion_matrix_csv(path: Path, labels, matrix):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + labels)
        for label, row in zip(labels, matrix):
            writer.writerow([label] + row.tolist())


def evaluate_predictions(trues, preds, labels):
    accuracy = accuracy_score(trues, preds)
    precision_macro = precision_score(trues, preds, labels=labels, average="macro", zero_division=0)
    recall_macro = recall_score(trues, preds, labels=labels, average="macro", zero_division=0)
    f1_macro = f1_score(trues, preds, labels=labels, average="macro", zero_division=0)
    f1_weighted = f1_score(trues, preds, labels=labels, average="weighted", zero_division=0)
    conf_mat = confusion_matrix(trues, preds, labels=labels)
    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "confusion_matrix": conf_mat,
    }


def main():
    config  = load_config("config.yaml")
    runtime = resolve_cfg(config)

    if not runtime["data_path"].exists():
        raise FileNotFoundError(f"JSON file not found: {runtime['data_path']}")
    if not runtime["images_dir"].exists():
        raise FileNotFoundError(f"Images directory not found: {runtime['images_dir']}")

    print(f"Device: {DEVICE}")
    print(f"Model: {runtime["clip_model"]}")

    model     = CLIPModel.from_pretrained(runtime["clip_model"]).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(runtime["clip_model"], use_fast=False)

    inputs_text = processor(text=runtime["label_prompts"], return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        text_embeds = model.get_text_features(**inputs_text)
        text_embeds = to_feature_tensor(text_embeds)
        text_embeds = F.normalize(text_embeds, dim=-1)

    items, missing_images, unknown_labels = load_items(runtime["data_path"], runtime["images_dir"], runtime["labels"])

    if runtime["max_samples"] is not None:
        items = items[: runtime["max_samples"]]

    if not items:
        raise RuntimeError("No valid samples found. Check JSON format and image folder.")

    print(f"Loaded valid samples: {len(items)}")
    print(f"Skipped because image not found: {missing_images}")
    print(f"Skipped because section not in 24 labels: {unknown_labels}")

    preds = []
    trues = []

    total_batches = (len(items) + runtime["batch_size"] - 1) // runtime["batch_size"]
    for batch in tqdm(batchify(items, runtime["batch_size"]), total=total_batches):
        images = []
        for item in batch:
            try:
                image = Image.open(item["image"]).convert("RGB")
            except Exception:
                image = Image.new("RGB", (224, 224), color=(128, 128, 128))
            images.append(image)

        inputs_image = processor(text=None, images=images, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            image_embeds = model.get_image_features(**inputs_image)
            image_embeds = to_feature_tensor(image_embeds)
            image_embeds = F.normalize(image_embeds, dim=-1)
            similarities = image_embeds @ text_embeds.T
            pred_indices = similarities.argmax(dim=1).cpu().tolist()

        for i, item in enumerate(batch):
            preds.append(runtime["labels"][pred_indices[i]])
            trues.append(item["label"])

    metrics = evaluate_predictions(trues, preds, runtime["labels"])
    correct = sum(1 for pred, true in zip(preds, trues) if pred == true)

    print("Evaluation metrics (Zero-shot CLIP):")
    print(f"- Accuracy: {metrics['accuracy']:.4f} ({correct}/{len(trues)})")
    print(f"- Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"- Recall (macro): {metrics['recall_macro']:.4f}")
    print(f"- F1 (macro): {metrics['f1_macro']:.4f}")
    print(f"- F1 (weighted): {metrics['f1_weighted']:.4f}")

    if runtime["print_cls_report"]:
        print("\nPer-class report:")
        print(classification_report(trues, preds, labels=runtime["labels"], zero_division=0))

    # if runtime["confusion_matrix_path"] is not None:
    #     write_confusion_matrix_csv(runtime["confusion_matrix_path"], runtime["labels"], metrics["confusion_matrix"])
    #     print(f"Saved confusion matrix to: {runtime['confusion_matrix_path']}")


if __name__ == "__main__":
    main()
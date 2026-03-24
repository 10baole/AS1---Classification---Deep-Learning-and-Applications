# Assignment 1 - Classification

## 1. Folder Structure

```text
.
|-- config.yaml
|-- few_shot.py
|-- zero_shot.py
|-- requirements.txt
|-- README.md
|-- data/
|   |-- imgs/
|   \-- news/
|       |-- nytimes_dataset.json
|       |-- nytimes_train.json
|       |-- nytimes_dev.json
|       |-- nytimes_test.json
|       \-- ...
|-- outputs/
|   \-- best_model.pth
\-- __pycache__/
```

Meanings:
- data/news: chua cac file json dataset va cac split train/dev/test.
- data/imgs: chua anh tuong ung image_id trong json.
- outputs: luu checkpoint mo hinh (few-shot).
- config.yaml: cau hinh duong dan du lieu, model, thong so train va runtime.

## 2. Set up Libraries

Requirements:
- Python 3.9+ 

Create and activate virtual environment (Windows PowerShell):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Libraries install:

```powershell
pip install -r requirements.txt
```

## 3. Data Config

Kiem tra file config.yaml, toi thieu cac muc sau phai dung voi may cua ban:
- paths.data_path
- paths.train_path
- paths.val_path
- paths.test_path
- paths.images_dir

Mac dinh project dang de:
- data_path: data/news/nytimes_dataset.json
- train_path: data/news/nytimes_train.json
- val_path: data/news/nytimes_dev.json
- test_path: data/news/nytimes_test.json
- images_dir: data/imgs

## 4. Chay zero-shot

Lenh chay:

```powershell
python zero_shot.py
```

Script se:
- Doc toan bo mau tu file paths.data_path.
- Tao prompt theo danh sach labels trong config.yaml.
- Su dung CLIP de du doan nhan tu anh.
- In cac chi so: Accuracy, Precision macro, Recall macro, F1 macro, F1 weighted.

Tuy chon nhanh trong config.yaml:
- runtime.max_samples: gioi han so mau de test nhanh.
- runtime.batch_size: batch cho infer.
- runtime.cls_report: bat/tat classification report tung lop.

## 5. Chay few-shot

Lenh chay:

```powershell
python few_shot.py
```

Script se:
- Doc train/dev/test theo cac duong dan split trong config.yaml.
- Lay mau few-shot theo runtime.shots_per_class tren train split.
- Huan luyen mo hinh ket hop Vision Transformer + RoBERTa.
- Luu checkpoint tot nhat vao training.save_path (mac dinh outputs/best_model.pth).
- Nap checkpoint tot nhat va danh gia tren test split.

Thong so hay dieu chinh trong config.yaml:
- runtime.shots_per_class
- training.epochs
- training.batch_size
- training.lr
- training.save_path

## 6. Tham khao dataset

N24News paper: https://aclanthology.org/2022.lrec-1.729
Use notebooks only for EDA, debugging, and explainability.

Recommended split:
- `01_eda.ipynb`: dataset inspection and class imbalance analysis
- `02_train_debug.ipynb`: small local training/debug runs that call code from `src/`
- `03_explainability.ipynb`: Grad-CAM and attention rollout

Do not keep the main training pipeline only in notebooks.

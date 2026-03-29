from __future__ import annotations

import torch


class EarlyStopping:
    def __init__(self, patience: int = 5, mode: str = "max", min_delta: float = 0.0, save_path: str = "best_model.pth"):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.save_path = save_path
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            torch.save(model.state_dict(), self.save_path)
            return

        improved = score > self.best_score + self.min_delta if self.mode == "max" else score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

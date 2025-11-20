import matplotlib.pyplot as plt
from pathlib import Path

def plot_learning_curves(history, title: str, out_path: Path | None = None):
    """
    history: dict with keys like 'train_loss', 'val_loss', 'val_f1'
    """
    plt.figure(figsize=(10, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    if "val_f1" in history:
        plt.plot(history["val_f1"], label="Val F1", linestyle="--")

    plt.xlabel("Epoch")
    plt.ylabel("Loss / F1")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150)
    plt.show()

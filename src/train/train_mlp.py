import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def train_mlp(
    model,
    train_loader,
    val_loader,
    epochs: int,
    lr: float,
    device: str,
    logger,
    checkpoint_path,
):
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [],
        "fair_train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
    }

    best_val_f1 = -1.0

    for epoch in range(1, epochs + 1):
        # ---------- Training ----------
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x).squeeze()
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / max(len(train_loader), 1)

        # ---------- Fair Train Loss ----------
        model.eval()
        fair_train_loss = 0.0
        with torch.no_grad():
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x).squeeze()
                loss = criterion(outputs, y)
                fair_train_loss += loss.item()
        fair_train_loss /= max(len(train_loader), 1)

        # ---------- Validation ----------
        val_loss = 0.0
        y_true, y_pred = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x).squeeze()
                loss = criterion(outputs, y)
                val_loss += loss.item()
                preds = (outputs > 0.5).long()
                y_true.extend(y.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
        val_loss /= max(len(val_loader), 1)

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        history["train_loss"].append(avg_train_loss)
        history["fair_train_loss"].append(fair_train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(acc)
        history["val_precision"].append(prec)
        history["val_recall"].append(rec)
        history["val_f1"].append(f1)

        logger.info(
            f"[MLP] Epoch {epoch:03d} | "
            f"Train: {avg_train_loss:.4f} (fair {fair_train_loss:.4f}) | "
            f"Val: loss={val_loss:.4f}, acc={acc:.4f}, prec={prec:.4f}, rec={rec:.4f}, f1={f1:.4f}"
        )

        if f1 > best_val_f1:
            best_val_f1 = f1
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"[MLP] ✅ New best model saved (val_f1={best_val_f1:.4f})")

    return history, best_val_f1


def mlp_evaluation(model, test_loader, device: str, logger):
    model.to(device)
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device).float()
            outputs = model(x).squeeze()
            preds = (outputs > 0.5).long()
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    logger.info(
        f"[MLP] Test metrics | acc={acc:.4f}, prec={prec:.4f}, rec={rec:.4f}, f1={f1:.4f}"
    )
    return acc, prec, rec, f1

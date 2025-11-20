import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def train_gnn(
    model,
    train_loader,
    val_loader,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: str,
    logger,
    checkpoint_path,
):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

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

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch, batch.edge_type)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / max(len(train_loader), 1)

        # ---------- Fair Train Loss ----------
        model.eval()
        fair_train_loss = 0.0
        with torch.no_grad():
            for batch in train_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch, batch.edge_type)
                loss = criterion(out, batch.y)
                fair_train_loss += loss.item()
        fair_train_loss /= max(len(train_loader), 1)

        # ---------- Validation ----------
        val_loss = 0.0
        preds, trues = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch, batch.edge_type)
                loss = criterion(out, batch.y)
                val_loss += loss.item()
                pred = out.argmax(dim=1).cpu().numpy()
                preds.extend(pred)
                trues.extend(batch.y.cpu().numpy())

        val_loss /= max(len(val_loader), 1)
        acc = accuracy_score(trues, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(trues, preds, average="binary")

        # ---------- History & Logging ----------
        history["train_loss"].append(avg_train_loss)
        history["fair_train_loss"].append(fair_train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(acc)
        history["val_precision"].append(precision)
        history["val_recall"].append(recall)
        history["val_f1"].append(f1)

        logger.info(
            f"[GNN] Epoch {epoch:03d} | "
            f"Train: {avg_train_loss:.4f} (fair {fair_train_loss:.4f}) | "
            f"Val: loss={val_loss:.4f}, acc={acc:.4f}, prec={precision:.4f}, rec={recall:.4f}, f1={f1:.4f}"
        )

        # ---------- Checkpoint ----------
        if f1 > best_val_f1:
            best_val_f1 = f1
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"[GNN] ✅ New best model saved (val_f1={best_val_f1:.4f})")

    return history, best_val_f1


def gnn_evaluation(model, test_loader, device: str, logger):
    model.to(device)
    model.eval()

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    preds, trues = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch, batch.edge_type)
            pred = logits.argmax(dim=1).cpu().numpy()
            preds.extend(pred)
            trues.extend(batch.y.cpu().numpy())

    acc = accuracy_score(trues, preds)
    precision = precision_score(trues, preds)
    recall = recall_score(trues, preds)
    f1 = f1_score(trues, preds)

    logger.info(
        f"[GNN] Test metrics | acc={acc:.4f}, prec={precision:.4f}, rec={recall:.4f}, f1={f1:.4f}"
    )
    return acc, precision, recall, f1

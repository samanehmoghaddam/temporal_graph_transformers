import json
from pathlib import Path

import torch
from torch_geometric.data import DataLoader as GeoDataLoader

from src.config import (
    GRAPHS_DIR,
    USE_CODEBERT,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    set_seed,
    CHECKPOINT_DIR,
    RESULTS_DIR,
)
from src.utils.logger import get_logger
from src.utils.plot_utils import plot_learning_curves
from src.data.graph_loader import load_graphs_from_json
from src.models.temporal_gnn import TemporalGraphTransformer
from src.models.node_mlp import NodeDataset, NodeMLP
from src.train.train_gnn import train_gnn, gnn_evaluation
from src.train.train_mlp import train_mlp, mlp_evaluation


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed()

    logger = get_logger("train", CHECKPOINT_DIR.parent / "logs" / "train.log")
    logger.info(f"Using device: {device}")

    # ---------- Load graphs ----------
    graphs = load_graphs_from_json(GRAPHS_DIR, use_codebert=USE_CODEBERT)
    n = len(graphs)
    logger.info(f"Total graphs: {n}")

    import random
    random.shuffle(graphs)

    from src.config import TRAIN_RATIO, VAL_RATIO

    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    train_graphs = graphs[:n_train]
    val_graphs = graphs[n_train : n_train + n_val]
    test_graphs = graphs[n_train + n_val :]

    logger.info(f"Split: train={len(train_graphs)}, val={len(val_graphs)}, test={len(test_graphs)}")

    # ---------- GNN ----------
    train_loader_gnn = GeoDataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    val_loader_gnn = GeoDataLoader(val_graphs, batch_size=BATCH_SIZE)
    test_loader_gnn = GeoDataLoader(test_graphs, batch_size=BATCH_SIZE)

    in_dim = train_graphs[0].x.size(1)
    gnn_model = TemporalGraphTransformer(in_dim)

    gnn_ckpt = CHECKPOINT_DIR / "gnn_best.pt"
    gnn_history, gnn_best_val_f1 = train_gnn(
        gnn_model,
        train_loader_gnn,
        val_loader_gnn,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        device=device,
        logger=logger,
        checkpoint_path=gnn_ckpt,
    )

    plot_learning_curves(
        gnn_history,
        title="GNN Learning Curves",
        out_path=RESULTS_DIR / "gnn_learning_curves.png",
    )

    # reload best checkpoint
    gnn_model.load_state_dict(torch.load(gnn_ckpt, map_location=device))
    gnn_test_metrics = gnn_evaluation(gnn_model, test_loader_gnn, device=device, logger=logger)

    # ---------- Baseline MLP ----------
    train_dataset_mlp = NodeDataset(train_graphs)
    val_dataset_mlp = NodeDataset(val_graphs)
    test_dataset_mlp = NodeDataset(test_graphs)

    train_loader_mlp = torch.utils.data.DataLoader(
        train_dataset_mlp, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader_mlp = torch.utils.data.DataLoader(val_dataset_mlp, batch_size=BATCH_SIZE)
    test_loader_mlp = torch.utils.data.DataLoader(test_dataset_mlp, batch_size=BATCH_SIZE)

    mlp_model = NodeMLP(input_dim=train_dataset_mlp.features.shape[1])
    mlp_ckpt = CHECKPOINT_DIR / "mlp_best.pt"

    mlp_history, mlp_best_val_f1 = train_mlp(
        mlp_model,
        train_loader_mlp,
        val_loader_mlp,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        device=device,
        logger=logger,
        checkpoint_path=mlp_ckpt,
    )

    plot_learning_curves(
        mlp_history,
        title="MLP Baseline Learning Curves",
        out_path=RESULTS_DIR / "mlp_learning_curves.png",
    )

    mlp_model.load_state_dict(torch.load(mlp_ckpt, map_location=device))
    mlp_test_metrics = mlp_evaluation(mlp_model, test_loader_mlp, device=device, logger=logger)

    # ---------- Save histories & summary ----------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "gnn_history.json", "w", encoding="utf-8") as f:
        json.dump(gnn_history, f, indent=2)
    with open(RESULTS_DIR / "mlp_history.json", "w", encoding="utf-8") as f:
        json.dump(mlp_history, f, indent=2)

    summary = {
        "gnn_best_val_f1": gnn_best_val_f1,
        "gnn_test_metrics": {
            "accuracy": gnn_test_metrics[0],
            "precision": gnn_test_metrics[1],
            "recall": gnn_test_metrics[2],
            "f1": gnn_test_metrics[3],
        },
        "mlp_best_val_f1": mlp_best_val_f1,
        "mlp_test_metrics": {
            "accuracy": mlp_test_metrics[0],
            "precision": mlp_test_metrics[1],
            "recall": mlp_test_metrics[2],
            "f1": mlp_test_metrics[3],
        },
    }
    with open(RESULTS_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"✅ Training complete. Summary saved to {RESULTS_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()

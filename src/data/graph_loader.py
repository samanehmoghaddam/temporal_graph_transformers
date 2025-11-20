import json
import datetime
from pathlib import Path

import torch
from torch_geometric.data import Data
from transformers import AutoTokenizer, AutoModel

EDGE_TYPES = {"temporal_method": 0, "temporal_commit": 1, "spatial_call": 2, "belongs_to": 3}


def _load_codebert():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModel.from_pretrained("microsoft/codebert-base")
    model.eval()
    return tokenizer, model


def encode_code_snippet(snippet: str, tokenizer, model) -> torch.Tensor:
    inputs = tokenizer(snippet, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze(0)


def node_to_feature_vector(
    node: dict,
    use_codebert: bool = False,
    tokenizer=None,
    model=None,
    min_time: float | None = None,
    max_time: float | None = None,
) -> torch.Tensor:
    commit_version = node["features"].get("commit_version", 0)
    num_lines = node["features"].get("num_lines", 0)
    num_tokens = node["features"].get("num_tokens", 0)
    num_calls = node["features"].get("num_calls", 0)

    contains_features = node["features"].get("contains", {})
    contains_vals = [
        int(contains_features.get(k, False))
        for k in ["deserialization", "sql", "logger", "env", "printf"]
    ]
    type_feature = 0 if node["type"] == "commit" else 1

    numeric_features = [
        commit_version,
        num_lines,
        num_tokens,
        num_calls,
        type_feature,
    ] + contains_vals

    # (timestamp normalization commented out in your original code)

    if use_codebert and node["type"] == "method":
        numeric_features += encode_code_snippet(node["code_snippet"], tokenizer, model).tolist()

    return torch.tensor(numeric_features, dtype=torch.float)


def load_graphs_from_json(data_dir: Path, use_codebert: bool = False):
    graphs = []

    if use_codebert:
        tokenizer, model = _load_codebert()
    else:
        tokenizer, model = None, None

    for path in Path(data_dir).glob("*.json"):
        with open(path, "r", encoding="utf-8") as f:
            g = json.load(f)

        node_id_map = {node["id"]: i for i, node in enumerate(g["nodes"])}
        edge_index_list, edge_type_ids = [], []

        for e in g["edges"]:
            src, tgt = node_id_map[e["source"]], node_id_map[e["target"]]
            edge_index_list.append([src, tgt])
            edge_type_ids.append(EDGE_TYPES.get(e.get("type", "spatial_call"), 0))

        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_type_ids = torch.tensor(edge_type_ids, dtype=torch.long)

        timestamps = []
        for node in g["nodes"]:
            if "timestamp" in node:
                ts = datetime.datetime.fromisoformat(node["timestamp"]).timestamp()
            else:
                ts = 0.0
            timestamps.append(ts)
        min_time, max_time = (min(timestamps), max(timestamps)) if any(timestamps) else (0.0, 1.0)

        x = torch.stack(
            [
                node_to_feature_vector(
                    node,
                    use_codebert=use_codebert,
                    tokenizer=tokenizer,
                    model=model,
                    min_time=min_time,
                    max_time=max_time,
                )
                for node in g["nodes"]
            ]
        )

        label_value = g.get("is_vulnerable")
        if label_value is None:
            label_value = 1 if g.get("label", "").lower() == "vulnerable" else 0

        y = torch.tensor(label_value, dtype=torch.long)

        graphs.append(
            Data(
                x=x,
                edge_index=edge_index,
                y=y,
                edge_type=edge_type_ids,
            )
        )

    print(f"✅ Loaded {len(graphs)} graphs from {data_dir}")
    return graphs

# Temporal Graph Transformers for Code Vulnerability Detection

Temporal Graph Transformers for code vulnerability detection. Includes CWE-based mutation engine, CEPG graph construction, and a temporal GNN with baseline MLP. Provides full training pipeline, logs, checkpoints, and notebooks for mutation and classification analysis.

This repository provides an end-to-end framework for generating synthetic vulnerable code samples, building **Code-Evolution Provenance Graphs (CEPGs)**, and training a **Temporal Graph Transformer** to classify vulnerable vs. benign evolutionary patterns in code. The system integrates CWE-based mutation, temporal graph reasoning, structured logging, checkpoints, and reproducible experiment pipelines.

---

## 📌 Features

### **CEPG Graph Construction**
- Extracts accepted Java submissions from Project CodeNet  
- Applies CWE-aligned vulnerable mutations and benign edits  
- Builds multi-commit provenance graphs capturing:
  - commit evolution  
  - method-level changes  
  - static call relationships  
  - semantic + structural node features  

### **Temporal Graph Transformer**
- Edge-type-aware TransformerConv layers  
- GRU-based temporal encoder  
- Graph-level classification  
- Optional CodeBERT feature integration  

### **Baseline MLP**
- Node-level classifier for comparison  
- Provides a non-graph baseline  

### **Training Pipeline**
- Structured logs (`logs/train.log`)  
- Automatic checkpointing on best validation F1  
- JSON-based result storage  
- Train/Val/Test splitting  
- Matplotlib learning curves  

### **Notebooks**
- `01_mutation_analysis.ipynb`  
- `02_classification_results.ipynb`  

---

## 📁 Project Structure
~~~~
temporal_graph_transformers/
├── src/
│   ├── data/
│   │   └── codenet_mutator_cepg.py
│   ├── models/
│   │   ├── edge_transformer.py
│   │   ├── temporal_gnn.py
│   │   └── node_mlp.py
│   ├── train/
│   │   ├── train_gnn.py
│   │   ├── train_mlp.py
│   │   └── main_train.py
│   ├── utils/
│   │   ├── plot_utils.py
│   │   └── logger.py
│   ├── config.py
│   └── __init__.py
├── notebooks/
│   ├── 01_mutation_analysis.ipynb
│   └── 02_classification_results.ipynb
├── data/
├── logs/
├── checkpoints/
├── results/
├── requirements.txt
└── README.md
~~~~

---

## ⚙️ Installation

~~~~bash
git clone https://github.com/<your-username>/temporal_graph_transformers.git
cd temporal_graph_transformers
pip install -r requirements.txt
~~~~

Optional (GPU-enabled):

~~~~bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
~~~~

---

## 🚀 Usage

### 1. Generate CEPG Graphs

~~~~bash
python src/data/codenet_mutator_cepg.py
~~~~

Outputs saved to: data/Mutated_CodeNet/graphs_multiple_no_timestamp/*.json


---

### 2. Train the Models (GNN + Baseline MLP)

~~~~bash
python src/train/main_train.py
~~~~

This will:
- load CEPG graphs  
- train the Temporal Graph Transformer  
- train the baseline MLP  
- save checkpoints in `checkpoints/`  
- log to `logs/train.log`  
- generate plots in `results/`  

---

## 📊 Results & Visualization

Training histories:  
- `results/gnn_history.json`  
- `results/mlp_history.json`

Learning curves:  
- `results/gnn_learning_curves.png`  
- `results/mlp_learning_curves.png`

Summary metrics:  
- `results/summary.json`

---

## 📓 Notebooks

### 01_mutation_analysis.ipynb
Includes:
- benign vs. vulnerable distribution  
- CWE frequency  
- mutation statistics  
- graph-level stats  

### 02_classification_results.ipynb
Includes:
- GNN vs MLP comparisons  
- per-epoch metrics  
- confusion matrices  
- checkpoint comparisons  

---

## 🧪 Dataset

This project uses Java submissions from Project CodeNet, filtered by:

- language = Java  
- status = Accepted  

Only synthetic mutation-based graphs are stored.  
The original CodeNet dataset is **not redistributed**.

---

## 📜 License

Distributed under the **MIT License**.  
See `LICENSE` for details.


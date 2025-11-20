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


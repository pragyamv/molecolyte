# MoleColyte

Chemical molecule toxicity prediction using 3D Equivariant Graph Neural Networks, pre-trained on QM9 and fine-tuned on Tox21 with physics-optimised molecular graphs and Functional Group Node augmentation.

---

## Overview

MoleColyte is an end-to-end deep learning pipeline for molecular toxicity prediction. Each molecule is converted from a SMILES string into a physics-optimised 3D graph, augmented with virtual Functional Group Nodes (FGNs) that encode hierarchical chemical structure, and processed by an Equivariant Graph Neural Network (EGNN) that reasons over both atomic features and 3D geometry simultaneously.

The model follows a two-stage transfer learning strategy. It is first pre-trained on the QM9 quantum chemistry dataset to learn fundamental 3D molecular geometry, then fine-tuned on the Tox21 dataset to predict 12 toxicity assay outcomes.

---

## Architecture

### EGNN — Equivariant Graph Neural Network

The core model is built on E(n) Equivariant Graph Neural Networks (Satorras et al., 2021). Unlike standard GNNs that treat 3D coordinates as static edge features, EGNN updates both node hidden states and atomic positions during every message passing step. This makes the model natively equivariant to rotations, reflections, and translations — the same molecule in any orientation produces the same prediction.

Each EGNN layer performs three operations:

- **Edge MLP** — computes messages from node hidden states, live pairwise distances, and bond type features
- **Node MLP** — aggregates neighbour messages to update each atom's hidden state
- **Coordinate MLP** — produces scalar weights that nudge atomic positions along relative position vectors

Three EGNN layers are stacked, giving each atom a 3-hop receptive field. SiLU activations are used throughout the EGNN layers for smooth gradient flow during coordinate updates. The final molecule representation is obtained via global mean pooling over all node hidden states.

### Functional Group Nodes (FGNs)

Before any neural network processing, each molecule's graph is augmented with virtual FGN nodes. Twelve SMARTS patterns are used to detect functional groups including benzene rings, carbonyls, hydroxyls, amines, halogens, and others. For each match:

- A virtual node is appended with features equal to the mean of its member atoms
- Its 3D position is set to the geometric centroid of its member atoms
- Bidirectional edges connect the FGN to every member atom

FGNs allow information to travel across chemically meaningful sub-structures in fewer message passing steps and provide a hierarchical representation of molecular topology.

### Transfer Learning

The model is first pre-trained on QM9 to predict internal energy at 0K (U0) — a quantum property that requires deep understanding of 3D molecular geometry. The EGNN layers trained on 133,885 molecules develop rich geometric representations that transfer directly to toxicity prediction. Only the final prediction head is replaced when fine-tuning on Tox21.

---

## Datasets

### Tox21

7,831 molecules with 12 binary toxicity labels across nuclear receptor and stress response assays. Labels represent experimental results from high-throughput screening — 1 indicates activity (toxic), 0 indicates inactivity, and missing values indicate untested assays.

The 12 assays are:

| Assay | Type | Biological Target |
|---|---|---|
| NR-AR | Nuclear Receptor | Androgen Receptor |
| NR-AR-LBD | Nuclear Receptor | Androgen Receptor Ligand Binding Domain |
| NR-AhR | Nuclear Receptor | Aryl Hydrocarbon Receptor |
| NR-Aromatase | Nuclear Receptor | Aromatase Enzyme |
| NR-ER | Nuclear Receptor | Estrogen Receptor |
| NR-ER-LBD | Nuclear Receptor | Estrogen Receptor Ligand Binding Domain |
| NR-PPAR-gamma | Nuclear Receptor | Peroxisome Proliferator Activated Receptor |
| SR-ARE | Stress Response | Antioxidant Response Element |
| SR-ATAD5 | Stress Response | DNA Damage Indicator |
| SR-HSE | Stress Response | Heat Shock Element |
| SR-MMP | Stress Response | Mitochondrial Membrane Potential |
| SR-p53 | Stress Response | DNA Damage / Cancer Suppression |

### QM9

133,885 small organic molecules with up to 9 heavy atoms, each with 19 quantum mechanical properties computed using Density Functional Theory (DFT). Pre-training uses U0 — internal energy at 0K — as the regression target.

---

## Pipeline

### File Structure

```
├── dataset/
│   └── tox21.csv
│
├── preprocessing/
│   ├── SMILES_to_3D.py
│   └── QM9_refactor.py
│   └── inspect.py
│
├── refrence-papers/
│    └── (reference papers)
└── Training/
    ├── egnn_layer.py
    ├── data_loader.py
    ├── train_QM9.py
    ├── train_Tox21.py
    └── eval_molecolyte.py

```

### Graph Format

Every molecule is stored as a DGL graph with the following tensors:

| Tensor | Shape | Description |
|---|---|---|
| `g.ndata['x']` | (N+K, 8) | Node features: 7 chemistry features + is_fgn flag |
| `g.ndata['pos']` | (N+K, 3) | 3D coordinates in Angstroms |
| `g.edata['edge_attr']` | (E+E_new, 5) | Bond type one-hot + virtual bond flag |
| `y` | (12,) for Tox21, (1,19) for QM9 | Labels |

Where N = real atoms, K = FGN nodes, E = directed chemical bonds, E_new = FGN bipartite edges.

### Node Features

| Index | Feature |
|---|---|
| 0 | Atomic number |
| 1 | Formal charge |
| 2 | Is aromatic |
| 3 | Number of hydrogens |
| 4 | SP hybridization |
| 5 | SP2 hybridization |
| 6 | SP3 hybridization |
| 7 | is_fgn flag (0 = real atom, 1 = virtual FGN) |

### Edge Features

| Index | Feature |
|---|---|
| 0 | Single bond |
| 1 | Double bond |
| 2 | Triple bond |
| 3 | Aromatic bond |
| 4 | Virtual bond flag (0 = real, 1 = FGN edge) |

---

## Training

### QM9 Pre-training

- **Task** — Regression (internal energy U0)
- **Loss** — Mean Squared Error (MSE)
- **Optimiser** — Adam (lr=1e-3)
- **Epochs** — 10
- **Batch size** — 32
- **Split** — 80 / 10 / 10

### Tox21 Fine-tuning

- **Task** — Multi-label binary classification (12 assays)
- **Loss** — Binary Cross Entropy with Logits (BCEWithLogitsLoss)
- **Optimiser** — Adam (lr=1e-3)
- **Epochs** — 20
- **Batch size** — 32
- **Split** — 80 / 10 / 10
- **Class imbalance** — Dynamic positive weights computed per assay (`negatives / positives`)
- **Missing labels** — NaN masking applied during loss computation and evaluation

### Model Configuration

| Hyperparameter | Value |
|---|---|
| Input node features | 8 |
| Hidden dimension | 128 |
| Edge attribute dimension | 5 |
| EGNN layers | 3 |
| Activation (EGNN) | SiLU |
| Activation (head) | ReLU |
| Output (QM9) | 1 |
| Output (Tox21) | 12 |

---

## Evaluation

Performance is measured using AUC-ROC per assay on the held-out test set. Missing labels are excluded from scoring. A mean AUC-ROC is reported across all valid assays.

- **0.5** — equivalent to random guessing
- **0.75+** — considered meaningful predictive performance
- **0.9+** — strong performance for molecular toxicity

---

## Installation

```bash
# Create a Python 3.10 virtual environment
py -3.10 -m venv venv
venv\Scripts\activate

# Install dependencies
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install dgl==2.2.1+cu121 -f https://data.dgl.ai/wheels/cu121/repo.html
pip install torchdata==0.6.0 --force-reinstall --no-deps
pip install pandas rdkit torch-geometric scikit-learn numpy==1.26.4
```

---

## Usage

Run the pipeline in order:

```bash
# Step 1 — Preprocess Tox21
python SMILES_to_3D.py

# Step 2 — Preprocess QM9
python QM9_refactor.py

# Step 3 — Verify data loading
python data_loader.py

# Step 4 — Pre-train on QM9
python train_QM9.py

# Step 5 — Fine-tune on Tox21
python train_Tox21.py

# Step 6 — Evaluate
python eval_molecolyte.py
```

---

## Dependencies

| Package | Purpose |
|---|---|
| PyTorch | Deep learning framework |
| DGL | Graph neural network library |
| RDKit | Cheminformatics — SMILES parsing, 3D embedding, MMFF optimisation |
| PyTorch Geometric | QM9 dataset download and processing |
| scikit-learn | AUC-ROC evaluation |
| NumPy / Pandas | Data handling |

---

## References

- Satorras, V. G., Hoogeboom, E., & Welling, M. (2021). E(n) Equivariant Graph Neural Networks. *ICML 2021*
- Tox21 Data Challenge — https://tripod.nih.gov/tox21
- QM9 Dataset — Ramakrishnan et al. (2014), Scientific Data
- PyTorch Geometric QM9 — https://pytorch-geometric.readthedocs.io

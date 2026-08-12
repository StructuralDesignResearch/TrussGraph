# TrussGraph GNN Examples

[![Dataset DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21841194.svg)](https://doi.org/10.5281/zenodo.21841194)

This repository contains PyTorch Geometric examples for learning from planar
truss graphs. It supports node-level displacement prediction, edge-level axial
force prediction, graph-level static-action evaluation, and a compact tutorial
for importing and manipulating the raw JSON dataset.

The dataset contains planar trusses from multiple typologies:

- Fink
- Howe
- KTruss
- Pratt
- Warren

Each typology is organized by panel count, seed, and graph JSON files:

```text
data/
  raw/
    Pratt/
      p_8_seed_7/
        graph_id.json
```

## Project Layout

```text
project_root/
  src/
    dataset.py          PyG dataset classes and JSON import logic
    model.py            GNN architectures
    training.py         splitting, training, checkpoint, and metric helpers
    visualization.py    plotting utilities
  notebooks/
    import_manipulate_dataset.ipynb
    example_node_level.ipynb
    example_edge_level.ipynb
    example_graph_level.ipynb
  data/
    raw/                downloaded or manually placed JSON folders
    processed/          PyG processed caches
  checkpoints/          trained model checkpoints
```

## Dataset Classes

The shared importer is `_TrussJsonDataset` in `src/dataset.py`. It handles:

- JSON discovery under `data/raw`
- typology, panel, seed, and design-count selection
- optional Zenodo download for missing selected typology folders
- parsing raw JSON into common graph tensors
- processed-cache creation under `data/processed`

Task-specific datasets inherit from `_TrussJsonDataset` and implement
`build_data(...)`:

- `TrussDisplacementDataset`: node-level displacement prediction
- `AxialForceDataset`: edge/member-level axial force prediction
- `StaticActionDataset`: graph-level static-action evaluation from predicted
  member forces

The tutorial notebook `import_manipulate_dataset.ipynb` shows how to create a
small custom base dataset by inheriting from `_TrussJsonDataset`.

## Installation

Create an environment with PyTorch, PyTorch Geometric, and the plotting/notebook
dependencies:

```bash
pip install torch torch-geometric numpy pandas matplotlib jupyterlab
```

Depending on your CUDA/PyTorch setup, install PyTorch and PyTorch Geometric from
their official installation selectors if you need GPU-specific wheels.

## Data

The dataset is available on Zenodo:

- Record: https://zenodo.org/records/21841194
- DOI: https://doi.org/10.5281/zenodo.21841194

The expected raw folder structure is:

```text
data/raw/
  Fink/
  Howe/
  KTruss/
  Pratt/
  Warren/
```

The dataset classes can download missing selected typology folders when
`download_if_missing=True`, which is the default.

## Quick Start

Open the tutorial first:

```bash
jupyter lab notebooks/import_manipulate_dataset.ipynb
```

Then run one of the learning examples:

```bash
jupyter lab notebooks/example_node_level.ipynb
jupyter lab notebooks/example_edge_level.ipynb
jupyter lab notebooks/example_graph_level.ipynb
```

Run notebooks from either `project_root/` or
`project_root/notebooks/`. 

## Typical Dataset Usage

```python
from pathlib import Path
from dataset import TrussDisplacementDataset

dataset = TrussDisplacementDataset(
    root=Path("data"),
    typologies=["Pratt"],
    panels=[8],
    seeds=[7],
    max_designs_per_seed=500,
    plane="xz",
)

print(len(dataset))
print(dataset[0])
```

Processed caches are configuration-specific. Changing typologies, panels,
seeds, sampling limits, plane, or dataset class produces a different cache file
under `data/processed`.

## Checkpoints

Training notebooks use configuration-specific checkpoint helpers from
`src/training.py`. Checkpoint filenames are hashed from:

- task name
- dataset configuration
- model configuration
- random seed

If a matching checkpoint exists, the notebook loads it and skips retraining.

## Citation



```bibtex

```


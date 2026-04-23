# NeuraGEM

NeuraGEM (Neural Gradient-based Expectation-Maximization) is a computational framework for learning latent task structure with fast latent updates and slower synaptic weight updates. This repository a bared down to the essential version with some code reorganization and a template to create a new task 'dataset'.

## System requirements

The software was run using Python version 3.12. Pytorch version 2.3.1. Python packages and versions in requirements.txt.
The model can run on any standard computer with RAM. No GPU is required, but is GPUs are supported. 

## Quick Start

The fastest way to run the different models:

Clone this repository:
```bash
git clone https://github.com/hummosa/NeuraGEM.git
cd NeuraGEM
```

Install requirements (5 minutes to install):
See instructions to install PyTorch on official website https://pytorch.org/. 
```bash
pip install matplotlib tqdm seaborn pandas scikit-learn statsmodels
```
Run simulations:
```bash
python run_training_behavior.py
```

This runs several behavioral conditions (NeuraGEM and RNN baselines) and exports compact figures seen in supp Fig 10 (10 minutes to run).

For a flexible sandbox script, use:

```bash
python run_example.py
```

In `run_example.py` you can:

- Load either `ContextualSwitchingTaskConfig` (contextual switching task) or `seq_learnConfig` (sequence learning task).
- Run NeuraGEM (default) or an RNN baseline by setting `config.LU_lr = 0`.
- Run a long-horizon RNN baseline by setting `config.LU_lr = 0` and `config.seq_len = 50`.


## Disclaimer

Code was refactored and organized as a last step using an LLM agent. Some minor discrepancies are possible. 

## Citation

If you build on this codebase, please cite the accompanying manuscript:

```bibtex
@article{Hummos2026.03.31.715618,
	author = {Hummos, Ali and Wang, Mien Brabeeba and Lu, Qihong and Norman, Kenneth A. and Jazayeri, Mehrdad},
	title = {A neural mechanism for online discovery of latent contexts},
	elocation-id = {2026.03.31.715618},
	year = {2026},
	doi = {10.64898/2026.03.31.715618},
	URL = {https://www.biorxiv.org/content/early/2026/04/02/2026.03.31.715618},
	journal = {bioRxiv}
}
```

## License

MIT License. See `LICENSE`.

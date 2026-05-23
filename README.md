<<<<<<< HEAD
<<<<<<< HEAD
https://drive.google.com/drive/folders/1q4FjgIlwDW4TpfvQb8L4CTGVNrg3Gtk5?usp=drive_link
=======
=======
>>>>>>> cd983a986e6992fffce46132bec8e2a4e0b342b8
Graduate Together — UAV-assisted Ambient Backscatter Anti-Jamming

Overview

This repository implements IA-MADDPG and several baselines for a UAV-assisted ambient backscatter anti-jamming problem. Core scripts live in the src/ directory.

Requirements

- Python 3.10+ recommended
- Install dependencies from src/requirements.txt (includes torch, numpy, matplotlib, scipy, pytest, tqdm)

Quick setup

```bash
# from the repo root
cd src
python -m venv .venv
source .venv/bin/activate    # on Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Running training

From src/ (activate the venv first):

# Train default method (IA-MADDPG+UAV)
python train.py

# Train a specific method
python train.py --method maddpg --episodes 600 --seed 42 --save_dir results/

# Train all methods sequentially
python train.py --method all --save_dir results/

Trained checkpoints and history are written under the save_dir (default: results/). Training curves and plots are saved when matplotlib is available.

Running evaluation

From src/ (after training or with existing results/):

python evaluate.py --results_dir results/ --n_episodes 50 --plot_all

This produces eval_results.json and several PNG plots in results/.

Running tests

From src/:

pytest -q

Notes

- If you have a CUDA-capable GPU and installed a CUDA-enabled torch build, training may use GPU implicitly (if code uses torch tensors). Otherwise CPU-only execution is supported for most scripts.
- Adjust hyperparameters in src/config.py (episodes, steps_per_episode, N, K, etc.).
- If a requirements.txt file is moved, update the install path accordingly.

<<<<<<< HEAD
If you want this README tuned (add badges, examples, or CI), tell me which details to include.
>>>>>>> 5baa267 (feat: update reports)
=======
If you want this README tuned (add badges, examples, or CI), tell me which details to include.
>>>>>>> cd983a986e6992fffce46132bec8e2a4e0b342b8

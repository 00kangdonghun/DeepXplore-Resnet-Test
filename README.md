# DeepXplore-Resnet-Test

## Environment Setup
conda env create -f environment.yml

conda activate deepxplore-env

## Project Structure
```
assignment2_deepxplore/
├── models/
│   ├── resnet50_a.py        # ResNet50 model with standard fc
│   └── resnet50_b.py        # ResNet50 model with dropout fc
├── data/
│   └── cifar10_loader.py    # CIFAR-10 dataloader with resizing
├── utils/
│   ├── neuron_coverage.py   # Neuron coverage tracking
│   └── attack_utils.py      # Gradient-based adversarial attack
├── run_deepxplore.py        # Main executable script
└── logs/                    # Saved adversarial/divergent samples
```

## How to Run
python run_deepxplore.py

## Features
- Uses **two pretrained ResNet50** models (from torchvision).
- Applies **neuron coverage analysis** (per-layer ReLU tracking).
- Performs **gradient-based adversarial attacks** to induce model divergence.
- Saves **up to 30 divergent/adversarial samples** into `logs/`.

## Output Example
- `logs/diverge_4_orig.png`: original image with divergence
- `logs/diverge_4_adv.png`: adversarially modified image

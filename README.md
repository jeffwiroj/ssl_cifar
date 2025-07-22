# SSL CIFAR-10

This repo contains the implementations of Self-Supervised Learning (SSL) methods tailored for CIFAR-10

## 🚀 Features

- **Multiple SSL Methods**: Currently supports SimSiam and Barlow Twins
- **Flexible Backbones**: ResNet-18 and MobileNetV4-Medium support
- **Modern Training**: Mixed precision training, learning rate warmup, and wandb logging
- **Easy Configuration**: YAML-based configuration system for reproducible experiments
- **Two-Stage Pipeline**: Separate pretraining and finetuning workflows

## 📁 Project Structure

```
ssl_cifar/
├── ssl_cifar/          # Main package
├── examples/           # Training scripts
│   ├── pretrain.py    # Pretraining script
│   └── finetune.py    # Finetuning script
└── configs/           # Configuration files
    ├── base_simsiam_pretrain.yaml
    └── base_simsiam_finetune.yaml
```

## 🛠️ Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management. Make sure you have uv installed:
Then install the project dependencies:

```bash
uv sync
```

## 🔧 Quick Start

### Pretraining

Train a SimSiam model on CIFAR-10:

```bash
uv run examples/pretrain.py -p configs/base_simsiam_pretrain.yaml
```

### Finetuning

After pretraining, finetune the learned representations:

```bash
uv run examples/finetune.py -p configs/base_simsiam_finetune.yaml
```

## ⚙️ Configuration

The project uses YAML configuration files for easy experiment management. Here's the structure:

### Pretraining Configuration

```yaml
train:
  backbone: resnet18              # Backbone architecture
  ssl_model: simsiam             # SSL method
  backbone_dim: 512              # Backbone output dimension
  epochs: 800                    # Training epochs
  lr: 0.06                       # Learning rate
  wd: 5e-4                       # Weight decay
  momentum: 0.9                  # SGD momentum
  lr_warmup: 0                   # Warmup epochs
  use_mixed_precision: true      # Enable mixed precision
  batch_size: 512                # Training batch size
  eval_batch_size: 2048          # Evaluation batch size

experiment:
  project_name: cifar_ssl        # Project name for logging into W&B
  weight_path: checkpoints/      # Local Checkpoint save directory
  data_path: "data/"            # Dataset path
  resume_from_checkpoint: null   # Path to resume from
  use_wandb: true               # Enable W&B logging
  eval_frequency: 10            # Evaluation frequency (epochs)
  save_frequency: 10            # Checkpoint save frequency (epochs)
  verbose: false                # Verbose logging
```

### Supported Configurations

- **Backbones**: Must specify either [`resnet18`, `mobilenet`]
    - For mobilenet we are using: 'mobilenetv4_conv_medium.e500_r256_in1k'   
    - For renset18, we removed the first pooling layer and changed the initial conv layer from 7x7 to 3x3
    - When changing the backbone, make sure to change the backbone_dim as well. This is the shape of the flattened output from the backbone
- **SSL Models**: [`simsiam`, `barlow_twins`] 
- **Mixed Precision**: Automatic mixed precision training support
    - Only enable if you have cuda
- **Experiment Tracking**: Weights & Biases integration

## 📊 Results
### CIFAR-10 Performance

- We report the final epoch KNN and Linear Eval Accuracy
- We pre-train all models for 800 epochs and finetune the linear layer for 100 epochs.

| Backbone | SSL Method | KNN (k=50) Accuracy % | Linear Eval Accuracy %|
|----------|------------|--------------------|--------------------|
| ResNet-18 | SimSiam | 90.59 | 91.6 |
| MobileNetV4-Medium | SimSiam | - | - |


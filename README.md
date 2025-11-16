BloodMNIST — Self-Supervised vs Fine-Tuning vs Scratch (ResNet-50)

CMP722 Assignment • Visual Representation Learning on BloodMNIST (MedMNIST)

This repo compares three training paradigms on a medical image dataset:

Scratch — train ResNet-50 from random init

ImageNet-FT — fine-tune ImageNet-pretrained ResNet-50

SSL+FT — SimCLR self-supervised pretraining on BloodMNIST, then supervised fine-tuning

We keep the same backbone (ResNet-50) in all scenarios and change only the initialization to isolate the effect of representation learning.

| Scenario | Init                                  | Method                               |  Test Loss | Test Accuracy | Test Macro-F1 |
| -------- | ------------------------------------- | ------------------------------------ | ---------: | ------------: | ------------: |
| **A**    | Random                                | From-scratch supervised              |     0.2291 |    **0.9366** |    **0.9290** |
| **B**    | ImageNet-pretrained                   | Transfer learning (fine-tune)        |     0.1970 |    **0.9617** |    **0.9588** |
| **C**    | SSL-pretrained (SimCLR on BloodMNIST) | Self-supervised pretrain + fine-tune | **0.1708** |    **0.9591** |    **0.9533** |


Takeaways

Both ImageNet-FT and SSL+FT outperform Scratch by ~2–3 points (Acc / Macro-F1).

ImageNet-FT is slightly best on Accuracy/F1; SSL+FT attains the lowest test loss (more calibrated/confident predictions).

📦 Dataset

BloodMNIST (from MedMNIST): ~17k images, 8 classes of blood cells

We use MedMNIST’s official train/val/test splits.

Images are loaded as RGB and resized to 64×64.

MedMNIST will auto-download via the Python API on first run.


🧰 Environment

Python ≥ 3.10

PyTorch & TorchVision

medmnist

scikit-learn, matplotlib, numpy, pandas

pip install torch torchvision torchaudio
pip install medmnist scikit-learn matplotlib pandas


⚙️Configuration

Backbone: ResNet-50 (TorchVision)

Input: 64×64, RGB, Normalize mean=std=0.5

Batch size: e.g., 128 (adjust to GPU RAM)

Optimizer: Adam, lr=1e-3

Supervised epochs: 30 (keep best val macro-F1 checkpoint)

SSL epochs (SimCLR): 50

SSL temperature (τ): 0.5

SSL augment: RandomResizedCrop, Flip, ColorJitter, RandomGrayscale (+Normalize)

Note on SSL split: I used train split only as unlabeled data (not train+val).

# <p align=center> Dynamic Cross-Modal Distillation with Modality Compensation for Multimodal Anomaly Detection </p>
#### Kaiyue Wang, Lu Zhang, Jieru Chi, Chenglizhao Chen, Teng Yu</sup>

<img width="1262" height="735" alt="FIG1" src="https://github.com/user-attachments/assets/d1a2134f-977f-4632-ae7a-528fdecbe36f" />

## Overview

**DCMD** (Dynamic Cross-Modal Distillation) is a novel multimodal anomaly detection framework designed to address the challenges of sensor failures and resource constraints in industrial environments. Unlike traditional fusion-based methods that assume full modality availability, DCMD integrates a **Modality Confidence Estimation (MCE)** module and a **Partial Modality Compensator (PMC)** into a dual-teacher–dual-student distillation framework.

Key features include:
* **Dynamic Adaptation:** Dynamically assesses modality quality to adjust fusion weights.
* **Bidirectional Distillation:** Enables symmetric and equitable knowledge exchange between RGB and Depth/Geometry modalities.
* **Modality Compensation:** Effectively restores features in partially missing or degraded modalities using cross-modal attention.

## Quick Start

### Environment Configuration

**1. Create virtual environment (optional):**

```bash
conda create -n anomaly-detection python=3.7
conda activate anomaly-detection
```

**2. Install dependencies:**

```bash
# Basic dependencies
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install numpy==1.21.6 scipy==1.7.3 matplotlib==3.5.3 scikit-image==0.19.3
pip install pillow==9.1.1 tqdm==4.64.1 pandas==1.3.5 opencv-python==4.6.0.66
pip install scikit-learn==1.0.2

# DCMD specific dependencies
pip install noise  # For noise generation if needed
```

### Data Preparation

1. **MVTec 3D-AD Dataset**
* Visit the [MVTec 3D-AD official website](https://www.mvtec.com/company/research/datasets/mvtec-3d-ad).
* Download and extract the dataset (contains 10 industrial categories with RGB and Depth maps).


2. **Eyecandies Dataset**
* Download the [Eyecandies dataset](https://eyecan-ai.github.io/eyecandies/).
* Used for robust evaluation under complex illumination and occlusion.


3. **Directory Structure Example**
```
/path/to/data/
├── mvtec_3d_ad/          # MVTec 3D-AD dataset
│   ├── bagel/
│   ├── cable_gland/
│   └── ...
└── eyecandies/           # Eyecandies dataset
    ├── CandyCane/
    ├── ChocoCookie/
    └── ...

```

### Training Execution

1. **Configure parameters:**
Modify parameters at the end of `train.py` (or `test.py` depending on your file structure):
```python
root_dir = "path/to/mvtec_3d_ad"        # Dataset root directory
category = "bagel"                      # Category to train
dataset_type = "mvtec3d"                # Dataset type choice

```


2. **Start training:**
```bash
python train.py

```

### Inference Testing

After training, the model is automatically saved as `dcmd_{category}_best.pth`. Use the following code for inference:

```python
# Load model
model = DCMDModel().to(DEVICE)
model.load_state_dict(torch.load(f"dcmd_{category}_best.pth"))
model.eval()

# Get test data
test_loader = get_dataloader(root_dir, category, train=False)

# Perform validation
i_auroc, p_auroc, pro = validate_dcmd(model, test_loader)
print(f"Test Results - I-AUROC: {i_auroc:.4f}, P-AUROC: {p_auroc:.4f}, PRO: {pro:.4f}")

```

## Model Architecture

The DCMD framework consists of three main innovative modules:

1. **Modality Confidence Estimation (MCE)**
* Dynamically evaluates the quality and reliability of input modalities (RGB and Depth).
* Adaptively allocates fusion weights to handle fluctuations in sensor data quality.


2. **Bidirectional Symmetric Distillation**
* A dual-teacher–dual-student structure.
* Facilitates balanced knowledge transfer between RGB and geometric modalities, overcoming the limitations of unidirectional master-slave structures.


3. **Partial Modality Compensator (PMC)**
* Uses attention-driven cross-modal feature restoration.
* Explicitly repairs locally missing modal regions to prevent feature flow disruption.

## Training Process

### Loss Functions

1. **Reconstruction Loss:** Standard MSE/SSIM for feature reconstruction.
2. **Distillation Loss:** Enforces consistency between teacher and student networks across modalities.

### Training Strategy

* Batch size: 32 (Adjustable based on GPU)
* Optimizer: Adam
* Learning rate scheduler: Cosine annealing
* Training epochs: Configurable (e.g., 200-600)

## Evaluation Metrics

* **Image-level AUROC (I-AUROC):** Evaluates anomaly detection performance at the image level.
* **Pixel-level AUROC (P-AUROC):** Evaluates anomaly localization accuracy at the pixel level.
* **Per-Region Overlap (PRO):** Measures overlap rate between predicted and ground truth anomaly regions (FPR < 30%).

## Visualization

Visualization results are automatically generated during inference:

* Original RGB/Depth images
* Ground truth anomaly mask
* Predicted anomaly heatmap (Confidence-weighted fusion)

Visualization results are saved in the output directory with naming format:

```
dcmd_{category}_{image_name}.png

```

## Customization and Extensions

### 1. Adjusting MCE Sensitivity

```python
def forward_mce(self, rgb_feat, depth_feat):
    # Adjust sigmoid scaling or thresholds for confidence estimation
    confidence_map = self.confidence_network(torch.cat([rgb_feat, depth_feat], dim=1))
    return confidence_map

```

### 2. Modifying Backbone Network

```python
# Switch backbones for different feature extraction capabilities
self.rgb_backbone = models.wide_resnet50_2(pretrained=True)
self.depth_backbone = models.wide_resnet50_2(pretrained=True)

```

## Experimental Results

### MVTec 3D-AD Dataset Performance

Our method achieves state-of-the-art results on the MVTec 3D-AD benchmark, significantly outperforming existing fusion-based methods in both detection (I-AUROC and P-AUROC) and localization (PRO) metrics.

**TABLE I: Anomaly Detection (P-AUROC) on MVTec 3D-AD**
    *DCMD achieves the highest mean P-AUROC of 99.4%.*

<img width="1113" height="483" alt="FIG2" src="https://github.com/user-attachments/assets/7f31e21f-74de-4e3d-ace6-8dfc80622856" />

**TABLE II: Per-Region Overlap (PRO) on MVTec 3D-AD**
*DCMD demonstrates superior localization performance with a mean PRO of 98.5%.*

<img width="1109" height="306" alt="FIG3" src="https://github.com/user-attachments/assets/79c660f6-f915-4bd6-bee5-8201a428872c" />

**TABLE III: Anomaly Detection (I-AUROC) on Eyecandies**
*DCMD demonstrates robust performance on complex datasets.*

<img width="1098" height="541" alt="FIG4" src="https://github.com/user-attachments/assets/7d9de37d-a0c5-4abc-ba38-3fbecf5e67d0" />

## Related Resources

* [MVTec 3D-AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-3d-ad)
* [Eyecandies Dataset](https://eyecan-ai.github.io/eyecandies/)
* [PyTorch Documentation](https://pytorch.org/docs/)

## Academic Applications

DCMD is suitable for research in:

* Multimodal Anomaly Detection
* Industrial Quality Inspection (RGB-D)
* Cross-Modal Knowledge Distillation
* Robust Sensing under Sensor Failure

## Common Issues

1. **Memory OOM:**
* The dual-branch structure (RGB + Depth) consumes more memory. Try reducing batch size or using mixed precision (`fp16`).


2. **Dataset Alignment:**
* Ensure RGB and Depth maps are properly aligned/registered before training.


3. **Slow Convergence:**
* If the confidence module oscillates, try lowering the learning rate for the MCE sub-network initially.


```

```

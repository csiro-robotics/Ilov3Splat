<div align="center">
<h1>Ilov3Splat: Instance-Level Open-Vocabulary 3D Scene Understanding in Gaussian Splatting</h1>

<h2> Accepted at ICPR 2026 </h2>

[**Binh Long Nguyen**](https://scholar.google.com.au/citations?user=MELpg_AAAAAJ)<sup>1,2</sup> , [**Kien Nguyen**](https://scholar.google.com.au/citations?user=18HsR5EAAAAJ)<sup>1</sup> , [**Sridha Sridharan**](https://scholar.google.com.au/citations?user=v8-lMdUAAAAJ)<sup>1</sup> , [**Clinton Fookes**](https://scholar.google.com.au/citations?user=VpaJsNQAAAAJ)<sup>1</sup> , [**Peyman Moghadam**](https://scholar.google.com.au/citations?user=QAVcuWUAAAAJ)<sup>1,2</sup>

<sup>1</sup>Queensland University of Technology&emsp;&emsp;&emsp;<sup>2</sup>CSIRO Robotics
<br>

<a href="https://arxiv.org/pdf/2605.04506.pdf"><img src='https://img.shields.io/badge/Paper-PDF-red' alt='Paper PDF on arXiv'></a>
<a href="https://arxiv.org/abs/2605.04506"><img src='https://img.shields.io/badge/arXiv-2605.04506-b31b1b' alt='arXiv abstract'></a>
<a href="https://csiro-robotics.github.io/Ilov3Splat"><img src='https://img.shields.io/badge/Project_Page-Ilov3Splat-green' alt='Project Page'></a>
<a href="https://github.com/csiro-robotics/Ilov3Splat"><img src='https://img.shields.io/badge/Code-GitHub-blue' alt='Code on GitHub'></a>
</div>

This repository hosts the project page and supporting materials for **Ilov3Splat**, an instance-level open-vocabulary 3D scene understanding framework built on Gaussian Splatting and accepted at ICPR 2026.

![overview](media/overview.png)

## 1. News

- **March 2026:** Ilov3Splat is accepted to ICPR 2026.
- **May 2026:** [arXiv preprint](https://arxiv.org/abs/2605.04506) is available.
- **June 2026:** Source code released on [GitHub](https://github.com/csiro-robotics/Ilov3Splat).

## 2. Installation

**Tested on:** Python 3.10, CUDA 12.1, Conda

1. **Create [conda](https://docs.conda.io/en/latest/) environment**

   ```bash
   conda create --name ilov3splat -y python=3.10
   conda activate ilov3splat
   ```

2. **Install [Nerfstudio](https://docs.nerf.studio/quickstart/installation.html) (with matching PyTorch / CUDA)**

   ```bash
   # install dependencies
   pip install torch torchvision
   pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

   # install nerfstudio
   git clone https://github.com/nerfstudio-project/nerfstudio.git
   cd nerfstudio
   pip install -e .
   ```

3. **Install Ilov3Splat**

   Ilov3Splat (including cuML for GPU clustering) is installed via `pyproject.toml`. From this repository:

   ```bash
   git clone https://github.com/csiro-robotics/Ilov3Splat.git
   cd Ilov3Splat
   pip install -e .
   ```


## 3. Data Preparation

Directory structure:

```
[DATA_ROOT]
├── lerf_ovs/
│   ├── figurines/ ramen/ teatime/ waldo_kitchen/
│   │   ├── images/
│   │   └── transforms.json
│   └── label/
├── scannet/
│   ├── scene0000_00/
│   │   ├── images/
│   │   ├── transforms.json
│   │   ├── *_vh_clean.ply                  -> points3d.ply
│   │   ├── *_vh_clean_2.labels.ply
│   │   ├── *_vh_clean.aggregation.json
│   │   └── *_vh_clean_2.0.010000.segs.json
│   └── ...
```

### LERF-OVS

- Download the dataset from [Kaggle: LERF-OVS](https://www.kaggle.com/datasets/claire100/lerf-ovs).
- Convert to Nerfstudio format with `ns-process-data` (same workflow as [custom data](https://docs.nerf.studio/quickstart/custom_dataset.html)):

  ```bash
  ns-process-data images --data <path/to/raw_scene/images> --output-dir <path/to/nerfstudio_scene>
  ```

### ScanNet

<!-- - Download our pre-processed data: **[OneDrive](https://)** / **[Dropbox](https://)**. -->
- The ScanNet dataset requires permission for use; follow the [ScanNet instructions](https://github.com/ScanNet/ScanNet) to apply for dataset access.
- To process additional scenes:
  1. Download `.sens` files using the official ScanNet `download-scannet.py` script.
  2. Extract RGB + poses using [`preprocess_2d_scannet.py`](https://github.com/pengsongyou/openscene/blob/main/scripts/preprocess/preprocess_2d_scannet.py).
  3. Convert to Nerfstudio format using `ns-process-data`.

### Custom Data

- Capture video → sample frames → COLMAP → convert to Nerfstudio format.
  See [Nerfstudio with custom data](https://docs.nerf.studio/quickstart/custom_dataset.html) for reference.

## 4. Training

Training is run via CLI commands (no bundled shell scripts yet).

### 1) SAM mask preprocessing

Download a SAM checkpoint from the [Segment Anything model checkpoints](https://github.com/facebookresearch/segment-anything#model-checkpoints). We use `sam_vit_h_4b8939.pth` (`vit_h`) in our experiments:

```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

Install the SAM preprocessing extra if needed:

```bash
pip install -e ".[preprocess-sam]"
```

Set the checkpoint path and extract masks for your Nerfstudio scene:

```bash
export SAM_CHECKPOINT=/path/to/sam_vit_h_4b8939.pth

ilov3splat-extract-sam-masks /path/to/nerfstudio_scene \
  --img-subdir images \
  --output-subdir sam \
  --sam-model vit_h \
  --compress
```

This writes `<scene_root>/sam/<image_stem>.npz` and `config.yaml`.

Defaults: `--levels` and `--sort score` (multi-level NPZ with score-sorted masks). Training loads the `whole` level via `instance_mask_npz_key` (see `Ilov3SplatDataManagerConfig`).

<!-- - Each NPZ includes `default`, `subpart`, `part`, and `whole` mask id maps (`[H, W]`, `-1` for background).
- Use `--no-levels` for a single `default` key only.
- Add `--binary-mask` to save per-mask stacks (`[N, H, W]`) instead of merged id maps. -->

The datamanager loads per-frame dense instance id maps from `<scene_root>/sam/<image_stem>.npz` (default key: `whole`). Configure via `Ilov3SplatDataManagerConfig` (`instance_mask_subdir`, `instance_mask_npz_key`).

### 2) Train

```bash
ns-train ilov3splat --data /path/to/nerfstudio_scene
```

After training, note the run directory (contains `config.yml` and `nerfstudio_models/`).

### 3) Cluster Gaussians

```bash
ilov3splat-cluster-gaussians /path/to/run_dir
```

For the LERF-OVS dataset, add `--no-assign-noise`:

```bash
ilov3splat-cluster-gaussians /path/to/run_dir --no-assign-noise
```

Clustering artifacts are saved to `/path/to/run_dir/clustering/` by default (including `labels.npy`).

## 5. Evaluation

### Visualization

```bash
# Viewer
ns-viewer --load-config <path/to/config.yml>
```

Open the Nerfstudio viewer during or after training. Available controls:

- **Run HDBSCAN** — cluster Gaussians interactively
- **Toggle RGB/Cluster** — switch between scene RGB and cluster-colored overlay
- **Load saved features** — load `clustering/labels.npy` from the run directory
- **Toggle lang 3D highlight** — turn green 3D highlight overlay on/off (run a query first)

The renderer also exposes an `instance` output (PCA-colored instance embedding map).

### LERF Evaluation (Open-Vocabulary 3D Object Selection)

Requires GT labels and clustering artifacts:

```bash
export LERF_OVS_LABEL_PATH=/path/to/lerf_ovs_labels   # the `label/` folder in the downloaded LERF-OVS dataset (contains figurines/, ramen/, etc.)
ilov3splat-eval-lerf-ovs /path/to/model_output
```

### ScanNet Evaluation (Category-agnostic 3D Instance Segmentation)

After clustering, load `clustering/labels.npy` from the run directory in the viewer (**Load saved features**) to visualize per-instance cluster assignments on the reconstructed scene. Use **Toggle RGB/Cluster** to switch between the RGB render and the cluster-colored overlay.

## 6. Acknowledgement

We would like to acknowledge the following repositories: [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio), [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), [OpenSplat3D](https://github.com/VisualComputingInstitute/opensplat3d), [LangSplat](https://github.com/minghanqin/LangSplat), [GARField](https://github.com/chungmin99/garfield), [FMGS](https://github.com/google-research/foundation-model-embedded-3dgs), [CLIP](https://github.com/openai/CLIP) and [SAM](https://segment-anything.com/).

This work was supported in part by the Australian Research Council Discovery Project under Grant DP250103634, and in part by the Commonwealth Scientific and Industrial Research Organisation (CSIRO). The authors acknowledge continued support from the CSIRO's Embodied AI Cluster.

## Citation

If you find this repository useful, please cite:

```
@inproceedings{nguyen2026ilov3splat,
  author    = {Nguyen, Binh Long and Nguyen, Kien and Sridharan, Sridha and Fookes, Clinton and Moghadam, Peyman},
  title     = {Ilov3Splat: Instance-Level Open-Vocabulary 3D Scene Understanding in Gaussian Splatting},
  booktitle = {International Conference on Pattern Recognition (ICPR)},
  year={2026}
}
```

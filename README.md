<div align="center">

# LAD-Drive: Bridging Language and Trajectory with Action-Aware Diffusion Transformers

## IROS 2026

[![arXiv](https://img.shields.io/badge/arXiv-2603.02035-b31b1b.svg?style=for-the-badge)](https://arxiv.org/abs/2603.02035)

[**Fabian Schmidt**](https://rl.uni-freiburg.de/people/schmidt)<sup>1,2</sup> · [**Karol Fedurko**](https://scholar.google.com/citations?user=FIpyuF8AAAAJ)<sup>1</sup> · [**Markus Enzweiler**](https://markus-enzweiler.de/)<sup>1</sup> · [**Abhinav Valada**](https://rl.uni-freiburg.de/people/valada)<sup>2</sup>

<sup>1</sup> **Esslingen University of Applied Sciences** · <sup>2</sup> **University of Freiburg**

---

</div>

## 📢 Updates
- **[2026/03]** Paper released on arXiv!
- **[2026/06]** LAD-Drive has been accepted to IROS 2026. See you in Pittsburgh!
- **[2026/08]** Release of full training code, fine-tuned model, and evaluation scripts.

---

## 📝 Citation
If you find our work useful in your research, please consider citing:

```bibtex
@misc{schmidt2026laddrive,
      title={LAD-Drive: Bridging Language and Trajectory with Action-Aware Diffusion Transformers}, 
      author={Fabian Schmidt and Karol Fedurko and Markus Enzweiler and Abhinav Valada},
      year={2026},
      eprint={2603.02035},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={[https://arxiv.org/abs/2603.02035](https://arxiv.org/abs/2603.02035)}, 
}
```

## Table of Contents
1. [Setup](#setup)
2. [Model Weights](#lmdrive-weights)
3. [Data Collection](#data-collection)
4. [Training](#training)
   1. [Vision encoder pre-training](#vision-encoder-pre-training)
   2. [Instruction finetuning](#instruction-finetuning)
5. [Evaluation](#evaluation)
6. [HPC / SLURM Cluster Execution](#hpc--slurm-cluster-execution)
8. [License](#license)

## Setup

Our project is built on three parts: (1) vision encoder (corresponding repo: timm); (2) vision LLM (corresponding repo: LAVIS); (3) data collection, agent controller (corresponding repo: InterFuser, Leaderboard, ScenarioRunner). 

### Paths to Configure
Before running training or evaluation, please make sure to adjust the hardcoded absolute paths to match your current workspace structure. The following files contain paths that need to be updated:

1. **`leaderboard/team_code/lad_drive_config.py`**
   - Update `lad_drive_ckpt` to point to your trained model checkpoint.
   - Update `preception_model_ckpt` to the checkpoint path of the vision encoder.
   - Update `llm_model` to the checkpoint path of the LLM (LLaMA/Vicuna/LLaVA).
2. **`leaderboard/scripts/run_evaluation_lad_drive.sh`**
   - Update `PYTHONPATH` to point to the correct `vision_encoder` directory in your current workspace.
   - Verify `CARLA_ROOT` points to your CARLA installation.
3. **`LAVIS/lavis/projects/lad_drive/lad_drive.yaml`**
   - Update `preception_model_ckpt` (vision encoder checkpoint).
   - Update `llm_model` (base LLM path).
   - Update `llm_pretrained_ckpt` (LMDrive LLM pre-trained checkpoint).
   - Update `plan_anchor_path` to point to the `anchors/` directory.
4. **`LAVIS/run_lad_drive.sh`**
   - Update `PYTHONPATH` and `CONFIG_BASE_PATH` to point to your respective workspaces.
   - Update `SRC_TARS`, `IDX_SRC`, `NAV_SRC`, `NOTICE_SRC` to point to the dataset location.

Install anaconda
```Shell
wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
bash Anaconda3-2020.11-Linux-x86_64.sh
source ~/.bashrc
```

Clone the repo and build the environment

```Shell
git clone https://github.com/iis-esslingen/LAD-Drive.git
cd lad-drive
conda create -n lad_drive python=3.8
conda activate lad_drive
cd vision_encoder
pip3 install -r requirements.txt
python setup.py develop # if you have installed timm before, please uninstall it
cd ../LAVIS
pip3 install -r requirements.txt
python setup.py develop # if you have installed LAVIS before, please uninstall it

pip install flash-attn --no-build-isolation # optional
```

Download and setup CARLA 0.9.10.1
```Shell
chmod +x setup_carla.sh
./setup_carla.sh
pip install carla
```

> If you encounter some problems related to Carla, please refer to [Carla Issues](https://github.com/carla-simulator/carla/issues) and [InterFuser Issues](https://github.com/opendilab/InterFuser) first.

## LAD-Drive Weights

| Version | Size |  Checkpoint | VisionEncoder | LLM-base | 
|---------|------|------------|----------------|-----------|
| LAD-Drive | 7B | [LAD-Drive](https://huggingface.co/iis-esslingen/LAD-Drive) | [R50](https://huggingface.co/OpenDILabCommunity/LMDrive-vision-encoder-r50-v1.0) | [LLaVA-v1.5-7B](https://huggingface.co/liuhaotian/llava-v1.5-7b) 
| LMDrive-1.0 (LLaVA-v1.5-7B) | 7B |  [LMDrive-llava-v1.5-7b-v1.0](https://huggingface.co/OpenDILabCommunity/LMDrive-llava-v1.5-7b-v1.0) | [R50](https://huggingface.co/OpenDILabCommunity/LMDrive-vision-encoder-r50-v1.0) | [LLaVA-v1.5-7B](https://huggingface.co/liuhaotian/llava-v1.5-7b) 

Download pretrained models
```Shell
git lfs install

git clone https://huggingface.co/liuhaotian/llava-v1.5-7b
git clone https://huggingface.co/OpenDILabCommunity/LMDrive-llava-v1.5-7b-v1.0
git clone https://huggingface.co/OpenDILabCommunity/LMDrive-vision-encoder-r50-v1.0
```

##### Data Collection

For details on data collection, see the [LMDrive dataset section](https://github.com/opendilab/LMDrive#dataset).

## Training

LAD-Drive vision encoder backbone is initialized using the official LMDrive checkpoint: https://huggingface.co/OpenDILabCommunity/LMDrive-vision-encoder-r50-v1.0

LAD-Drive's training consists of two stages: 1) mask the action decoder's loss contribution to facilitate spatial grounding; 2) achieve semantic alignment by modulating spatial refinement based on the probabilistic action conditioning.

LAD-Drive is trained on 8 L40s GPUs with 48GB memory. To train on fewer GPUs, you can reduce the `batch-size` and the `learning-rate` while maintaining their proportion.
    
### Vision encoder pre-training

> [NOTE]
> This section does not need to be done to achieve LAD-Drive's performance as we initialize the vision encoder using LMDrive's checkpoint.

Pretrain takes around 2~3 days for the visual encoder on 8x A100 (80G). Once the training is completed, you can locate the checkpoint of the vision encoder in the `output/` directory.

```bash
cd vision_encoder
bash scripts/train.sh
```

Some options to note:

- `GPU_NUM`:  the number of GPUs you want to use. By default, it is set to 8.
- `DATASET_ROOT`: the root directory for storing the dataset.
- `--model`: the structure of visual model. You can choose memfuser_baseline_e1d3_r26 which replaces ResNet50 with ResNet26. It's also possible to create new model variants in `visual_encoder/timm/models/memfuser.py`
- `--train-towns/train-weathers`: the data filter for the training dataset. Similarly, there are corresponding options, `val-towns/val-weathers` to filter the validation dataset accordingly.

### Finetuning

Finetuning takes around 10 hours on 8x L40s (48GB). Once the training is completed, you can locate the checkpoint of the adapters and qformer in the `lavis/output/` directory.

```bash
cd LAVIS
sbatch run_lad_drive.sh
```

Some options in the config.yaml to note:

- `preception_model`:  the model architecture of the vision encoder.
- `preception_model_ckpt`: the checkpoint path of the vision encoder.
- `llm_model`: the checkpoint path of the llm (Vicuna/LLaVA).
- `plan_anchor_path`: the path to the diffusion decoder's anchors.
- `split_section_num_for_visual_encoder`: the number of sections the frames are divided into during the forward encoding of visual features. Higher values can save more memory, and it needs to be a factor of `token_max_length`.
- **datasets:**
  - `storage`: the root directory for storing the dataset.
  - `towns/weathers`: the data filter for training/evaluating.
  - `token_max_length`: the maximum number of frames, if the number of frames exceeds this value, they will be truncated.
  - `sample_interval`: the interval at which frames are sampled.

## Evaluation
Start a CARLA server (described above) and run the required agent. The adequate routes and scenarios files are provided in ```leaderboard/data``` and the required variables need to be set in ```leaderboard/scripts/run_evaluation.sh```.

## HPC / SLURM Cluster Execution

We provide specialized SLURM submission scripts for environments backed by high-performance computing clusters:

- **Training**: `cd LAVIS && sbatch run_lad_drive.sh`
  - Submits an 8-node job that automatically stages dataset files to node-local scratch storage (`/localscratch`), overrides configs dynamically, and spawns distributed PyTorch training across SLURM ranks.
- **Data Collection**: `bash sbatch_data_collection.sh`
  - Scans for pending data collection bash scripts (e.g., `data_collection/bashs/sub-*/todo_*.sh`) and schedules an independent `sbatch` job for each route segment.
- **Evaluation**: `bash sbatch_eval_lad_drive.sh`
  - Dispatches separate parallel SLURM jobs for `benchmark_tiny`, `benchmark_short`, and `benchmark_long`. Each job starts a headless CARLA simulation using `run_evaluation_lad_drive.sh`.

Run the final evaluation for all benchmarks:

```shell
./sbatch_eval_lad_drive.sh
```

## License
All code within this repository is under [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

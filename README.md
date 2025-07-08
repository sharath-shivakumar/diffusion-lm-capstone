# Diffusion-LM Setup and Training Guide

This guide outlines the complete setup and execution workflow for Diffusion-LM, a language model using diffusion-based generation and classifier-guided control.

## Environment Setup

```bash
# 1. Create Conda Environment
conda create -n diffusion_lm_ce python=3.9 -y
conda activate diffusion_lm_ce

# 2. Install System Dependencies
sudo apt update
sudo apt install -y build-essential gfortran

# 3. Install PyTorch with CUDA 12.1 Support
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia

# 4. Clone Local Modules (assumes current directory is repo root)
pip install -e improved-diffusion/
pip install -e transformers/

# 5. Install Required Python Packages
pip install numpy==1.26.4
pip install \
  datasets==2.8.0 \
  wandb<0.15.0 \
  pydantic==1.7.4 \
  spacy==3.2.4 \
  thinc==8.0.17 \
  tokenizers==0.11.6 \
  protobuf==3.20.3

# 6. NLP Tools
python -m spacy download en_core_web_sm
pip install stanza spacy_stanza benepar scikit-learn
python -c "import stanza; stanza.download('en')"
python -c "import benepar; benepar.download('benepar_en3')"

# 7. Sanity Checks
python -c "import torch; import numpy; import spacy; import thinc; import pydantic; import datasets; print(torch.__version__, numpy.__version__, spacy.__version__)"
python -c "import torch; print(torch.cuda.is_available())"
python -c "from datasets import load_metric; print('Datasets OK')"
python -c "import benepar; benepar.load('benepar_en3'); print('benepar ready')"
python -c "import sklearn; print('sklearn ready')"
```

---

## Training Pipeline

### 1. Train a Diffusion-LM model (sqrt schedule)
```bash
python scripts/run_train.py \
  --diff_steps 2000 \
  --model_arch transformer \
  --lr 0.0001 \
  --lr_anneal_steps 200000 \
  --seed 102 \
  --noise_schedule sqrt \
  --in_channel 16 \
  --modality e2e-tgt \
  --submit no \
  --padding_mode block \
  --app "--predict_xstart True --training_mode e2e --vocab_size 821 --e2e_train ../datasets/e2e_data" \
  --notes xstart_e2e
```

### 2. Decode Generated Outputs
```bash
mkdir -p generation_outputs
python scripts/batch_decode.py \
  ~/Capstone/Diffusion-LM-main/improved-diffusion/diffusion_models/diff_e2e-tgt_block_rand16_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd102_xstart_e2e \
  -1.0 ema
```

### 3. Train Classifier Models for Control

#### Tree Control:
```bash
python train_run.py \
  --experiment e2e-tgt-tree \
  --app "--init_emb ~/Capstone/Diffusion-LM-main/improved-diffusion/diffusion_models/diff_e2e-tgt_block_rand16_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd102_xstart_e2e --n_embd 16 --learned_emb yes" \
  --pretrained_model bert-base-uncased \
  --epoch 6 \
  --bsz 10
```

#### POS Control:
```bash
python train_run.py \
  --experiment e2e-tgt-pos \
  --app "--init_emb ~/Capstone/Diffusion-LM-main/improved-diffusion/diffusion_models/diff_e2e-tgt_block_rand16_transformer_lr0.0001_0.0_2000_cosine_Lsimple_h128_s2_d0.1_sd102_xstart_cosine --n_embd 16 --learned_emb yes" \
  --pretrained_model bert-base-uncased \
  --epoch 6 \
  --bsz 10
```

---

## Infill Decoding with Classifier Constraints

```bash
python scripts/infill.py \
  --model_path ~/Capstone/Diffusion-LM-main/improved-diffusion/diffusion_models/diff_e2e-tgt_block_rand16_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd102_xstart_e2e/model050000.pt \
  --eval_task_ 'control_tree' \
  --use_ddim True \
  --notes "tree_adagrad" \
  --eta 1. \
  --verbose pipe \
  --classifier_path /home/exouser/Capstone/Diffusion-LM-main/classifier_models/e2e-tgt-tree_e=6_b=10_m=bert-base-uncased_wikitext-103-raw-v1_101_wp_None
```

Repeat similarly for cosine schedule:
```bash
python scripts/infill.py \
  --model_path ~/Capstone/Diffusion-LM-main/improved-diffusion/diffusion_models/diff_e2e-tgt_block_rand16_transformer_lr0.0001_0.0_2000_cosine_Lsimple_h128_s2_d0.1_sd102_xstart_cosine/model050000.pt \
  --eval_task_ 'control_tree' \
  --use_ddim True \
  --notes "tree_adagrad_cosine" \
  --eta 1. \
  --verbose pipe \
  --classifier_path /home/exouser/Capstone/Diffusion-LM-main/classifier_models/e2e-tgt-tree_e=6_b=10_m=bert-base-uncased_wikitext-103-raw-v1_101_wp_None \
  --num_samples 1
```

## Notes:
- Ensure target control files are placed in `improved-diffusion/control_gen`.
- Modify `eval_control.py` for dynamic path resolution of `EVALB`.

This pipeline enables training, decoding, and control evaluation for E2E NLG tasks using Diffusion-LM.

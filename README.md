# refine
Code for our paper **Rethinking Fine-Tuning when Scaling Test-Time Compute: Limiting Confidence Improves Mathematical Reasoning**.

# Installation
Download and install conda.
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
...
```
## Create env
```
conda create -y --name refine python=3.10
conda activate refine
conda install -n refine ipykernel --update-deps --force-reinstall
```
## Install the package
```
git clone git@github.com:allanraventos/refine.git
cd refine
pip install -e .
pip install flash-attn --no-build-isolation
```
Check deepspeed installation
```
python -c "import deepspeed; deepspeed.ops.adam.cpu_adam.CPUAdamBuilder().load()"
python -c "import deepspeed; deepspeed.ops.adam.fused_adam.FusedAdamBuilder().load()"
```
## Login
```
huggingface-cli login
wandb login
```
##  Additional dependencies for analysis
```
pip install antlr4-python3-runtime==4.11
pip install scikit-learn
```
# Math finetuning
## Run training
Running `export NCCL_CUMEM_ENABLE=0` is required in some cases; also change to `vllm==0.6.3.post1` if facing problems in some cluster setups. Set up wandb with:
```
export WANDB_PROJECT=<wandb_project>
export WANDB_ENTITY=<wandb_entity>
```
To launch training, from the `refine` directory, run:
```
accelerate launch --config_file default_config.yaml training/train.py --with_tracking --num_train_epochs 6 --dataset_name $HOME/refine/refine/data/math --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 16 --model_name_or_path meta-llama/Meta-Llama-3-8B --prompt_template base_format --gradient_checkpointing --seed 10 --N 1 --filter_strategy cot --cot_filter_N 64 --threshold 0.1 --learning_rate 2e-5
```
The default behavior with an 8-GPU setup is that two vllm engines are constructed, one to run on ranks 0-3 with current model weights broadcasted to it from rank 4, and another to run on ranks 4-7 with weights broadcasted from rank 0. This can be changed manually in `train.py`, to run four engines for example, by changing `control_gpus` and the internal `CUDA_VISIBLE_DEVICES` logic.

Checkpoints will be saved to `exps/<wandb_run_id>`. In order to run inference on the trained model, run:
```
python training/test.py --model $HOME/refine/refine/exps/<wandb_run_id>/epoch_<epoch_number>  --dataset_name $HOME/refine/refine/data/math --prompt_template base_format --N <256 / 1024> &
```
(We will add support for DeepSpeed ZeRO Stage 3 evaluation soon, for which the checkpoints need to be converted prior to evaluation.)

To actually get evaluation numbers, step through the `eval.ipynb` notebook.

# Acknowledgements
Our code builds on https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm_no_trainer.py. We thank the huggingface contributors for their work.

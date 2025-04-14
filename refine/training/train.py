#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.
# adapt from https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm_no_trainer.py

import argparse
import json
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path
import numpy as np
import deepspeed
import datasets
import torch
import torch.nn as nn
import torch.distributed as dist
import accelerate
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset, load_from_disk
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
import wandb
import hashlib

from refine.training.NCO import custom_loss_function1
from refine.training.NCO import f as fac
from refine.training.filter_samples import (
    filter_training_data,
    filter_training_data_NCO,
)

from refine.training.loss_functions import (
    compute_mean_loss_per_example,
    custom_eval_function1,
    custom_loss_function2,
)
from refine.training.arg_parser import parse_args, create_wandb_config
from refine.training.registry import prompt_registry, loss_registry, filter_registry

# TODO: could import this optionally if filter strategy requires it
from vllm import SamplingParams
from refine.training.vllm_engine import broadcast_weights, construct_inference_engine

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.45.0.dev0")

logger = get_logger(__name__)

require_version(
    "datasets>=3.1.0",
    "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt",
)

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def main():
    args = parse_args()
    custom_loss_function = loss_registry[args.loss_function]
    prompt = prompt_registry[args.prompt_template]
    config = create_wandb_config(args)
    expname = hashlib.md5(str(config).encode()).hexdigest()
    args.output_dir = "exps/" + expname
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.json"), "w") as ff:
        json.dump(config, ff)

    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    # if args.seed is not None:
    #     set_seed(args.seed, device_specific=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_seedable_sampler=True,
        **accelerator_log_kwargs,
    )

    # Technically could always use GPU 4 as main, but this is ok
    if args.filter_strategy == "cot":
        # is_main_process = accelerator.process_index == tensor_parallel_size
        is_main_process = accelerator.is_main_process

        # TODO: check that this is ok; now we should be able to set the choice
        # of which GPUs to run, as long as CUDA_VISIBLE_DEVICES is not a problem
    else:
        is_main_process = accelerator.is_main_process

    filter_fn = filter_registry[args.filter_strategy]

    # wandb
    if is_main_process:
        wandb.init(project=os.getenv("WANDB_PROJECT"), entity=os.getenv("WANDB_ENTITY"), name=expname, config=config)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        if "gsm8k" in args.dataset_name:
            raise NotImplementedError
        elif "math" in args.dataset_name:
            raw_datasets = load_from_disk(args.dataset_name)
        if "leandojo" in args.dataset_name:
            raw_datasets = load_from_disk(args.dataset_name)
    else:
        raise NotImplemented

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(
            args.config_name,
            trust_remote_code=args.trust_remote_code,
        )
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=args.trust_remote_code,
            attn_implementation="flash_attention_2",
        )
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name,
            use_fast=not args.use_slow_tokenizer,
            trust_remote_code=args.trust_remote_code,
        )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            use_fast=not args.use_slow_tokenizer,
            trust_remote_code=args.trust_remote_code,
        )

    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    if tokenizer.pad_token_id is None or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    if args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            trust_remote_code=args.trust_remote_code,
            torch_dtype=torch.bfloat16,  # NOTE: added. This might be problematic but needed for flash attention
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(
            config, trust_remote_code=args.trust_remote_code
        )

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    prefix_column = "question"
    completion_column = "answer_s" if args.no_COT else "solution"
    answer_column = "answer"

    if "leandojo" in args.dataset_name:
        prefix_column = "state"
        completion_column = answer_column = "tactic"

    def tokenize_function(examples):

        if "gsm8k" in args.dataset_name:
            raise NotImplementedError
        elif "math" in args.dataset_name or "leandojo" in args.dataset_name:
            question_answer = [
                prompt.format(
                    qe=ex[prefix_column],
                    ans=ex[completion_column],
                )
                + tokenizer.eos_token
                for ex in examples
            ]

            question = [
                prompt.format(
                    qe=ex[prefix_column],
                    ans="",
                )
                for ex in examples
            ]

            answer = [ex[answer_column] for ex in examples]

        tokenized_state_tactic = tokenizer(
            question_answer,
            # padding="max_length",
            padding="longest",
            max_length=8_192,  # FIXME: should this be 8192?
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        batch = {}
        batch["input_ids"] = tokenized_state_tactic.input_ids[:, : args.max_length]
        batch["attention_mask"] = tokenized_state_tactic.attention_mask[
            :, : args.max_length
        ]
        batch["labels"] = tokenized_state_tactic.input_ids.clone()[:, : args.max_length]
        tokenized_state_tactic = tokenizer(
            question,
            padding="longest",
            # padding="max_length",
            max_length=8192,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        batch["answer"] = answer
        batch["question"] = tokenized_state_tactic.input_ids

        # Mask the 'state' and 'pad' portion in the 'labels'
        for idx, ex in enumerate(examples):
            # Tokenize 'state' separately to determine the number of state tokens
            tokenized_state = tokenizer(
                prompt.format(qe=ex[prefix_column], ans=""),
                add_special_tokens=True,
            )
            state_length = len(tokenized_state.input_ids)
            if "gsm8k" in args.dataset_name:
                raise NotImplementedError
            elif "math" in args.dataset_name or "leandojo" in args.dataset_name:
                non_pad_tokens = tokenizer(
                    prompt.format(
                        qe=ex[prefix_column],
                        ans=ex[completion_column],
                    )
                    + tokenizer.eos_token,
                    add_special_tokens=True,
                )
            non_pad_length = len(non_pad_tokens.input_ids)

            if "leandojo" in args.dataset_name:
                batch["labels"][idx, :state_length] = -100
                batch["labels"][idx, non_pad_length:] = -100
            else:

                batch["labels"][idx, : state_length - 1] = -100
                batch["labels"][idx, non_pad_length:] = -100

        if "leandojo" in args.dataset_name:
            batch["labels"] = batch["labels"][:, :2048]
            batch["input_ids"] = batch["input_ids"][:, :2048]
            batch["attention_mask"] = batch["attention_mask"][:, :2048]

        return batch

    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["test"]

    if args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > config.max_position_embeddings:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                f"Using block_size={min(1024, config.max_position_embeddings)} instead. You can change that default value by passing --block_size xxx."
            )
            block_size = min(1024, config.max_position_embeddings)
    else:
        if args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({args.block_size}) is larger than the maximum length for the model "
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(args.block_size, tokenizer.model_max_length)

    global_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    # TODO: this setup is a bit long, might want to factor out
    if args.filter_strategy == "cot":
        control_gpus = (0, 2, 4, 6)
        tensor_parallel_size = accelerator.num_processes // len(control_gpus)
        is_control_process = accelerator.process_index in control_gpus
        if is_main_process:
            assert is_control_process

        sleep_group = dist.new_group(
            ranks=list(range(accelerator.num_processes)), backend="gloo"
        )

        if is_control_process:
            original_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
            assert original_cuda_visible_devices is None

            if accelerator.process_index == 0:
                controller_id = 0
                os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
            elif accelerator.process_index == 2:
                controller_id = 1
                os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"
            elif accelerator.process_index == 4:
                controller_id = 2
                os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"
            elif accelerator.process_index == 6:
                controller_id = 3
                os.environ["CUDA_VISIBLE_DEVICES"] = "0,7"
            else:
                raise ValueError

            model_update_group, inference_engine = construct_inference_engine(
                args.model_name_or_path,
                # args.seed,  # FIXME: use same seed?
                1234,
                tensor_parallel_size=tensor_parallel_size,
                backend="nccl",
                max_model_length=8_192,
                gpu_memory_utilization=0.3,
            )

            sampling_params = SamplingParams(
                temperature=1,
                n=args.cot_filter_N,
                logprobs=0,
                max_tokens=1_024,  # NOTE: fixing this here (seems 512 might be insufficient)
            )

            ##########################################################################
            # Test inference
            ##########################################################################
            # bs_per_engine = 64 // len(control_gpus)
            # t0 = time.time()
            # start_index = np.random.randint(0, len(train_dataset) - bs_per_engine)
            # sample_list = [
            #     prompt.format(qe=train_dataset[_i]["question"], ans="")
            #     for _i in range(start_index, start_index + bs_per_engine)
            # ]
            # outputs = ray.get(
            #     inference_engine.generate.remote(sample_list, sampling_params)
            # )
            # print(f"Time for inference: {time.time() - t0:.2f} seconds")
        else:
            # More just a convenience for when `filter_fn` is called
            sampling_params = None
            inference_engine = None
            controller_id = None

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=tokenize_function,
        batch_size=args.per_device_train_batch_size,
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=tokenize_function,
        batch_size=args.per_device_eval_batch_size,
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )

    if args.max_train_steps is None:
        args.max_train_steps = (
            len(train_dataset) * args.num_train_epochs // global_batch_size
        )
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=(
            args.max_train_steps
            if overrode_max_train_steps
            else args.max_train_steps * accelerator.num_processes
        ),
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = (
        accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )
    )

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config[
            "lr_scheduler_type"
        ].value
        accelerator.init_trackers("test", experiment_config)

    # Train!
    total_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not is_main_process)

    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        raise NotImplementedError

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    world_size = accelerator.num_processes

    per_device_batch_size = global_batch_size // world_size

    def train_on_a_batch(sample_indices, probs):
        # FIXME: don't love using "global" variables here

        local_tracked_loss = torch.tensor(0.0).to(accelerator.device)
        local_batch_size = len(sample_indices)

        assert local_batch_size % args.per_device_train_batch_size == 0

        sample_indices = sample_indices.reshape(-1, args.per_device_train_batch_size)
        sample_probs = probs.reshape(-1, args.per_device_train_batch_size)

        for indices, p in zip(sample_indices, sample_probs):
            # if True:
            batch = tokenize_function(train_dataset.select(indices))
            batch["inputs"] = batch["input_ids"].cuda()
            batch["labels"] = batch["labels"].cuda()
            batch["attention_mask"] = batch["attention_mask"].cuda()

            outputs = model(
                input_ids=batch["inputs"],
                labels=batch["labels"],
                attention_mask=batch["attention_mask"],
            )

            if args.filter_strategy == "cot":

                per_sample_losses = custom_loss_function(
                    outputs.logits, batch["labels"], 1
                )  # pass 1 so that it is the CELoss
                with torch.no_grad():
                    #
                    fac_sample = fac(p, args.N)
                per_sample_losses = per_sample_losses * fac_sample

                local_avg_loss = per_sample_losses.mean()

            elif args.filter_strategy == "NCO_formal":
                # for NCO_formal, the p is the factor
                # print("NCO formal")
                per_sample_losses = custom_loss_function(
                    outputs.logits, batch["labels"], 1
                )  # pass 1 so that it is the CELoss
                per_sample_losses = per_sample_losses * p
                local_avg_loss = per_sample_losses.mean()

            else:

                per_sample_losses = custom_loss_function(
                    outputs.logits, batch["labels"], args.N
                )

                local_avg_loss = per_sample_losses.mean()
            # scaled_loss = local_avg_loss * (
            #     local_batch_size * world_size / global_batch_size
            # )

            scaled_loss = (
                local_avg_loss * local_batch_size / global_batch_size * world_size
            )

            # Track losses
            # local_tracked_loss += local_avg_loss.detach()
            local_tracked_loss += scaled_loss.detach()

            # Backward step
            accelerator.backward(local_avg_loss)
            # accelerator.deepspeed_engine_wrapped.engine.backward(scaled_loss)
        # accelerator.deepspeed_engine_wrapped.engine.step()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        local_tracked_loss /= len(sample_indices)
        global_tracked_loss = accelerator.reduce(local_tracked_loss, reduction="mean")

        return global_tracked_loss

    def eval():
        model.eval()
        losses = []
        celosses = []
        celosses2 = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(
                    input_ids=batch["input_ids"],
                    labels=batch["labels"],
                    attention_mask=batch["attention_mask"],
                )
            logits = outputs.logits

            # Compute custom loss
            if args.N == 1:
                loss = outputs.loss

            else:
                loss = custom_loss_function(logits, batch["labels"], args.N)

            loss = torch.where(
                torch.isnan(loss), torch.tensor(0.0, device=loss.device), loss
            )

            celoss2 = custom_loss_function2(logits, batch["labels"], 1)
            celoss2 = torch.where(
                torch.isnan(celoss2), torch.tensor(0.0, device=celoss2.device), celoss2
            )

            celosses2.append(
                accelerator.gather_for_metrics(
                    celoss2.repeat(args.per_device_eval_batch_size)
                )
            )
            # Append loss for each batch (replicate across devices for metrics)
            losses.append(
                accelerator.gather_for_metrics(
                    loss.repeat(args.per_device_eval_batch_size)
                )
            )

            celoss = outputs.loss
            celoss = torch.where(
                torch.isnan(celoss), torch.tensor(0.0, device=celoss.device), celoss
            )

            celosses.append(
                accelerator.gather_for_metrics(
                    celoss.repeat(args.per_device_eval_batch_size)
                )
            )

        losses = torch.cat(losses)
        eval_loss = torch.mean(losses)

        celosses = torch.cat(celosses)
        eval_celoss = torch.mean(celosses)

        celosses2 = torch.cat(celosses2)
        eval_celoss2 = torch.mean(celosses2)

        model.train()

        return eval_loss, eval_celoss, eval_celoss2

    global_discarded_samples = 0
    completed_steps = 0

    sample_indices_unused = []
    unused_indices = []  # These are all hard
    unused_probs = []

    for epoch in range(args.num_train_epochs):
        model.train()

        last_used = 0

        sample_indices = list(range(len(train_dataset)))
        random.shuffle(sample_indices)
        if is_main_process:
            sample_indices = torch.tensor(sample_indices, dtype=int).to(
                accelerator.device
            )
        else:
            sample_indices = torch.zeros(len(sample_indices), dtype=int).to(
                accelerator.device
            )
        sample_indices = (
            accelerator.reduce(sample_indices, reduction="sum").cpu().tolist()
        )

        # See (*)
        sample_indices = sample_indices_unused + sample_indices

        while True:
            # (*) if running out of data, then just prepend remaining samples to next epoch dataset
            if last_used + global_batch_size > len(sample_indices):
                sample_indices_unused = sample_indices[last_used:]
                break
            if is_main_process:
                progress_bar.update(1)

            # indices to run inference on (NOTE: * always * global batch size)
            inds = sample_indices[last_used : last_used + global_batch_size]
            last_used += global_batch_size

            if args.filter_strategy == "cot":
                # Do not filter in first 20 steps (model is bad and inference will be slow)
                if completed_steps < 20:
                    hard_indices, probs = (
                        inds,
                        torch.zeros(len(inds)).float().cpu().tolist(),
                    )
                else:
                    hard_indices, probs = filter_fn(
                        inference_engine,
                        inds,
                        train_dataset,
                        tokenizer,
                        prompt,
                        controller_id,
                        len(control_gpus),
                        sampling_params,
                        accelerator,
                        sleep_group,
                        threshold=args.threshold,
                    )
            else:
                hard_indices, probs = filter_fn(
                    inds=inds,
                    model=model,
                    accelerator=accelerator,
                    dataset=train_dataset,
                    collate_fn=tokenize_function,
                    global_batch_size=global_batch_size,
                    is_main_process=is_main_process,
                    N=args.N,
                    threshold=args.threshold,
                )

            global_discarded_samples += global_batch_size - len(hard_indices)

            unused_indices += hard_indices
            unused_probs += probs

            if len(unused_indices) < global_batch_size:

                if last_used >= len(sample_indices):
                    break
                else:
                    continue
            else:
                batch_indices = unused_indices[:global_batch_size]
                unused_indices = unused_indices[global_batch_size:]
                batch_probs = unused_probs[:global_batch_size]
                unused_probs = unused_probs[global_batch_size:]

            assert len(unused_indices) == len(unused_probs)

            inds = (
                torch.tensor(batch_indices)
                .reshape(per_device_batch_size, world_size)
                .to(accelerator.device)
            )

            probs = (
                torch.tensor(batch_probs)
                .to(accelerator.device)
                .reshape(per_device_batch_size, world_size)
            )

            global_tracked_loss = train_on_a_batch(
                inds[:, accelerator.process_index].cpu().numpy(),
                probs[:, accelerator.process_index],
            )

            # TODO: I guess only do this starting after 20 steps?
            # if args.filter_strategy == "cot" and is_main_process:
            if args.filter_strategy == "cot" and is_control_process:
                unwrapped_model = accelerator.unwrap_model(model)
                broadcast_weights(unwrapped_model, inference_engine, model_update_group)

            completed_steps += 1

            if args.with_tracking and is_main_process:
                wandb.log(
                    {
                        "discarded_samples": global_discarded_samples,
                        "train_loss": global_tracked_loss.item(),
                        "step": completed_steps,
                        "lr": optimizer.param_groups[0]["lr"],
                        "p": torch.mean(probs).item(),
                    },
                    step=completed_steps,
                )

            global_discarded_samples = 0

            # check if it is int
            if (
                args.checkpointing_steps.isdigit()
                and completed_steps % int(args.checkpointing_steps) == 0
                and not args.turn_off_save
            ):
                # if args.checkpointing_steps == "epoch" and not args.turn_off_save:
                output_dir = f"step_{completed_steps}"
                if args.output_dir is not None:
                    output_dir = os.path.join(args.output_dir, output_dir)

                if accelerator.is_main_process:
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(
                        output_dir,
                        is_main_process=True,
                        save_function=accelerator.save,
                    )
                    tokenizer.save_pretrained(output_dir)

                    torch.save(
                        sample_indices[last_used:],
                        os.path.join(output_dir, "sample_indices_unused.pkl"),
                    )

        if args.checkpointing_steps == "epoch" and not args.turn_off_save:
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)

            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    output_dir,
                    is_main_process=True,
                    save_function=accelerator.save,
                )
                tokenizer.save_pretrained(output_dir)
            if args.save_optimizer:
                accelerator.save_state(output_dir)

        accelerator.wait_for_everyone()

        eval_loss, eval_celoss, eval_celoss2 = eval()
        if is_main_process:
            logger.info(f"epoch {epoch}: eval_loss: {eval_loss}")

            if args.with_tracking:
                wandb.log(
                    {
                        "eval_loss": eval_loss,
                        "step": completed_steps,
                        "eval_celoss": eval_celoss,
                        "eval_celoss2": eval_celoss2,
                    },
                    step=completed_steps,
                )

    if args.with_tracking:
        accelerator.end_training()


if __name__ == "__main__":
    main()

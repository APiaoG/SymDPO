""" Main training script """

import argparse
import glob
import os
import random

import numpy as np
import torch
import wandb
from data import get_data
from dataclasses import dataclass, field
from distributed import dpo_init_distributed_device, world_info_from_env
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from train_utils import (
    train_one_epoch,
    get_mp_policy_dtype,
    save_final_checkpoint,
)
import transformers
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from torch.distributed.fsdp import (
    CPUOffload,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointWrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
from torch.distributed.fsdp._init_utils import _init_intra_and_inter_node_groups
from torch.distributed.distributed_c10d import _get_default_group
import functools

from open_flamingo import create_model_and_transforms, create_ref_model_and_transforms
from dpo_data import make_dpo_data_module, DPODataCollator
from openflamingo_trainer import OpenFlamingoDPOTrainer
from dpo_data import rank0_print


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    double_quant: bool = field(default=True, metadata={"help": "Compress the quantization statistics through double quantization."})
    quant_type: str = field(default="nf4", metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."})
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    group_by_varlen: bool = field(default=False)
    group_by_modality_length: bool = field(default=False)
    group_by_modality_length_auto: bool = field(default=False)
    auto_find_batch_size: bool = field(default=False)
    gradient_checkpointing: bool = field(default=True)
    verbose_logging: bool = field(default=False)
    dpo_alpha: float = field(default=1.0)
    beta: float = field(default=0.1)
    gamma: float = field(default=0.0)
    generate_during_eval: bool = field(default=False)
    precompute_ref_log_probs: bool = field(default=False)


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ["vision_encoder", "perceiver"]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    if "wte" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)

# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

def main():
    parser = argparse.ArgumentParser()
    # model configuration args
    parser.add_argument("--vision_encoder_path", default="ViT-L-14", type=str)
    parser.add_argument("--vision_encoder_pretrained", default="openai", type=str)
    parser.add_argument("--lm_path", default="facebook/opt-1.3b", type=str)
    parser.add_argument("--model_path", default="path_to_checkpoint.pt", type=str)
    parser.add_argument(
        "--tokenizer_path",
        default="facebook/opt-30b",
        type=str,
        help="path to tokenizer",
    )
    parser.add_argument(
        "--cross_attn_every_n_layers",
        type=int,
        default=1,
        help="how often to add a cross-attention layer after each transformer layer",
    )

    # training args
    parser.add_argument(
        "--run_name",
        type=str,
        default="openflamingo3B",
        help="used to name saving directory and wandb run",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        help="path to checkpoint to resume from, this should contain model, optimizer, and lr_scheduler states. if there exists a checkpoint in the dir named run_name, we will resume from that checkpoint by default",
        default=None,
    )
    parser.add_argument(
        "--delete_previous_checkpoint",
        action="store_true",
        help="delete previous checkpoint when saving new checkpoint",
    )
    # parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    # parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument(
        "--lr_scheduler",
        default="constant",
        type=str,
        help="constant, linear, or cosine",
    )
    parser.add_argument("--warmup_steps", default=5000, type=int)
    # parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument(
        "--precision",
        choices=["amp_bf16", "amp_bfloat16", "bf16", "fp16", "fp32"],
        default="fp32",
        help="Floating point precision.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="whether to train with gradient/activation checkpointing",
    )
    # parser.add_argument(
    #     "--num_epochs",
    #     type=int,
    #     default=1,
    #     help="we define an 'epoch' as a fixed number of examples (train_num_samples_mmc4, train_num_samples_laion), not a pass through the entire dataset",
    # )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--freeze_lm_embeddings",
        action="store_true",
        help="if True, we freeze the LM embeddings during training. Otherwise, we train the <image> and <|endofchunk|> embeddings.",
    )
    # parser.add_argument(
    #     "--logging_steps", type=int, default=100, help="log loss every n steps"
    # )

    # data args
    
    parser.add_argument("--workers", type=int, default=1)
    
    parser.add_argument("--dataset_resampled", action="store_true")

    # distributed training args
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--horovod",
        default=False,
        action="store_true",
        help="Use horovod for distributed training.",
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).",
    )
    parser.add_argument(
        "--fsdp",
        default=False,
        action="store_true",
        help="Use FullyShardedDataParallel for distributed training.",
    )
    parser.add_argument(
        "--fsdp_use_orig_params",
        default=False,
        action="store_true",
        help="Passed into the FSDP constructor. Enables param_groups and gradient masking for weight_decay. Does not work with OPT.",
    )
    parser.add_argument(
        "--fsdp_sharding_strategy", default="full", type=str, choices=["full", "hybrid"]
    )

    # wandb args
    parser.add_argument("--report_to_wandb", default=False, action="store_true")
    parser.add_argument(
        "--wandb_project",
        type=str,
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
    )
    parser.add_argument(
        "--save_checkpoints_to_wandb",
        default=False,
        action="store_true",
        help="save checkpoints to wandb",
    )

    parser.add_argument(
        "--data_path",
        type=str,
        help="data path",
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        help="data path",
    )
    parser.add_argument(
        "--precompute_ref_log_probs",
        default=False,
        action="store_true",
        help="precompute_ref_log_probs",
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="adamw_torch",
        help="data path",
    )
    parser.add_argument(
        "--remove_unused_columns",
        default=False,
        action="store_true",
        help="remove_unused_columns",
    )
    parser.add_argument(
        "--model_max_length",
        default=2048,
        type=int,
        help="min number of images per sequence in mmc4 / chatgpt",
    )
    parser.add_argument(
        "--bits",
        default=16,
        type=int,
        help="bits",
    )
    parser.add_argument(
        "--lora_enable",
        default=False,
        action="store_true",
        help="lora_enable",
    )
    parser.add_argument(
        "--group_by_varlen",
        default=False,
        action="store_true",
        help="group_by_varlen",
    )
    parser.add_argument(
        "--group_by_modality_length",
        default=False,
        action="store_true",
        help="group_by_modality_length",
    )
    parser.add_argument(
        "--group_by_modality_length_auto",
        default=False,
        action="store_true",
        help="group_by_modality_length_auto",
    )
    parser.add_argument(
        "--auto_find_batch_size",
        default=False,
        action="store_true",
        help="auto_find_batch_size",
    )
    # parser.add_argument(
    #     "--gradient_checkpointing",
    #     default=True,
    #     action="store_false",
    #     help="gradient_checkpointing",
    # )
    parser.add_argument(
        "--verbose_logging",
        default=False,
        action="store_true",
        help="verbose_logging",
    )

    parser.add_argument(
        "--dpo_alpha",
        default=1.0,
        type=float,
        help="dpo_alpha",
    )
    parser.add_argument(
        "--beta",
        default=0.1,
        type=float,
        help="beta",
    )
    parser.add_argument(
        "--gamma",
        default=1.0,
        type=float,
        help="gamma",
    )
    parser.add_argument(
        "--generate_during_eval",
        default=False,
        action="store_true",
        help="generate_during_eval",
    )
    parser.add_argument(
        "--num_train_epochs",
        default=1,
        type=int,
        help="num_train_epochs",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        default=1,
        type=int,
        help="per_device_train_batch_size",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        default=1,
        type=int,
        help="per_device_eval_batch_size",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        default=1,
        type=int,
        help="gradient_accumulation_steps",
    )
    parser.add_argument(
        "--evaluation_strategy",
        type=str,
        default="no",
        help="evaluation_strategy",
    )
    parser.add_argument(
        "--save_strategy",
        type=str,
        default="steps",
        help="save_strategy",
    )
    parser.add_argument(
        "--save_total_limit",
        default=1,
        type=int,
        help="save_total_limit",
    )
    parser.add_argument(
        "--save_steps",
        default=1250,
        type=int,
        help="save_steps",
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-6,
        type=float,
        help="learning_rate",
    )
    parser.add_argument(
        "--weight_decay",
        default=0.1,
        type=float,
        help="weight_decay",
    )
    parser.add_argument(
        "--warmup_ratio",
        default=0.01,
        type=float,
        help="warmup_ratio",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        help="lr_scheduler_type",
    )
    parser.add_argument(
        "--logging_steps",
        default=1,
        type=int,
        help="logging_steps",
    )
    parser.add_argument(
        "--bf16",
        default=True,
        action="store_false",
        help="bf16",
    )
    parser.add_argument(
        "--tf32",
        default=True,
        action="store_false",
        help="tf32",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        default=4,
        type=int,
        help="dataloader_num_workers",
    )
    parser.add_argument(
        "--train_num_samples",
        default=470796,
        type=int,
        help="train_num_samples",
    )
    parser.add_argument(
        "--lora_r",
        default=64,
        type=int,
        help="lora_r",
    )
    parser.add_argument(
        "--lora_alpha",
        default=16,
        type=int,
        help="lora_alpha",
    )
    parser.add_argument(
        "--lora_dropout",
        default=0.05,
        type=float,
        help="lora_dropout",
    )
    parser.add_argument(
        "--lora_weight_path",
        type=str,
        default="",
        help="lora_weight_path",
    )
    parser.add_argument(
        "--lora_bias",
        type=str,
        default="none",
        help="lora_bias",
    )
    parser.add_argument(
        "--quant_type",
        type=str,
        default="nf4",
        help="Quantization data type to use. Should be one of `fp4` or `nf4`.",
    )
    parser.add_argument(
        "--double_quant",
        default=True,
        action="store_false",
        help="double_quant",
    )

    args = parser.parse_args()
    training_args = TrainingArguments(save_steps=args.save_steps, output_dir=f"{args.run_name}/", dataloader_num_workers=args.dataloader_num_workers, tf32=args.tf32, bf16=args.bf16, logging_steps=args.logging_steps, lr_scheduler_type=args.lr_scheduler_type, warmup_ratio=args.warmup_ratio, weight_decay=args.weight_decay, learning_rate=args.learning_rate, save_total_limit=args.save_total_limit, precompute_ref_log_probs=args.precompute_ref_log_probs, optim=args.optim, remove_unused_columns=args.remove_unused_columns, model_max_length=args.model_max_length, bits=args.bits, group_by_varlen=args.group_by_varlen, group_by_modality_length=args.group_by_modality_length, group_by_modality_length_auto=args.group_by_modality_length_auto, auto_find_batch_size=args.auto_find_batch_size, gradient_checkpointing=args.gradient_checkpointing, verbose_logging=args.verbose_logging, dpo_alpha=args.dpo_alpha, beta=args.beta, gamma=args.gamma, generate_during_eval=args.generate_during_eval, num_train_epochs=args.num_train_epochs, per_device_train_batch_size=args.per_device_train_batch_size, per_device_eval_batch_size=args.per_device_eval_batch_size, gradient_accumulation_steps=args.gradient_accumulation_steps, evaluation_strategy=args.evaluation_strategy, save_strategy=args.save_strategy, double_quant=args.double_quant, quant_type=args.quant_type, lora_bias=args.lora_bias, lora_weight_path=args.lora_weight_path, lora_dropout=args.lora_dropout, lora_alpha=args.lora_alpha, lora_r=args.lora_r, lora_enable=args.lora_enable)


    # 存不存wandb
    if args.save_checkpoints_to_wandb and not args.report_to_wandb:
        raise ValueError("save_checkpoints_to_wandb requires report_to_wandb")

    # 使用FullyShardedDataParallel 但是没有为weight_decay启用参数组和渐变掩码。
    if args.fsdp and not args.fsdp_use_orig_params:
        print(
            "Warning: FSDP is running without fsdp_use_orig_params flag. "
            + "This is not recommended because it means we will use uniform weight decay"
            + " and train all embeddings, not just the newly added ones. "
            + "Note: OPT models are not compatible with fsdp_use_orig_params flag."
        )

    if args.fsdp and args.fsdp_sharding_strategy == "hybrid":
        print(
            "Warning: As of torch=2.0.1, the FSDP logic for optim_state_dict() is broken for hybrid sharding."
            + "To make this method work, we need to modify torch.distributed.fsdp._optim_utils.py"
            + "Copy and paste the code from the _optim_utils.py in this repo into the torch file."
            + "The main issue was the missing group kwarg on line 1596 in _all_gather_optim_state."
        )

    # 每个epoch中 mmc4 和 laion 的数据数量应该相同
    
    # Set up distributed training
    if args.offline:
        os.environ["WANDB_MODE"] = "offline"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    device_id = dpo_init_distributed_device(args)
    random_seed(args.seed)

    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)

    bnb_model_from_pretrained_args = {}
    if args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig

        bnb_model_from_pretrained_args.update(
            dict(
                device_map={"": training_args.device},
                # load_in_4bit=training_args.bits == 4,
                # load_in_8bit=training_args.bits == 8,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type,  # {'fp4', 'nf4'}
                ),
            )
        )

    # Initialize model
    # 初始化模型
    model, image_processor, tokenizer = create_model_and_transforms(
        args.vision_encoder_path,
        args.vision_encoder_pretrained,
        args.lm_path,
        args.tokenizer_path if args.tokenizer_path else args.lm_path,
        cross_attn_every_n_layers=args.cross_attn_every_n_layers,
        use_local_files=args.offline,
        gradient_checkpointing=args.gradient_checkpointing,
        freeze_lm_embeddings=args.freeze_lm_embeddings,
        bnb_model_from_pretrained_args=bnb_model_from_pretrained_args,
    )
    ref_model, ref_image_processor, ref_tokenizer = create_ref_model_and_transforms(
        args.vision_encoder_path,
        args.vision_encoder_pretrained,
        args.lm_path,
        args.tokenizer_path if args.tokenizer_path else args.lm_path,
        cross_attn_every_n_layers=args.cross_attn_every_n_layers,
        use_local_files=args.offline,
        gradient_checkpointing=args.gradient_checkpointing,
        freeze_lm_embeddings=args.freeze_lm_embeddings,
        bnb_model_from_pretrained_args=bnb_model_from_pretrained_args,
    )
    random_seed(args.seed, args.rank)

    # Initialize logging
    print(f"Start running training on rank {args.rank}.")
    if args.rank == 0 and args.report_to_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=vars(args),
        )

    if args.model_path is not None:
        if args.rank == 0:
            print(f"Loading checkpoint from {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location="cpu")
        # model.load_state_dict(torch.load(checkpoint_path), strict=False)
        # print(checkpoint.keys())
        # import time 
        # time.sleep(1000)
        msd = checkpoint
        msd = {k.replace("module.", ""): v for k, v in msd.items()}

        # model.load_state_dict(msd, False)
        # ref_model.load_state_dict(msd, False)

        # for fsdp, only one rank needs to load the state dict
        if not args.fsdp or args.rank == 0:
            model.load_state_dict(msd, False)
            ref_model.load_state_dict(msd, False)

    # Load model checkpoint on CPU
    #  如果没指定resume 指出可用项
    # if os.path.exists(f"{args.run_name}") and args.resume_from_checkpoint is None:
    #     # if args do not specify a checkpoint to resume from, check if checkpoints exist for this run
    #     # and automatically resume from the latest checkpoint
    #     checkpoint_list = glob.glob(f"{args.run_name}/checkpoint_*.pt")
    #     if len(checkpoint_list) == 0:
    #         print(f"Found no checkpoints for run {args.run_name}.")
    #     else:
    #         args.resume_from_checkpoint = sorted(
    #             checkpoint_list, key=lambda x: int(x.split("_")[-1].split(".")[0])
    #         )[-1]
    #         print(
    #             f"Found checkpoint {args.resume_from_checkpoint} for run {args.run_name}."
    #         )

    # 指定加载路径，从哪个epoch恢复训练
    resume_from_epoch = 0
    if args.resume_from_checkpoint is not None:
        if args.rank == 0:
            print(f"Loading checkpoint from {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
        # model.load_state_dict(torch.load(checkpoint_path), strict=False)
        msd = checkpoint["model_state_dict"]
        msd = {k.replace("module.", ""): v for k, v in msd.items()}
        resume_from_epoch = checkpoint["epoch"] + 1

        # for fsdp, only one rank needs to load the state dict
        if not args.fsdp or args.rank == 0:
            model.load_state_dict(msd, False)

    # Initialize FSDP / DDP, and ensure the model is on GPU
    print(f"Initializing distributed training with {args.world_size} GPUs.")

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training

        model.config.torch_dtype = torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
    
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)
    
    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer

        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if "norm" in name:
                module = module.to(torch.float32)
            if "lm_head" in name or "embed_tokens" in name or "wte" in name:
                if hasattr(module, "weight"):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)
    
    if args.fsdp:
        print(
            f"Before FSDP parameter num: {sum(p.numel() for p in model.parameters())} on rank {args.rank}"
        )
        print(
            f"Before FSDP parameter num: {sum(p.numel() for p in ref_model.parameters())} on rank {args.rank}"
        )

        # 获得对应的数据类型，例如torch.float32
        # init MixedPrecision
        if args.precision != "fp32":
            cast_dtype = get_mp_policy_dtype(args.precision)
            mp_policy = MixedPrecision(
                param_dtype=torch.float32,
                reduce_dtype=cast_dtype,  # gradient communication
                buffer_dtype=cast_dtype,
            )
        else:
            mp_policy = None

        # init process groups
        if args.fsdp_sharding_strategy == "hybrid":
            intra_node_group, inter_node_group = _init_intra_and_inter_node_groups(
                _get_default_group()
            )
            args.my_group = intra_node_group  # for optimizer saving
            process_group = (intra_node_group, inter_node_group)  # for FSDP init
        else:
            args.my_group = None  # for optimizer saving
            process_group = None  # for FSDP init

        # init FSDP
        wrapper_kwargs = dict(
            process_group=process_group,
            cpu_offload=CPUOffload(offload_params=False),
            device_id=device_id,
            sync_module_states=True,  # broadcast loaded ckpt from rank 0 -> all ranks
            sharding_strategy=ShardingStrategy.FULL_SHARD
            if args.fsdp_sharding_strategy == "full"
            else ShardingStrategy.HYBRID_SHARD,
            use_orig_params=args.fsdp_use_orig_params,
            mixed_precision=mp_policy,
            forward_prefetch=True,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            limit_all_gathers=True,
        )
        model.wrap_fsdp(wrapper_kwargs, device_id)
        ref_model.wrap_fsdp(wrapper_kwargs, device_id)
        ddp_model = model
        ref_ddp_model = ref_model

        print(
            f"After FSDP parameter num: {sum(p.numel() for p in model.parameters())} on rank {args.rank}"
        )
        print(
            f"After FSDP {torch.cuda.memory_allocated()/1024**3:.3} GB on rank {args.rank}"
        )
        print(
            f"After FSDP parameter num: {sum(p.numel() for p in ref_model.parameters())} on rank {args.rank}"
        )
        print(
            f"After FSDP {torch.cuda.memory_allocated()/1024**3:.3} GB on rank {args.rank}"
        )

    else:
        # 否则使用 DDP
        model = model.to(device_id)
        ref_model = ref_model.to(device_id)
        # ddp_model = DDP(model, device_ids=[device_id])
        # ref_ddp_model = DDP(ref_model, device_ids=[device_id])

    # Initialize gradient checkpointing
    # 梯度检查点
    # if args.gradient_checkpointing:
    #     non_reentrant_wrapper = functools.partial(
    #         checkpoint_wrapper,
    #         offload_to_cpu=True,
    #         checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    #     )
    #     apply_activation_checkpointing(
    #         ddp_model,
    #         checkpoint_wrapper_fn=non_reentrant_wrapper,
    #         check_fn=lambda m: getattr(m, "_use_gradient_checkpointing", False)
    #         and not isinstance(m, FSDP)
    #         and not isinstance(m, CheckpointWrapper),
    #     )

    # Initialize optimizer
    # 设置需要optimize的参数
    params_to_optimize = model.named_parameters()
    params_to_optimize = list(
        filter(
            lambda x: x[1].requires_grad
            and not getattr(x[1], "exclude_from_optimizer", False),
            params_to_optimize,
        )
    )
    if not args.fsdp or args.fsdp_use_orig_params:
        # apply weight decay only to params in the xattn layers
        # 只对 xattn层 部分的进行weight decay
        def get_grouped_params(model):
            params_with_wd, params_without_wd = [], []
            for n, p in params_to_optimize:
                if "gated_cross_attn" in n:
                    params_with_wd.append(p)
                else:
                    params_without_wd.append(p)
            return [
                {"params": params_with_wd, "weight_decay": args.weight_decay},
                {"params": params_without_wd, "weight_decay": 0.0},
            ]

        optimizer = torch.optim.AdamW(
            get_grouped_params(params_to_optimize), lr=args.learning_rate
        )
    else:
        # unclear if we should be using no weight decay or small weight decay for all parameters
        # 对所有参数weight decay
        optimizer = torch.optim.AdamW(
            (p for _, p in params_to_optimize),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

    # load optimizer checkpoint
    # 加载 断点的 optimizer
    # if args.resume_from_checkpoint is not None:
    #     osd = checkpoint["optimizer_state_dict"]
    #     if args.fsdp:
    #         osd = FSDP.optim_state_dict_to_load(osd, model, optimizer)
    #     optimizer.load_state_dict(osd)
    # Initialize lr scheduler
    # 设置scheduler
    total_training_steps = int((
        (args.train_num_samples) / (args.per_device_train_batch_size * args.gradient_accumulation_steps * args.world_size)
    ) * args.num_train_epochs)
    
    if args.lr_scheduler_type == "linear":
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(args.warmup_ratio*total_training_steps),
            num_training_steps=total_training_steps,
        )
    elif args.lr_scheduler_type == "cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(args.warmup_ratio*total_training_steps),
            num_training_steps=total_training_steps,
        )
    else:
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=int(args.warmup_ratio*total_training_steps)
        )

    
    # Initialize data loaders
    train_dataset = make_dpo_data_module(data_path=args.data_path, tokenizer=tokenizer, image_folder=args.image_folder, image_processor=image_processor)
    data_collator = DPODataCollator(
        tokenizer,
        label_pad_token_id=-100,
        pad_token_id=tokenizer.pad_token_id,
    )

    trainer = OpenFlamingoDPOTrainer(
        model,
        ref_model,
        optimizers=(optimizer,lr_scheduler),
        args=training_args,
        dpo_alpha=training_args.dpo_alpha,
        beta=training_args.beta,
        gamma=training_args.gamma,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
        tokenizer=tokenizer,
        max_length=training_args.model_max_length,
        generate_during_eval=False,  # training_args.generate_during_eval,
        precompute_ref_log_probs=training_args.precompute_ref_log_probs,
    )

    if args.resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), training_args.lora_bias)
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            if hasattr(model, "config"):
                model.config.save_pretrained(training_args.output_dir)
            if hasattr(model, "generation_config"):
                model.generation_config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_trainables.bin"))
    # else:
        # safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    save_final_checkpoint(model, optimizer, lr_scheduler, args)
    rank0_print(f"Model saved")


    # laion_dataset = get_data(args, image_processor, tokenizer, "image_text")
    # mmc4_dataset = get_data(args, image_processor, tokenizer, "mmc4")
    # # 总共的训练步数
    # total_training_steps = (
    #     (args.train_num_samples_mmc4) // (args.batch_size_mmc4 * args.world_size)
    # ) * args.num_epochs

    # if args.rank == 0:
    #     print(f"Total training steps: {total_training_steps}")

    # # Initialize lr scheduler
    # # 设置scheduler
    # if args.lr_scheduler == "linear":
    #     lr_scheduler = get_linear_schedule_with_warmup(
    #         optimizer,
    #         num_warmup_steps=args.warmup_steps,
    #         num_training_steps=total_training_steps,
    #     )
    # elif args.lr_scheduler == "cosine":
    #     lr_scheduler = get_cosine_schedule_with_warmup(
    #         optimizer,
    #         num_warmup_steps=args.warmup_steps,
    #         num_training_steps=total_training_steps,
    #     )
    # else:
    #     lr_scheduler = get_constant_schedule_with_warmup(
    #         optimizer, num_warmup_steps=args.warmup_steps
    #     )

    # # load lr scheduler checkpoint
    # # 加载断点处的scheduler
    # if args.resume_from_checkpoint is not None:
    #     lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

    # # 开始训练
    # # Start training!
    # ddp_model.train()

    # # 为每个epoch设置数据 并保存ckpt
    # for epoch in range(resume_from_epoch, args.num_epochs):
    #     laion_dataset.set_epoch(epoch)
    #     laion_loader = laion_dataset.dataloader
    #     mmc4_dataset.set_epoch(epoch)
    #     mmc4_loader = mmc4_dataset.dataloader

    #     train_one_epoch(
    #         args=args,
    #         model=ddp_model,
    #         epoch=epoch,
    #         tokenizer=tokenizer,
    #         optimizer=optimizer,
    #         lr_scheduler=lr_scheduler,
    #         laion_loader=laion_loader,
    #         mmc4_loader=mmc4_loader,
    #         device_id=device_id,
    #         wandb=wandb,
    #     )
    #     save_checkpoint(ddp_model, optimizer, lr_scheduler, epoch, args)

    # # 保存最后的ckpt
    # # save final checkpoint
    # save_checkpoint(ddp_model, optimizer, lr_scheduler, epoch, args)


if __name__ == "__main__":
    main()

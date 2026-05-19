"""
Generate a large batch of video samples from a model.
"""
import argparse
import os
import time
import numpy as np
import torch as th
import torch.distributed as dist
import pytorch_lightning as pl

from diffusion.resample import create_named_schedule_sampler
from diffusion import dist_util, logger
from diffusion.cmgn_script_util_ti import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

# from diffusion.cmgn_train_util import TrainLoop
from diffusion.cmgn_train_util_ti import TrainLoop
from diffusion.dist_util import save_video_grid
from cmgn import VideoData
from einops import rearrange, repeat

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    # # ===================== 打印模型参数量 =====================
    # def count_parameters(model):
    #     total = sum(p.numel() for p in model.parameters())
    #     trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #     print(f"\n=============================================")
    #     print(f"模型总参数量: {total:,}")
    #     print(f"可训练参数量: {trainable:,}")
    #     print(f"=============================================\n")
    #
    # count_parameters(model)
    
    logger.log("loading dataset...")
    data = VideoData(args)
    data = data.train_dataloader()

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        save_dir=args.save_dir,
        sequence_length=args.sequence_length,
        audio_emb_model=args.audio_emb_model,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        model_path="",
        spatial_model_path="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        # batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = VideoData.add_data_specific_args(parser)
    parser.add_argument('--save_dir', type=str)
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

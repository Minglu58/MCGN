"""
Generate a large batch of video samples from a model.
"""
import sys
sys.path.append('/fs/data/home/zhaoml_fengszlab/program/mcgn')
import argparse
import os
import time
import numpy as np
import torch as th
import torch.distributed as dist
import pytorch_lightning as pl

from diffusion.resample import create_named_schedule_sampler
from diffusion import dist_util, logger
from diffusion.cmgn_script_util_colla import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

from diffusion.cmgn_train_util_colla import TrainLoop
from diffusion.dist_util import save_video_grid
from cmgn import VideoData
from einops import rearrange, repeat

from vitlens.open_clip import ModalityType
from vitlens.mm_vit_lens import ViTLens

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

    if args.model_path:
         # load original model parameters
        original_model_dict = dist_util.load_state_dict(args.model_path, map_location="cpu")
        # get new model parameters dictionary
        new_model_dict = model.state_dict()

        pretrained_dict = {}
        for key, weight in original_model_dict.items():
             # 跳过旧的双模态融合层（绝对不能加载！）
            if "dualattention" in key:
                 continue
             # 匹配条件：新模型有这个key + 形状完全相同
            if key in new_model_dict and weight.shape == new_model_dict[key].shape:
                 pretrained_dict[key] = weight

        new_model_dict.update(pretrained_dict)
        model.load_state_dict(new_model_dict, strict=False)

        for name, param in model.named_parameters():
             # ❄️ 冻结空间层（第一阶段学到的）
             if "transformers.0" in name or "transformers.1" in name:
                 if "attn2.to_q" in name or "attn2.to_k" in name or "attn2.to_v" in name:
                     param.requires_grad = True
                 else:
                     param.requires_grad = False

             # 🔥 只训练时序 + 三模态融合
             elif "transformers.2" in name or "tripleattention" in name or "temporal_conv" in name:
                 param.requires_grad = True

             # ⚠️ 其余层（UNet backbone）——建议弱训练 or 冻结
             else:
                 param.requires_grad = False

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

    trainable_params = [
        {
            "params": [p for n, p in model.named_parameters()
                       if ("attn2.to_q" in n or "attn2.to_k" in n or "attn2.to_v" in n)
                       and ("transformers.0" in n or "transformers.1" in n)
                       and p.requires_grad],
            "lr": args.lr * 0.5,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if ("transformers.2" in n or "tripleattention" in n or "temporal_conv" in n)
                       and p.requires_grad],
            "lr": args.lr,
            },
        # 3. UNet基础卷积/嵌入层：中等学习率 = 基础lr × 0.3
        # {
        #     "params": [p for n, p in model.named_parameters()
        #                 if "transformers" not in n and "tripleattention" not in n and "temporal_conv" not in n],
        #         "lr": args.lr * 0.3,
        # },
    ]

    logger.log("loading dataset...")
    data = VideoData(args)
    data = data.train_dataloader()

    if args.colla_model:
        colla_model = ViTLens(modality_loaded=[ModalityType.IMAGE, ModalityType.AUDIO, ModalityType.TEXT])
        colla_model.to(dist_util.dev())
    else:
        colla_model = None

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        custom_params=trainable_params,
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
        spatial_model_path=args.spatial_model_path,
        colla_model=colla_model,
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
    parser.add_argument('--colla_model', type=str, default=False)
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

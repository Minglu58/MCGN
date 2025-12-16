"""
Generate a large batch of video samples from a model.
"""
import sys
sys.path.append('/home/apulis-dev/userdata/program/CMGN')
import argparse
import os
import time
import random
import numpy as np
import torch as th
import torch.distributed as dist
import torchvision.transforms.functional as F
import torch.nn.functional as Func

from diffusion import dist_util, logger
from diffusion.cmgn_script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

from diffusion.dist_util import save_video_grid
from cmgn import VideoData, AudioCLIP
from einops import rearrange, repeat
import wav2clip
from beats.BEATs import BEATs, BEATsConfig

import transformers.image_transforms
from PIL import Image
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel

import soundfile
from shutil import copyfile

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def d_clip_loss(x, y, use_cosine=False):
    x = th.nn.functional.normalize(x, dim=-1)
    y = th.nn.functional.normalize(y, dim=-1)

    if use_cosine:
        distance = 1 - (x @ y.t()).squeeze()
    else:
        distance = (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

    return distance


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()
    
    processor = CLIPProcessor.from_pretrained("cmgn/modules/cache/clip-vit-large-patch14")
    clipmodel = CLIPModel.from_pretrained("cmgn/modules/cache/clip-vit-large-patch14").to(dist_util.dev())
    
    logger.log("loading dataset...")
    data = VideoData(args)
    data = data.test_dataloader()
    
    # load audio
    logger.log("loading audio embedding model...")
    # if args.audio_emb_model == 'audioclip':
    #     audioclip_model = AudioCLIP(pretrained=f'saved_ckpts/AudioCLIP-Full-Training.pt')
    #     audioclip_model = audioclip_model.to(dist_util.dev())
    # elif args.audio_emb_model == 'wav2clip':
    #     wav2clip_model = wav2clip.get_model()
    #     wav2clip_model = wav2clip_model.to(dist_util.dev())
    #     for p in wav2clip_model.parameters():
    #         p.requires_grad = False
    if args.audio_emb_model == 'beats':
        checkpoint = th.load('saved_ckpts/BEATs_iter3_plus_AS20K.pt')
        cfg = BEATsConfig(checkpoint['cfg'])
        BEATs_model = BEATs(cfg)
        BEATs_model = BEATs_model.to(dist_util.dev())
        BEATs_model.load_state_dict(checkpoint['model'])
        BEATs_model.eval()

    # random_numbers = [random.randint(0, 3239 - 1) for _ in range(args.num_samples)]
    # for i in random_numbers:
    # sampling
    for i in range(args.num_samples):
        batch = data.dataset.__getitem__(i) #sample_id
        # get text from test_data
        c_t = batch['text'].to(dist_util.dev()) #torch.Size([1, 77, 768])
        c_t = repeat(c_t, "b n d -> (b f) n d", f=batch['video'].shape[1])
        image = batch['video'][:,0]+0.5
        image = image.unsqueeze(0)
        image_cat=None
        for j in range(image.shape[0]):
            image_j = transformers.image_transforms.to_pil_image(image[j])
            image_input = processor(images=image_j, return_tensors="pt", padding=True).to(dist_util.dev())
            with th.no_grad():
                image_features = clipmodel.get_image_features(image_input.pixel_values)

                if image_cat is None:
                    image_cat = image_features.unsqueeze(0)
                else:
                    image_cat = th.concat((image_cat, image_features), dim=0) #torch.Size([1, 1, 768])
        c_i = repeat(image_cat, "b n d -> (b f) n d", f=batch['video'].shape[1])
        
        if args.audio_emb_model == 'STFT':
            stft = batch['stft']
        else:
            audio = batch['audio'].to(dist_util.dev()) 
               
        # if args.audio_emb_model == 'audioclip':
        #     ((audio_embed, _, _), _), _ = audioclip_model(audio=audio)
        #     c_temp = audio_embed.unsqueeze(0) #(1,16,1024)
        # elif args.audio_emb_model == 'wav2clip':
        #     audio_embed = th.from_numpy(wav2clip.embed_audio(audio.cpu().numpy(), wav2clip_model)) #(16,512)
        #     c_temp = audio_embed.unsqueeze(1) #(16,1,512)
        if args.audio_emb_model == 'STFT':
            c_temp = stft
        elif args.audio_emb_model == 'beats':
            audio = rearrange(audio.unsqueeze(0), "b f g -> (b f) g")
            c_temp = BEATs_model.extract_features(audio, padding_mask=None)[0] #torch.Size([16, 8, 768])

        c = th.concat((c_t, c_i, c_temp), dim=1)
        c = c.to(dist_util.dev())

        # a = th.rand(args.batch_size*16, args.in_channels, args.resolution, args.resolution)
        # init_video = th.zeros_like(a)
        # init_video[0] = batch['video'][:,0]

        init_video = th.zeros(
            (args.batch_size*16, args.in_channels, args.resolution, args.resolution),
            device=batch['video'].device,  # 与输入视频同设备（GPU/CPU）
            dtype=batch['video'].dtype     # 与输入视频同数据类型
            )
        first_frames = batch['video'][:, 0]
        for k in range(args.batch_size):
            frame_idx = k * 16  # 每个视频的第0帧在总帧数中的索引
            init_video[frame_idx] = first_frames[k]  # 赋值对应视频的初始帧
        
        logger.log("sampling...")
        t1 = time.time()    
        model_kwargs = {}

        sample_fn = (diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop)
        sample = sample_fn(
            model,
            (args.batch_size*16, args.in_channels, args.resolution, args.resolution),
            c,
            clip_denoised=args.clip_denoised,
            cond_fn=None,
            model_kwargs=model_kwargs,
            progress=True,
            skip_timesteps=10,
            init_image=init_video.to(dist_util.dev()),
            # init_image=None,
        )
        
        #sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        #sample = sample.permute(0, 2, 3, 4, 1)
        #sample = sample.contiguous()
        #print('samples:', sample.shape) #torch.Size([1, 256, 16, 8, 8])

        sample = rearrange(sample, '(b f) c h w -> b c f h w', f=16)
        sample_recon = th.clamp(sample, -0.5, 0.5)
            
        logger.log("save to mp4 format...")
        os.makedirs('./results/%d_%s/real/'%(args.run, args.dataset), exist_ok=True)
        save_video_grid(th.clamp(batch['video'].unsqueeze(0), -0.5, 0.5) + 0.5, os.path.join('./results/%d_%s/real/'%(args.run, args.dataset), 'groundtruth_%d.mp4'%(i)), nrow=1,fps=6)
        
        os.makedirs("./results/%d_%s/fake1_6fps"%(args.run, args.dataset), exist_ok=True)
        save_video_grid(sample_recon+0.5, os.path.join("./results/%d_%s"%(args.run, args.dataset), "fake1_6fps", f"video_%d.mp4"%(i)), nrow=1, fps=6)

        os.makedirs("./results/%d_%s/fake1_30fps" % (args.run, args.dataset), exist_ok=True)
        save_video_grid(sample_recon+0.5, os.path.join("./results/%d_%s" % (args.run, args.dataset), "fake1_30fps", f"video_%d.mp4" % (i)), nrow=1,fps=30)

        os.makedirs('./results/%d_%s/txt/'%(args.run, args.dataset), exist_ok=True)
        copyfile(batch['path'].replace("/mp4_3s/", "/txt_3s/").replace(".mp4", ".txt"), os.path.join('./results/%d_%s/txt/'%(args.run, args.dataset), 'groundtruth_%d.txt'%(i)))
        
        os.makedirs('./results/%d_%s/audio/'%(args.run, args.dataset), exist_ok=True)
        soundfile.write(os.path.join('./results/%d_%s/audio/'%(args.run, args.dataset), 'groundtruth_%d.wav'%(i)), batch['audio'].reshape(-1).numpy(), 48000)

        
    #dist.barrier()
    logger.log("sampling complete")
    t2 = time.time()
    sampling_time = t2 - t1
    logger.log(f"sampling time: {sampling_time:.2f} seconds.")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10,
        #batch_size=8,
        use_ddim=False,
        model_path="",
        run=0,
        dataset="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    parser = VideoData.add_data_specific_args(parser)
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

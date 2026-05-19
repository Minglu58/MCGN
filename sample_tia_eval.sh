#!/bin/bash

#SBATCH --job-name=sample_eval #作业名
#SBATCH --partition=g_v100 #分区名称
#SBATCH -G 1 #GPU卡数
#SBATCH -N 1 #节点个数
#SBATCH --ntasks-per-node 1 #单节点任务个数
#SBATCH --output=output.%j.out #结果输出文件
#SBATCH --error=output.%j.err #错误信息输出文件

python scripts/sample_tia_eval_aist.py --resolution 64 \
                                --image_size 64 \
                                --batch_size 1 \
                                --diffusion_steps 4000 \
                                --noise_schedule cosine \
                                --num_channels 64 \
                                --num_res_blocks 2 \
                                --class_cond False \
                                --model_path saved_ckpts/YOUR_FINAL_CKPT \
                                --num_samples 50 \
                                --learn_sigma True \
                                --text_stft_cond \
                                --audio_emb_model beats \
                                --data_path datasets/YOUR_DATASET \
                                --load_vid_len 90 \
                                --in_channels 3 \
                                --clip_denoised True \
                                --use_temporal_conv True \
                                --dataset DATASET_NAME \
                                --run 0

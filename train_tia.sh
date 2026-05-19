#!/bin/bash

#SBATCH --job-name=train_tia #作业名
#SBATCH --partition=g_v100 #分区名称
#SBATCH -G 1 #GPU卡数
#SBATCH -N 1 #节点个数
#SBATCH --ntasks-per-node 1 #单节点任务个数
#SBATCH --output=output.%j.out #结果输出文件
#SBATCH --error=output.%j.err #错误信息输出文件

python scripts/train_tia.py --num_workers 8 \
                            --batch_size 2 \
                            --data_path datasets/YOUR_DATASETS \
                            --load_vid_len 90 \
                            --save_dir saved_ckpts/YOUR_SAVE_DIR \
                            --resolution 64 \
                            --image_size 64 \
                            --sequence_length 16 \
                            --text_stft_cond \
                            --audio_emb_model beats \
                            --diffusion_steps 4000 \
                            --noise_schedule cosine \
                            --num_channels 64 \
                            --num_res_blocks 2 \
                            --class_cond False \
                            --learn_sigma True \
                            --in_channels 3 \
                            --lr 5e-5 \
                            --log_interval 50 \
                            --save_interval 5000 \
                            --gpus 1 \
                            --use_temporal_conv False 

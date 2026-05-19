#!/bin/bash

#SBATCH --job-name=train_colla #作业名
#SBATCH --partition=g_v100 #分区名称
#SBATCH -G 1 #GPU卡数
#SBATCH -N 1 #节点个数
#SBATCH --ntasks-per-node 1 #单节点任务个数
#SBATCH --output=output.%j.out #结果输出文件
#SBATCH --error=output.%j.err #错误信息输出文件

python scripts/train_tia_colla.py --num_workers 8 \
                            --batch_size 1 \
                            --data_path datasets/post_aist_3s \
                            --load_vid_len 30 \
                            --save_dir saved_ckpts/aist/0519_tia_colla_128 \
                            --model_path saved_ckpts/aist/0512_ti_128/model230000.pt \
                            --resolution 128 \
                            --image_size 128 \
                            --sequence_length 16 \
                            --avs \
                            --audio_emb_model beats \
                            --diffusion_steps 4000 \
                            --noise_schedule cosine \
                            --num_channels 64 \
                            --num_res_blocks 2 \
                            --class_cond False \
                            --learn_sigma True \
                            --in_channels 3 \
                            --lr 5e-5 \
                            --log_interval 10 \
                            --save_interval 1000 \
                            --gpus 1 \
                            --use_temporal_conv True \
                            --colla_model True
                            # --resume_checkpoint saved_ckpts/aist/1123_tia_colla_64/model005000.pt
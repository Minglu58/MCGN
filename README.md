# MCGN

This is the official implement of our MCGN model. We introduce Multimodal Collaborative Generative Network (MCGN), which integrates a Triple-Modality Cross-Interaction (TMCI) module, an Adaptive Multimodal Weight Allocation (AMWA) mechanism, and two cross-modal collaborators to enhance semantic alignment and temporal coherence.

<img width="959" height="377" alt="colla_model" src="https://github.com/user-attachments/assets/20ec9a08-1c1f-4750-87f6-9e2df71e7063" />


## ⚙️ Setup
1. Create the virtual environment
```bash
conda create -n mcgn python==3.9
conda activate mcgn
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
pip install -r requirements.txt 
```

2. Create a `saved_ckpts` folder to download pretrained checkpoints.

| Dataset | checkpoint |
| --------------- | --------------- |
| URMP-VAT | [URMP_checkpoint](https://studentmust-my.sharepoint.com/:u:/g/personal/3220000901_student_must_edu_mo/IQA8dBW2H2MaTryYlgpSIzcHARUu6aMgNw0K-nUl19ExIs4?e=zw3iAO) 
| Landscape-VAT | [Landscape_checkpoint](https://studentmust-my.sharepoint.com/:u:/g/personal/3220000901_student_must_edu_mo/IQApTEy0-TnASL6NaPUNaL28AVHNI_UBn81gWz4mcR06E4c?e=5agJGZ) 

3. Create a `vitlens` folder for downloading the omni-modal representations of multimodalities.

You can access the source at [Vit-Lens](https://github.com/TencentARC/ViT-Lens/tree/main/vitlens/src) repository.

## 📊 Datasets
The datasets used in this experiment can be downloaded from [URMP-VAT](https://studentmust-my.sharepoint.com/:u:/g/personal/3220000901_student_must_edu_mo/IQAs0Egt-kv3QJzrTsormgmfAZoSXk9QgB55tlsnOWd5Q1g) and [Landscape-VAT](https://studentmust-my.sharepoint.com/:u:/g/personal/3220000901_student_must_edu_mo/IQDlZuoqMJgoTLrgpBntAKUmARW2MSOxGy1u3iYagSDotDk).

You can download these processed datasets to `datasets` folder.

## 📚 Testing
### Sample Short-Time Videos
- `resolution`: resolution used in crop data 
- `model_path`: path to pre-trained checkpoint
- `n_sample`: the number of videos need to be sampled
- `text_stft_cond`: load text-audio-video data
- `audio_emb_model`: model to encode audio, choices: `audioclip`, `wav2clip`, `beats`
- `text_emb_model`: model to encode text, choices: `bert`, `clip`
- `data_path`: path to dataset, `post_URMP` for URMP-VAT dataset, and `post_landscape` for Landscape-VAT dataset
- `load_vid_len`: for URMP-VAT, it is set to `90` (fps=30); for Landscape-VAT, it is set to `30` (fps=10)
- `use_temporal_con`: whether to use temporal_conv layers in sampling procedure; if not, the generated content will not have temporal information
- `dataset`: dataset used in each run
- `run`: index for each run
```
python scripts/sample_tia_eval.py --resolution 64 \
                                --image_size 64 \
                                --batch_size 4 \
                                --diffusion_steps 4000 \
                                --noise_schedule cosine \
                                --num_channels 64 \
                                --num_res_blocks 2 \
                                --class_cond False \
                                --model_path saved_ckpts/path_to_your_checkpoint.pt \
                                --num_samples 50 \
                                --learn_sigma True \
                                --text_stft_cond \
                                --audio_emb_model beats \
                                --text_emb_model clip \
                                --data_path datasets/path_to_your_data_folder \
                                --load_vid_len 90 \
                                --in_channels 3 \
                                --clip_denoised True \
                                --use_temporal_conv True \
                                --dataset urmp \
                                --run 0
```
### Evaluation
The generated samples will be organized in this form:
```
0_urmp
  |---real
    |---video_0.mp4
    |---video_1.mp4
    |---...
    |---video_49.mp4
  |---fake
    |---video_0.mp4
    |---video_1.mp4
    |---...
    |---video_49.mp4
  |---audio
    |---video_0.wav
    |---video_1.wav
    |---...
    |---video_49.wav
  |---txt
    |---video_0.txt
    |---video_1.txt
    |---...
    |---video_49.txt
```

- `exp_tag`: folder name of generated samples
- `real_folder`: ground-truth video folder name
- `fake_folder`/`video_folder`: generated video folder name
- `audio_folder`: original audio folder name
- `txt_folder`: original text folder name
* **FVD**
```
python calculation/fvd_fid/fvd.py --exp_tag 0_urmp --real_folder real --fake_folder fake
```
* **FID**
```
python calculation/fvd_fid/fid_pytorch.py --exp_tag 0_urmp --real_folder real --fake_folder fake
```
* **AV-align**
```
python calculation/audio_video_align.py --exp_tag 0_urmp --audio_folder audio --video_folder fake
```
* **CLIP-audio**
```
python calculation/clip_score/clip_audio.py --exp_tag 0_urmp --audio_folder audio --video_folder fake
```
* **CLIP-text**
```
python calculation/clip_score/clip_text.py --exp_tag 0_urmp --txt_folder txt --video_folder fake
```

## Training
### Training Spatial Layers

### Training Temporal Layers


## 🎢 Demos

https://github.com/user-attachments/assets/51de058f-d388-420b-9ed6-adaf6d493448

https://github.com/user-attachments/assets/05cb998b-ace1-44c6-babe-98953e2bd3dd

https://github.com/user-attachments/assets/c6aa94be-66bc-46e0-a149-836db34f88f5

https://github.com/user-attachments/assets/9c35abf7-298c-4a6d-9963-3203fdadf9c0

https://github.com/user-attachments/assets/ac4d6608-743b-47e4-a64a-a1935f9b5aef


## 🙏 Acknowledgement

The code is based on [Latent-Diffusion](https://github.com/CompVis/latent-diffusion) and [Vit-Lens](https://github.com/TencentARC/ViT-Lens). Thanks to the authors for their significant contributions.

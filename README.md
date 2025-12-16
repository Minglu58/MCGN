# MCGN

This is the official implement of our MCGN model. We introduce Multimodal Collaborative Generative Network (MCGN), which integrates a Triple-Modality Cross-Interaction (TMCI) module, an Adaptive Multimodal Weight Allocation (AMWA) mechanism, and two cross-modal collaborators to enhance semantic alignment and temporal coherence.

<img width="959" height="377" alt="colla_model" src="https://github.com/user-attachments/assets/20ec9a08-1c1f-4750-87f6-9e2df71e7063" />


## Setup
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

## Datasets
The datasets used in this experiment can be downloaded from [URMP-VAT](https://studentmust-my.sharepoint.com/:u:/g/personal/3220000901_student_must_edu_mo/IQAs0Egt-kv3QJzrTsormgmfAZoSXk9QgB55tlsnOWd5Q1g) and [Landscape-VAT](https://studentmust-my.sharepoint.com/:u:/g/personal/3220000901_student_must_edu_mo/IQDlZuoqMJgoTLrgpBntAKUmARW2MSOxGy1u3iYagSDotDk).

You can download these processed datasets to `datasets` folder.

## Demos
### Music Performance Videos

![video_1](https://github.com/user-attachments/assets/fdbad527-92b2-4125-b3b2-54982d5dfd51)
![video_3](https://github.com/user-attachments/assets/78a4f303-1bb7-4d3d-8e26-03799d341da5)
![video_16](https://github.com/user-attachments/assets/5634f354-2108-44b3-be7c-3f9129c43e9d)
![video_17](https://github.com/user-attachments/assets/41373957-1fce-4726-8d78-9c9b1b6694b8)
![video_28](https://github.com/user-attachments/assets/ace99557-606b-4cb2-b384-afd864828fd0)


### Landscape Videos
![video_43](https://github.com/user-attachments/assets/927ed211-4280-42df-83bd-605a816e215c)
![video_47](https://github.com/user-attachments/assets/d5f0ae5a-5fa3-4f05-a76a-13699d86cee9)

## Acknowledgement

The code is based on [Latent-Diffusion](https://github.com/CompVis/latent-diffusion) and [Vit-Lens](https://github.com/TencentARC/ViT-Lens). Thanks to the authors for their significant contributions.

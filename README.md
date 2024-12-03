# DiffTR-Flow3D
This repository contains the source code DiffTR-Flow3D for our paper: "Generative AI models for exploring diffusion dynamics from time-resolved particle tracking".

## Reproduction

1. Find a device with GPU support. Our experiment is conducted on two RTX 24GB GPUs and in the Linux system.

2. Install Python 3.8 The following script can be convenient.
```bash
conda create -n difftr-flow3d python=3.8.19
conda activate difftr-flow3d
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 -c pytorch
pip install -r requirements.txt
```

3. Download the TR-Flow3D dataset from [[Figshare]](https://figshare.com/articles/dataset/TR-Flow3D/27617541). And place them under the './TR-Flow3D' folder. 

4. Train the model with the following scripts:
```shell
CUDA_VISIBLE_DEVICES=0,1 python difftr_flow3d_main.py --train_dataset TR-Flow3D --val_dataset TR-Flow3D --lr 4e-5 --train_batch_size 8 --test_batch_size 4 --num_epochs 500 --result_dir results
```

5. Evaluate the model with the following scripts:
```shell
CUDA_VISIBLE_DEVICES=0,1 python difftr_flow3d_main.py --train_dataset TR-Flow3D --val_dataset TR-Flow3D --lr 4e-5 --train_batch_size 8 --test_batch_size 4 --num_epochs 500 --result_dir results --resume model_best.pt --eval
```

## Citation

If you find our work useful in your research, please consider citing:

```
@article{
  title={{Generative AI models for exploring particle diffusion from time-resolved particle tracking}},
  author={Jianan Wei and Yi Yang and Wenguan Wang},
}
```

## Contact
If you have any questions or suggestions, feel free to contact Jianan Wei (weijianan.gm@gmail.com).



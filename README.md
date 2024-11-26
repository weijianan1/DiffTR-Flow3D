# DiffTR-Flow3D
This repository contains the source code DiffTR-Flow3D for our paper: "Generative AI models for exploring diffusion dynamics from time-resolved particle tracking".

## Reproduction

1. Find a device with GPU support. Our experiment is conducted on two RTX 24GB GPUs and in the Linux system.

2. Install Python 3.8 The following script can be convenient.
```bash
pip install -r requirements.txt
```

3. Download the TR-Flow3D dataset from [[Figshare]](https://figshare.com/articles/dataset/TR-Flow3D/27617541). And place them under the './TR-Flow3D' folder. 

4. Train and evaluate the model with the following scripts.
```shell
bash ./scripts/run.sh
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



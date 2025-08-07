# [ICCV 2025 Highlight]⭐ISP2HRNet
PyTorch implementation of paper "ISP2HRNet: Learning to Reconstruct High Resolution Image from Irregularly Sampled Pixels via Hierarchical Gradient Learning".
## Environment
We conduct our experiments on NVIDIA 3090 GPUs in an environment configured as follows:
* Python 3.9.0
* Pytorch 1.13.1 + cu116
  
Readers do not need to replicate our setup exactly.
## Train
```
python train_liif.py --config configs/train-irregular/train_edsr-baseline-liif_irregular.yaml --name save_name --gpu gpu_id
```
Remember to modify the `root_path` in train_edsr-baseline-liif_irregular.yaml. All training parameters can be reconfigured in this file.

## Reproducing Experiments

*  ### EX1🌱Reconstruction from Incomplete Image with Random Missing Pixels
```
. test_ex1.sh
```
*  ### EX2🌱Super Resolution from Incomplete Image with Random Missing Pixels
```
. test_ex2.sh
```
*  ### EX3🌱Reconstruction from Irregularly Sampled Pixels
```
. test_ex3.sh
```
*  ### EX4🌱Super Resolution from a Fixed Number of Irregularly Sampled Pixels
```
. test_ex4.sh
```

## Acknowledgements
This code is built upon [LIIF](https://github.com/yinboc/liif). We thank the authors for sharing the codes.

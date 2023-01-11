# Dual Conditional Normalization Pyramid Network for Face Photo-Sketch Synthesis (DCNP)

Pytorch Code for "Dual Conditional Normalization Pyramid Network for Face Photo-Sketch Synthesis."

The code is coming soon!

### Requirements

+ Ubuntu 18.04
+ Anaconda (Python, Numpy, PIL, etc.)
+ PyTorch 1.7.1
+ TorchVision 0.8.2

### Results

The results of the baseline methods and our method can be downloaded from [Google Drive](https://drive.google.com/file/d/1iLesbjhFp5oYkOTSKzwgO_wUvTZ61Z9-/view?usp=sharing).

### Evaluation Metrics

We provide a [MATLAB code](https://github.com/Tony0720/Dual-Conditional-Normalization-Pyramid-Network-for-Face-Photo-Sketch-Synthesis/blob/main/compute_fsim.m) for calculating the FSIM score.

We use a PyTorch code from this [repository](https://github.com/mseitzer/pytorch-fid) to calculate the FID score.

We use a PyTorch code from this [repository](https://github.com/richzhang/PerceptualSimilarity) to calculate the LPIPS score.

### Acknowledgments

This code borrows heavily from the **[pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)** repository. Thanks for open-sourcing!

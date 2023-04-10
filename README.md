## DCNP

Pytorch Code for "Dual Conditional Normalization Pyramid Network
for Face Photo-Sketch Synthesis".

![framework](/imgs/framework.png)

![network](/imgs/network.png)

### Requirements

+ Ubuntu 18.04
+ Anaconda (Python, Numpy, PIL, etc.)
+ PyTorch 1.7.1
+ torchvision 0.8.2

### Prepare data

1. Creat folder '/data/'.

2. Download the datasets from [Google Drive](https://drive.google.com/file/d/1K9EXuHCu2zeQ1WP2JVhAc3yWfN0rIjtE/view?usp=sharing) and put them into '/data'.

### Inference:

1. Create folder '/checkpoint/pretrained/'.

2. Download the pre-trained models from [Google Drive](https://drive.google.com/file/d/1_S3Iy22RLfeG9dCCBq8tsbwTUmvpeFyh/view?usp=sharing) and put them into '/checkpoint/pretrained/'.

3. Configure the dataset from ['cuhk', 'ar', 'xmwvts', 'cuhk_feret', 'WildSketch'].

4. Run:

```
python inference.py 
```
5. Check the results under './results/pretrained/'.

### Train:

1. Configure the dataset from ['cuhk', 'ar', 'xmwvts', 'cuhk_feret', 'WildSketch'] and name your output_path.

2. Run:

```
python train.py 
```

### Test:

1. Configure the dataset from ['cuhk', 'ar', 'xmwvts', 'cuhk_feret', 'WildSketch'] and confirm your output_path to be consistent with the name at the train stage.

2. Run:

```
python test.py 
```

3. Check the results under './results/'.

### Results

Our final results can be downloaded [here](https://drive.google.com/file/d/1iLesbjhFp5oYkOTSKzwgO_wUvTZ61Z9-/view?usp=sharing).

### Evaluation

Matlab is requested to compute the FSIM metrics in [compute_fsim.m](https://github.com/Tony0720/Dual-Conditional-Normalization-Pyramid-Network-for-Face-Photo-Sketch-Synthesis/blob/main/compute_fsim.m).
The evaluation of FID can be referred to [here](https://github.com/mseitzer/pytorch-fid).
The evaluation of LPIPS can be referred to [here](https://github.com/richzhang/PerceptualSimilarity).

### Acknowledgments

  * This code builds heavily on **[pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)**. Thanks for open-sourcing!

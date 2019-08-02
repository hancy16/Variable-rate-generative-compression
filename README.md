# Toward Variable-Rate Generative Compression by Reducing the Channel Redundancy

TensorFlow Implementation for paper "Toward Variable-Rate Generative Compression by Reducing the Channel Redundancy". The code is based on the implementation [https://github.com/Justin-Tan/generative-compression.git](Justin-Tan)  of method [Generative Adversarial Networks for Extreme Learned Image Compression](https://arxiv.org/abs/1804.02958). The arithmetic-coding is modified from [https://www.nayuki.io/page/reference-arithmetic-coding](https://www.nayuki.io/page/reference-arithmetic-coding). The tensorflow version of VGG comes from [https://github.com/machrisaa/tensorflow-vgg.git].

-----------------------------
### Dependencies
* Python 3.6
* [Pandas](https://pandas.pydata.org/)
* [TensorFlow 1.8](https://github.com/tensorflow/tensorflow)

Docker or anaconda is recommended for installing dependencies.

### Data / Setup
Training was done using the [ADE 20k dataset](http://groups.csail.mit.edu/vision/datasets/ADE20K/) and the [Cityscapes leftImg8bit dataset](https://www.cityscapes-dataset.com/). In the former case images are rescaled to width `512` px, and in the latter images are [resampled to `[512 x 1024]` prior to training](https://www.imagemagick.org/script/command-line-options.php#resample). An example script for resampling using `Imagemagick` is provided under `data/`. In each case, you will need to create a Pandas dataframe containing a single column: `path`, which holds the absolute/relative path to the images. This should be saved as a `HDF5` file, and you should provide the path to this under the `directories` class in `config.py`. Examples for the Cityscapes dataset are provided in the `data` directory. 

## Usage
```bash
# Clone
$ git clone git@github.com:hancy16/Variable-rate-generative-compression.git
$ cd variable-rate_compression

#Training
$ python3 train.py -h
# Run
$ python3 train.py -opt momentum --name my_network

#Testing
```bash
# Compress
$ python3 compress.py  --name my_network -r /path/to/model/checkpoint
```
### Quantized PCA
The code for quantized PCA is available at quantized_PCA/

### Environments 
The code is running on 4 TITAN XP GPUs. Modify the gpu_num in the script to change the number of GPUs.




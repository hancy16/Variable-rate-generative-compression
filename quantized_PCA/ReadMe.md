Quantized PCA for variable-rate compression and tsne-visualization for different methods.

##Usage
All scripts needed is included in test_pca.py.
More detailed scripts will be releasing soon.

1. Save the latent representations generated in the autoencoder network into numpy files and obtain the training data (split needed).

2. Apply pca on the training data and obtain the transform matrix A.

3. The function for solving quantized PCA is named as SGD_t.

4. Apply the matrix A and quantization on those latent representations to generate variable-rate compression.



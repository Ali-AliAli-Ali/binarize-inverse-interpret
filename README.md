# binarize-inverse-interpret

A tool for **interpreting** neural network image classifiers through activations **inversion** and analysis of **binarized** version of model output before classifier layer

### Branches

- **`network_inversion`**  
  Core branch with the full feature inversion pipeline for CNN and vision transformer models, including loss functions, logging, and visualization utilities.

- **`regnetx_bin_extractor_cpu`**  
  CPU-adapted implementation for extracting binary codebooks from RegNetX models.
  
- **`resnet18_bin_extractor_cpu`**  
  CPU-optimized extraction of binary codebooks from ResNet-18.

- **`regnetx_bin`** *[archived]*  
  Analysis and extraction of binary features from the weights of RegNetX architectures.

- **`regnety_bin`** *[archived]*  
  Binary feature analysis for RegNetY variants, which include Squeeze-and-Excitation blocks.

- **`resnet18_bin`** *[archived]*  
  Binary feature analysis specifically for the ResNet-18 architecture.

### Experimental networks sets

Binarization and SVD analysis:
- `RegNetX_3_2`
- `ResNet18`

Inversion:
- CNNs:
  - `ResNet` (versions larger than ResNet18 are recommended)
  - `RegNetX`
  - `EfficientNetV2`
  - `ConvNexT`
  - *[will be improved soon]* `Wide_ResNet` (reconstructs pictures with regular structures)
- visual transformers:
  - ViT
  - *[will be improved soon]* `Swin` (reconstructs random noise with with minor objects silhouettes) 

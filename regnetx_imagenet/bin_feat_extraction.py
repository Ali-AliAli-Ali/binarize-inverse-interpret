import torch
import torch.nn as nn
from torchvision import models
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from collections import defaultdict
import openai  # for semantic analytics


class BinaryFeaturesWrapper(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.model = original_model
        self.layer_before_avgpool = self.model.trunk_output[-1]
        
    def forward(self, x):
        # forward pass to the last layer before avgpool
        x = self.model.stem(x)
        x = self.model.trunk_output[:-1](x)
        # activations binarization
        x = self.layer_before_avgpool(x)
        x = self.binarize_features(x)
        # the remaining part of model
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)
        return x

    def binarize_features(x, threshold=0.0):
        return torch.where(x > threshold, 1.0, -1.0)

# 3. Accuracy test (simplified)
def validate_accuracy(model, dataloader, device='cuda'):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


# 4. avgpool and linear fusing
def fuse_avgpool_linear(model):
    W = model.model.fc.weight.data  # [1000, 368]
    b = model.model.fc.bias.data if model.model.fc.bias is not None else torch.zeros(W.shape[0])
    
    # Определяем размеры после avgpool
    with torch.no_grad():
        dummy = torch.randn(1, 3, 224, 224).to(next(model.parameters()).device)
        features = model.model.stem(dummy)
        features = model.model.trunk_output[:-1](features)
        features = model.model.trunk_output[-1](features)
        features = model.model.avgpool(features)
        H, W = features.shape[-2:]
    
    divisor = H * W
    W_fused = W.repeat(1, divisor) / divisor
    
    return W_fused.cpu().numpy(), b.cpu().numpy()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    regnetx_3_2 = models.regnet_x_3_2gf(weights=models.RegNet_X_3_2GF_Weights.DEFAULT)
    regnetx_16 = models.regnet_x_16gf(weights=models.RegNet_X_16GF_Weights.DEFAULT)
    regnetx_3_2.eval()
    regnetx_16.eval()
    
    # 2. last layer before avgpool binarization
    binary_regnetx_3_2 = BinaryFeaturesWrapper(regnetx_3_2)

    # accuracy = validate_accuracy(binary_regnetx_3_2, imagenet_val_loader)
    # print(f"Accuracy after binarization: {accuracy:.2f}")


    W_fused, b_fused = fuse_avgpool_linear(binary_regnetx_3_2)
    print(f"Fused weight matrix shape: {W_fused.shape}")  # [1000, 368*H*W]

    # 5. SVD декомпозиция
    svd = TruncatedSVD(n_components=50)
    svd.fit(W_fused)
    print(f"Explained variance ratio: {np.sum(svd.explained_variance_ratio_):.2f}")

    # 6. Выделение бинарных признаков
    components = svd.components_  # [50, 368*H*W]
    binary_features = (components > 0).astype(int)

    # 7. Подсчет существенных признаков
    significant_features = np.sum(np.abs(svd.components_) > 0.1)
    print(f"Number of significant binary features: {significant_features}")




if __name__ == '__main__':
    main()

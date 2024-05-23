import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

class CamEncoder(nn.Module):
    def __init__(self):
        super(CamEncoder, self).__init__()

        # Initialize ResNet50 backbone
        resnet50 = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet50.children())[:-2])  # Remove last two layers (avgpool and fc)
        
    def forward(self, input_images):
        B, N, C, H, W = input_images.size()
        
        # Reshape the size to fit ResNet50
        input_images = input_images.view(B * N, C, H, W)
        features = self.backbone(input_images)
        features = features.view(B, N, -1, features.size(2), features.size(3))
        return features

if __name__ == "__main__":
    num_views = 4
    input_channels = 3
    camera_encoder = CamEncoder()
    input_images = torch.randn(2, num_views, input_channels, 224, 224)
    output_features = camera_encoder(input_images)
    print("Output features shape:", output_features.shape)

import torch
from torch import nn
from torchvision import models

# ResNet50
class beetleResNet50(nn.Module):
    def __init__(self, num_classes=8):
        super(beetleResNet50, self).__init__()
        # self.model = models.resnet50(pretrained=True)
        self.model = models.resnet50(pretrained=False)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)
    
if __name__ == '__main__':
    model = beetleResNet50(8)

    # input random image
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(output.shape)
import torch
from torch import nn
from torchvision import models
from fvcore.nn import FlopCountAnalysis, flop_count_table
import time


# ResNet50
class beetleResNet50(nn.Module):
    def __init__(self, num_classes=8, pretrained=True):
        super(beetleResNet50, self).__init__()
        if pretrained:
            weights = models.ResNet50_Weights.DEFAULT
        else:
            weights = None
        self.model = models.resnet50(weights=weights)

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)

class beetleResNet101(nn.Module):
    def __init__(self, num_classes=8, pretrained=True):
        super(beetleResNet101, self).__init__()
        if pretrained:
            weights = models.ResNet101_Weights.DEFAULT
        else:
            weights = None
        self.model = models.resnet101(weights=weights)
        
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)

class beetleEfficientNetV2S(nn.Module):
    def __init__(self, num_classes=8, pretrained=True):
        super(beetleEfficientNetV2S, self).__init__()
        if pretrained:
            weights = models.EfficientNet_V2_S_Weights.DEFAULT
        else:
            weights = None
        self.model = models.efficientnet_v2_s(weights=weights)

        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)

class beetleEfficientNetV2M(nn.Module):
    def __init__(self, num_classes=8, pretrained=True):
        super(beetleEfficientNetV2M, self).__init__()
        if pretrained:
            weights = models.EfficientNet_V2_M_Weights.DEFAULT
        else:
            weights = None
        self.model = models.efficientnet_v2_m(weights=weights)

        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)

class beetleSwinB(nn.Module):
    def __init__(self, num_classes=8, pretrained=True):
        super(beetleSwinB, self).__init__()
        if pretrained:
            weights = models.Swin_B_Weights.DEFAULT
        else:
            weights = None
        self.model = models.swin_b(weights=weights)
        
        in_features = self.model.head.in_features
        self.model.head = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)

class beetleViTB16(nn.Module):
    def __init__(self, num_classes=8, pretrained=True):
        super(beetleViTB16, self).__init__()
        if pretrained:
            weights = models.ViT_B_16_Weights.DEFAULT
        else:
            weights = None
        self.model = models.vit_b_16(weights=weights)
        
        in_features = self.model.heads[-1].in_features
        self.model.heads[-1] = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)

class beetleMobileNetV3L(nn.Module):
    def __init__(self, num_classes=8, pretrained=True):
        super(beetleMobileNetV3L, self).__init__()
        if pretrained:
            weights = models.MobileNet_V3_Large_Weights.DEFAULT
        else:
            weights = None
        self.model = models.mobilenet_v3_large(weights=weights)
        
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)

class beetleMobileNetV3S(nn.Module):
    def __init__(self, num_classes=8, pretrained=True):
        super(beetleMobileNetV3S, self).__init__()
        if pretrained:
            weights = models.MobileNet_V3_Small_Weights.DEFAULT
        else:
            weights = None
        self.model = models.mobilenet_v3_small(weights=weights)
        
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)
    
class beetleRegNetY128GF(nn.Module):
    def __init__(self, num_classes=8, pretrained=True):
        super(beetleRegNetY128GF, self).__init__()
        if pretrained:
            weights = models.RegNet_Y_128GF_Weights.DEFAULT
        else:
            weights = None
        self.model = models.regnet_y_128gf(weights=weights)
        
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)
    
class beetleResNet50Dropout(nn.Module):
    def __init__(self, num_classes=8, pretrained=True):
        super(beetleResNet50Dropout, self).__init__()
        if pretrained:
            weights = models.ResNet50_Weights.DEFAULT
        else:
            weights = None
        self.model = models.resnet50(weights=weights)

        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)
    
if __name__ == '__main__':
    # model = beetleResNet50(8)
    # model = beetleResNet101(8)
    # model = beetleSwinB(8)
    # model = beetleViTB16(8)
    # model = beetleEfficientNetV2S(8)
    # model = beetleEfficientNetV2M(8)
    # model = beetleMobileNetV3L(8)
    model = beetleMobileNetV3S(8)

    # input random image
    x = torch.randn(1, 3, 224, 224)
    flops = FlopCountAnalysis(model, x)

    print(flop_count_table(flops))

    # Compute inference time
    # Except the first run, the time will be faster
    with torch.no_grad():
        model(x)

        times = []
        for _ in range(10):
            start = time.time()
            model(x)
            times.append(time.time() - start)
        # print as ms
        print(f"Average inference time: {sum(times) / len(times) * 1000:.2f} ms")

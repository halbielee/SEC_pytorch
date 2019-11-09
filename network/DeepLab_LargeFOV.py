import torch
import torch.nn as nn

__all__ = [
    'DeepLabLargeFOV', 'deeplab_large_fov',
]

class DeepLabLargeFOV(nn.Module):
    """
    DeepLab v2: Dilated ResNet + ASPP
    Output stride is fixed at 8
    """

    def __init__(self, cfg, num_classes):
        super(DeepLabLargeFOV, self).__init__()
        self.features = make_layers(cfg)

        self.fc6 = nn.Conv2d(512, 1024, kernel_size=3, padding=12, dilation=12)
        self.relu6 = nn.ReLU()
        self.dropout6 = nn.Dropout2d(0.5)
        self.fc7 = nn.Conv2d(1024, 1024, kernel_size=1)
        self.relu7 = nn.ReLU()
        self.dropout7 = nn.Dropout2d(0.5)
        self.fc8 = nn.Conv2d(1024, num_classes, kernel_size=1)

        nn.init.normal_(self.fc8.weight, mean=0, std=0.01)
        nn.init.constant_(self.fc8.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.fc6(x)
        x = self.relu6(x)
        x = self.dropout6(x)
        x = self.fc7(x)
        x = self.relu7(x)
        x = self.dropout7(x)
        x = self.fc8(x)

        return x


def make_layers(cfg):
    layers = []
    in_channels = 3
    for i, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        elif v == 'N':
            layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]
        elif v == 'A':
            layers += [nn.AvgPool2d(kernel_size=3, stride=1, padding=1)]
        else:
            if i >= 14:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=2, dilation=2)
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfgs = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
          512, 512, 512, 'N', 512, 512, 512, 'N', 'A'],
}

def deeplab_large_fov(num_classes):
    return DeepLabLargeFOV(cfgs['D'], num_classes=num_classes)

if __name__ == "__main__":
    model = DeepLabLargeFOV(cfgs['D'], num_classes=21)
    model.eval()
    image = torch.randn(1, 3, 321, 321)

    print(model)
    print("input:", image.shape)
    print("output:", model(image).shape)
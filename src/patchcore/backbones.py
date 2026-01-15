from torchvision.models import (
    resnext101_32x8d, ResNeXt101_32X8D_Weights,
    wide_resnet50_2, Wide_ResNet50_2_Weights,
    wide_resnet101_2, Wide_ResNet101_2_Weights,
    densenet201, DenseNet201_Weights,
)


def load(name):
    match name:
        case "resnext101":
            return resnext101_32x8d(weights=ResNeXt101_32X8D_Weights.DEFAULT)
        case "wideresnet50":
            return wide_resnet50_2(weights=Wide_ResNet50_2_Weights.DEFAULT)
        case "wideresnet101":
            return wide_resnet101_2(weights=Wide_ResNet101_2_Weights.DEFAULT)
        case "densenet201":
            return densenet201(weights=DenseNet201_Weights.DEFAULT)

    return None

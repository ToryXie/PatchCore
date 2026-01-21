from torchvision.models import (ConvNeXt_Base_Weights, ConvNeXt_Large_Weights, ConvNeXt_Small_Weights,
                                ConvNeXt_Tiny_Weights, DenseNet201_Weights, ResNeXt101_32X8D_Weights,
                                Wide_ResNet101_2_Weights, Wide_ResNet50_2_Weights, convnext_base, convnext_large,
                                convnext_small, convnext_tiny, densenet201, resnext101_32x8d, wide_resnet101_2,
                                wide_resnet50_2)


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
        case "convnext_tiny":
            return convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        case "convnext_small":
            return convnext_small(weights=ConvNeXt_Small_Weights.DEFAULT)
        case "convnext_base":
            return convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)
        case "convnext_large":
            return convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT)

    return None

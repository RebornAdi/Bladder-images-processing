"""DenseNet-121 U-Net. Uses ImageNet-pretrained encoder; falls back to
random init if weights can't download (offline)."""
import segmentation_models_pytorch as smp

def build_model(pretrained=True):
    try:
        return smp.Unet(encoder_name="densenet121",
                        encoder_weights="imagenet" if pretrained else None,
                        in_channels=3, classes=1, activation=None)
    except Exception as e:
        print(f"[model] pretrained weights unavailable ({e}); random init.")
        return smp.Unet(encoder_name="densenet121", encoder_weights=None,
                        in_channels=3, classes=1, activation=None)
"""Grad-CAM from the final ResNet feature map using KernelLeaf autograd."""

import argparse
from pathlib import Path

import numpy as np
import kernelleaf as kl

from .dataset import preprocess_rgb_frame
from .drive import AutoDrivePolicy


def grad_cam(model, inputs, head="steering"):
    if head not in {"steering", "throttle"}:
        raise ValueError("head must be steering or throttle")
    if inputs.shape[0] != 1:
        raise ValueError("Grad-CAM requires one image")
    training = model.training
    parameters = model.parameters()
    previous_grads = [getattr(parameter, "grad", None) for parameter in parameters]
    model.eval()
    try:
        # The framework retains leaf gradients only. Make the final feature
        # map a leaf on the same device, then differentiate the real heads.
        features = kl.Tensor(model.forward_features(inputs).realize_cached_data(),
                             device=inputs.device, requires_grad=True)
        outputs = model.forward_heads(features)
        outputs[0 if head == "steering" else 1].sum().backward()
        values = features.realize_cached_data()
        gradients = features.grad.realize_cached_data()
        xp = features.device.xp
        weights = gradients.mean(axis=(2, 3), keepdims=True)
        heatmap = xp.maximum((weights * values).sum(axis=1)[0], 0)
        heatmap = heatmap / xp.maximum(heatmap.max(), 1e-12)
        # Explicit transfer at the visualization boundary, after GPU reduction.
        return kl.Tensor(heatmap, device=features.device, requires_grad=False).numpy()
    finally:
        for parameter, gradient in zip(parameters, previous_grads):
            parameter.grad = gradient
        if training:
            model.train()


def overlay_heatmap(frame, heatmap, alpha=0.45):
    from PIL import Image
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    pixels = np.asarray(frame)
    if pixels.ndim != 3 or pixels.shape[-1] != 3:
        raise ValueError("expected HWC RGB frame")
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    resized = np.asarray(Image.fromarray(np.asarray(heatmap, dtype=np.float32)).resize(
        (pixels.shape[1], pixels.shape[0]), resampling))
    colors = np.stack((resized, np.zeros_like(resized), 1 - resized), axis=-1) * 255
    return np.clip((1 - alpha) * pixels + alpha * colors, 0, 255).astype(np.uint8)


def main():
    from PIL import Image
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="runs/gradcam.png")
    parser.add_argument("--head", choices=("steering", "throttle"), default="steering")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    policy = AutoDrivePolicy(args.config, device=kl.cuda(0) if args.device == "cuda" else kl.cpu())
    with Image.open(args.image) as image:
        frame = np.asarray(image.convert("RGB"))
    p = policy.preprocessing
    values = preprocess_rgb_frame(frame, p["image_size"], p["mean"], p["std"])
    inputs = kl.Tensor(values[None], device=policy.device, requires_grad=True)
    heatmap = grad_cam(policy.model, inputs, args.head)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay_heatmap(frame, heatmap)).save(output)
    print(str(output))


if __name__ == "__main__":
    main()

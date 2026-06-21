import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path

from representations import (
    create_feature_extractor,
    visualize_samples,
    FeaturesDataset,
    visualize_features_tsne,
    train_linear_probe,
    train_finetune_probe,
    evaluate_linear,
    evaluate_finetune,
    plot_losses,
)

device = "cuda:0"

feature_extractors = {
    "clip": create_feature_extractor("vit_base_patch16_clip_224", "openai", device=device),
    "dino": create_feature_extractor("vit_base_patch16_224", "dino", device=device),
    "mae":  create_feature_extractor("vit_base_patch16_224", "mae", device=device),
}

linear_probes = torch.load("probes/probes.pt", weights_only=False)

# Finetune probe training (init from linear probe)
finetuned_models = {}
for method_name, fx in feature_extractors.items():
    print(f"Fine-tuning {method_name} (init from linear probe)...")
    finetuned_models[method_name], ft_losses = train_finetune_probe(
        "photo_train",
        fx,
        pretrained_linear_probe=linear_probes[method_name],
        device=device,
    )
    plot_losses(ft_losses, f"{method_name}_finetune")

# Evaluate finetuned models on photo_val (implement evaluate_finetune from scratch)
for method_name, fx in feature_extractors.items():
    print(f"Evaluating {method_name} finetuned model...")
    acc = evaluate_finetune(finetuned_models[method_name], "photo_val", fx, device=device)
    print(f"{method_name} finetune photo-val accuracy: {acc:.4f}")

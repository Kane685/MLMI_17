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
    plot_losses,
)

device = "cuda:0"

feature_extractors = {
    "clip": create_feature_extractor("vit_base_patch16_clip_224", "openai", device=device),
    "dino": create_feature_extractor("vit_base_patch16_224", "dino", device=device),
    "mae":  create_feature_extractor("vit_base_patch16_224", "mae", device=device),
}

train_features_datasets = torch.load("train_features/train_features.pt", weights_only=False)

# Linear probe training
linear_probes = {}
for method_name in feature_extractors.keys():
    print(f"Training linear probe for {method_name}...")
    linear_probes[method_name], losses = train_linear_probe(
        train_features_datasets[method_name], device=device
    )
    plot_losses(losses, method_name)

# Evaluate linear probes on photo_val (implement evaluate_linear from scratch)
for method_name, probe in linear_probes.items():
    val_feats = FeaturesDataset.create("photo_val", feature_extractors[method_name], device=device)
    acc = evaluate_linear(probe, val_feats, device=device)
    print(f"{method_name} linear probe photo-val accuracy: {acc:.4f}")

Path("probes").mkdir(parents=True, exist_ok=True)
data_saving = Path("probes") / "probes.pt"
torch.save(linear_probes, data_saving)
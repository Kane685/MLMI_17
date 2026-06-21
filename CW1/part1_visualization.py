import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch

from representations import (
    create_feature_extractor,
    visualize_samples,
    FeaturesDataset,
    visualize_features_tsne,
)

device = "cuda:0"

feature_extractors = {
    "clip": create_feature_extractor("vit_base_patch16_clip_224", "openai", device=device),
    "dino": create_feature_extractor("vit_base_patch16_224", "dino", device=device),
    "mae":  create_feature_extractor("vit_base_patch16_224", "mae", device=device),
}

# Visualise some samples
visualize_samples("photo_val", seed=0)

# Extract features for training set
train_features_datasets = {}
for method_name, fx in feature_extractors.items():
    print(f"Extracting features for training set using {method_name}...")
    train_features_datasets[method_name] = FeaturesDataset.create(
        "photo_train", fx, device=device
    )
    assert train_features_datasets[method_name].features.shape[0] == 13000
    assert train_features_datasets[method_name].features.ndim == 2

# Visualize features using t-SNE
for method_name in feature_extractors.keys():
    print(f"Visualizing features for {method_name}...")
    visualize_features_tsne(
        train_features_datasets[method_name],
        title=f"Features t-SNE ({method_name})"
    )

Path("train_features").mkdir(parents=True, exist_ok=True)
data_saving = Path("train_features") / "train_features.pt"
torch.save(train_features_datasets, data_saving)


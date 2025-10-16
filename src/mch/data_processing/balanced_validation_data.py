import polars as pl
import numpy as np
from typing import Set, List
from sklearn.model_selection import train_test_split
from mch.core.disease_tree import DiseaseTree

def split_validation_samples(
    node: DiseaseTree,
    test_size=0.2,
    min_samples_per_class=2
):
    if node.children:
        # Build sample-to-label map and count per class
        sample_to_label = {}
        label_counts = defaultdict(int)

        for child in node.children:
            child_samples = child.samples if child.samples else child.get_samples_recursive()
            for sid in child_samples:
                sample_to_label[sid] = child.name
                label_counts[child.name] += 1

        # Filter samples by class sample count
        valid_sample_ids = [sid for sid in sample_to_label if label_counts[sample_to_label[sid]] >= min_samples_per_class]
        valid_labels = [sample_to_label[sid] for sid in valid_sample_ids]

        # Proceed only if there are at least 2 classes with enough samples
        if len(set(valid_labels)) > 1 and len(valid_sample_ids) > 1:
            _, val_ids = train_test_split(valid_sample_ids, test_size=test_size, stratify=valid_labels)
            node.validation_samples = val_ids

    # Recurse on children
    for child in node.children:
        split_validation_samples(child, test_size=test_size, min_samples_per_class=min_samples_per_class)
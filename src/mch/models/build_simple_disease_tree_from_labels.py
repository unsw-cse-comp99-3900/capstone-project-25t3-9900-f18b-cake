#!/usr/bin/env python3
"""
Build a simplified disease tree from labelswithhealthy.csv and save as joblib.
- Root node name: ZERO2
- Child node: unique disease name from label field (including Healthy)
- The samples of each child node are the corresponding biosampleid list
Output: results/simplediseaseTree.joblib
"""

import os
import sys
import joblib
import pandas as pd
from dataclasses import dataclass, field
from typing import List
from pathlib import Path

@dataclass
class SimpleDiseaseTree:
    name: str
    children: List['SimpleDiseaseTree'] = field(default_factory=list)
    samples: List[str] = field(default_factory=list)

    def get_child_names(self) -> List[str]:
        return [c.name for c in self.children]

    def get_samples_recursive(self) -> List[str]:
        all_samples = list(self.samples)
        for c in self.children:
            all_samples.extend(c.get_samples_recursive())
        return all_samples


def build_tree_from_labels(labels_csv: str) -> SimpleDiseaseTree:
    df = pd.read_csv(labels_csv)
    required_cols = {"biosample_id", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Required column is missing: {missing}")

    root = SimpleDiseaseTree(name="ZERO2")

    # Create a child node for each label
    for disease_name, group in df.groupby("label"):
        samples = group["biosample_id"].dropna().astype(str).tolist()
        if len(samples) == 0:
            continue
        child = SimpleDiseaseTree(name=str(disease_name), children=[], samples=samples)
        root.children.append(child)

    return root


def main():
    labels_file = "labels_with_healthy.csv"
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "simple_diseaseTree.joblib"

    try:
        tree = build_tree_from_labels(labels_file)
        joblib.dump(tree, out_file)
        print(f"Disease tree saved to:{out_file}")
        print(f"Root node: {tree.name}, Number of child nodes: {len(tree.children)}")
        total_samples = len(tree.get_samples_recursive())
        print(f"Total number of samples: {total_samples}")
    except Exception as e:
        print(f"Failed to build disease tree: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

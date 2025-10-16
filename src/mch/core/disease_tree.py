"""
Core disease tree functionality used across the package.
"""

import pandas as pd
import polars as pl

import random

from dataclasses import dataclass, field
from typing import Optional, List

from mch.db.database import Database 

@dataclass
class DiseaseTree:
    name: str
    children: list
    samples: list
    training_samples: list = field(default_factory=list)
    validation_samples: list = field(default_factory=list)
    selected_features: list = field(default_factory=list)

    def is_leaf(self):
        return len(self.children) == 0 and len(self.samples) > 0


    def split_validation_training(self, tree, validation_ratio=0.2, random_seed=42):
        """
        Populates the validation_samples and training_samples lists for each node in the tree.
        
        Args:
            tree: The root node of the disease tree
            validation_ratio: Fraction of samples to use for validation (default: 0.2 or 20%)
            random_seed: Random seed for reproducibility
        """
        import random
        random.seed(random_seed)
        
        def process_node(node):
            if node.is_leaf():
                # This is a leaf node with samples
                samples = node.samples.copy()
                random.shuffle(samples)
                
                # Calculate how many samples to take for validation
                n_validation = max(1, int(len(samples) * validation_ratio))
                
                # Split the samples
                node.validation_samples = samples[:n_validation]
                node.training_samples = samples[n_validation:]
            else:
                # For non-leaf nodes, first process all children
                for child in node.children:
                    process_node(child)
                
                # Then aggregate samples from children
                node.validation_samples = []
                node.training_samples = []
                
                for child in node.children:
                    node.validation_samples.extend(child.validation_samples)
                    node.training_samples.extend(child.training_samples)
        
        # Start processing from the root
        process_node(tree)
        
        # Return the root node with updated validation and training samples
        return tree
        
    def find_sample(self, sample_id: str, path: Optional[List[str]] = None) -> Optional[List[str]]:
        if path is None:
            path = []
        path.append(self.name)
        if sample_id in self.samples:
            return path
        for child in self.children:
            result = findSample(sample_id)
            if result is not None:
                return result
        path.pop()
        return None

    def delete_node(self, target_name):
        if self.name == target_name:
            # If the current node matches the target, return None to delete it
            return None
        
        # Recursively search for the target node in children
        new_children = [child.delete_node(target_name) for child in self.children]
        # Remove None (deleted nodes) from children list
        new_children = [child for child in new_children if child is not None]

        # Update children list with modified list
        self.children = new_children
        
        return self  # Return modified node


    def find_node_by_name(self, target_name):
        #print(target_name)
        if self.name == target_name:
            return self

        for child in self.children:
            result = child.find_node_by_name(target_name)
            if result:
                return result
        return None


    def get_child_names(self):
        return [child.name for child in self.children]

    def get_samples_recursive(self, sample_type="all"):
        # Start with the samples of the current node
        # Recursively collect samples from children
        if sample_type == "all":
            collected_samples = self.samples.copy()
        elif sample_type == "training":
            collected_samples = self.training_samples.copy()
        elif sample_type == "validation":
            collected_samples = self.validation_samples.copy()
        else:
            raise ValueError(f"Unknown sample_type: {sample_type}")

        if not self.children:
            return collected_samples
        else:
            for child in self.children:
                collected_samples.extend(child.get_samples_recursive(sample_type=sample_type))

            return collected_samples

    def get_nodes_at_level(self, level: int) -> list['DiseaseTree']:
        # Create a list to store nodes at the specified level
        nodes_at_level = []
        # Recursive function to traverse the tree and collect nodes
        def collect_nodes(node: 'DiseaseTree', current_level: int):
            if current_level == level:
                nodes_at_level.append(node)
            elif current_level < level:
                for child in node.children:
                    collect_nodes(child, current_level + 1)

        # Start the traversal from the root node
        collect_nodes(self, 1)

        return nodes_at_level


    def filter_tree_by_depth(node, target_depth):
        if target_depth == 0:
            return [node.name]
        elif target_depth > 0:
            result = []
            for child in node.children:
                child_result = filter_tree_by_depth(child, target_depth - 1)
                if child_result:
                    result.extend(child_result)
            return result
        else:
            return []


    def build_disease_tree(self, node_name, disease_tree_df=None, sample_df=None):
    #def build_disease_tree(self, node_name):
        if disease_tree_df is None  or sample_df is None:
            db = Database()
            disease_tree_df, sample_df = db.get_disease_tree_components()
        
        name = node_name
        
        children = list(disease_tree_df[disease_tree_df.parent == node_name].child)
        samples = list(sample_df[sample_df.diseaseName == node_name].sample_id)
        
        # Recursively build child nodes
        #child_nodes = [self.build_disease_tree(child) for child in children]
        child_nodes = [self.build_disease_tree(child, disease_tree_df, sample_df) for child in children]

        # Remove nodes with no samples and no children
        child_nodes = [node for node in child_nodes if node.samples or node.children]
        
        return DiseaseTree(node_name, child_nodes, samples)


    def get_samples_at_level(self, level: int) -> pl.DataFrame:
        # Initialize an empty list to store the results
        dfs = []
        nodes = self.get_nodes_at_level(level)
        
        for tree in nodes:
            samples = tree.get_samples_recursive()
            # Create a Polars DataFrame directly instead of Pandas
            df = pl.DataFrame({
                "sample_id": samples,
                "cancerType": [tree.name] * len(samples)  # Replicate tree.name for each sample
            })
            dfs.append(df)
        
        # Use Polars concat instead of Pandas concat
        if dfs:
            results = pl.concat(dfs)
        else:
            # Return an empty Polars DataFrame with the correct schema if no data
            results = pl.DataFrame({"sample_id": [], "cancerType": []})
        
        return results


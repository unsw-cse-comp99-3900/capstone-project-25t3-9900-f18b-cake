from mch.core.disease_tree import DiseaseTree
from typing import Optional, List


def cohortSampleList():
    """Returns list of samples under a node in diseaseTree."""
    tree = mainTree.find_node_by_name(cancerType)
    samples = tree.get_samples_recursive()
    return(samples)

def zero2FinalDiagnosis(sample_id):
    """Returns final diagnosis for a sample_id."""
    z2 = findSample(mainTree, sample_id)
    return z2

def findSample(tree: DiseaseTree, sample_id: str, path: Optional[List[str]] = None) -> Optional[List[str]]:
    """Returns location of a sample in diseaseTree."""
    if path is None:
        path = []
    path.append(tree.name)
    #print(f"visiting: {tree.name}, current path: {path}")
    if sample_id in tree.samples:
        #print(f"found sample in {path}")
        return path
    for child in tree.children:
        result = findSample(child, sample_id)
        if result is not None:
            return result
    path.pop()
    return None

def collect_tree_structure(node):
    """Traverse the tree and collect (parent, child, count) for each edge."""
    if node is None:  # safety guard
        return []

    edges = []
    for child in node.children or []:  # empty list if no children
        if child is None:
            continue  # skip bad data

        count = len(child.get_samples_recursive())
        edges.append((node.name, child.name, count))
        edges.extend(collect_tree_structure(child))  # recurse down

    return edges
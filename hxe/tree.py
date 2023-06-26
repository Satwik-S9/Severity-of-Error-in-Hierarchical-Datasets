import os
import numpy as np
from math import exp, fsum
from nltk.tree import Tree
from copy import deepcopy
import networkx as nx
from typing import Optional


# The CheXpert Hierarchy Tree :: NLTK Library
CHEXPERT_TREE = Tree('root', ['No Finding', Tree('Enlarged Cardiomediastinum', ['Cardiomegaly']), 
'Support Devices', 'Fracture', Tree('Lung Opacity', ['Edema', 'Consolidation', 'Pneumonia', 'Lesion', 'Atelectasis']), 
Tree('L1', ['Pleural Other', 'Pleural Effusion', 'Pneumothorax'])])

# Label Map
LABEL_MAP = { 'No Finding': 0,
            'Enlarged Cardiomediastinum': 1,
            'Support Devices': 2,
            'Fracture': 3,
            'Lung Opacity': 4,
            'Cardiomegaly': 5,
            'Pleural Effusion': 6,
            'Pleural Other': 7,
            'Pneumothorax': 8,
            'Edema': 9,
            'Consolidation': 10,
            'Pneumonia': 11,
            'Lung Lesion': 12,
            'Atelectasis': 13}

# Classes
CLASSES = list(LABEL_MAP.keys())

def get_label(node):
    if isinstance(node, Tree):
        return node.label()
    else:
        return node

def get_uniform_weighting(hierarchy: Tree, value):
    """
    Construct unit weighting tree from hierarchy.
    Args:
        hierarchy: The hierarchy to use to generate the weights.
        value: The value to fill the tree with.
    Returns:
        Weights as a nltk.Tree whose labels are the weights associated with the
        parent edge.
    """
    weights = deepcopy(hierarchy)
    for p in weights.treepositions():
        node = weights[p]
        if isinstance(node, Tree):
            node.set_label(value)
        else:
            weights[p] = value
    return weights


def get_exponential_weighting(hierarchy: Tree, value, normalize=True):
    """
    Construct exponentially decreasing weighting, where each edge is weighted
    according to its distance from the root as exp(-value*dist).
    Args:
        hierarchy: The hierarchy to use to generate the weights.
        value: The decay value.
        normalize: If True ensures that the sum of all weights sums
            to one.
    Returns:
        Weights as a nltk.Tree whose labels are the weights associated with the
        parent edge.
    """
    weights = deepcopy(hierarchy)
    all_weights = []
    for p in weights.treepositions():
        node = weights[p]
        weight = exp(-value * len(p))
        all_weights.append(weight)
        if isinstance(node, Tree):
            node.set_label(weight)
        else:
            weights[p] = weight
    total = fsum(all_weights)  # stable sum
    if normalize:
        for p in weights.treepositions():
            node = weights[p]
            if isinstance(node, Tree):
                node.set_label(node.label() / total)
            else:
                weights[p] /= total
    return weights


def get_weighting(hierarchy: Tree, weighting="uniform", **kwargs):
    """
    Get different weightings of edges in a tree.
    Args:
        hierarchy: The tree to generate the weighting for.
        weighting: The type of weighting, one of 'uniform', 'exponential'.
        **kwards: Keyword arguments passed to the weighting function.
    """
    if weighting == "uniform":
        return get_uniform_weighting(hierarchy, **kwargs)
    elif weighting == "exponential":
        return get_exponential_weighting(hierarchy, **kwargs)
    else:
        raise NotImplementedError("Weighting {} is not implemented".format(weighting))


def get_classes(hierarchy: Tree, output_all_nodes=False):
    """
    Return all classes associated with a hierarchy. The classes are sorted in
    alphabetical order using their label, putting all leaf nodes first and the
    non-leaf nodes afterwards.
    Args:
        hierarhcy: The hierarchy to use.
        all_nodes: Set to true if the non-leaf nodes (excepted the origin) must
            also be included.
    Return:
        A pair (classes, positions) of the array of all classes (sorted) and the
        associated tree positions.
    """

    def get_classes_from_positions(positions):
        classes = [get_label(hierarchy[p]) for p in positions]
        class_order = np.argsort(classes)  # we output classes in alphabetical order
        positions = [positions[i] for i in class_order]
        classes = [classes[i] for i in class_order]
        return classes, positions

    positions = hierarchy.treepositions("leaves")
    classes, positions = get_classes_from_positions(positions)

    if output_all_nodes:
        positions_nl = [p for p in hierarchy.treepositions() if p not in positions]
        classes_nl, positions_nl = get_classes_from_positions(positions_nl)
        classes += classes_nl
        positions += positions_nl

    return classes

def get_nx_chexpert_htree(save=False, save_path: Optional[str] = None):
    """loads the hierarchy tree of the ChexPert Dataset

    Args:
        save (bool, optional): Save the hierarchy tree in the default save location 
        provided in the config file. Defaults to True.

    Returns:
        htree: nx.Graph
    """    
    
    if save and save_path is None:
        raise ValueError("Please Provide a save-path")
    
    nodes_list = ['root', 'L1', 'No Finding', 'Enlarged Cardiomediastinum', 
                  'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 
                  'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 
                  'Pleural Effusion', 'Pleural Other', 'Fracture', 
                  'Support Devices']
    
    htree = nx.DiGraph()
    htree.add_nodes_from(nodes_list)
    
    # Level-1
    htree.add_edge('root', 'No Finding')
    htree.add_edge('root', 'Enlarged Cardiomediastinum')
    htree.add_edge('root', 'Support Devices')
    htree.add_edge('root', 'Fracture')
    htree.add_edge('root', 'Lung Opacity')
    htree.add_edge('root', 'L1')

    # Level-2
    htree.add_edge('Enlarged Cardiomediastinum', 'Cardiomegaly')
    htree.add_edge('L1', 'Pleural Effusion')
    htree.add_edge('L1', 'Pleural Other')
    htree.add_edge('L1', 'Pneumothorax')
    
    # Level-2 -- Lung Opacity Branch
    htree.add_edge('Lung Opacity', 'Edema')
    htree.add_edge('Lung Opacity', 'Consolidation')
    htree.add_edge('Lung Opacity', 'Pneumonia')
    htree.add_edge('Lung Opacity', 'Lung Lesion')
    htree.add_edge('Lung Opacity', 'Atelectasis')
    
    if save:
        path = os.path.join(save_path, 'cx_htree.graphml')
        print(f"Saving the Tree in '{path}'")
        nx.write_graphml_lxml(htree, path)
    
    return htree
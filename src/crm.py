from types import SimpleNamespace
from warnings import warn
from typing import Optional

import numpy as np
import networkx as nx
import torch

from .definitions import *
from .tree import get_chexpert_htree, lca


def get_cost_matrix(tree):
    """Calculates and returns the cost matrix for the CRM framework, given the predicted and ground truth values

    Args:
        tree (nx.Graph): Hierarchy Tree for the dataset
        node1 (listornp.ndarrayortorch.Tensor): Iterable containing node-1 information
        node2 (listornp.ndarrayortorch.Tensor): Iterable containing node-2 information
        config (SimpleNamespace): config file

    Returns:
        cost (np.ndarray): cost array as a numpy array
    """
    num_classes = len(LABEL_MAP)
    # common_lca_map = dict(nx.lowest_common_ancestors.all_pairs_lowest_common_ancestor(tree))
    cost = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            n1, n2 = REVERSE_LABEL_MAP[i], REVERSE_LABEL_MAP[j]
            cost[i, j] = lca(tree, n1, n2)

    return cost


def calc_crm_risk(
    output: torch.Tensor or np.ndarray,
    labels: torch.Tensor or np.ndarray,
    htree,
    hpath: Optional[str] = None,
    trim: bool = False,
):
    """Calculate the risk matrix from the outputs using CRM Framework

    Args:
        output (torch.Tensorornp.ndarray): _description_
        htree (Optional[Any]): Hierarchy Tree of the current dataset
        hpath (Optional[str]): Path to hierarchy tree in the json format
    """

    if htree is None and hpath is None:
        warn(
            "Please Specify a path to the tree or pass in the tree as an arg\
 loading the tree from default path ..."
        )
        htree = get_chexpert_htree()

    if htree is not None and hpath is not None:
        warn("Using passed in 'htree' ...")
    elif htree is None and hpath is not None:
        pass

    if isinstance(output, torch.Tensor):
        output = output.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    predicted = np.argmax(output, axis=1)
    cost = get_cost_matrix(htree)

    final = np.dot(output, cost)

    return -1 * final


def get_topk(prediction, target, htree, k=5):
    """Computing hierarchical distance@k"""
    if not isinstance(prediction, torch.Tensor):
        assert isinstance(prediction, list) or isinstance(
            prediction, np.ndarray
        ), "Invalid datatype for the prediction iterable {}".format(type(prediction))
        prediction = torch.tensor(prediction)

    whole = torch.topk(prediction, k)
    ind = whole.indices.detach().cpu().tolist()
    # print(ind)

    scores = []
    s1, s2 = 0, 0
    for i in ind:
        scores.append(lca(htree, REVERSE_LABEL_MAP[i], REVERSE_LABEL_MAP[target]))
    return scores


def get_metrics(output, target, htree, use_crm: bool = True, trim=False):
    if isinstance(output, torch.Tensor):
        output = output.detach().cpu()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu()

    # ##Random Shuffling if Required
    # if opts.shuffle_classes==1:
    #     np.random.seed(42)##Seed used to Train HXE/Soft-Labels. However, can be changed
    #     np.random.shuffle(classes)

    ##Apply CRM
    if use_crm:
        output = calc_crm_risk(output, target, htree, trim=trim)

    orig_top1 = []
    orig_mistake = []
    orig_avg_1 = []
    orig_avg_5 = []

    for i in range(len(output)):
        n1 = int(target[i])
        n2 = int(output[i].argmax())
        n3 = output[i]

        if n2 == n1:
            orig_top1.append(1)
        else:
            orig_top1.append(0)
            orig_mistake.append(
                lca(htree, REVERSE_LABEL_MAP[n1], REVERSE_LABEL_MAP[n2])
            )

        orig_avg_1.extend(get_topk(n3, n1, htree, 1))

        orig_avg_5.append(get_topk(n3, n1, htree, 5))

    # print("Top-1 Accuracy",np.array(orig_top1).mean())

    # print("Mistake Severity",np.array(orig_mistake).mean())

    # print("Hierarchical Distance@1",np.array(orig_avg_1).mean())

    # print("Hierarchical Distance@5",np.array(orig_avg_5).mean())

    result = {
        "top-1 error": np.array(orig_top1).mean(),
        "mistake severity": np.array(orig_mistake).mean(),
        "hierarchical distance@1": np.array(orig_avg_1).mean(),
        "hierarchical distance@5": np.array(orig_avg_5).mean(),
    }
    with open("temp_crm_results.txt", "a") as file:
        file.write(
            f"top-1 error: {np.array(orig_top1).mean()}, \
mistake severity: {np.array(orig_mistake).mean()}, \
hierarchical distance@1: {np.array(orig_avg_1).mean()}, \
hierarchical distance@5: {np.array(orig_avg_5).mean()}\n"
        )

    # print(result)

    return result

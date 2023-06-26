import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import statistics
import numpy as np
import networkx as nx
from typing import Any
from collections import defaultdict

from .crm import get_cost_matrix
from .definitions import REVERSE_LABEL_MAP, TREE_MAP, LABEL_MAP


def lca(htree: nx.DiGraph, node1: Any, node2: Any, mapping_dict: dict = None):
    if mapping_dict is not None:
        node1 = mapping_dict[node1]
        node2 = mapping_dict[node2]

    return TREE_MAP[nx.lowest_common_ancestor(G=htree, node1=node1, node2=node2)]


def calculate_multilabel_severity(
    label: torch.Tensor,
    ground_truth: torch.Tensor,
    htree,
    return_label_wise: bool = False,
):
    label = label.cpu().detach().tolist()
    ground_truth = ground_truth.cpu().detach().tolist()

    # Extract the labels from the one-hot vector
    label_group = [i for i, l in enumerate(ground_truth) if l == 1]

    # Get the matching and the different keys
    matched_keys = []
    diff_keys = []
    for (idx, l), (_, g) in zip(enumerate(label), enumerate(ground_truth)):
        if int(l) == 1 and int(g) == 1:
            matched_keys.append(idx)
        elif int(l) == 1 and int(g) == 0:
            diff_keys.append(idx)

    # Remove the matching keys of the label from the label_group
    for key in matched_keys:
        label_group.remove(key)

    # Calculate the Individual Severites and Sum there mean
    dd = defaultdict(list)
    severity: float = 0
    for l in diff_keys:
        temp = []
        for g in label_group:
            v = lca(htree, l, g, REVERSE_LABEL_MAP)
            temp.append(v)
            dd[REVERSE_LABEL_MAP[l]].append(v)
        severity += statistics.mean(temp) if len(temp) > 0 else severity

    if return_label_wise:
        return dd
    return severity


def batch_calculate_severity(
    label, gt, htree, device, crm: bool = False, return_label_wise: bool = False
):
    severities = []
    if crm:
        label = label.to(torch.float64)
        C = torch.tensor(get_cost_matrix(htree)).to(device)
        label = torch.matmul(label, C)
        # label = F.softmax(label)

    for l, g in zip(label, gt):
        l = torch.where(l > torch.mean(l), 1.0, 0.0)
        severities.append(calculate_multilabel_severity(l, g, htree, return_label_wise))

    if return_label_wise:
        label_wise_sev = {key: [] for key in LABEL_MAP.keys()}
        for d in severities:
            for key, value in d.items():
                label_wise_sev[key] += value
        # print(label_wise_sev)
        return label_wise_sev

    return severities

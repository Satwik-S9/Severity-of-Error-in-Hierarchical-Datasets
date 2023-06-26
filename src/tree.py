import os
from collections import defaultdict
from tqdm.auto import tqdm

import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional

from .definitions import REV_COMP_LABEL_MAP, ROOT_DIR
from .utils import load_config
from .dataset import LABEL_MAP, REVERSE_LABEL_MAP, TREE_MAP


## === FUNCTIONS === ##
def get_chexpert_htree(save=True, save_path: Optional[str] = None):
    """loads the hierarchy tree of the ChexPert Dataset

    Args:
        save (bool, optional): Save the hierarchy tree in the default save location
        provided in the config file. Defaults to True.

    Returns:
        htree: nx.Graph
    """

    if save and save_path is None:
        raise ValueError("Please Provide a save-path")

    nodes_list = [
        "root",
        "L1",
        "No Finding",
        "Enlarged Cardiomediastinum",
        "Cardiomegaly",
        "Lung Opacity",
        "Lung Lesion",
        "Edema",
        "Consolidation",
        "Pneumonia",
        "Atelectasis",
        "Pneumothorax",
        "Pleural Effusion",
        "Pleural Other",
        "Fracture",
        "Support Devices",
    ]

    htree = nx.DiGraph()
    htree.add_nodes_from(nodes_list)

    # Level-1
    htree.add_edge("root", "No Finding")
    htree.add_edge("root", "Enlarged Cardiomediastinum")
    htree.add_edge("root", "Support Devices")
    htree.add_edge("root", "Fracture")
    htree.add_edge("root", "Lung Opacity")
    htree.add_edge("root", "L1")

    # Level-2
    htree.add_edge("Enlarged Cardiomediastinum", "Cardiomegaly")
    htree.add_edge("L1", "Pleural Effusion")
    htree.add_edge("L1", "Pleural Other")
    htree.add_edge("L1", "Pneumothorax")

    # Level-2 -- Lung Opacity Branch
    htree.add_edge("Lung Opacity", "Edema")
    htree.add_edge("Lung Opacity", "Consolidation")
    htree.add_edge("Lung Opacity", "Pneumonia")
    htree.add_edge("Consolidation", "Pneumonia")
    htree.add_edge("Lung Opacity", "Lung Lesion")
    htree.add_edge("Lung Opacity", "Atelectasis")

    if save:
        path = os.path.join(save_path, "cx_htree.graphml")
        print(f"Saving the Tree in '{path}'")
        nx.write_graphml_lxml(htree, path)

    return htree


def lca(tree, node1, node2, mapping: dict or str = TREE_MAP):
    if isinstance(node1, int):
        node1 = REVERSE_LABEL_MAP[node1]
    if isinstance(node2, int):
        node2 = REVERSE_LABEL_MAP[node2]
    
    a = nx.lowest_common_ancestor(tree, node1, node2)
    return mapping[a]


def lca_batch(
    htree, labels, ground_truths, images, minmax: bool = True, sample_id: bool = True
):
    avg_lca = 0
    lca_list = []

    for label, ground_truth in zip(labels, ground_truths):
        l = REVERSE_LABEL_MAP[label.cpu().item()]
        g = REVERSE_LABEL_MAP[ground_truth.cpu().item()]
        # print(f"l: {l} || g: {g}")
        avg_lca += lca(htree, l, g)
        lca_list.append(lca(htree, l, g))

    avg_lca = avg_lca / labels.cpu().size()[0]
    result = {"avg_lca": avg_lca}
    if minmax:
        result["min_lca"] = min(lca_list)
        result["max_lca"] = max(lca_list)

    result["median"] = lca_list[len(lca_list) // 2]

    if sample_id:
        images_list = images.detach().cpu().tolist()
        label_list = labels.detach().cpu().tolist()
        gt_list = ground_truths.detach().cpu().tolist()

        min_idx = lca_list.index(min(lca_list))
        max_idx = lca_list.index(max(lca_list))

        t1 = np.array(images_list[min_idx])
        t2 = np.array(images_list[max_idx])
        result["min_sample"] = np.transpose(t1, (1, 2, 0))
        result["max_sample"] = np.transpose(t2, (1, 2, 0))
        result["min_label"] = label_list[min_idx]
        result["max_label"] = label_list[max_idx]
        result["min_gt"] = gt_list[max_idx]
        result["max_gt"] = gt_list[max_idx]

    return result


def batch_wise_lca(labels, gts, htree, crm: bool = False):
    if not crm:
        node_wise_lca = defaultdict(list)
        for label_list, gt_list in tqdm(zip(labels, gts)):
            for label, gt in zip(label_list, gt_list):
                # print(label.size())
                l = REVERSE_LABEL_MAP[label.cpu().item()]
                g = REVERSE_LABEL_MAP[gt.cpu().item()]

                node_wise_lca[g].append(lca(htree, l, g))

        for key in node_wise_lca.keys():
            node_wise_lca[key] = sum(node_wise_lca[key]) / len(node_wise_lca[key])

        return node_wise_lca
    else:
        ...


def collate_batch_data(
    htree, results, labels, ground_truths, comp: bool, plot=True, fname: str = None
):
    if not comp:
        if plot and fname is None:
            raise ValueError("Please provide fname")

        min_lca = results["min_lca"]
        max_lca = results["max_lca"]

        min_lca_nodes = []
        max_lca_nodes = []
        node_lca_info = {REVERSE_LABEL_MAP[i]: [] for i in range(14)}
        node_avg_lca_info = {REVERSE_LABEL_MAP[i]: 0 for i in range(14)}
        node_median_lca_info = {REVERSE_LABEL_MAP[i]: 0 for i in range(14)}

        # count = 0
        for l, g in zip(labels, ground_truths):
            # print(idx, l, g)
            l1 = REVERSE_LABEL_MAP[l.item()]
            g1 = REVERSE_LABEL_MAP[g.item()]

            lca_val = lca(htree, l1, g1)
            node_lca_info[g1].append(lca_val)
            node_avg_lca_info[g1] += lca_val
            # count += 1

            if lca_val == min_lca:
                min_lca_nodes.append((l.item(), g.item()))
            elif lca_val == max_lca:
                max_lca_nodes.append((l.item(), g.item()))

        # print(count)
        for key, value in node_avg_lca_info.items():
            #! Issue in this part of the code :: What choose as denom.
            node_avg_lca_info[key] = value / 8

        if plot:
            keys, values = list(node_avg_lca_info.keys()), list(
                node_avg_lca_info.values()
            )
            fig = plt.figure(figsize=(10, 10))
            plt.bar(keys, values)
            plt.xticks(rotation=90)

            fname += "avg_lca_labels.png"
            plt.savefig(fname)

        return {
            "min_lca_nodes": min_lca_nodes,
            "max_lca_nodes": max_lca_nodes,
            "node_lca_info": node_lca_info,
            "node_avg_lca_info": node_avg_lca_info,
        }

    else:
        if plot and fname is None:
            raise ValueError("Please provide fname")

        min_lca = results["min_lca"]
        max_lca = results["max_lca"]

        min_lca_nodes = []
        max_lca_nodes = []
        node_lca_info = {REV_COMP_LABEL_MAP[i]: [] for i in range(5)}
        node_avg_lca_info = {REV_COMP_LABEL_MAP[i]: 0 for i in range(5)}
        node_median_lca_info = {REV_COMP_LABEL_MAP[i]: 0 for i in range(5)}

        # count = 0
        for l, g in zip(labels, ground_truths):
            # print(idx, l, g)
            l1 = REV_COMP_LABEL_MAP[l.item()]
            g1 = REV_COMP_LABEL_MAP[g.item()]

            lca_val = lca(htree, l1, g1)
            node_lca_info[g1].append(lca_val)
            node_avg_lca_info[g1] += lca_val
            # count += 1

            if lca_val == min_lca:
                min_lca_nodes.append((l.item(), g.item()))
            elif lca_val == max_lca:
                max_lca_nodes.append((l.item(), g.item()))

        # print(count)
        for key, value in node_avg_lca_info.items():
            #! Issue in this part of the code :: What choose as denom.
            node_avg_lca_info[key] = value  # /8

        if plot:
            keys, values = list(node_avg_lca_info.keys()), list(
                node_avg_lca_info.values()
            )
            fig = plt.figure(figsize=(10, 10))
            plt.bar(keys, values)
            plt.xticks(rotation=90)

            fname += "avg_lca_labels.png"
            plt.savefig(fname)

        return {
            "min_lca_nodes": min_lca_nodes,
            "max_lca_nodes": max_lca_nodes,
            "node_lca_info": node_lca_info,
            "node_avg_lca_info": node_avg_lca_info,
        }

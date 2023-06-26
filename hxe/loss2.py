"""
    # TODOS:
        todo: [ ] Implement the weighting solution for uniform and exponential weights
"""

from math import exp
import torch
import torch.nn as nn 
import networkx as nx 
from typing import Any, Iterable, Union
import torch.nn.functional as F

class HXELoss(nn.Module):
    def __init__(self, htree: nx.DiGraph, num_classes: int, 
                 ignore_nodes: list = None, label_map: dict = None, cuda=False,
                 weight_policy: str = 'uniform', alpha: float = 9.0) -> None:
        # Initialize the super class
        super().__init__()
        # Initialize Basic Members
        self.num_classes: int = num_classes
        self.htree: nx.Graph = htree
        self.ignore_nodes: list = ignore_nodes
        self.LABEL_MAP: dict = label_map
        self.REVERSE_LABEL_MAP: dict = {value: key for key, value in label_map.items()}
        self.weight_policy: str = weight_policy
        self.alpha: float = alpha
        # Initialize the derived members
        self.distances: dict = dict(nx.shortest_path_length(htree))
        self.__htree_nodes: list = list(self.htree.nodes())
        self.device = None
        
        # Setup Device
        if cuda:
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        
    ### HELPER METHODS ###
    def _get_hierarchy(self, node: Any):
        H = []
        x = node
        while True:
            x = list(self.htree.predecessors(x))[0]
            H.append(x)
            if x == self.__htree_nodes[0]:
                break
        
        return H

    def _get_parent(self, node: Any):
        return list(self.htree.predecessors(node))[0]

    def _get_child(self, node: Any):
        return list(self.htree.successors(node))[0]

    def _get_leaves(self, node: Any, label_map: dict = None):
        # Get the leaves of the current node if the node is not the root node
        leaves = []
        if node != self.__htree_nodes[0]:
            p = self._get_parent(node)
            # print(p, distances[p])
            for key, dist in self.distances[p].items():
                if dist == 1:
                    leaves.append(key)
        
        # remove all the nodes that are to be ignored
        for ignode in self.ignore_nodes:         
            if ignode in leaves:
                leaves.remove(ignode)
        
        # apply a label map if provided
        if label_map is not None:
            for idx, leaf in enumerate(leaves):
                leaves[idx] = label_map[leaf]
            
        return leaves
    
    def _generate_class_weights(self, policy: str, targets: Union[Iterable, torch.Tensor] ):
        batch_size: int = targets.size()[0]
        weights: list = []
        if policy == 'uniform':
            weights = [self.alpha for _ in range(batch_size)]
        else:
            heights = self.distances[self.__htree_nodes[0]]
            for target in targets.cpu().detach().tolist():
                weight = exp(-self.alpha * heights[self.REVERSE_LABEL_MAP[target]])
                weights.append(weight)
                assert len(weights) == batch_size
        return weights
    
    def forward(self, model_outputs, labels):
        batch_size = model_outputs.size()[0]

        # -> Going Over the Numerator
        # > Start with the target leaves
        label_leaves: list = []
        for label in labels.cpu().detach().tolist():
            label_leaves.append(self._get_leaves(self.REVERSE_LABEL_MAP[label], self.LABEL_MAP))
        # print(label_leaves, len(label_leaves))

        # > Generating One Hot Encoding per label :: Numerator
        one_hot_num = torch.zeros((batch_size, self.num_classes, 1), requires_grad=False, device=self.device)
        for idx, _ in enumerate(labels):
            one_hot_num[idx, torch.tensor(label_leaves[idx], device=self.device)] = 1
        # print(one_hot_num.size(), one_hot_num[18])

        # -> Going over the denomenator
        # > Generating the Parent
        label_parents: list = []
        for label in labels.cpu().detach().tolist():
            p = self._get_parent(self.REVERSE_LABEL_MAP[label])
            if p not in self.ignore_nodes:
                label_parents.append(p)
            else:
                label_parents.append(None)
        # print(label_parents)

        # > Generating Leaves of Parents
        label_parent_leaves: list = []
        for label in label_parents:
            if label is not None:
                label_parent_leaves.append(self._get_leaves(label, self.LABEL_MAP))
            else:
                label_parent_leaves.append(None)
        # print(label_parent_leaves)

        # > Generating One Hot Encoding per label :: Denomenator
        one_hot_den = torch.zeros((batch_size, self.num_classes, 1), requires_grad=False, device=self.device)
        for idx, _ in enumerate(label_parents):
            if label_parent_leaves[idx] is not None:
                one_hot_den[idx, torch.tensor(label_parent_leaves[idx], device=self.device)] = 1
            else:
                one_hot_den[idx, :] = 1
        # print(one_hot_den)

        # -> Unsqueeze and Apply Softmax to Model Outputs to a temporary tensor
        tmo = F.softmax(model_outputs, 1)
        tmo = tmo.unsqueeze(dim=1)
        tmo.size()

        # -> Calculate the numerator and the denomenator according to the formula in the paper by Bernitto et. al.
        num = torch.bmm(tmo, one_hot_num)
        den = torch.bmm(tmo, one_hot_den)

        # -> FINAL RESULT
        res = -torch.log(num / den)
        return torch.mean(res)
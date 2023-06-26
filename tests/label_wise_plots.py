### Append the root directory to the system path ###
import os
import sys
from pathlib import Path
ROOT_DIR = Path(Path.cwd().root) / "workspace/utsa/MTP-SatwikSrivastava"
sys.path.insert(1, str(ROOT_DIR))

### Regular Imports ###
## PyTorch
import torch
from torch.utils.data import DataLoader

## Standard Libarary and Other Small Modules
import yaml
import pickle
import statistics
import networkx as nx
from barbar import Bar
from typing import Union
from dataclasses import dataclass
from yaml.loader import SafeLoader
from argparse import ArgumentParser

## Custom Imports
import src.cx_models as cxm
from src.dataset import CheXpert2
from src.severity import batch_calculate_severity
from src.utils import set_global_seed
from src.tree import get_chexpert_htree


@dataclass
class Config:
    model_path: Union[str, Path] = Path(Path.cwd().root) / "workspace/utsa/chexpert_results/models/hxe"
    batch_size: int = 32
    num_classes: int = 14
    seed: int = 42
    htree: nx.DiGraph = get_chexpert_htree(False)
    data_dir: Union[Path, str] = Path("/workspace/DATASETS/CheXpert-v1.0-small/")
    save_dir: Union[Path, str] = Path(Path.cwd().root) / "workspace/utsa/chexpert_results/MTP/MultiLabelSeverity/hxe"
    crm: bool = False
    till = -1
    
def test():
    ...

    
if __name__ == '__main__':
    # Get the available models
    with open(Path(ROOT_DIR) / "info/model_info.yaml", "rb") as file:
        available_models = yaml.load(file, SafeLoader)
    
    # Parse the arguments
    parser = ArgumentParser()
    parser.add_argument("--model", 
                        type=str, 
                        action="store", 
                        required=True, 
                        choices=available_models
                    )
    parser.add_argument("--seed", 
                        type=int, 
                        action="store", 
                        default=42
                    )
    parser.add_argument("--crm", 
                        action="store_true", 
                    )
    parser.add_argument("--till", 
                        action="store", 
                        type=int, 
                        default=-1)
    
    args = parser.parse_args()
    
    # Setup the config
    Config.model_path = Config.model_path / (args.model + ".pth")
    Config.seed = args.seed
    Config.crm = args.crm
    Config.till = args.till
    
    # Set the gloabl seed
    set_global_seed(Config.seed)
    
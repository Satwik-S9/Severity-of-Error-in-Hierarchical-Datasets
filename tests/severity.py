# Append to system path
import os
import sys
from pathlib import Path
ROOT_DIR = Path(Path.cwd().root) / "workspace/utsa/MTP-SatwikSrivastava"
sys.path.insert(1, str(ROOT_DIR))

# Regular Imports
## PyTorch
import torch
from torch.utils.data import DataLoader

## Standard Imports
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
from src.definitions import LABEL_MAP


@dataclass
class Config:
    model_path: Union[str, Path] = Path(Path.cwd().root) / "workspace/utsa/chexpert_results/models"
    batch_size: int = 32
    num_classes: int = 14
    seed: int = 42
    htree: nx.DiGraph = get_chexpert_htree(False)
    data_dir: Union[Path, str] = Path("/workspace/DATASETS/CheXpert-v1.0-small/")
    save_dir: Union[Path, str] = Path(Path.cwd().root) / "workspace/utsa/chexpert_results/MTP/MultiLabelSeverity"
    crm: bool = False
    till: int = -1
    label_wise: bool = False

def load_model(model_name, hxe):
    if model_name == 'densenet121':
        if hxe:
            model = cxm.Densenet121HXE(Config.num_classes)
        else:
            model = cxm.DenseNet121(Config.num_classes)
    elif model_name == 'resnet18':
        if hxe:
            model = cxm.Resnet18HXE(Config.num_classes)
        else:
            model = cxm.Resnet18(Config.num_classes)
    elif model_name == 'resnet50':
        model = cxm.Resnet50(Config.num_classes)
    elif model_name == 'mobilenet':
        model = cxm.MobileNet(Config.num_classes)
    elif model_name == 'shufflenet':
        model = cxm.ShuffleNet1_0(Config.num_classes)
    elif model_name == 'effnetb4':
        model = cxm.EffnetB4(Config.num_classes)
    elif model_name == 'wideresnet':
        model = cxm.WideResnet50(Config.num_classes)
        
    return model

def test(model):
    # Load the model
    model = load_model(model, Config.hxe)
    state = torch.load(Config.model_path)
    model.load_state_dict(state['state_dict'])
    
    # Load the Datset and the DataLoader
    dataset = CheXpert2(Config.data_dir, split='valid', till=Config.till)
    dataloader = DataLoader(dataset, 
                        Config.batch_size, 
                        pin_memory=True, 
                        num_workers=8)

    # init the device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # Calculate the severities
    severities = []
    model = model.to(device)
    model.eval()
    for images, labels in Bar(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        
        logits = model(images)
        
        if Config.label_wise:
            severities.append(batch_calculate_severity(logits, 
                                                labels, 
                                                Config.htree, 
                                                device, 
                                                Config.crm,
                                                Config.label_wise))
        else:    
            severities += batch_calculate_severity(logits, 
                                                labels, 
                                                Config.htree, 
                                                device, 
                                                Config.crm,
                                                Config.label_wise)
        
    if Config.label_wise:
        # print(severities)
        label_wise_sev = {key: [] for key in LABEL_MAP.keys()}
        for d in severities:
            for key, value in d.items():
                label_wise_sev[key] += value
        
        return label_wise_sev
    
    return severities

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
    parser.add_argument("--lb", 
                        action="store_true"
                    )
    parser.add_argument("--hxe", 
                        action="store_true"
                    )
    args = parser.parse_args()
    
    # Setup the config
    Config.seed = args.seed
    Config.crm = args.crm
    Config.till = args.till
    Config.label_wise = args.lb
    Config.hxe = args.hxe
    if args.hxe:
        print("RUNNING EXPERIMENT FOR HXE")
        Config.save_path = Config.save_dir / "hxe"
        Config.model_path = Config.model_path / "hxe"
    else:
        Config.model_path = Config.model_path / "main"
        
    
    Config.model_path = Config.model_path / (args.model + ".pth")
        
    # Set the gloabl seed
    set_global_seed(Config.seed)    
    
    # Get the result
    severities = test(args.model)
    # print(severities)
    if Config.label_wise:
        mv = {key: statistics.mean(value) for key, value in severities.items()}
        stdv = {key: statistics.stdev(value) for key, value in severities.items()}
    if not Config.label_wise:
        mv = statistics.mean(severities)
        stdv = statistics.stdev(severities)
        print(f"The average severity of the model is {mv} | StdDev is: {stdv}")

    if not os.path.exists(Config.save_dir):
        os.makedirs(Config.save_dir)
    
    if args.crm:
        path = Config.save_dir / f"{args.model}_crm_results"
    else:
        path = Config.save_dir / f"{args.model}_results"
        
    if Config.label_wise:
        path = str(path) + "_lb"
        
    # Save the results    
    with open(str(path) + ".pkl", "wb") as file:
        pickle.dump(severities, file)
    
    with open(str(path) + ".txt", "w") as file:
        file.write(str(severities) 
                + f"\n => The average severity of the model is: {mv}\n"
                + f"\n => The StdDev is: {stdv}\n")
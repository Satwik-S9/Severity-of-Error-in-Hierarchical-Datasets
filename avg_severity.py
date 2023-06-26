import os
import pickle
import random
import statistics
import numpy as np
from pprint import pprint
from barbar import Bar
from pathlib import Path
from typing import DefaultDict
from argparse import ArgumentParser
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

import src.cx_models as cxm
from src.definitions import REVERSE_LABEL_MAP
from src.dataset import CheXpert2 as CheXpert
from src.tree import lca, get_chexpert_htree
from src.crm import get_cost_matrix

# UTILITY FUNCTION
RESULTS_PATH = "/workspace/utsa/chexpert_results/BCE_RESULTS_SEV_HXE.pkl"
STATS_PATH = "/workspace/utsa/chexpert_results/BCE_STATS_SEV_HXE.pkl"
def load_pkl(path: str):
    try:
        with open(path, "rb") as file:
            results: DefaultDict[dict] = pickle.load(file) 
        return results
    except:
        return defaultdict(dict)
    
def save_pkl(results: defaultdict, path: str):
    with open(path, "wb") as file:
        pickle.dump(results, file)

# CONSTANTS
IDX_MAP = {
    "densenet121": 0,
    "wideresnet": 1,
    "resnet50": 2,
    "effnetb4": 3,
    "resnet18": 4,
    "shufflenet": 5,
    "mobilenet": 6 
}
REV_IDX_MAP = {value:key for key, value in IDX_MAP.items()}

MODELS = [cxm.Densenet121HXE, 
          cxm.WideResnet50,
          cxm.Resnet50,
          cxm.EffnetB4,
          cxm.Resnet18HXE,
          cxm.ShuffleNet1_0,
          cxm.MobileNet]

PATH = "/workspace/utsa/chexpert_trainer/results/models"

temp = os.listdir(PATH)

MODEL_PATHS = [os.path.join(PATH, mp, "hxe") for mp in temp]
DATA_DIR = "/workspace/DATASETS/CheXpert-v1.0-small/"
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
BATCH_SIZE = 64
HTREE = get_chexpert_htree(False)
CRM = False


def main(IDX: int):
    # Load the model
    print(f"Loading Model {REV_IDX_MAP[IDX]}...\n")
    model = MODELS[IDX](14)
    
    # Load the model paths
    temp = os.listdir(MODEL_PATHS[IDX])
    model_paths = [os.path.join(MODEL_PATHS[IDX], mp) for mp in temp if mp != "history"]

    
    # Load Dataset and Dataloader
    print("Preparing Dataloader ...\n")
    dataset = CheXpert(DATA_DIR, split="test", policy="none")
    dataloader = DataLoader(dataset, BATCH_SIZE, pin_memory=True, num_workers=8)
    
    # Load a random choice from the list
    mp = random.choice(model_paths)
    state = torch.load(mp)
    model.load_state_dict(state['state_dict'])
    
    
    print("Evaluating...\n")
    
    node_wise_lca = defaultdict(list)
    model = model.to(DEVICE)
    model.eval()
    for images, labels in Bar(dataloader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        
        logits = model(images)
        if CRM:
            C = get_cost_matrix(HTREE)
            logits = torch.tensor(
                -1*np.dot(logits.cpu().detach().numpy(), C)
            )
        
        preds = torch.argmax(logits, dim=1)
        
        for p, gt in zip(preds, labels):
            l = REVERSE_LABEL_MAP[p.cpu().item()]
            g = REVERSE_LABEL_MAP[gt.cpu().item()]

            node_wise_lca[g].append(lca(HTREE, l, g))
    
    RESULTS[REV_IDX_MAP[IDX]] = node_wise_lca
    
    node_wise_mean = {key: statistics.mean(value) for key, value in node_wise_lca.items()}
    node_wise_stddev = {key: statistics.stdev(value) for key, value in node_wise_lca.items()}
    temp_full = []
    for l in node_wise_lca.values():
        temp_full += l
    model_mean = statistics.mean(temp_full)
    model_std = statistics.stdev(temp_full)
    
    stats = {
        "model_mean": model_mean,
        "model_std": model_std,
        "node_wise_mean": node_wise_mean,
        "node_wise_std": node_wise_stddev
    }
    
    return node_wise_lca, stats


if __name__ == '__main__':
    # Parse Arguments
    parser = ArgumentParser()
    parser.add_argument("--model", action="store", type=str, help="specif the model")
    parser.add_argument("--batch-size", action="store", type=int, help="specify the model", default=64)
    parser.add_argument("--crm", action="store_true", help="Use CRM")
    args = parser.parse_args()
    
    
    # SET THE RUN-TIME CONSTANTS
    IDX: int = IDX_MAP[args.model]
    BATCH_SIZE = args.batch_size
    CRM = args.crm
    if args.crm:
        RESULTS_PATH = "/workspace/utsa/chexpert_results/BCE_RESULTS_SEV_CRM.pkl"
        STATS_PATH = "/workspace/utsa/chexpert_results/BCE_STATS_SEV_CRM.pkl"
    
    # Load the relevant storages
    RESULTS = load_pkl(RESULTS_PATH)
    STATS_ALL = load_pkl(STATS_PATH)
    
    # START THE EVALUATION
    print(f"\n> Evaluating for model: {args.model}")
        
    RESULTS[args.model], STATS_ALL[args.model] = main(IDX)
    save_pkl(RESULTS, RESULTS_PATH)
    save_pkl(STATS_ALL, STATS_PATH)
    print("The Stats for the model: \n\n")
    pprint(STATS_ALL, indent=4)
    
        
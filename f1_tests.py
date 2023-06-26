import os
import pickle
import statistics
from pprint import pprint
from barbar import Bar
from pathlib import Path
from typing import DefaultDict
from argparse import ArgumentParser
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassF1Score

import src.cx_models as cxm
from src.dataset import CheXpert2 as CheXpert


# UTILITY FUNCTION
RESULTS_PATH = "/workspace/utsa/chexpert_results/BCE_RESULTS_F1_2.pkl"
def load_results(path: str = RESULTS_PATH):
    try:
        with open(path, "rb") as file:
            results: DefaultDict[dict] = pickle.load(file) 
        return results
    except:
        return defaultdict(dict)
    
def save_results(results: defaultdict):
    with open(RESULTS_PATH, "wb") as file:
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

MODEL_PATHS = [os.path.join(PATH, mp) for mp in temp]
RESULTS = load_results()
DATA_DIR = "/workspace/DATASETS/CheXpert-v1.0-small/"
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
BATCH_SIZE = 64


def main(IDX: int):
    # Load the model
    print(f"Loading Model {REV_IDX_MAP[IDX]}...\n")
    model = MODELS[IDX](14)
    
    # Load the model paths
    model_paths = []
    temp = os.listdir(MODEL_PATHS[IDX])
    for mp in temp:
        if mp == "history" or mp == "hxe":
            continue
        model_paths.append(os.path.join(MODEL_PATHS[IDX], mp))
    
    # Load Dataset and Dataloader
    print("Preparing Dataloader ...\n")
    dataset = CheXpert(DATA_DIR, split="test", policy="none")
    dataloader = DataLoader(dataset, BATCH_SIZE, pin_memory=True, num_workers=8)
    
    # Store all the models
    temp_models = []
    for mp in model_paths:
        state = torch.load(mp)
        model.load_state_dict(state['state_dict'])
        temp_models.append(model)
    
    print("Evaluating...\n")
    f1_scores = {}
    for idx, m in enumerate(temp_models):
        f1_score = 0
        m = m.to(DEVICE)
        count = 0
        m.eval()
        for images, labels in Bar(dataloader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            logits = m(images)
            metric = MulticlassF1Score(14).to(DEVICE)
            f1_score += metric(logits, labels).cpu().item()
            count += 1
        
        f1_scores[idx] = (f1_score / count)*10
        pprint(f1_scores, indent=4)
        torch.cuda.empty_cache()
    
    # ADD Mean and StdDev Info
    temp_v = list(f1_scores.values())
    f1_scores["mean"] = statistics.mean(temp_v)
    f1_scores["stddev"] = statistics.stdev(temp_v)
    
    RESULTS[REV_IDX_MAP[IDX]] = f1_scores
    return RESULTS

if __name__ == '__main__':
    # Parse Arguments
    parser = ArgumentParser()
    # parser.add_argument("--model", action="store", type=str, help="specif the model")
    parser.add_argument("--batch-size", action="store", type=int, help="specif the model", default=64)
    args = parser.parse_args()
    
    for model in IDX_MAP.keys():
        IDX: int = IDX_MAP[model]
        BATCH_SIZE = args.batch_size
        
        print(f"\n> Evaluating for model: {model}")
            
        results = main(IDX)
        save_results(results)
        print("The Results uptil now are: \n\n")
        pprint(results, indent=4)
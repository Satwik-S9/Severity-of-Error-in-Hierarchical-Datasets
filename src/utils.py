import os
import json
import random
import yaml
from time import perf_counter
from types import SimpleNamespace
from typing import List
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import torch
from torchvision.models import densenet121
import torchxrayvision as xrv

from . import cx_models as cxm
from .definitions import REV_COMP_LABEL_MAP, REVERSE_LABEL_MAP, ROOT_DIR


@dataclass()
class MakeUp:
    # STYLES
    RESET: str = "\033[0m"
    BOLD: str = "\033[1m"
    LIGHT: str = "\033[2m"
    ITALIC: str = "\033[3m"
    UNDERLINE: str = "\033[4m"

    # COLOURS
    BLACK: str = "\033[30m"
    RED: str = "\033[31m"
    GREEN: str = "\033[32m"
    YELLOW: str = "\033[33m"
    BLUE: str = "\033[34m"
    MAGENTA: str = "\033[35m"
    CYAN: str = "\033[36m"
    WHITE: str = "\033[37m"


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    else:
        return False


#! DEPRECATED
def load_config(config_path: str):
    with open(config_path, "r") as config_file:
        config_dict = yaml.safe_load(config_file)

    config_dict["labels"] = [
        "No Finding",
        "Enlarged Cardiomediastinum",
        "Support Devices",
        "Fracture",
        "Lung Opacity",
        "Cardiomegaly",
        "Pleural Effusion",
        "Pleural Other",
        "Pneumothorax",
        "Edema",
        "Consolidation",
        "Pneumonia",
        "Lung Lesion",
        "Atelectasis",
    ]
    # Alter some params
    config_dict["cuda"] = torch.cuda.is_available()
    config_dict["device"] = (
        torch.device("cuda") if config_dict["cuda"] else torch.device("cpu")
    )
    config_dict["save_path"] = os.path.join(ROOT_DIR, config_dict["save_path"])

    # Convert to SimpleNamespace for easy access
    config = SimpleNamespace(**config_dict)
    return config


def load_model(args: SimpleNamespace):
    # Load the list of all trained and available models
    model_info_path: str = os.path.join(ROOT_DIR, "info/model_info.yaml")
    with open(model_info_path, "r") as file:
        available_models: List[str] = yaml.safe_load(file)

    # Error Checking
    if args.model not in available_models:
        raise ValueError("Invalid parameter model_name !!")

    # Setup Model root directory
    if args.criterion == "hxe":
        model_root = os.path.join(ROOT_DIR, "models", "hxe")
    else:
        model_root = os.path.join(ROOT_DIR, "models", "pretrained")

    # create model path
    path: str = str(None)
    for m in os.listdir(model_root):
        if args.model in m:
            path = os.path.join(model_root, m)

    # Print the model loading path
    print("Loading model from path: {}".format(path))

    # Load the base classifier
    model = None
    best_auc: float = float()
    if args.model in ["densenet", "densenet121"]:
        if args.criterion == "hxe":
            model = cxm.Densenet121HXE(args.num_classes)
        else:
            model = cxm.DenseNet121(args.num_classes)

    elif args.model == "resnet18":
        if args.criterion == "hxe":
            model = cxm.Resnet18HXE(args.num_classes)
        else:
            model = cxm.Resnet18HXE(args.num_classes)

    elif args.model in ["effnetb4", "resnet152"]:
        if args.criterion == "hxe":
            model = cxm.EffnetB4(args.num_classes)
        else:
            model = cxm.load_base_classifier(args.model)

    elif args.model == "mobilenet":
        model = cxm.MobileNet(args.num_classes)

    elif args.model == "resnet50":
        model = cxm.Resnet50(args.num_classes)

    elif args.model == "googlenet":
        model = cxm.GoogleNet(args.num_classes)

    else:
        raise NotImplementedError("Not implemented yet for {}".format(args.model))

    # Load state-dict and best-auc
    state = torch.load(path)
    model.load_state_dict(state["state_dict"])
    best_auc = (
        state["avg_auc"]
        if (args.model in ["effnetb4", "resnet152"] and args.criterion != "hxe")
        else state["best_auc"]
    )

    return model, best_auc


def load_model_from_file(model_type: str, model_path: str, num_classes: int):
    # Load the list of all trained and available models
    model_info_path: str = os.path.join(ROOT_DIR, "info/model_info.yaml")
    with open(model_info_path, "r") as file:
        available_models: List[str] = yaml.safe_load(file)

    if model_type not in available_models:
        raise ValueError(
            f"Invalid Mode Type !!\nOnly the following models are available: {available_models}"
        )

    # print("loading models ...")
    if model_type in ["densenet", "densenet121"]:
        model = cxm.Densenet121HXE(num_classes)

    elif model_type == "resnet18":
        model = cxm.Resnet18HXE(num_classes)

    elif model_type == "resnet50":
        model = cxm.Resnet50(num_classes)

    elif model_type == "effnetb4":
        model = cxm.EffnetB4(num_classes)

    elif model_type in ["wideresnet", "wideresnet50"]:
        model = cxm.WideResnet50(num_classes)

    elif model_type == "shufflenet":
        model = cxm.ShuffleNet1_0(num_classes)

    elif model_type == "mobilenet":
        model = cxm.MobileNet(num_classes)

    else:
        raise NotImplementedError("Not implemented yet for {}".format(model_type))

    # Load state-dict and best-auc
    state = torch.load(model_path)
    model.load_state_dict(state["state_dict"])
    best_auc = state["best_auc"]

    return model, best_auc


def load_model_from_folder(args):
    print(args.model_dir)
    if not os.path.isdir(args.model_dir):
        raise ValueError("Not a directory")

    hxe = True if args.criterion == "hxe" else False
    model_dirs = os.listdir(os.path.join(args.model_dir, args.model))

    models, best_aucs = [], []
    for md in model_dirs:
        if md == "history" or md == "hxe":
            continue
        model_path = os.path.join(args.model_dir, args.model, md)
        m, b = load_model_from_file(args.model, model_path, args.num_classes)
        models.append(m)
        best_aucs.append(b)

    return models, best_aucs


def get_accuracy(logits, labels, verbose=True):
    predicted = torch.argmax(logits, 1)
    acc = torch.sum(predicted == labels) / labels.size()[0]
    if verbose:
        if acc <= 0.6:
            print(
                f"Accuracy of the Model: {MakeUp.BOLD}{MakeUp.RED}{acc.item():.4f}{MakeUp.RESET}\n"
            )
        elif acc > 0.6 and acc < 0.8:
            print(
                f"Accuracy of the Model: {MakeUp.BOLD}{MakeUp.YELLOW}{acc.item():.4f}{MakeUp.RESET}\n"
            )
        else:
            print(
                f"Accuracy of the Model: {MakeUp.BOLD}{MakeUp.GREEN}{acc.item():.4f}{MakeUp.RESET}\n"
            )

    return acc.item()


def plot_sample(sample, fname, label=None, gt=None, comp: bool = False):
    if not comp:
        fig, ax = plt.subplots(1)
        ax.imshow(sample, cmap=cm.gray)
        if label is not None and gt is not None:
            ax.text(
                10,
                10,
                f"L: {REVERSE_LABEL_MAP[label]} | GT: {REVERSE_LABEL_MAP[gt]} ",
                bbox={"facecolor": "white", "pad": 10},
            )
        plt.savefig(fname)
    else:
        fig, ax = plt.subplots(1)
        ax.imshow(sample, cmap=cm.gray)
        if label is not None and gt is not None:
            ax.text(
                10,
                10,
                f"L: {REV_COMP_LABEL_MAP[label]} | GT: {REV_COMP_LABEL_MAP[gt]} ",
                bbox={"facecolor": "white", "pad": 10},
            )
        plt.savefig(fname)


if __name__ == "__main__":
    config = load_config()
    print("CONFIG: train.csv path:: {}\n\n".format(config.train_csv_path))

    path = os.path.join(config.model_root, config.model_name)
    start = perf_counter()
    model = load_model(path, "densenet121")
    stop = perf_counter()
    print(f"\n\n MODEL --- \n\n  {model.parameters()} \n\n")

    print(f"Model Loaded in {stop-start:.4f}s")

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
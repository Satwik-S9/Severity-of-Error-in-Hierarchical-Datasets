import os
import json
import time
import atexit
from argparse import ArgumentParser

import torch
from runs import severity_run
from runner import Runner
from src.definitions import ROOT_DIR

global sr_time, st_time
sr_time = time.perf_counter()

###* KEEP COUNT OF RUNS *###
def read_counter():
    if os.path.exists("counter.json"):
        with open("counter.json", "r") as file:
            c = json.loads(file.read())
            c = int(c) if c is not None else 0
            return c + 1
    else:
        return 0


counter = read_counter()


def write_counter():
    with open("counter.json", "w") as f:
        f.write(json.dumps(counter))


atexit.register(write_counter)

###* PARSE COMMAND LINE ARGUMENTS *###
def parse_args():
    parser = ArgumentParser(description="Parse Arguments for Input Values")

    # Add Parser Arguments
    ## Directories
    parser.add_argument(
        "--save-dir",
        action="store",
        type=str,
        help="path to the config.yaml/yml file, (default: results2)",
        default="results2",
    )

    parser.add_argument(
        "--data-dir",
        action="store",
        type=str,
        help="path to the chexpert dataset",
        default="/workspace/DATASETS/CheXpert-v1.0-small",
    )

    parser.add_argument(
        "--model-dir",
        action="store",
        type=str,
        help="path to the models directory",
        default="/workspace/utsa/chexpert_trainer/results/models",
    )

    parser.add_argument(
        "--model-path",
        action="store",
        type=str,
        help="path to the models directory",
        default="/workspace/utsa/chexpert_trainer/results/models/densenet121/densenet121_20230219192728.pth",
    )

    ## Experiment configurations
    parser.add_argument(
        "--epochs",
        action="store",
        type=int,
        help="No. of epochs for which the training should run, (default: 10)",
        default=10,
    )

    parser.add_argument(
        "--batch-size",
        action="store",
        type=int,
        help="Batch size for training",
        default=32,
    )

    parser.add_argument(
        "--lr",
        action="store",
        type=float,
        default=0.001,
        help="Learning rate to be used in the optimizer",
    )

    parser.add_argument(
        "--img-size",
        action="store",
        type=int,
        default=224,
        help="Image size that we will resize the input images to",
    )

    parser.add_argument(
        "--num-classes",
        action="store",
        type=int,
        default=14,
        help="Number of classes for the dataset",
    )

    parser.add_argument(
        "--no-pin",
        action="store_false",
        help="raise this flag if you don't want to pin the memory for dataloaders",
    )

    parser.add_argument(
        "--workers",
        action="store",
        type=int,
        default=6,
        help="Workers used to create the dataloader",
    )

    parser.add_argument(
        "--model",
        action="store",
        type=str,
        help="Pretrained Model (default: densenet)",
        default="densenet121",
    )

    parser.add_argument(
        "--criterion",
        action="store",
        type=str,
        default="xentropy",
        help="Loss function to use, (default: xentropy)",
        choices=["default", "xentropy", "mse", "hxe", "bce"],
    )

    parser.add_argument(
        "--no-paint",
        action="store_false",
        help="Location to save the results, (default: False)",
        default=True,
    )

    parser.add_argument(
        "--exp", action="store", type=str, default="crm", help="experiment type"
    )

    parser.add_argument(
        "--comp", action="store_true", help="Run this model for 5 competetion labels"
    )
    parser.add_argument(
        "--crm",
        action="store_true",
        help="Run the model with crm framework to reduce severity",
    )

    parser.add_argument(
        "--device",
        action="store",
        type=str,
        default="cuda:0",
        help="device to be used: GPU or CPU",
    )

    parser.add_argument(
        "--no-multirun",
        action="store_true",
        help="Run for Mutliple Instances of a model trained with different seeds",
    )

    args = parser.parse_args()

    if args.comp:
        args.num_classes = 5

    if args.device in ["cuda:0", "gpu", "cuda"] and torch.cuda.is_available():
        args.device = torch.device(args.device)
    else:
        print("Using CPU ...\n")
        args.device = torch.device("cpu")
    args.no_paint = False

    return args


if __name__ == "__main__":
    args = parse_args()
    print(f"\nParsed Args:\n {args}")

    if not args.no_multirun:
        runner = Runner(args, counter)
        results, extras = runner.run()
    else:
        result, detailed_metrics = severity_run(args, counter)

    # save the result if not for the multirun
    if args.no_multirun:
        result_locn = os.path.join(args.save_dir, "results.json")
        with open(result_locn, "a") as result_file:
            json_str = "{\n    main: {\n"
            json_str += json.dumps(result, indent=4)
            json_str += "\n}}\n"
            result_file.write(json_str)

    st_time = time.perf_counter()

    print(f"TIME TAKEN FOR THE RUN: {st_time-sr_time:.2f}s\n")

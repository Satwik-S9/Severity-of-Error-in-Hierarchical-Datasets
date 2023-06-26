import os
import time
import json
from pprint import pprint
from types import SimpleNamespace
from typing import List, Optional, Any
from collections import defaultdict

from src.dataset import CheXpert2, CheXpert
from src.utils import get_accuracy, load_model, plot_sample, load_model_from_folder
from src.tree import get_chexpert_htree, lca_batch, collate_batch_data, lca
from src.crm import get_metrics
from src.definitions import ROOT_DIR, REVERSE_LABEL_MAP
from hxe import HXELoss
from hxe.tree import get_nx_chexpert_htree, LABEL_MAP

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader, Dataset
import torchvision.transforms as T
from torchmetrics.classification import MulticlassAUROC


def severity_run(args: SimpleNamespace, counter: int):
    trim = False
    if args.comp:
        trim = True

    ## Load the criterion
    if args.criterion not in ["default", "xentropy", "mse", "hxe", "bce"]:
        raise ValueError("Invalid Loss fn.")

    criterion = None
    if args.criterion == "default" or args.criterion == "xentropy":
        criterion = nn.CrossEntropyLoss()
    elif args.criterion == "hxe":
        htree_hxe = get_nx_chexpert_htree()
        criterion = HXELoss(
            htree=htree_hxe,
            num_classes=args.num_classes,
            ignore_nodes=["root", "L1"],
            label_map=LABEL_MAP,
            cuda=True,
        )
    else:
        criterion = nn.MSELoss()

    ## Load the Model
    start = time.perf_counter()
    model, best_auc = load_model(args)
    model = model.to(args.device)

    checkpoint1 = time.perf_counter()

    ## Model::Print Info
    print(f"\n\n\n{args.model} Loaded ...\nModel Loaded in: {checkpoint1 - start:.2f}s")

    ## Load The Dataset and Prepare the Dataloader
    t = T.Compose([T.Resize(size=(args.img_size, args.img_size)), T.ToTensor()])

    val_ds = CheXpert(args=args, split="valid", transform=t, trim=trim)
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=(not args.no_pin),
        num_workers=args.workers,
    )

    ## Load the hierarchy tree
    htree = get_chexpert_htree(save=False)

    ## Evaluate the Validation Set and Calculate AUCROC
    severity = []
    more_metrics = []
    batch_wise_median = []
    aurocs = []
    val_loss = 0
    val_acc = 0
    count = 0

    ## Run the experiment in Val mode
    with torch.no_grad():
        model.eval()
        val_start = time.perf_counter()
        for images, labels in val_dl:
            images = images.to(args.device)
            labels = labels.to(args.device)

            auroc_calc = MulticlassAUROC(14)
            logits = model(images)
            loss = criterion(logits, labels)
            auroc = auroc_calc(logits, labels)
            auroc = auroc.detach().cpu().item()
            aurocs.append(auroc)

            val_loss += loss.item()
            val_acc += get_accuracy(logits, labels, verbose=False)
            predicted = torch.argmax(logits, dim=1)
            xg = lca_batch(htree, predicted, labels, images)
            xs = xg["avg_lca"]
            batch_wise_median.append(xg["median"])
            severity.append(xs)

            ## Plot the values
            pname = f"{counter}_{args.model}"
            if not os.path.isdir(os.path.join(ROOT_DIR, args.save_dir, pname)):
                print(
                    "Making directory ... {}\n".format(
                        os.path.join(args.save_dir, pname)
                    )
                )
                os.mkdir(os.path.join(ROOT_DIR, args.save_dir, pname))

            if not args.no_paint:
                min_lca = xg["min_lca"]
                max_lca = xg["max_lca"]

                min_label = xg["min_label"]
                max_label = xg["max_label"]
                min_gt = xg["min_gt"]

                max_gt = xg["max_gt"]
                min_sample = xg["min_sample"]
                max_sample = xg["max_sample"]

                # print(type(min_sample))

                fname_min = os.path.join(args.save_dir, pname, f"{pname}_min.png")
                fname_max = os.path.join(args.save_dir, pname, f"{pname}_max.png")

                plot_sample(min_sample, fname_min, min_label, min_gt, args.comp)
                plot_sample(max_sample, fname_max, max_label, max_gt, args.comp)

                print(
                    f"Min. Severity for the batch: {min_lca} | Max Severity: {max_lca}"
                )

            metric = get_metrics(logits, labels, htree, args.crm, trim)
            fname = os.path.join(args.save_dir, pname, f"{pname}_")
            more_metrics.append(
                collate_batch_data(
                    htree, xg, predicted, labels, args.comp, plot=True, fname=fname
                )
            )

            print(
                f"\nRan Batch {count}:\n\tLoss: {val_loss}\n\tSeverity: {severity}\n\tCurrent AUROC: {auroc}"
            )
            count += 1

            print(f"CRM Based Metrics:\n{metric}")

        val_end = time.perf_counter()

    ## Collate the Important stats for result
    severity_stats = {
        "avg_severity": sum(severity) / len(severity),
        "min_severity_batch": min(severity),
        "max_severity_batch": max(severity),
        "%-severity": sum(severity) / (3 * len(severity)) * 100,
    }

    print(f"\nBATCH-COUNT: {count}\n")
    result = {
        f"run_no. {counter}": {
            "exp": args.exp,
            "model": args.model,
            "crm_used": args.crm,
            "val_loss": val_loss / len(val_ds),
            "val_acc": val_acc / len(val_dl),
            "val_time": f"{(val_end - val_start):.4f}s",
            "Best AUC": best_auc,
            "Current Best AUC": 1 - max(aurocs),
            "severity_stats": severity_stats,
            "crm_metrics": metric,
            "batch_wise_median": batch_wise_median,
            "model_median": sum(batch_wise_median) / len(batch_wise_median),
        }
    }

    pprint(f"Final Stats after the run: {result}", indent=4)
    print("\n\n\n")

    return result, more_metrics
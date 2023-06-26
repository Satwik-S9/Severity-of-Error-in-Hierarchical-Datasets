import os
import time
import json
from barbar import Bar
from pprint import pprint
from types import SimpleNamespace
from typing import Any, Optional
from collections import defaultdict
import numpy as np


from src.dataset import CheXpert2, CheXpert
from src.utils import plot_sample, load_model_from_folder, load_model_from_file
from src.tree import (
    get_chexpert_htree,
    lca_batch,
    collate_batch_data,
    lca,
    batch_wise_lca,
)
from src.crm import get_metrics, get_cost_matrix
from src.definitions import ROOT_DIR, REVERSE_LABEL_MAP
from hxe import HXELoss
from hxe.tree import get_nx_chexpert_htree, LABEL_MAP

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader, Dataset
import torchvision.transforms as T
from torchmetrics.classification import MulticlassAUROC, MulticlassF1Score


class Runner:
    def __init__(self, args: SimpleNamespace, counter: int) -> None:
        self.args: SimpleNamespace = args
        self.counter: int = counter

    def __warmup(self):
        start = time.perf_counter()
        # Load the models
        self.models, self.best_aucs = load_model_from_folder(self.args)

        # Load the Loss function
        self.criterion = None
        if self.args.criterion == "default" or self.args.criterion == "xentropy":
            self.criterion = nn.CrossEntropyLoss()

        elif self.args.criterion == "bce":
            self.criterion = nn.BCELoss()

        elif self.args.criterion == "hxe":
            htree_hxe = get_nx_chexpert_htree()
            self.criterion = HXELoss(
                htree=htree_hxe,
                num_classes=self.args.num_classes,
                ignore_nodes=["root", "L1"],
                label_map=LABEL_MAP,
                cuda=True,
            )
        else:
            self.criterion = nn.MSELoss()

        # Load the Hierarchy Tree
        self.htree = get_chexpert_htree(save=False)

        stop = time.perf_counter()
        print(f"Warmup done in ... {stop - start:.2f}s")

    def __load_dataloaders(self):
        ## Load The Dataset and Prepare the Dataloader
        t = T.Compose(
            [T.Resize(size=(self.args.img_size, self.args.img_size)), T.ToTensor()]
        )

        self.ds = CheXpert2(
            data_dir=self.args.data_dir, split="test", transform=t, policy="none"
        )
        self.dl = DataLoader(
            self.ds,
            batch_size=self.args.batch_size,
            shuffle=True,
            pin_memory=(not self.args.no_pin),
            num_workers=self.args.workers,
        )

    @classmethod
    def _save(cls, counter, args, result):
        print(f"Saving the results for run: {counter} ...\n")
        file_name = os.path.join(
            ROOT_DIR,
            args.save_dir,
            args.model,
            f"{args.exp}_{counter}_results_{args.model}.json",
        )
        with open(file_name, "a") as file:
            json_str = json.dumps(result, indent=4)
            json_str += "\n"
            file.write(json_str)

    @classmethod
    def _sev_run(
        cls, args, counter, model, criterion, dl, ds, best_auc, htree, idx: int = 0
    ):
        sv_pth = os.path.join(ROOT_DIR, args.save_dir, args.model)
        if not os.path.isdir(sv_pth):
            os.makedirs(sv_pth)

        aurocs = []
        more_metrics = []
        gts = []
        predicted = []
        val_loss = 0
        f1_score = 0

        model.eval()
        with torch.no_grad():
            for images, labels in Bar(dl):
                # transfer data to the device
                images = images.to(args.device)
                labels = labels.to(args.device)
                # Initialize AUC Calculator
                auroc_calc = MulticlassAUROC(args.num_classes)
                # f1_calc = MulticlassF1Score(args.num_classes)

                # Predict
                logits = model(images)
                if args.crm:
                    C = get_cost_matrix(htree)
                    logits = torch.tensor(
                        np.dot(logits.cpu().detach().numpy(), C), device=args.device
                    )

                preds = torch.argmax(logits, dim=1)

                # Loss and AUC
                loss = criterion(logits, labels)
                auroc = auroc_calc(logits, labels)
                auroc = auroc.detach().cpu().item()
                aurocs.append(auroc)
                # f1_score += f1_calc(logits, labels).cpu().item()

                # Val Loss and Predicted Values
                val_loss += loss.item()

                gts.append(labels.cpu())
                predicted.append(preds.cpu())

                pname = f"{counter}_{args.model}"
                # metric = get_metrics(logits, labels, htree, args.crm, False)
                fname = os.path.join(
                    ROOT_DIR, args.save_dir, args.model, pname, f"{pname}_{idx}_"
                )
            print("\nCalculating Node wise and Model Severity Stats ...\n")
            node_wise_lca = batch_wise_lca(predicted, gts, htree)
            severity_model = sum(list(node_wise_lca.values())) / len(
                list(node_wise_lca.values())
            )

        print(f"Severity Stats Calculated for run: {counter}...\n")

        result = {
            "exp": args.exp,
            "model": args.model,
            "crm_used": args.crm,
            "val_loss": val_loss / len(ds),
            "Best AUC": best_auc,
            "auroc": sum(aurocs) / len(aurocs),
            "model_severity": severity_model,
            # "crm_metrics": metric,
            "label_wise_severity": dict(node_wise_lca),
        }

        pprint(result)
        # cls._save(counter, args, result)
        return result, more_metrics

    def run(self):
        # Load The models and dataloaders
        self.__warmup()
        self.__load_dataloaders()

        results: defaultdict[Any, dict] = defaultdict(dict)
        extras: list = []
        for (idx, model), (_, best_auc) in zip(
            enumerate(self.models), enumerate(self.best_aucs)
        ):
            print(f"Calculating for Model#{idx}...\n")

            r, e = self._sev_run(
                args=self.args,
                counter=self.counter,
                model=model.to(self.args.device),
                criterion=self.criterion,
                dl=self.dl,
                ds=self.ds,
                best_auc=best_auc,
                htree=self.htree,
                idx=idx,
            )

            results["main"][f"run_{self.counter}_{idx}"] = r
            extras.append(e)

        self._save(self.counter, self.args, results)

        return results, extras


class RunnerOld:
    def __init__(self, args: SimpleNamespace, counter: int):
        self.args: SimpleNamespace = args
        self.counter: int = counter

    def __warmup(self):
        start = time.perf_counter()
        # Load the models
        self.models, self.best_aucs = load_model_from_folder(self.args)

        # Load the Loss function
        self.criterion = None
        if self.args.criterion == "default" or self.args.criterion == "xentropy":
            self.criterion = nn.CrossEntropyLoss()

        elif self.args.criterion == "bce":
            self.criterion = nn.BCELoss()

        elif self.args.criterion == "hxe":
            htree_hxe = get_nx_chexpert_htree()
            self.criterion = HXELoss(
                htree=htree_hxe,
                num_classes=self.args.num_classes,
                ignore_nodes=["root", "L1"],
                label_map=LABEL_MAP,
                cuda=True,
            )
        else:
            self.criterion = nn.MSELoss()

        self.htree = get_chexpert_htree(save=False)

        stop = time.perf_counter()
        print(f"Warmup done in ... {stop - start:.2f}s")

    def __load_dataloaders(self):
        ## Load The Dataset and Prepare the Dataloader
        t = T.Compose(
            [T.Resize(size=(self.args.img_size, self.args.img_size)), T.ToTensor()]
        )

        self.ds = CheXpert2(
            data_dir=self.args.data_dir, split="test", transform=t, policy="none"
        )
        self.dl = DataLoader(
            self.ds,
            batch_size=self.args.batch_size,
            shuffle=True,
            pin_memory=(not self.args.no_pin),
            num_workers=self.args.workers,
        )

    @staticmethod
    def _plot(pname, args, xg, idx):
        # Plot the values
        path = os.path.join(ROOT_DIR, args.save_dir, args.model, pname)
        if not os.path.isdir(path):
            print("Making directory ... {}\n".format(path))
            os.makedirs(os.path.join(path))

        min_lca = xg["min_lca"]
        max_lca = xg["max_lca"]

        min_label = xg["min_label"]
        max_label = xg["max_label"]
        min_gt = xg["min_gt"]

        max_gt = xg["max_gt"]
        min_sample = xg["min_sample"]
        max_sample = xg["max_sample"]

        fname_min = os.path.join(path, f"{pname}_{idx}_min.png")
        fname_max = os.path.join(path, f"{pname}_{idx}_max.png")

        plot_sample(min_sample, fname_min, min_label, min_gt, args.comp)
        plot_sample(max_sample, fname_max, max_label, max_gt, args.comp)

        print(f"Min. Severity for the batch: {min_lca} | Max Severity: {max_lca}")

    @classmethod
    def _sev_run(
        cls, args, counter, model, criterion, dl, ds, best_auc, htree, idx: int = 0
    ):
        sv_pth = os.path.join(ROOT_DIR, args.save_dir, args.model)
        if not os.path.isdir(sv_pth):
            os.makedirs(sv_pth)

        severity = []
        batch_wise_median = []
        aurocs = []
        more_metrics = []
        val_loss = 0

        model.eval()
        with torch.no_grad():
            for images, labels in dl:
                # transfer data to the device
                images = images.to(args.device)
                labels = labels.to(args.device)
                # Initialize AUC Calculator
                auroc_calc = MulticlassAUROC(args.num_classes)

                # Predict
                logits = model(images)

                # Loss and AUC
                loss = criterion(logits, labels)
                auroc = auroc_calc(logits, labels)
                auroc = auroc.detach().cpu().item()
                aurocs.append(auroc)

                # Val Loss and Predicted Values
                val_loss += loss.item()

                predicted = torch.argmax(logits, dim=1)
                xg = lca_batch(htree, predicted, labels, images)
                xs = xg["avg_lca"]
                batch_wise_median.append(xg["median"])
                severity.append(xs)

                pname = f"{counter}_{args.model}"
                if not args.no_paint:
                    cls._plot(pname, args, xg, idx)

                metric = get_metrics(logits, labels, htree, args.crm, False)
                fname = os.path.join(
                    ROOT_DIR, args.save_dir, args.model, pname, f"{pname}_{idx}_"
                )

                print("\n\nCollating\n\n")
                more_metrics.append(
                    collate_batch_data(
                        htree, xg, predicted, labels, args.comp, plot=True, fname=fname
                    )
                )

        print(f"Severity Stats Calculated for run: {counter}...\n")
        severity_stats = {
            "avg_severity": sum(severity) / len(severity),
            "min_severity_batch": min(severity),
            "max_severity_batch": max(severity),
            "%-severity": sum(severity) / (3 * len(severity)) * 100,
        }

        result = {
            "exp": args.exp,
            "model": args.model,
            "crm_used": args.crm,
            "val_loss": val_loss / len(ds),
            "Best AUC": best_auc,
            "aurocs": aurocs,
            "severity_stats": severity_stats,
            "crm_metrics": metric,
            "batch_wise_median": batch_wise_median,
            "model_median": sum(batch_wise_median) / len(batch_wise_median),
        }

        # cls._save(counter, args, result)
        return result, more_metrics

    @staticmethod
    def _plot_label_wise_sev_stats(labels, ground_truths, htree):
        node_lca_info = {REVERSE_LABEL_MAP[i]: [] for i in range(14)}
        for l, g in zip(labels, ground_truths):
            # print(idx, l, g)
            l1 = REVERSE_LABEL_MAP[l.item()]
            g1 = REVERSE_LABEL_MAP[g.item()]

            lca_val = lca(htree, l1, g1)
            node_lca_info[g1].append(lca_val)

            print(node_lca_info)

    @classmethod
    def _save(cls, counter, args, result):
        print(f"Saving the results for run: {counter} ...\n")
        file_name = os.path.join(
            ROOT_DIR,
            args.save_dir,
            args.model,
            f"{args.exp}_{counter}_results_{args.model}.json",
        )
        with open(file_name, "a") as file:
            json_str = json.dumps(result, indent=4)
            json_str += "\n"
            file.write(json_str)

    def run(self):
        # Load The models and dataloaders
        self.__warmup()
        self.__load_dataloaders()

        results: defaultdict[Any, dict] = defaultdict(dict)
        extras: list = []
        for (idx, model), (_, best_auc) in zip(
            enumerate(self.models), enumerate(self.best_aucs)
        ):
            print(f"Calculating for Model#{idx}...\n")

            r, e = self._sev_run(
                args=self.args,
                counter=self.counter,
                model=model.to(self.args.device),
                criterion=self.criterion,
                dl=self.dl,
                ds=self.ds,
                best_auc=best_auc,
                htree=self.htree,
                idx=idx,
            )

            results["main"][f"run_{self.counter}_{idx}"] = r
            extras.append(e)

        self._save(self.counter, self.args, results)

        return results, extras

import os
from datetime import datetime
import json
import statistics
from scipy.stats import ttest_ind
from typing import Optional, List
from collections import defaultdict
from .definitions import ROOT_DIR


def _load_json_files(model: str, path: str) -> List[str]:
    tmp_path: str = os.path.join(ROOT_DIR, path, model)
    json_names: List[str] = [
        json_file for json_file in os.listdir(tmp_path) if json_file.endswith(".json")
    ]
    return sorted(json_names)


def load_data(model: str, path: str, mode: str):
    if mode not in ["crm2", "crm", "nocrm"]:
        raise ValueError("Invalid mode choose from {}".format(["crm2", "crm", "nocrm"]))

    tmp_path: str = os.path.join(ROOT_DIR, path, model)
    json_name = filter(lambda x: x.startswith(mode), _load_json_files(model, path))
    json_path: str = os.path.join(tmp_path, next(json_name))

    with open(json_path, "rb") as file:
        obj = json.load(file)
        obj = obj["main"]

    for key in obj.keys():
        yield obj[key]["model_severity"]


def t_test_all(path: str, save_dir: str = "results2"):
    models = os.listdir(os.path.join(ROOT_DIR, path))
    results = defaultdict(dict)
    for model in models:
        data_crm = list(load_data(model, path, "crm"))
        data_nocrm = list(load_data(model, path, "nocrm"))
        v = ttest_ind(data_crm, data_nocrm)
        
        results["main"][f"{model}"] = {"p-value": v.pvalue, "statistic": v.statistic}
        
        print("For model: {}\n\tpvalue: {}\n\tstatistic:{}\n\n".format(
            model, v.pvalue, v.statistic
        ))
        
    uid = datetime.now().strftime('%Y%m%d%H%M%S')
    with open(os.path.join(ROOT_DIR, save_dir, f"t_test_results_{uid}.json"), "w") as file:
        json.dump(results, file, indent=4)
        
    return results
    
    
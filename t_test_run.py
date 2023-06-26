import pickle
from scipy.stats import ttest_ind
from collections import defaultdict
from typing import DefaultDict


RESULTS_PATH_CRM = "/workspace/utsa/chexpert_results/MTP/MultiLabelSeverity/densenet121_crm_results.pkl"
RESULTS_PATH = "/workspace/utsa/chexpert_results/MTP/MultiLabelSeverity/densenet121_results.pkl"

def load_pkl(path: str):
    with open(path, "rb") as file:
        results = pickle.load(file) 
    return results
    
def save_pkl(results: defaultdict, path: str):
    with open(path, "wb") as file:
        pickle.dump(results, file)

# T_TEST_ALL = defaultdict(dict)
def main():
    dcrm = load_pkl(RESULTS_PATH_CRM)
    dnocrm = load_pkl(RESULTS_PATH)
    
    # for model in dcrm.keys():
        # data_crm = []
        # data_no_crm = []
        # for l1, l2 in zip(dcrm[model].values(), dnocrm[model].values()):
        #     data_crm += l1
        #     data_no_crm += l2
        
    v = ttest_ind(dcrm, dnocrm)
    T_TEST_ALL = {"pvalue": v.pvalue, "statistic": v.statistic}
    print(f"pvalue: {v.pvalue}, statistic: {v.statistic}")
        
    return T_TEST_ALL


if __name__ == '__main__':
    results = main()
    save_pkl(results, "/workspace/utsa/chexpert_results/t_test_ml.pkl")
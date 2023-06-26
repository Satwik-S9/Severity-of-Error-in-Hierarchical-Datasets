import os

# * ROOT WD
ROOT_DIR: str = os.getcwd()
if ROOT_DIR[-3:] == "src":
    ROOT_DIR = os.path.normpath(os.getcwd() + os.sep + os.pardir)
print(f"Using ROOT-DIR: {ROOT_DIR}")


# * GLOBAL MAPS
LABEL_MAP = {
    "No Finding": 0,
    "Enlarged Cardiomediastinum": 1,
    "Support Devices": 2,
    "Fracture": 3,
    "Lung Opacity": 4,
    "Cardiomegaly": 5,
    "Pleural Effusion": 6,
    "Pleural Other": 7,
    "Pneumothorax": 8,
    "Edema": 9,
    "Consolidation": 10,
    "Pneumonia": 11,
    "Lung Lesion": 12,
    "Atelectasis": 13,
}

REVERSE_LABEL_MAP = dict((value, key) for key, value in LABEL_MAP.items())

TREE_MAP = {
    "root": 2,
    "L1": 1,
    "No Finding": 1,
    "Enlarged Cardiomediastinum": 1,
    "Support Devices": 1,
    "Fracture": 1,
    "Lung Opacity": 1,
    "Cardiomegaly": 0,
    "Pleural Effusion": 0,
    "Pleural Other": 0,
    "Pneumothorax": 0,
    "Edema": 0,
    "Consolidation": 0,
    "Pneumonia": 0,
    "Lung Lesion": 0,
    "Atelectasis": 0,
}

COMP_LABEL_MAP = {
    "Atelectasis": 0,
    "Cardiomegaly": 1,
    "Consolidation": 2,
    "Edema": 3,
    "Pleural Effusion": 4,
}

REV_COMP_LABEL_MAP = {
    0: "Atelectasis",
    1: "Cardiomegaly",
    2: "Consolidation",
    3: "Edema",
    4: "Pleural Effusion",
}

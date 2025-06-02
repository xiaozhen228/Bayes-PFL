import sys
import shutil
import os
from datasets.VisA import Visa_dataset
from datasets.MVTec import Mvtec_dataset
from datasets.BTAD import BTAD_dataset
from datasets.KSDD2 import KSDD2_dataset
from datasets.RSDD import RSDD_dataset
from datasets.DAGM import DAGM_dataset
from datasets.DTD import DTD_dataset
from datasets.HeadCT import HeadCT_dataset
from datasets.BrainMRI import BrainMRI_dataset
from datasets.Br35H import Br35H_dataset
from datasets.ISIC import ISIC_dataset
from datasets.ClinicDB import ClinicDB_dataset
from datasets.ColonDB import ColonDB_dataset
from datasets.Endo import Endo_dataset
from datasets.Kvasir import Kvasir_dataset






def move(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def process_dataset(dataset_cls, src_root, des_root, id_start=0, binary=True, to_255=True):

    move(des_root)
    dataset = dataset_cls(src_root)
    return dataset.make_VAND(binary=binary, to_255=to_255, des_path_root=des_root, id=id_start)


if __name__ == "__main__":
    id_counter = 0


    datasets_config = [
        {
            "name": "visa",
            "class": Visa_dataset,
            "src": "Path to your root/visa",
            "des": "./dataset/mvisa/data/visa"
        },
        {
            "name": "mvtec",
            "class": Mvtec_dataset,
            "src": "Path to your root/MVTec",
            "des": "./dataset/mvisa/data/mvtec"
        },
        {
            "name": "BTAD",
            "class": BTAD_dataset,
            "src": "Path to your root/BTech_Dataset_transformed",
            "des": "./dataset/mvisa/data/BTAD"
        },
        {
            "name": "KSDD2",
            "class": KSDD2_dataset,
            "src": "Path to your root/KSDD2",
            "des": "./dataset/mvisa/data/KSDD2"
        },
        {
            "name": "RSDD",
            "class": RSDD_dataset,
            "src": "Path to your root/RSDD",
            "des": "./dataset/mvisa/data/RSDD"
        },
        {
            "name": "DAGM",
            "class": DAGM_dataset,
            "src": "Path to your root/DAGM2007",
            "des": "./dataset/mvisa/data/DAGM"
        },
        {
            "name": "DTD",
            "class": DTD_dataset,
            "src": "Path to your root/DTDSynthetic",  # From AdaCLIP
            "des": "./dataset/mvisa/data/DTD"
        },
        {
            "name": "HeadCT",
            "class": HeadCT_dataset,
            "src": "Path to your root/HeadCT_anomaly_detection",   # From AdaCLIP
            "des": "./dataset/mvisa/data/HeadCT"
        },
        {
            "name": "BrainMRI",
            "class": BrainMRI_dataset,
            "src": "Path to your root/BrainMRI",   # From AdaCLIP
            "des": "./dataset/mvisa/data/BrainMRI"
        },
        {
            "name": "Br35H",
            "class": Br35H_dataset,
            "src": "Path to your root/Br35h_anomaly_detection",   # From AdaCLIP
            "des": "./dataset/mvisa/data/Br35H"
        },
        {
            "name": "ISIC",
            "class": ISIC_dataset,
            "src": "Path to your root/ISIC",   # From AdaCLIP
            "des": "./dataset/mvisa/data/ISIC"
        },
        {
            "name": "ClinicDB",
            "class": ClinicDB_dataset,
            "src": "Path to your root/ClinicDB",   # From AdaCLIP
            "des": "./dataset/mvisa/data/ClinicDB"
        },
        {
            "name": "ColonDB",
            "class": ColonDB_dataset,
            "src": "Path to your root/ColonDB",   # From AdaCLIP
            "des": "./dataset/mvisa/data/ColonDB"
        },
        {
            "name": "Endo",
            "class": Endo_dataset,
            "src": "Path to your root/Endo",   # From AnomalyCLIP
            "des": "./dataset/mvisa/data/Endo"
        },
        {
            "name": "Kvasir",
            "class": Kvasir_dataset,
            "src": "Path to your root/Kvasir",   # From AnomalyCLIP
            "des": "./dataset/mvisa/data/Kvasir"
        },
    ]

    for config in datasets_config:
        print(f"Processing {config['name']}...")
        id_counter = process_dataset(
            dataset_cls=config["class"],
            src_root=config["src"],
            des_root=config["des"],
            id_start= 0
        )
        print(f"Finished {config['name']}, next ID: {id_counter}")

    
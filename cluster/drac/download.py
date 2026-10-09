"""Download the raw datasets the given recipe configurations use, without preprocessing.

    python cluster/drac/download.py bnci2014001_eegnet lee2019mi_eegnet ...
"""

import sys
from pathlib import Path

import moabb.datasets
import yaml

HPARAMS = Path(__file__).resolve().parents[2] / "recipes" / "moabb" / "hparams"

datasets = {}
for name in sys.argv[1:]:
    data = yaml.safe_load((HPARAMS / f"{name}.yaml").read_text())["data"]
    key = (data["dataset"], repr(data.get("dataset_kwargs")))
    datasets.setdefault(key, (data, set()))[1].update(data["subjects"] or [None])
for (dataset, _), (data, subjects) in datasets.items():
    instance = getattr(moabb.datasets, dataset)(**(data.get("dataset_kwargs") or {}))
    subject_list = instance.subject_list if None in subjects else sorted(subjects)
    print(f"{dataset}: {len(subject_list)} subjects", flush=True)
    instance.download(subject_list=subject_list)

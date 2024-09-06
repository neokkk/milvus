import os
import sys
sys.path.append(os.path.dirname(__file__))
print(sys.path)

from datasets import DATASETS, download, get_dataset_fn

if __name__ == "__main__":
    for dataset in DATASETS.keys():
        hdf5_filename = get_dataset_fn(dataset)
        dataset_url = f"https://ann-benchmarks.com/{dataset}.hdf5"
        download(dataset_url, hdf5_filename)

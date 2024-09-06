from datasets import DATASETS, get_dataset

if __name__ == "__main__":
    for dataset in enumerate(DATASETS):
        print(dataset)
        get_dataset(dataset)

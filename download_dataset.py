import os
from kaggle.api.kaggle_api_extended import KaggleApi

if __name__ == '__main__':
    dataset = 'ahmedhamada0/brain-tumor-detection'

    dataset_dir = './dataset'
    os.makedirs(dataset_dir, exist_ok=True)

    if not os.listdir(dataset_dir):
        api = KaggleApi()
        api.authenticate()

        # Download dataset from Kaggle
        api.dataset_download_files(dataset, path=dataset_dir, unzip=True)

        print(f'Dataset {dataset} scaricato e decompresso in {dataset_dir}')
    else:
        print(f'Il dataset {dataset} è già presente nella directory {dataset_dir}.')
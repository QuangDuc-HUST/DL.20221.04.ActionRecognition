#
# File for build the dataset
# 
import os
import argparse

from utils import runcmd


def get_dataset_arg():

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_folder', type=str, default='./data/')
    parser.add_argument('--kaggle_config', type=str, default='./script/kaggle_config.sh')
    parser.add_argument('--dataset', required=True, type=str, choices=['ucf101', 'hmdb51'])
    parser.add_argument('--process_type', type=str, default='5_frames_uniform', choices=['5_frames_uniform', '5_frames_conse_rand', '16_frames_conse_rand'] )
    
    return parser.parse_args()

def get_data_id(dataset, process_type):

    DATASET_URLS = {
    "ucf101_5_frames_uniform": "17DHqu9-ySzg0nh55QCGqT6gFd38UZDPR",
    "ucf101_16_frames_conse_rand": "1-6zxq5eV0NFLimoasGzztRT35q3LtepF",
    "hmdb51_5_frames_uniform": "1jBV4ZFuWDxUgPezXdoLiX15ZyICnVW9X",
    "hmdb51_16_frames_conse_rand": "1-AzbJDJ3KQdnnXSXLB8AuDisds4a5O_J",
    }

    return DATASET_URLS[dataset + '_' + process_type]


if __name__ == '__main__':
    args = get_dataset_arg()

    # Create dataset folder if not exists
    
    final_data_folder = os.path.join(args.data_folder, 
                                     args.dataset.upper(), 
                                     args.process_type)

    if not os.path.exists(final_data_folder):
        os.makedirs(final_data_folder)

    print("-" * 20)
    # Kaggle setting
    runcmd(f'sh {args.kaggle_config}')

    # Download file
    print("Downloading the dataset...")

    # Create temp folder for zip file
    if not os.path.exists('./temp/'):
        os.makedirs('./temp/')

    zip_file_name = args.dataset + '_processed.zip'

    download_id = get_data_id(args.dataset, args.process_type)


    # runcmd(f'wget {download_url} -P "./temp/"', is_wait=True)

    runcmd(f'sh script/googledown.sh {download_id} "./temp/{zip_file_name}"', is_wait=True)
    
    print("Unzip the dataset...")


    runcmd(f'unzip -qo ./temp/{zip_file_name} -d {final_data_folder} \
            && rm -rf ./temp/{zip_file_name} \
            && mv {final_data_folder}/kaggle/temp/*/* {final_data_folder}/ \
            && rm -rf {final_data_folder}/kaggle', 
            is_wait=True)

    print("--DONE--")
    print("-" * 20)

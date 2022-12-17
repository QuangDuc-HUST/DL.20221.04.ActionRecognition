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
    parser.add_argument('--kaggle', action='store_true')
    parser.add_argument('--google', action="store_true")
    parser.add_argument('--google_access_key', type=str, default='')
    parser.add_argument('--dataset', required=True, type=str, choices=['ucf101', 'hmdb51'])
    parser.add_argument('--process_type', type=str, default='5_frames_uniform', choices=['5_frames_uniform', '5_frames_conse_rand', '16_frames_conse_rand'] )
    
    return parser.parse_args()



def get_data_google_id(dataset, process_type):

    DATASET_IDS = {
    "ucf101_5_frames_uniform": "17DHqu9-ySzg0nh55QCGqT6gFd38UZDPR",
    "ucf101_16_frames_conse_rand": "1-6zxq5eV0NFLimoasGzztRT35q3LtepF",
    "hmdb51_5_frames_uniform": "1jBV4ZFuWDxUgPezXdoLiX15ZyICnVW9X",
    "hmdb51_16_frames_conse_rand": "1-AzbJDJ3KQdnnXSXLB8AuDisds4a5O_J",
    }

    return DATASET_IDS[dataset + '_' + process_type]


def get_data_kaggle_id(dataset, process_type):

    DATASET_IDS = {
    "ucf101_5_frames_uniform": "17DHqu9-ySzg0nh55QCGqT6gFd38UZDPR",
    "ucf101_16_frames_conse_rand": "1-6zxq5eV0NFLimoasGzztRT35q3LtepF",
    "hmdb51_5_frames_uniform": "quangduc0703/hmdb51-5-uniform",
    "hmdb51_16_frames_conse_rand": "1-AzbJDJ3KQdnnXSXLB8AuDisds4a5O_J",
    }

    return DATASET_IDS[dataset + '_' + process_type]

if __name__ == '__main__':
    args = get_dataset_arg()

    if (args.google and args.kaggle) or (not args.google and not args.google):
        raise("Please choose one google or kaggle for download.")

    if args.google and args.google_access_key == '':
        raise("Input google access key.")


    # Create dataset folder if not exists
    final_data_folder = os.path.join(args.data_folder, 
                                     args.dataset.upper(), 
                                     args.process_type)

    if not os.path.exists(final_data_folder):
        os.makedirs(final_data_folder)

    print("-" * 20)
    # Kaggle setting
    if args.kaggle:
        runcmd(f'sh {args.kaggle_config}')

    # Download file
    print("Downloading the dataset...")

    # Create temp folder for zip file
    if not os.path.exists('./temp/'):
        os.makedirs('./temp/')

    zip_file_name = args.dataset + '_processed.zip'

    if args.google:
        download_id = get_data_google_id(args.dataset, args.process_type)
        runcmd(f'sh script/googledown.sh {download_id} "./temp/{zip_file_name}" {args.google_access_key}', is_wait=True)

    elif args.kaggle:
        download_id = get_data_kaggle_id(args.dataset, args.process_type)
        runcmd(f'kaggle datasets download -d {download_id} -p "./temp/"', is_wait=True)


    
    print("Unzip the dataset...")

    if not os.path.exists(os.path.join('./temp', zip_file_name)):
        zip_file_name = os.listdir('./temp')[0]


    runcmd(f'unzip -qo ./temp/{zip_file_name} -d {final_data_folder} \
            && rm -rf ./temp/{zip_file_name} \
            && mv {final_data_folder}/kaggle/temp/*/* {final_data_folder}/ \
            && rm -rf {final_data_folder}/kaggle', 
            is_wait=True)

    print("--DONE--")
    print("-" * 20)

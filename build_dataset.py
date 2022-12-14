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
    parser.add_argument('--dataset', required = True, type=str, choices=['ucf101', 'hmdb51'])
    parser.add_argument('--process_type', type=str, default='5_frames_uniform' )
    
    return parser.parse_args()

def get_data_url(dataset, process_type):

    DATASET_URLS = {
    "ucf101_5_frames_uniform": "https://www.kaggleusercontent.com/kf/113750491/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..sVUWgz6PD-iOFgBLORnxxQ.6xJ1TotkB85GzaNtzHf9Qbkkf2Va04Eyq-MESgW_ej9vFYUUi7NhB0He3Ts54j7z5HyG-2TlnrbOTWk2PkwvH4aFVzMy4UTsizIjy8jglK9-EtCaydcwO3f7AN0StoCMMqi8PL8KwD_OOdNTnzBEuyDVoGHPfYjI-sdMMdpW5mAHXyuyhn-h6QBUUZy-Zl4SMK5S90KqQ0ycx3RGEmRYWxR03Iz-WHXVDwjvuugOsx_ZdnNKD6OXGfEXUSNInBFy8KEBHQJXm7ilOVEnjAtFDIzQHhIAcK_lpEhKxJEACrFRMg4GYEEShl2WysNkpk0WmBxsash11JkLGROE4XOPTljUtg-X-gW85Ko5fdQWlRo2dFM45aFBHfJSBN2ELcQ8c1sdnPwuek1KT5TFKpCe3mBKMIt12mGps_cStXdZu_h7vloGXR22lw-WRpXoA3QtEIkohXz3ifKJyKfy243GfdI65ZovoFVH7uHMQrvM-GW2knhL_u7PdHyaky5jfqsZNQmRPeM6nF6_yYMXx9TAAYW8dkp5V7pRwtFB44xqjduLsM-dBse6YabfTmkQE2RkhKTz9RKBi9cIC9S9ltAC90uRDctKAYEK87DKBZW-VptkBNoqKd_n1yucgaJ2iWkgLNhJvA3qBEr_Z5BT2Ttc911sk5Vf-VagxtAMoWesBoMAkozgWCwh0H-r-WXvhnPu.7w-Ox8VE2awMs-OyFAPAzw/ucf101_processed.zip",
    "hmdb51_5_frames_uniform": "https://www.kaggleusercontent.com/kf/113750535/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..MDe5KA-drb4yMM_nSIBTgA.KfNjLEsTCjW8eCZHioObk-TSwor_j3YmErvt0U2rHSKWr9oFa9h2aNYjccpeB6nqrUGIGOv3mxpPYMEMKG1rpgMCOhVXqKmFZzFCYlIe8K0XkumTTTzEddrBieQBvnlhuIRc-Yrx6pMLwVmPQeFxTQlz0uuQfndjIzWfqI4y2Q8ln06kn4sOs4qiTBd_U79rFXTY7naU4UX1VrAT56MqM6ZYj0TMEaiNrwCR1eCJz1MnZEA4w2eIs4_nnhsCv3eObVuluGBTSA4A1oqX-bwTsBIwXgHSFmsXHQU8xYfI96lJsamGUvM1jfHxdvln7bLpvnpbTSuJWWOZuYIiPhWcf7I41c_44cQ2NpVZOewQvl0aCBEW7O3AAFzJvWbV2r-dOcqfvj8JnYsEFsXXdZPgIQ6cYI6HLu4-Eyjm4w9n0-ZPlGlo8XQJzNzC5atgJTVFt2mT9c9j1dbYIgB74wPIBZw0QOCkM_N0vMqTvhLFNCRS9jLIVHgnnvHoxue6RP5AM9l5K6lGYeH0-veSifg8j-wbeU_xty1Em3MSsxrsS8wZe86RxfKM7dJ45MX4RXmql85fGMQuL5a3qXWawXM2B_cGvyXl1HBB9PReYY4ifx2KmuPES19ezCd_kMifm-X5MHei730IkgBe6pEnZcKhJtf74GCjZThZaypYMQj38N4NVZdaLPISmR4N8KkD2BA3.akfjbHPAI6_VONt7d5Fcnw/hmdb51_processed.zip",   
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

    download_url = get_data_url(args.dataset, args.process_type)
    runcmd(f'wget {download_url} -P "./temp/"', is_wait=True)

    print("Unzip the dataset...")

    zip_file_name = args.dataset + '_processed.zip'

    runcmd(f'unzip -qo ./temp/{zip_file_name} -d {final_data_folder} \
            && rm -rf ./temp/{zip_file_name} \
            && mv {final_data_folder}/kaggle/temp/*/* {final_data_folder}/ \
            && rm -rf {final_data_folder}/kaggle', 
            is_wait=True)

    print("--DONE--")
    print("-" * 20)
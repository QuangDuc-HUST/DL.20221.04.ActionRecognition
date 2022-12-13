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
    "hmdb51_5_frames_uniform": "https://www.kaggleusercontent.com/kf/113750535/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..1svNMkONRLLEaw-7cFkAzw.LptlltPfIRvwyn-rcK-g0q2F-pOFIXKl6IuiBaNZ3BXv1E9nTwrFq5JdR6EamaoGsjTU1phaw3lDxpk6UMztRggR2ftAK51C1rNIH8NsKJjr-2DefdhJhriIwTcuansye9r7mAUE59MHaDJuJvAPkfrgx0zKSgvdAhC-LGWGDk-ro6PlV4-1JxulGO1xd6aWJxdtqi62HkigkVP8KDCA1YKzkfhteRSaW6rYpjHI7CDE1j9jTg80_UQyXYQAE89p8wucC5j-h9ynNDu6m1E7y_Ca8kjkImQD3mAmzDIPMy9W2c-jvkjIghT4BNFhmAemh6qMzR2sY1vPrnlxOe96D1ERmczl17TpfYHAZgrhUI6_rbbmChVBC-elLm9jkXfLyJ9zsc1TDnOX0ODfgCH8sS_4azyOwS2XZh3hV3Z3I-p8yESLJaH_GYKapALyYzzOJBYllrRfAsOv5Gm9aiMMrEZkB3u6MRSaxLl8msStxm_FEecDDEFEi5nNwlY6WPtWXYJkx_CNgGGMQt6o5hu--KyoxH3PZVwNg-YT5lWEUu4p1-nDMek4F1UsxXjG5aH4pd9Z61LbSoCOEyVCGy0jBZvxOSt8YE3W1MvRntxGTPkvJBwiUsKJLCIUasynoSlTOYoyqFAkHWbOFvK07aQmk_T4dWdC6p8sAyQa5a-G1TvXGeab_YO96iJ3rmhPTw-q.D6kEnKztk0kSR5Fzn0l4fQ/hmdb51_processed.zip",   
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
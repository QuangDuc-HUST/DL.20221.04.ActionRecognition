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
    "ucf101_5_frames_uniform": "https://www.kaggleusercontent.com/kf/113725945/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..AXcTEd81ToCt1f3JPP_EVA.EAm91AnyqPKOl4UMWTAQdjOo6c7OiXuqO2Unl3wnBFjbXpg_-Sqd6A-bNsdBlLabsmqjfj1UgkgVHpqcWNUCvV5ZxMQH1z1Vii11rwesabfnX7V56lcw5SCbo8JSUuZt2r23JBMSqiId6UxOMK6MfJaZAZJpkQ6Fe16E5ZwSGGCZhhoDkMa7hMaB_PDGoPsQg2gpDWQFgaYXJINffExJjlA-cDmP9zgGDHncDq8YdcrRyaB-RNQQLNGjIXv1MZA71zTwrSk_5fEXjYVzgW1hHb82ufzL34cy_nLw5ZpktxGBedobfx8nVZQsGOT52o1LMLmP9C_0QUjxewvemLxGYZlTd_qaPxtviBT-7oZpJvXYT4DzkMOQsuCEbOMII4tDhrnnLCSDFEvEBEhxl0V35ViQfNoi5RdT9YQwzbG1YYazBqIvOx3G6gr_PpPYAxznZmWbP6UJAP6d1AAxGCKweYSvw_UGss8YuoN5TlXF3pzcfg_K2-Z3-ZAwGRJXhSOMMjYnENrT2lslzYrr-hlOEGuZTqaod2hhmdHhUqL5bUClKYOShDgy1kZFZ61yBaTko3nrZRdP2C0UgrOStIMQn4Wzl2XOqPce6AYRpK4CDPhYAdP-V6mDjKwEId-bJdXDzannY9e6BdHXYKlR_Q9lDpBCmnNYFYm1r9L2smlCk7HQG92y-X-xUsXGfbsLN-0r.eyNHJXiBa9Zh0KtI8GiDIg/ucf101_processed.zip",
    "ucf101_10_frames_uniform": "https://www.kaggleusercontent.com/kf/113724047/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..arn681V48S_REmkgsm264w.DO1W1kdogrg3wuSmz6Cx6GXeAr95Fi3gSQAZ2vz60svirIhnUeVW0_mn1zlv2OTotsep199WRrhwEzQVXkg4ut-YRDlz11JnUqeKirKbhZTZd3B6eJKEmkdcEx5XEYhdqUX7foNSzqci4tYoEKAH2Jn-tVK0l8RLakAPGqFULnd2Uq_efANMHHveM0zCh5fUC7Zy_uhkwrOyzw9Kjq3qJEc1XwcHAtZWqQSFcuYIHLJvBSC6SWwc2ugGdxRShNbGaNV9y3gxs0h3L-zfuLP1Anpr4Kw2y5TrgtF5rhaP-BA_vPlIve1xqaIgRCRcbABtbR3FLcxUbu44bnLyweoUkYMvpdZGC9tBH-5DeAQOx7OlIZyGjMDK_vct6FCuCudddd7xK0nWpAiWIBsHqg-kVyk3PyuaQ36IIrf7WiwXbQbHltjUc4KlxOTdyqal8j3zYp5sNNibs4Yffifdv6-AGKXv_IEk4RG4owXm1Hc3Fxs3xCdjkdALuF3nniplqt4VRpK7vnG8_8mthu0t5ttHCUuqvBrkwkIQA5OwP2SlyM2l1uvI0e520RaJNuBj3HjsB4TDhf5NPRLFlGYS3oB2K_kbBZ9Bdd4vPtbQLQvnoPpH1TQdtowjBgPjfFzh53QJmelRMp6MLGEYbes44slvy-fMk-NGRk1oNR3knhf4ww8V8xuws1HLc2TK-KSIJ2zy.DgMK8AF3E5vk8nUGbBlBTw/ucf101_processed.zip",
    "hmdb51_5_frames_uniform": "https://www.kaggleusercontent.com/kf/113731246/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..nRMOifFZVrvO9x33AM-iRw.R2iFcTv4_-_fmPNlztLkQZ1WFtLm0Xp7Wbl9GvdIEl2KbFEaPoyWvDRL8zhCkpcDIsKp7KGYVde7lonP0GrYNvU6FWxRoEZE5i72yu4FbM6WQ_2OhiY-wGHx9JaTalZ6ypgMihXj9LlmEnT6eAkopEz0lpSj1ZV-VOsMcdGbfGN5PuLwLHOUoaKh9pSRtepYmn6-qka8K3SyJaotqjCRHkBaJZ9AulRDaUxL9vHLk0Xm_dSzS_b77LbynOrScx7qVZQJ6Jd7LOBuX21naspAvX_9H8-kzZwwTGcYWcBM07uZlLwwbnYINb_XeZOYFf2SClsCwBIeFVakbhnf4a4-lK-KsIb2CLfhpx5BdAtLik3mXkLZcaXlZeN98V1ZZgZpNLMmfy5_wcG-3dCfkkBeAe4a4si-guI6Ges30gUOxf1K_xsIzFfPuHzcNxzqarn8bKc1obnZOxSQ215rE-GsS34e5Wa6N90iM48NUfLqLLl6QfpY8AEqSHPxXJTcm8rhU07N0CX2hcplfL4rK1WmUI6cCGHFmLKV9pu3UyQ1g5iDa0ZeD2riIPlPa1D8qKQumr-LDc4WRvTnpQJ8d1UEQds5d5fSVVAWRTb0bEkh99ncUWPjUOz3gNFGiBqo_5k41fr-VSYeAuGMrprCMBKDeF11KXLZiFgewnhN6ndjhB2DTdyuTp2nyEKvWOwOleZL.nxTxd3V1F6EUnBQGPEFvCw/ucf101_processed.zip",   
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
    runcmd(f'unzip -qo ./temp/ucf101_processed.zip -d {final_data_folder} && rm -rf ./temp/ucf101_processed.zip', 
            is_wait=True)

    print("--DONE--")
    print("-" * 20)
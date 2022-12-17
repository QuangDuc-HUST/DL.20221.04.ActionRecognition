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

def get_data_url(dataset, process_type):

    DATASET_URLS = {
    "ucf101_5_frames_uniform": "https://www.kaggleusercontent.com/kf/113750491/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..sVUWgz6PD-iOFgBLORnxxQ.6xJ1TotkB85GzaNtzHf9Qbkkf2Va04Eyq-MESgW_ej9vFYUUi7NhB0He3Ts54j7z5HyG-2TlnrbOTWk2PkwvH4aFVzMy4UTsizIjy8jglK9-EtCaydcwO3f7AN0StoCMMqi8PL8KwD_OOdNTnzBEuyDVoGHPfYjI-sdMMdpW5mAHXyuyhn-h6QBUUZy-Zl4SMK5S90KqQ0ycx3RGEmRYWxR03Iz-WHXVDwjvuugOsx_ZdnNKD6OXGfEXUSNInBFy8KEBHQJXm7ilOVEnjAtFDIzQHhIAcK_lpEhKxJEACrFRMg4GYEEShl2WysNkpk0WmBxsash11JkLGROE4XOPTljUtg-X-gW85Ko5fdQWlRo2dFM45aFBHfJSBN2ELcQ8c1sdnPwuek1KT5TFKpCe3mBKMIt12mGps_cStXdZu_h7vloGXR22lw-WRpXoA3QtEIkohXz3ifKJyKfy243GfdI65ZovoFVH7uHMQrvM-GW2knhL_u7PdHyaky5jfqsZNQmRPeM6nF6_yYMXx9TAAYW8dkp5V7pRwtFB44xqjduLsM-dBse6YabfTmkQE2RkhKTz9RKBi9cIC9S9ltAC90uRDctKAYEK87DKBZW-VptkBNoqKd_n1yucgaJ2iWkgLNhJvA3qBEr_Z5BT2Ttc911sk5Vf-VagxtAMoWesBoMAkozgWCwh0H-r-WXvhnPu.7w-Ox8VE2awMs-OyFAPAzw/ucf101_processed.zip",
    "ucf101_16_frames_conse_rand": "https://www.kaggleusercontent.com/kf/113916742/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..QBGMtkVjPn97CqNiXl96EQ.XAXBnHhJMytDPshol2tUJPWAeADzLwv-Fz1HqlSBhkOJq0BPQJ-CeBAU378vXq_oOKdmyObpKSemdIC-0WnybRdSOrEw9Xc3uNbQixpFhk79KllObawMZpRHDIpOZg_QYn0bgHdfUxoGiKpSEAIlmqXBszGVHsnP55Q2488QxvHOoxr63IMakaNo0QG1XcfWdXfDr-NTIbAxHdBLfe5wB3FzbUdUhaxnhRz2q4R3SNG84PLLLp0tzGUpHK7X4a-TotqZY_jCgqT0xVyGwiviSYEfqkc8gNhp5xcgkM_S3px5iCnd1Zbs5DzICKFzaFdsio7vYLl5htuNovNCcBsimfRAg9zObKHIQ8qoK2PjARqMh2fGdT1pPhmZAh148M4X6-6e1aLWn9IXXTsG7TSkqXEMKKpd2OczLWb7WWBzcyFBwoUU1VImBGFxTaktSpugZXu8FEBqMqBVy2Xpn9auBrvnepKW6jyLWQ7XWwbFrkAxD1peV6n7asdiGErv0V964H3s9BcUki5MeH-ZsgHoRVkh5J1Heo3Whb1JVwxQyAwqnIMdJeBxrSdVUgWsaRr6ZEOzWb8xilzdsG0_zm8softREnGGEMSzhLcNyIA-54pArduBVV8gpP6ee9j5rzs7eFRpLj7R2QYZPWdova7i7vJ6aMKMGroHHAqlxSS1K9COIK2tg5qCcU4E_2JTCT4T.E5OiXSHeaoO4re5Z4dFiOg/ucf101_processed.zip",
    "hmdb51_5_frames_uniform": "https://www.kaggleusercontent.com/kf/113750535/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..yEwntYz6GeduAXt4__Xw2Q.OL4z_mpXG_uWQNPFmXo70BTJ1SjZZMH750VZDa03Ss53n1-uNR6YPnkR-G83UxvEHyTZDi3HlNQ6BQ8DzoMWZOuJTDj_n_GjIc36W_CJLxn7ZIzxhuR3MPmvO3OoaTHqxXk_vKiJ0SMq5i-GIjAOFqAHgXeQV44E3Eraunzqwq9ugEtS-SuGFsl8LSB9tbaWjbZdP9Ou9ng8jBkGnX0DrUV8A0cVFOeypQ5yQFiQTqFd7cZP0e17Sl3g3_sFHft1OndXQuZc17rrOm1eatYtfCaczhHSSejrizMeCznGbYAWUYZW2ha9Zs8lfOcc4u5vPerOD3Bp9RHf2n1-ceK55SW_OetE-S6pvffJRF1voOyHY6W-Ceq1iuiiCLAB53Z5Cb--5SB1_mF6MxmbgHUBH4vrbycGfJ3b6xGmBQbC5QmwXsqaJBYh_5h7an6854Lf8FOoV9r2I4kcQUu8zKA4b1fhkEh8ChplowA__eE0cnlbt0WgEy_h-cmJYh_nzuECxueH8AEN3k21rE_vg8w7xos3b3KupUXhr4ahR180u_xg46RJSlD9CAvovHpKLv2HXH0qH5t8pTYnIx5wbnjAUuTKlA8RpmmpEzoYoocSs0VmnEq6LmlOFWI5_FxYA_OrX7YEcTwe6LBsH7ScfKhqjXidqeqppk9Y3cPO1YgOvNUjuWFM5nuZNHOuxTfxExzM.ISLmYBiObw1XRRQttF8wkA/hmdb51_processed.zip",
    "hmdb51_5_frames_conse_rand": "https://www.kaggleusercontent.com/kf/113879914/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..5iXzBCRy7sZ3N5q-z6yBgg.26HMI-tUOIJFHmeKGvpojKVfnTcfgnltKEB5md40P9TTMlMRELBXHlb4P2CsNCd4yUtW_SBq4vJkhrfwAR03B34_coBzAO0b8e_qOuLZGWlKn-6R42DyIQtjFN6mDBBdO53k-Cf0XYU7ishW9LenZFsJCABuBIOIt2ymvYwW4OSyfYIWtqEghbhfz4nM0yCgD-VjSwOVrblCbUUJGH471jvbO_IM_SCPPvv72Zg7odmiskN-45Ia90YPEB7NvrIWCEHWe84f5pNbxqi4y0axnldB84dl8ADHSm2TCYP5LFSs4Srtr3Ozhp-2jejC7nwgsYtLvz9W7jBciWI1qayFusZSc6_ztYn21rVlS9VV7H098XiP0DO0shLLepa-SbbI8lA6fT5cPmo4ctihEvUmSDUeBO4ncketMMa6j0lhksn4oSfKEPdXzhLQWl-JR10fg6Ju1X0qQ1jfeB1jOJhexQKgVC7fPPl-fqO0SkPjo89CBRtxzs4ynJ3b11aK1VyaZjkh7FCh3WwPbPOcodu6XKJvlvu5wxHFRvUyw0q7mldvvj3JCHyZZc_-0IxoYupGRz3_Xn8eE9ygActFIbyM_MB6dirRV-hTSSTFcPtYCEZkQrDyNABPobkVbwz5VRRfaYd5GHFVFIJQlGChaIlNz9TkZG82X249PrvKaIjIZOqB1c7iInyhAYhOWg2qgmZX.-GzwJnio1CYEIbAV2mSdFg/hmdb51_processed.zip",
    "hmdb51_16_frames_conse_rand": "https://www.kaggleusercontent.com/kf/113883974/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..nZBBTAT6Eq6BitgB1JHjTg.bqj3CUivldaZXmgY1xnckUJDv9n7OzXj252pbHlTpXg2HHsfPOdaUhDRAml7uSf8ndX1VUdZSbccMLguCY2-m9gLA7_0AkIwcTtQVuJ3YT58hjtUh4LNVFGyvQ99CPZeMPIp3_KkKY4wPtC8Hz2U8RirHLaaocKp6GrJDlIbUBX_fqnlw6olh4m-QNabZeUkeMCJtH1c14cBeSW07kkfdIluKIk4k9WicKqZG0H9NJfwebYy31gBXDx9cGbi18l4-0UdEQc0YFENEN_dv93UI3E15oL8BkbLTffu6V1ONoa8rRnYzlibPm1gLOu9F-NePwm-mctJOGkef0zM8OLgEZiu6Je28gY9-u0D0dzfttrd27fFDVUyqjz5xqI_DJ_3f4dB96ARvjSjBu0J6ZEcgT64xtM9GVFXp08Jtcm-X4SY9xpXbRSs9QelKWNekyhEztYQ6QRMg6JZQLu69Hq5li42SfU6gvSgqw-PiR88YXi6usCDtA6l38O6rT1drVZjUDtRdCVUX-ByrVjPXxSPIfXUqkXIoG3r3eH9hQEmfB4l_CTfBTtGRexpgxSpxn1GmhwmWCIoSHE2rNjkVuUfCVwR-UFU-PvoaouKcwPbDAMIcYOqdyPoUQtcvXbqkrA4kojklza__kEUy6MlqL18reXk3cerHNv3dtYaSOV_rCrnRtmC_kNchhY8oygGxCb3.-ki4TJVVn0oqGH75Lnylzg/hmdb51_processed.zip",
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

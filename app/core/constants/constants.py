import pandas as pd


NUM_CLASSES = 101

WEIGHT_FILE = 'best.pth'
NUM_WORKERS = 2

DATASET_51 = 'hmdb51'
DATASET_101 = 'ucf101'

LABEL = pd.read_csv('./data/UCF101/annotation/video_class_to_label.csv')[['video_class_id', 'label_id']].rename(columns={'video_class_id': 'label'})

ARTIFACT_NAME_LRCN = 'ucf101-lrcn-5f-uni-256-resnet-backbone-lr-5e-4-epochs-20'
LRCN_ARGS = {'model_name': 'lrcn',
             'latent_dim': 512,
             'hidden_size': 256,
             'lstm_layers': 2,
             'bidirectional': True,
             'sample_type': '5_frames_uniform',
             'resize_to': 256}

ARTIFACT_NAME_C3D = 'ucf101-c3d-16f-1c-112-pretrained-lr-1e-4-epochs-20'
C3D_ARGS = {'model_name': 'c3d',
            'drop_out': .5,
            'pretrain': False,
            'weight_folder': './model/weights/',
            'sample_type': '16_frames_conse_rand',
            'resize_to': 112
            }

ARTIFACT_NAME_I3D = 'ucf101-i3d-16f-5c-224-resnet-backbone-lr-1e-4-epochs-5'
I3D_ARGS = {'model_name': 'i3d',
            'sample_type': '16_frames_conse_rand',
            'resize_to': 224
            }

ARTIFACT_NAME_NON_LOCAL = 'ucf101-non-local-16f-5c-224-resnet-backbone-lr-1e-4-epochs-5'
NON_LOCAL_ARGS = {'model_name': 'non_local',
            'use_nl': True,
            'weight_folder': './model/weights/',
            'sample_type': '16_frames_conse_rand',
            'resize_to': 224
            }

ARTIFACT_NAME_LATE_FUSION = 'ucf101-late-fusion-5f-uni-256-resnet-backbone-lr-5e-4-epochs-20'
LATE_FUSION_ARGS = {'model_name': 'late_fusion',
            'latent_dim': 512,
            'sample_type': '5_frames_uniform',
            'resize_to': 256
            }

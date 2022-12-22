NUM_CLASSES = 101

MODEL_FILE = 'best.pth'
NUM_WORKERS = 2
RESIZE_LRCN = 256
RESIZE_C3D = 112

ARTIFACT_NAME_LRCN = 'ucf101-lrcn-5f-uni-256-resnet-backbone-lr-5e-4-epochs-20'
LRCN_ARGS = {
        'model_name': 'lrcn',
        'latent_dim': 512,
        'hidden_size': 256,
        'lstm_layers': 2,
        'bidirectional': True,
    }

ARTIFACT_NAME_C3D = 'ucf101-c3d-16f-1c-112-pretrained-lr-1e-4-epochs-20'
C3D_ARGS = {
        'model_name': 'c3d',
        'drop_out': .5,
        'pretrain': False,
        'weight_path': 'c3d.pickle',
    }
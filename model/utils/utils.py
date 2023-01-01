#
#
# Utils file for models
#
#
def download_weights(weight_folder, weight_file):
    try:
        import wandb
        wandb.login(anonymous='must')
    except:
        raise("Please install wandb for downloading weights")

    api = wandb.Api()

    artifact = api.artifact('dandl/dl_action_recognition/weights_collection:v0', type='weights')

    artifact.get_path(weight_file).download(weight_folder)

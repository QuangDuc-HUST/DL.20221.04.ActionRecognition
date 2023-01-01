#
# API for Late Fusion
#

from .utils import late_fusion_model_2_dataset


class LateFusion(late_fusion_model_2_dataset.LateFusion):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("late_fusion")
        parser.add_argument("--latent_dim", type=int, default=512)
        return parent_parser

    def __init__(self, n_class_1, n_class_2, latent_dim, *args, **kwargs):
        super(LateFusion, self).__init__(n_class_first=n_class_1, n_class_second=n_class_2, latent_dim=latent_dim)

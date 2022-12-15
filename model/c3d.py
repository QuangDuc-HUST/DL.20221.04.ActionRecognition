from .utils import c3d_model


class C3D(c3d_model.C3D):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("C3D")
        parser.add_argument("--drop_out", type=float, default=0.5)
        return parent_parser

    def __init__(self, drop_out, n_class,  *args, **kwargs):
        super(C3D, self).__init__(drop_out=drop_out, n_class=n_class)
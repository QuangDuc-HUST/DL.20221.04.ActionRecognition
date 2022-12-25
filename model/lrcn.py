#
# API for LRCN
#

from .utils import lrcn_model

class LRCN(lrcn_model.ConvLSTM):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LRCN")
        parser.add_argument("--latent_dim", type=int, default=512)
        parser.add_argument("--hidden_size", type=int, default=256)
        parser.add_argument("--lstm_layers", type=int, default=2)
        parser.add_argument("--bidirectional", action='store_false')
        return parent_parser

    def __init__(self, latent_dim, hidden_size, lstm_layers, bidirectional, n_class,  *args, **kwargs):
        super(LRCN, self).__init__(latent_dim, hidden_size, lstm_layers, bidirectional, n_class)
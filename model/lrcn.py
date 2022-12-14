import utils.lrcn_model as lrcn_model

class LRCN(lrcn_model.ConvLstm):
    def __init__(self, *args, **kwargs):
        super(LRCN, self).__init__(*args, **kwargs)
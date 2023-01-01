#
# API for I3D
# It's also based on https://github.com/PPPrior/i3d-pytorch
#

from .utils import i3d_resnet3d_model, i3d_resnet3d_backbones


class I3D(i3d_resnet3d_model.I3D):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("I3D")
        return parent_parser

    def __init__(self, num_classes, backbone_name="resnet50", progress=True, modality='RGB', pretrained2d=True, **kwargs):
        backbone = i3d_resnet3d_backbones.resnet3d(arch=backbone_name, progress=progress, modality=modality, pretrained2d=pretrained2d)
        classifier = i3d_resnet3d_model.I3DHead(num_classes=num_classes, in_channels=2048)
        super(I3D, self).__init__(backbone, classifier)

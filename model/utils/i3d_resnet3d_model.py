#
#
#  I3d Resnet3D Implementation 
#  Source: https://github.com/PPPrior/i3d-pytorch
# 

import torch.nn as nn

class I3D(nn.Module):
    """
    Implements a I3D Network for action recognition.

    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
        classifier (nn.Module): module that takes the features returned from the
            backbone and returns classification scores.
    """
    def __init__(self, backbone, classifier):
        super(I3D, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.backbone(x)
        x = self.classifier(x)
        return x


class I3DHead(nn.Module):
    """Classification head for I3D.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
    """

    def __init__(self, num_classes, in_channels, dropout_ratio=0.5):
        super(I3DHead, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.dropout_ratio = dropout_ratio
        # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # [N, in_channels, 4, 7, 7]
        x = self.avg_pool(x)
        # [N, in_channels, 1, 1, 1]
        if self.dropout is not None:
            x = self.dropout(x)
        # [N, in_channels, 1, 1, 1]
        x = x.view(x.shape[0], -1)
        # [N, in_channels]
        cls_score = self.fc_cls(x)
        # [N, num_classes]
        return cls_score


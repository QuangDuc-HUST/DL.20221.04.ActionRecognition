#
#
#   Late Fusion model Pytorch implemented by our team
#
#

import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class LateFusion(nn.Module):
    """
    Pytorch implementation of LRCN: PretrainConv -> Pooling -> MLP
    """
    def __init__(self, n_class, latent_dim, *args, **kwargs):

        super(LateFusion, self).__init__()
        self.conv_model = PretrainedConv(latent_dim)
        self.pooling_layer = nn.AdaptiveAvgPool1d(1)
        self.output_layer = nn.Linear(latent_dim, n_class)

    def forward(self, x):

        batch_size, timesteps, channel_x, h_x, w_x = x.shape
        conv_input = x.view(batch_size * timesteps, channel_x, h_x, w_x)
        conv_output = F.relu(self.conv_model(conv_input))
        pool_input = conv_output.view(batch_size, timesteps, -1).permute(0, 2, 1)
        pool_out = self.pooling_layer(pool_input).squeeze()
        output = self.output_layer(pool_out)

        return output


class PretrainedConv(nn.Module):

    def __init__(self, latent_dim):
        super(PretrainedConv, self).__init__()
        self.conv_model = models.resnet152(pretrained=True)
        # self.conv_model = models.resnet152(weights='DEFAULT')

        for param in self.conv_model.parameters():
            param.requires_grad = False

        self.conv_model.fc = nn.Linear(self.conv_model.fc.in_features, latent_dim)

    def forward(self, x):
        return self.conv_model(x)

#
#  Long-term Recurrent Convolutional Neural Networks (LRCN)
#  Based on (with modification): https://github.com/doronharitan/human_activity_recognition_LRCN
#
#

import torch.nn as nn
from torchvision import models


class ConvLSTM(nn.Module):
    """
    Pytorch implementation of LRCN: PretrainConv -> LSTM (get the last hidden state) -> MLP
    """
   
    def __init__(self, latent_dim, hidden_size, lstm_layers, bidirectional, n_class):

        super(ConvLSTM, self).__init__()

        self.conv_model = Pretrained_conv(latent_dim)

        self.Lstm = Lstm(latent_dim, hidden_size, lstm_layers, bidirectional)

        self.output_layer = nn.Linear(2 * hidden_size if bidirectional else hidden_size, n_class)

    def forward(self, x):

        batch_size, timesteps, channel_x, h_x, w_x = x.shape

        conv_input = x.view(batch_size*timesteps, channel_x, h_x, w_x)

        conv_output = self.conv_model(conv_input)

        lstm_input = conv_output.view(batch_size, timesteps, -1)

        lstm_output = self.Lstm(lstm_input)

        lstm_output = lstm_output[:, -1, :] #Last layer

        output = self.output_layer(lstm_output)

        return output

class Pretrained_conv(nn.Module):

    def __init__(self, latent_dim):
        super(Pretrained_conv, self).__init__()
        self.conv_model = models.resnet152(pretrained=True)
        for param in self.conv_model.parameters():
            param.requires_grad = False

        self.conv_model.fc = nn.Linear(self.conv_model.fc.in_features, latent_dim)

    def forward(self, x):
        return self.conv_model(x)

class Lstm(nn.Module):

    def __init__(self, latent_dim, hidden_size, lstm_layers, bidirectional):
        
        super(Lstm, self).__init__()

        self.Lstm = nn.LSTM(latent_dim, 
                            hidden_size=hidden_size, 
                            num_layers=lstm_layers, 
                            batch_first=True, 
                            bidirectional=bidirectional)

    def forward(self, x):
        output, _ = self.Lstm(x)
        return output
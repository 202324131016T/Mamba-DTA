import torch
import torch.nn as nn
from torch.autograd import Variable
from mamba_ssm import Mamba


def my_mamba(dim=128):
    # input: (N, L, C)
    my_mamba = Mamba(
        # This module uses roughly 3 * expand * d_model^2 parameters
        d_model=dim,  # Model dimension d_model
        d_state=16,  # SSM state expansion factor
        d_conv=4,  # Local convolution width
        expand=2,  # Block expansion factor
    ).to("cuda")
    # output: (N, L, C)
    return my_mamba


class SFF(nn.Module):
    '''
    Selective feature fusion SFF
    '''
    def __init__(self, channels=64, r=4):
        super(SFF, self).__init__()
        inter_channels = int(channels // r)
        # att
        self.att = nn.Sequential(
            nn.Linear(channels, inter_channels),
            # nn.LayerNorm(inter_channels),
            nn.ReLU(inplace=True),
            nn.Linear(inter_channels, channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        xa = x + y
        xa = self.att(xa)
        wei = self.sigmoid(xa)
        xi = x * wei + y * (1 - wei)
        return xi


class RegNet(nn.Module):
    def __init__(self):
        super(RegNet, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(128 * 2, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x, y):
        # BC
        r = torch.cat((x, y), dim=1)  # (256, 256*2)
        r = self.fc(r)  # (256, 1)
        r = r.squeeze()  # (256)
        return r

class net(nn.Module):
    def __init__(self, FLAGS):
        super(net, self).__init__()
        self.embedding1 = nn.Embedding(FLAGS.charsmiset_size, 128)
        self.embedding2 = nn.Embedding(FLAGS.charseqset_size, 128)

        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.d_pool = nn.AdaptiveMaxPool1d(100)
        self.t_pool = nn.AdaptiveMaxPool1d(1000)

        # extract feature layer
        C0 = 128
        C = 256
        self.d_mamba = my_mamba(dim=C0)
        self.t_mamba = my_mamba(dim=C0)
        # k = 5
        # self.d_conv1 = nn.Conv1d(in_channels=C0, out_channels=C, kernel_size=k, stride=1, padding=int((k - 1) / 2))
        # self.d_conv2 = nn.Conv1d(in_channels=C, out_channels=C, kernel_size=k, stride=1, padding=int((k - 1) / 2))
        # k = 7
        # self.t_conv1 = nn.Conv1d(in_channels=C0, out_channels=C, kernel_size=k, stride=1, padding=int((k - 1) / 2))
        # self.t_conv2 = nn.Conv1d(in_channels=C, out_channels=C, kernel_size=k, stride=1, padding=int((k - 1) / 2))

        # SFF
        # self.d_sff = SFF(channels=C, r=4)
        self.t_sff = SFF(channels=C0, r=4)

        # RegNet
        self.regnet = RegNet()

    def forward(self, x, y):
        # d_embedding
        x_init = Variable(x.long()).cuda()  # (B, L=100)
        x_embedding = self.embedding1(x_init)  # (B, L=100, C=128)
        x_embedding = x_embedding.permute(0, 2, 1)  # (B, C=256, L=100)
        x_embedding = self.d_pool(x_embedding)  # (B, C=256, L=100)
        # t_embedding
        y_init = Variable(y.long()).cuda()  # (B, L=1000)
        y_embedding = self.embedding2(y_init)  # (B, L=1000, C=128)
        y_embedding = y_embedding.permute(0, 2, 1)  # (B, C=256, L=1000)
        y_embedding = self.t_pool(y_embedding)  # (B, C=256, L=1000)

        # BCL -> BLC
        x_embedding = x_embedding.permute(0, 2, 1)
        x_embedding = self.d_mamba(x_embedding)
        x_embedding = x_embedding.permute(0, 2, 1)
        # BCL
        # x_embedding = self.d_conv1(x_embedding)
        # x_embedding = self.relu(x_embedding)
        # x_embedding = self.d_conv2(x_embedding)
        # x_embedding = self.relu(x_embedding)
        x_embedding = self.pool(x_embedding)
        x_embedding = x_embedding.squeeze()
        # BCL -> BLC
        y_embedding = y_embedding.permute(0, 2, 1)
        y_embedding = self.t_mamba(y_embedding)
        y_embedding = y_embedding.permute(0, 2, 1)
        # BCL
        # y_embedding = self.t_conv1(y_embedding)
        # y_embedding = self.relu(y_embedding)
        # y_embedding = self.t_conv2(y_embedding)
        # y_embedding = self.relu(y_embedding)
        y_embedding = self.pool(y_embedding)
        y_embedding = y_embedding.squeeze()

        # SFF BCL
        # d = self.d_sff(x_embedding, y_embedding)
        t = self.t_sff(y_embedding, x_embedding)
        d = x_embedding
        # RegNet BCL
        out = self.regnet(d, t)

        return out  # B affinity



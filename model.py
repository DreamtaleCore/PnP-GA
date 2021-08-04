import torch
import torch.nn as nn
from resnet import resnet18
import torch.nn.functional as F
from scipy.stats import norm


class GazeRes18(nn.Module):
    def __init__(self, drop_p=0.5):
        super(GazeRes18, self).__init__()
        self.img_feature_dim = 256  # the dimension of the CNN feature to represent each frame

        self.base_model = resnet18(pretrained=True)

        self.base_model.fc2 = nn.Linear(1000, self.img_feature_dim)

        # The linear layer that maps the LSTM with the 3 outputs
        self.last_layer = nn.Linear(self.img_feature_dim, 3)
        self.drop = nn.Dropout(drop_p)

    def forward(self, x_in):
        base_out = self.base_model(x_in["face"])
        base_out = torch.flatten(base_out, start_dim=1)
        output = self.drop(base_out)
        output = self.last_layer(output)
        angular_output = output[:, :2]

        return angular_output, base_out


class OutlierLoss(nn.Module):
    def __index__(self):
        sum(OutlierLoss, self).__init__()

    def forward(self, gaze, gaze_ema, significant, gamma=0.01):
        assert gaze.shape == gaze_ema.shape
        mean = torch.mean(gaze_ema, dim=2).reshape(-1, 2, 1)
        std = torch.std(gaze_ema, dim=2).reshape(-1, 2, 1)
        nd = torch.distributions.normal.Normal(mean, std)
        norm_gaze = (gaze - mean) / std
        outlier1 = norm_gaze < norm.ppf(significant)
        outlier2 = norm_gaze > norm.ppf(1 - significant)

        return torch.sum(torch.abs(norm_gaze[outlier1])) + torch.sum(torch.abs(norm_gaze[outlier2])) + \
               gamma * F.l1_loss(nd.cdf(gaze), nd.cdf(mean))

import torch
import torch.nn as nn
import torch.nn.functional as F

class STN_01(nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super(STN_01, self).__init__()
        self.is_training = True  # Flag to control the STN activation

        # Localization network
        self.conv1 = nn.Conv2d(input_channels+hidden_channels, 8, kernel_size=7)
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.relu1 = nn.ReLU(True)
        self.conv2 = nn.Conv2d(8, 10, kernel_size=5)
        self.maxpool2 = nn.MaxPool2d(2, stride=2)
        self.relu2 = nn.ReLU(True)

        # Regressor for the transformation parameters
        self.fc_loc1 = nn.Linear(10 * 3 * 3, 32)
        self.relu3 = nn.ReLU(True)
        self.fc_loc2 = nn.Linear(32, 3 * 2)  # 3x2 for affine transformation

        # Initialize the weights/bias with identity transformation
        #self.fc_loc2.weight.data.zero_()
        #self.fc_loc2.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))


    def forward(self, x):
        if self.is_training:  # Activate STN only during training
            # Localization network
            xs = self.conv1(x)
            xs = self.maxpool1(xs)
            xs = self.relu1(xs)
            xs = self.conv2(xs)
            xs = self.maxpool2(xs)
            xs = self.relu2(xs)

            xs = xs.view(-1, 10 * 3 * 3)
            theta = self.fc_loc1(xs)
            theta = self.relu3(theta)
            theta = self.fc_loc2(theta)
            theta = theta.view(-1, 2, 3)  # Affine transformation matrix

            # Apply the spatial transformation
            grid = F.affine_grid(theta, x.size())
            x = F.grid_sample(x, grid)

        return x
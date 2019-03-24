import torch
from torch.nn.functional import relu, max_pool3d
from torch.nn.modules.conv import Conv3d, ConvTranspose3d

from flow.base.model import BaseModel


class UNet3D(BaseModel):
    def __init__(self):
        super(UNet3D, self).__init__()
        # Define layers
        self.conv1 = Conv3d(1, 64, (3, 3, 3), padding=(1, 1, 1))
        self.conv2 = Conv3d(64, 64, (3, 3, 3), padding=(1, 1, 1))

        self.conv3 = Conv3d(64, 128, (3, 3, 3), padding=(1, 1, 1))
        self.conv4 = Conv3d(128, 128, (3, 3, 3), padding=(1, 1, 1))

        self.conv5 = Conv3d(128, 256, (3, 3, 3), padding=(1, 1, 1))
        self.conv6 = Conv3d(256, 256, (3, 3, 3), padding=(1, 1, 1))

        self.up7 = ConvTranspose3d(256, 128, (2, 2, 2), stride=(2, 2, 2))
        self.conv8 = Conv3d(256, 128, (3, 3, 3), padding=(1, 1, 1))
        self.conv9 = Conv3d(128, 128, (3, 3, 3), padding=(1, 1, 1))

        self.up10 = ConvTranspose3d(128, 64, (2, 2, 2), stride=(2, 2, 2))
        self.conv11 = Conv3d(128, 64, (3, 3, 3), padding=(1, 1, 1))
        self.conv12 = Conv3d(64, 64, (3, 3, 3), padding=(1, 1, 1))

        self.conv13 = Conv3d(64, 1, (1, 1, 1))

    def forward(self, x):
        x = relu(self.conv1(x))
        x1 = relu(self.conv2(x))
        x = max_pool3d(x1, (2, 2, 2), stride=(2, 2, 2))

        x = relu(self.conv3(x))
        x2 = relu(self.conv4(x))
        x = max_pool3d(x2, (2, 2, 2), stride=(2, 2, 2))

        x = relu(self.conv5(x))
        x = relu(self.conv6(x))

        x = self.up7(x)
        x = torch.cat((x, x2), dim=1)
        x = relu(self.conv8(x))
        x = relu(self.conv9(x))

        x = self.up10(x)
        x = torch.cat((x, x1), dim=1)
        x = relu(self.conv11(x))
        x = relu(self.conv12(x))

        x = self.conv13(x)

        x = torch.sigmoid(x)
        return x

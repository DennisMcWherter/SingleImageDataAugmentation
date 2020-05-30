import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

class TestMobilenetV2(nn.Module):

    def __init__(self, num_classes):
        super(TestMobilenetV2, self).__init__()

        # Freeze mobilenet
        self.mobilenetv2 = models.mobilenet_v2(pretrained=True)
        for param in self.mobilenetv2.parameters():
            param.requires_grad = False

        # Expect input size of (224,224)
        # TODO: We shouldn't need to hardcode this, but it works for our experiments.
        self.fc1 = nn.Linear(11520, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.avg_pool2d(self.mobilenetv2.features(x), 2)
        x = torch.flatten(x, start_dim=1, end_dim=3)
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.fc3(x))
        return F.softmax(x, dim=0)



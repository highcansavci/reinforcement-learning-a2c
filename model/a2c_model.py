import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transform


TRANSFORM = transform.Resize(size=(84, 84))


def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out = std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find("Linear") != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1])
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


class A2C(nn.Module):
    def __init__(self, num_inputs, action_space):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1),
            nn.ELU(True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ELU(True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ELU(True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ELU(True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ELU(True),
            nn.BatchNorm2d(32),
        )

        self.lstm = nn.LSTM(num_layers=1, input_size=1, hidden_size=256, batch_first=True)
        num_outputs = action_space
        self.critic_linear = nn.Linear(in_features=256, out_features=1)
        self.actor_linear = nn.Linear(in_features=256, out_features=num_outputs)

        self.apply(weights_init)
        self.train()

    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        x = self.convs(inputs)
        x = x.reshape(-1, 32 * 3 * 3).unsqueeze(2)

        hx, cx = self.lstm(x, (hx, cx))
        x = hx[:, -1, :]
        return self.critic_linear(x), self.actor_linear(x), (hx, cx)

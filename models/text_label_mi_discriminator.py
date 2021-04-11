import torch
import torch.nn as nn
import torch.nn.functional as F

class TextLabelMIDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.c0 = nn.Conv1d(300, 300, kernel_size=3)
        self.c1 = nn.Conv1d(300, 512, kernel_size=3)
        self.l0 = nn.Linear(512 + 300, 512)
        self.l1 = nn.Linear(512, 512)
        self.l2 = nn.Linear(512, 1)

    def forward(self, y, M):
        h = F.relu(self.c0(M))
        h = self.c1(h)
        h = torch.mean(h, dim=2)
        h = h.view(y.shape[0], -1)
        h = torch.cat((y, h), dim=1)
        h = F.relu(self.l0(h))
        h = F.relu(self.l1(h))
        return self.l2(h)


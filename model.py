import torch
import torch.nn as nn

class TANet(nn.Module):
    def __init__(self):
        super(TANet, self).__init__()
        self.embedding = nn.EmbeddingBag(1000, 64, sparse=False)
        self.fc = nn.Sequential(
            nn.Linear(64 + 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

    def forward(self, text, numeric_features):
        embedded = self.embedding(text)
        combined = torch.cat((embedded, numeric_features), dim=1)
        return self.fc(combined)

class SiaNet(nn.Module):
    def __init__(self):
        super(SiaNet, self).__init__()
        self.embedding_net = TANet()

    def forward(self, text1, num1, text2, num2):
        out1 = self.embedding_net(text1, num1)
        out2 = self.embedding_net(text2, num2)
        return out1, out2

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Implements the attention-based MIL pooling introduced by:
        Ilse et al. 'Attention-based Deep Multiple Instance Learning' (2018)
    """
    def __init__(self, L, num_classes):
        super(Attention, self).__init__()
        self.L = L
        self.D = 128
        self.K = 1
        self.num_classes = num_classes

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, self.num_classes)
        )

    def forward(self, H):

        A = self.attention(H)
        A = A.permute(0, 2, 1)
        A = F.softmax(A, dim=2)

        Z = torch.matmul(A, H)

        Y_logit = self.classifier(Z)
        Y_logit = Y_logit.squeeze(1)
        # Y_hat = torch.ge(torch.sigmoid(Y_logit), 0.5).float()
        Y_hat = torch.argmax(nn.functional.softmax(Y_logit, dim=1), dim=1).float()

        return Y_logit, Y_hat, A


if __name__ == '__main__':
    model = Attention(2048, 3)
    test_bag = torch.rand(size=(32, 16, 2048), dtype=torch.float32, requires_grad=True)

    out = model.forward(test_bag)

    print(out)


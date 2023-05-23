import torch


class SimdLoss(torch.nn.Module):
    def __init__(self):
        super(SimdLoss, self).__init__()
        self.similarity_loss = torch.nn.CosineSimilarity(dim=1)

    def forward(self, x, y):
        cosine_sim = self.similarity_loss(x, y)
        loss = torch.mean(1 - cosine_sim)
        return loss

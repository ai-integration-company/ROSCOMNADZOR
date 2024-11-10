from torch import nn


class Embedder(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(Embedder, self).__init__()
        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, x):
        return self.projection_head(x.last_hidden_state[:, 0, :])  
    
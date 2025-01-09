import torch
import torch.nn as nn

class Regressor(nn.Module):
    def __init__(self, input_size):
        super(Regressor, self).__init__()

        self.temporal_branch = nn.Sequential(
            nn.Linear(6, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.spatial_branch = nn.Sequential(
            nn.Linear(2, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.historical_branch = nn.Sequential(
            nn.Linear(7, 48),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.route_branch = nn.Sequential(
            nn.Linear(3, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.attention = nn.Sequential(
            nn.Linear(112, 112),
            nn.Tanh(),
            nn.Linear(112, 112),
            nn.Softmax(dim = 1)
        )

        self.regressor = nn.Sequential(
            nn.Linear(112, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        temporal = self.temporal_branch(x[:, :6])
        spatial = self.spatial_branch(x[:, 6:8])
        historical = self.historical_branch(x[:, 8:15])
        route = self.route_branch(x[:, 15:18])

        combined = torch.cat([temporal, spatial, historical, route], dim = 1)

        attention_weights = self.attention(combined)
        attended_features = combined * attention_weights

        return self.regressor(attended_features)
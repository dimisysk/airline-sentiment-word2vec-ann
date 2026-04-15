import torch
import torch.nn as nn
import torch.nn.functional as F

class ANNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, dropout_p1, dropout_p2):
        super(ANNClassifier, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)

        self.dropout1 = nn.Dropout(p=dropout_p1)
        self.dropout2 = nn.Dropout(p=dropout_p2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)

        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        x = self.fc3(x)
        return x
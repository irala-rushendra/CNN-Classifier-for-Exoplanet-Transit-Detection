import torch
import torch.nn as nn
import torch.nn.functional as F
class ExoplanetDualBranchCNN(nn.Module):
    def __init__(self, global_length=2001, local_length=201, num_classes=6):
        super(ExoplanetDualBranchCNN, self).__init__()

        # Global View Branch
        self.global_conv1_1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
        self.global_conv1_2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=5, padding=2)
        self.global_pool1 = nn.MaxPool1d(kernel_size=5, stride=2)
        self.global_bn1 = nn.BatchNorm1d(16)

        self.global_conv2_1 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.global_conv2_2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.global_pool2 = nn.MaxPool1d(kernel_size=5, stride=2)
        self.global_bn2 = nn.BatchNorm1d(32)
        
        self.global_conv3_1 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.global_conv3_2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, padding=2)
        self.global_pool3 = nn.MaxPool1d(kernel_size=5, stride=2)

        #Local View Branch

        self.local_conv1_1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
        self.local_conv1_2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=5, padding=2)
        self.local_pool1 = nn.MaxPool1d(kernel_size=5, stride=2)
        self.local_bn1 = nn.BatchNorm1d(16)
    
        self.local_conv2_1 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.local_conv2_2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.local_pool2 = nn.MaxPool1d(kernel_size=5, stride=2)

        with torch.no_grad():
            dummy_global = torch.zeros(1,1,global_length)
            dummy_local = torch.zeros(1,1,local_length)
            flat_global_dim = self._forward_global(dummy_global).shape[1]
            flat_local_dim = self._forward_local(dummy_local).shape[1]

        combined_dim = flat_global_dim + flat_local_dim

        self.fc1 = nn.Linear(combined_dim, 512)
        self.dropout1 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(256, 128)
        self.fc_out = nn.Linear(128, num_classes)

    def _forward_global(self, x):
        x = F.relu(self.global_conv1_1(x))
        x = F.relu(self.global_conv1_2(x))
        x = self.global_pool1(x)
        x = self.global_bn1(x)

        x = F.relu(self.global_conv2_1(x))
        x = F.relu(self.global_conv2_2(x))
        x = self.global_pool2(x)
        x = self.global_bn2(x)

        x = F.relu(self.global_conv3_1(x))
        x = F.relu(self.global_conv3_2(x))
        x = self.global_pool3(x)
        return torch.flatten(x, start_dim=1)

    def _forward_local(self, x):
        x = F.relu(self.local_conv1_1(x))
        x = F.relu(self.local_conv1_2(x))
        x = self.local_pool1(x)
        x = self.local_bn1(x)

        x = F.relu(self.local_conv2_1(x))
        x = F.relu(self.local_conv2_2(x))
        x = self.local_pool2(x)
        return torch.flatten(x, start_dim=1)

    def forward(self, global_input, local_input):
        x1 = self._forward_global(global_input)
        x2 = self._forward_local(local_input)
        merged = torch.cat((x1,x2), dim=1)

        z = F.relu(self.fc1(merged))
        z = self.dropout1(z)
        z = F.relu(self.fc2(z))
        z = self.dropout2(z)
        z = F.relu(self.fc3(z))

        return self.fc_out(z)

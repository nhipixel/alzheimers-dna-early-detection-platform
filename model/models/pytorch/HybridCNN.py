import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridCNN(nn.Module):
    def __init__(self, input_dim, conv_channels=[64, 128, 256], 
                 mlp_hidden_dims=[512, 256, 128], output_dim=3, dropout_rate=0.4):
        super(HybridCNN, self).__init__()
        
        self.conv_layers = nn.ModuleList()
        self.conv_bn = nn.ModuleList()
        
        in_channels = 1
        for out_channels in conv_channels:
            self.conv_layers.append(
                nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3, stride=1)
            )
            self.conv_bn.append(nn.BatchNorm1d(out_channels))
            in_channels = out_channels
        
        self.adaptive_pool = nn.AdaptiveAvgPool1d(32)
        conv_output_dim = conv_channels[-1] * 32
        
        self.mlp_layers = nn.ModuleList()
        self.mlp_bn = nn.ModuleList()
        
        prev_dim = conv_output_dim
        for hidden_dim in mlp_hidden_dims:
            self.mlp_layers.append(nn.Linear(prev_dim, hidden_dim))
            self.mlp_bn.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim
        
        # Output layer
        self.output_layer = nn.Linear(prev_dim, output_dim)
        
        # Dropout layers
        self.dropout_conv = nn.Dropout(dropout_rate * 0.5)
        self.dropout_mlp = nn.Dropout(dropout_rate)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Ensure input has correct dimensions
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add channel dimension
        
        # Convolutional feature extraction
        for conv, bn in zip(self.conv_layers, self.conv_bn):
            x = conv(x)
            x = bn(x)
            x = F.relu(x)
            x = F.max_pool1d(x, kernel_size=2, stride=2)
            x = self.dropout_conv(x)
        
        # Adaptive pooling to standardize output size
        x = self.adaptive_pool(x)
        
        # Flatten for MLP
        x = x.view(x.size(0), -1)
        
        # MLP layers
        for mlp, bn in zip(self.mlp_layers, self.mlp_bn):
            x = mlp(x)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout_mlp(x)
        
        # Output layer
        x = self.output_layer(x)
        
        return x
    
    def get_feature_maps(self, x):
        """Extract intermediate feature maps for analysis"""
        feature_maps = []
        
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        for i, (conv, bn) in enumerate(zip(self.conv_layers, self.conv_bn)):
            x = conv(x)
            x = bn(x)
            x = F.relu(x)
            feature_maps.append(x.clone())
            x = F.max_pool1d(x, kernel_size=2, stride=2)
            x = self.dropout_conv(x)
        
        return feature_maps
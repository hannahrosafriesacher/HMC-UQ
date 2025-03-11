import torch

class MLP(torch.nn.Module):
    def __init__(self, input_features, hidden_sizes, output_features, dropout):
        super().__init__()
        self.input_features = input_features
        self.hidden_sizes = hidden_sizes
        self.output_features = output_features
        self.input = torch.nn.Linear(in_features=input_features, out_features=hidden_sizes)
        self.tanh = torch.nn.Tanh()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.output = torch.nn.Linear(in_features=hidden_sizes, out_features=output_features)
        
    def forward(self, x):
        fc = self.input(x)
        a = self.tanh(fc)
        dr = self.dropout(a)
        out = self.output(dr)
             
        return out
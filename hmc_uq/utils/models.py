import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict


class MLP(nn.Module):
    def __init__(self, input_features, hidden_sizes, nr_layers, output_features, dropout):
        super().__init__()
        self.input_features = input_features
        self.hidden_sizes = hidden_sizes
        self.output_features = output_features
        self.nr_layers = nr_layers

        model = OrderedDict([])
        #First Layer
        model.update({f'0': nn.Linear(input_features,hidden_sizes)})
        model.update({f'activation' : nn.Tanh()})
        model.update({f'dropout' : nn.Dropout(dropout)})

        #add layers
        for i in range(nr_layers-1):
            model.update({f'{i+1}': nn.Linear(hidden_sizes,hidden_sizes)})
            model.update({f'activation' : nn.Tanh()})
            model.update({f'dropout' : nn.Dropout(dropout)})

        model.update({f'{nr_layers}': nn.Linear(hidden_sizes,output_features)})

        self.model = nn.Sequential(model)
        
    def forward(self, x):             
        return self.model(x)  
    
class BNN(nn.Module):
    """
    Re-implementation of the BNN in Bayes by Backprop (Blundell et al., 2015).
    """

    def __init__(
        self, 
        input_dim: int, 
        hidden_sizes: int, 
        output_dim: int, 
        nr_layers: int, 
        prior_mu: int, 
        prior_rho: int,
        prior_sig: float,
    ):
        super(BNN, self).__init__()
        self.input_dim = input_dim 
        self.hidden_sizes = hidden_sizes 
        self.output_dim = output_dim 
        self.nr_layers = nr_layers 
        self.prior_mu = prior_mu
        self.prior_rho = prior_rho
        self.prior_sig = prior_sig
                    
        hidden_layers = [input_dim]
        for i in range(nr_layers):
            hidden_layers.append(hidden_sizes)
        hidden_layers.append(output_dim)

        self.nr_layers = nn.ModuleList()
        for layer in range(len(hidden_layers)-2):
            self.nr_layers.append(
                BayesWeightLayer(hidden_layers[layer], hidden_layers[layer+1], prior_mu, prior_rho, prior_sig, activation='tanh')
            )

        self.nr_layers.append(
            BayesWeightLayer(hidden_layers[-2], hidden_layers[-1], prior_mu, prior_rho, prior_sig, activation='none')
        )

    def forward(self, x):
        net_kl = 0
        for layer in self.nr_layers:
            x, layer_kl = layer(x)
            net_kl += layer_kl
        return x, net_kl


class BayesWeightLayer(nn.Module):
    '''
    Layer used in BNN, heavily inspired by the implementation in 
    https://github.com/JavierAntoran/Bayesian-Neural-Networks/tree/master.
    '''

    def __init__(self, input_dim, output_dim, prior_mu, prior_rho, prior_sig, activation):
        super(BayesWeightLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.prior_mu = prior_mu
        self.prior_rho = prior_rho
        self.prior_sig = prior_sig
        
        # Instantiate a large Gaussian block to sample from, much faster than generating random sample every time
        self._gaussian_block = np.random.randn(10000) # Noise
        self._Var = lambda x: Variable(torch.from_numpy(x).type(torch.FloatTensor)) #Function that outputs TorchTensor

        # Learnable parameters (Parameters of surrogate distribution)
        self.W_mu = nn.Parameter(torch.Tensor(input_dim, output_dim).uniform_(-0.01, 0.01)) # SurrWeights: mean
        self.W_rho = nn.Parameter(torch.Tensor(input_dim, output_dim).uniform_(-3, -3)) #SurrWeights: Var

        self.b_mu = nn.Parameter(torch.Tensor(output_dim).uniform_(-0.01, 0.01)) #SurrBias: mean
        self.b_rho = nn.Parameter(torch.Tensor(output_dim).uniform_(-3, -3)) # SurrBias: Var
        self.activation = nn.Tanh() if activation == 'tanh' else None
    
    def forward(self, x):
        
        # calculate std
        std_w = 1e-6 + F.softplus(self.W_rho) #STD of weights = 1e-6 + softplus(SurrWeights:Var)
        std_b = 1e-6 + F.softplus(self.b_rho) #STD of bias = 1e-6 + softplus(SurrWeights:Bias)

        act_W_mu = torch.mm(x, self.W_mu)  # activated means of weights = input * mean of weight
        act_W_std = torch.sqrt(torch.mm(x.pow(2), std_w.pow(2))) # activated std of weights = sqrt(input² * std of weight²)
    
        eps_W = self._random(act_W_std.shape).to(x.device) #noise in shape of activated means of weights
        eps_b = self._random(std_b.shape).to(x.device) #noise in shape of STD of bias

        act_W_out = act_W_mu + act_W_std * eps_W  # activated means of weights + activated std of Weights * noise
        act_b_out = self.b_mu + std_b * eps_b  # means of bias + std of bias * noise   

        output = act_W_out + act_b_out.unsqueeze(0).expand(x.shape[0], -1) #add
        output = self.activation(output) if self.activation else output
        
        if not self.training: 
            return output, 0
                
        kld = BayesWeightLayer.KLD_cost(mu_p=0, sig_p=self.prior_sig, mu_q=self.W_mu, sig_q=std_w) + \
            BayesWeightLayer.KLD_cost(mu_p=0, sig_p=0.1, mu_q=self.b_mu, sig_q=std_b)
        
        return output, kld
    
    def _random(self, shape):
        n_eps = shape[0] * shape[1] if len(shape) > 1 else shape[0]
        eps = np.random.choice(self._gaussian_block, size=n_eps)
        eps = np.expand_dims(eps, axis=1).reshape(*shape)
        return self._Var(eps)
    
    @staticmethod
    def KLD_cost(mu_p, sig_p, mu_q, sig_q):
        KLD = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()
        # https://arxiv.org/abs/1312.6114 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        return KLD
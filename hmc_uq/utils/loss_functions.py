from torch.nn.modules.loss import _Loss

class BayesLoss(_Loss):
    '''
    Re-implementation of the Bayes by Backprop Loss from Blundell et al., (2015).
    '''
    def __init__(self, likelihood, batch_size) -> None:
        super().__init__()
        self.likelihood = likelihood
        self.batch_size = batch_size 

    def forward(self, input, target, kl):
        log_likelihood = self.likelihood(input, target).sum()
        kl *= (self.batch_size/len(target))
        return kl + log_likelihood
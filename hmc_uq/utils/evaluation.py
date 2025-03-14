import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, rankdata, norm
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import math
import random


class BaselinePredictivePerformance:
    def __init__(self, preds, labels, epoch, phase) -> None:
        self.loss = torch.nn.BCELoss()
        self.preds = F.sigmoid(preds).squeeze(dim = 1)
        self.labels = labels.squeeze(dim = 1).float()
        self.epoch = epoch
        self.phase = phase

    def transform_array(self, array):
        return np.asarray(array.cpu().detach())
        
    def epoch_performance(self):
        #NLL and AUC performance            
        nll = self.loss(self.preds, self.labels).detach().item()
        auc = roc_auc_score(self.transform_array(self.labels), self.transform_array(self.preds))

        log_dict = {'epoch' : self.epoch,
                    f'{self.phase}/loss/': nll,
                    f'{self.phase}/auc/': auc}
        
        return log_dict



class HMCPredictivePerformance:
    def __init__(self, preds_chains, labels) -> None:
        self.loss = torch.nn.BCELoss()
        self.preds_chains = F.sigmoid(preds_chains)
        self.labels = labels
        self.nr_chains, self.nr_samples, _ = preds_chains.shape
        self.PF = {'Chain' : [], '#HMC Samples' : [], 'NLL': [], 'AUC':[]}
        
    def calculate_performance(self):
        #NLL and AUC performance
        for chain in range(self.nr_chains):
            nll = 0
            auc = 0
            pred = self.preds_chains[chain]
            for sample in range(self.nr_samples):
                pred_sample = pred[sample]
                nll += self.loss(pred_sample, self.labels.squeeze(dim = 1).float()).item()
                auc += roc_auc_score(np.asarray(self.labels.cpu().detach().numpy()), np.asarray(pred_sample.cpu()))
                mean_nll = nll/(sample+1)
                mean_auc = auc/(sample+1)

                self.PF['#HMC Samples'].append(sample + 1)
                self.PF['Chain'].append(chain + 1)
                self.PF['NLL'].append(mean_nll) 
                self.PF['AUC'].append(mean_auc)
        self.PF = pd.DataFrame(self.PF)
        return self.PF
    
    def nll(self, return_plot):
        df = self.PF[self.PF['#HMC Samples'] == self.nr_samples]
        if return_plot:
            plt.cla()
            NLL_plot = sns.lineplot(data = self.PF, x = '#HMC Samples', y = 'NLL', hue = 'Chain')
            return df['NLL'].to_list(), NLL_plot
        else:
            return df['NLL'].to_list()
    
    
    def auc(self, return_plot):
        df = self.PF[self.PF['#HMC Samples'] == self.nr_samples]
        if return_plot:
            plt.cla()
            AUC_plot = sns.lineplot(data = self.PF, x = '#HMC Samples', y = 'AUC', hue = 'Chain')
            return df['AUC'].to_list(), AUC_plot
        else:
            return df['AUC'].to_list()

class HMCSampleEvaluation:
    def __init__(self, params_chains):
        self.params_chains = params_chains
        self.nr_chains, self.nr_samples,self.nr_params = params_chains.shape
        self.max_burnin = 50 #for plotting Burnin vs Metrics
        self.burnin_step = 5 #for plotting Burnin vs Metrics
        self.AC = {'Burn-in' : [], 'Chain':[], 'LAG' : [], 'AUTOCORR': []}
        self.IPS = {'Burn-in' : [], 'Chain': [], 'IPS':[]}
        self.RHAT = {'Burn-in' : [], 'Split-Rhat':[], 'rnSplit-Rhat':[]}

    def calculate_autocorrelation(self):
        for burnin in range(0, int(self.max_burnin), self.burnin_step):
            for chain in range(self.nr_chains):
                params_chain = self.params_chains[chain][burnin:]
                nr_samples = self.nr_samples - burnin
                max_lag = math.floor((self.nr_samples - burnin)/2)
                for lag in range(0, max_lag):
                    ac = np.mean(pearsonr(params_chain[lag:], params_chain[:(nr_samples-lag)], axis = 0)[0])
                    self.AC['Chain'].append(chain)
                    self.AC['LAG'].append(lag)
                    self.AC['AUTOCORR'].append(ac)
                    self.AC['Burn-in'].append(burnin)
        self.AC = pd.DataFrame(self.AC)

    
    def geweke(self):
        #Geweke's Diagnostic: computed per chain)
        gewekes = []
        for chain in range(self.nr_chains):
            params_chain = self.params_chains[chain]
            s_index = round(len(params_chain) * 0.5) 
            s1 = params_chain[:s_index]
            s2 = params_chain[-s_index:]
            means = np.mean(s1, axis = 0) - np.mean(s2, axis = 0) 
            errors = np.var(s1, axis = 0)/s_index + np.var(s2, axis = 0)/s_index
            gewekes.append(np.mean(means/np.sqrt(errors)))

        return gewekes



    def split_rhat(self, burnin, rank_normalized = False):
        #Gelman Rubin/ Potenitial Scale Reduction Factor (Split Rhat)
        #https://jbds.isdsa.org/public/journals/1/html/v2n2/p3/
        #file:///home/rosa/Downloads/20-BA1221.pdf      

        params_burnin = self.params_chains[:,burnin:]
        nr_samples = self.nr_samples - burnin
        if nr_samples % 2 != 0:
            params_burnin = params_burnin[:, :-1,]

        params_burnin_split = params_burnin.reshape(self.nr_chains *2, int(nr_samples/2), -1)
        nr_chains_split, nr_samples_split,_ = params_burnin_split.shape
        nr_total_samples = nr_samples_split * nr_chains_split

        if rank_normalized:
            params_burnin_split = self.rank_normalize(params_burnin_split, nr_total_samples, nr_chains_split, nr_samples_split)

        chains_mean = np.mean(params_burnin_split, axis = 1).reshape(nr_chains_split, 1, -1)
        grand_mean = np.mean(params_burnin_split.reshape(nr_total_samples,-1), axis = 0)

        within_chains = np.sum(np.square(np.subtract(params_burnin_split, chains_mean)).reshape(nr_total_samples, -1), axis = 0) / (nr_chains_split * (nr_samples_split-1))
        between_chains = np.sum(np.square(np.subtract(chains_mean, grand_mean)), axis = 0) / (nr_chains_split - 1)

        var = ((nr_samples_split -1) / nr_samples_split) * within_chains +  between_chains
        split_rhat = np.sqrt(var/within_chains)

        return split_rhat
    
    def rank_normalize(self, params_burnin_split, nr_total_samples, nr_chains, nr_samples):
        #Rank-Normalized Split Rhat (Vehtari et al (2021)
        #file:///home/rosa/Downloads/20-BA1221.pdf

        params_burnin_split = params_burnin_split.reshape(nr_total_samples, -1)
        ranks = rankdata(params_burnin_split, axis = 0)
        ranks = (ranks - 0.375) / (nr_total_samples + 0.25)
        param_rn = norm.ppf(ranks).reshape(nr_chains, nr_samples, -1)
        return param_rn
    
    
    def ess(self):
        #Traditional Effective Sample Size

        return len(self.params_chains)/(1 + 2*self.AC['AUTOCORR'].sum())
    

    def ips_per_chain(self, burnin):
        #Initial Positive Sequence Estimation
        ips_chains = []
        df_burnin = self.AC[self.AC['Burn-in'] == burnin]
        for chain in range(self.nr_chains):
            autocorr = 0
            max_lag = math.floor((self.nr_samples - burnin)/2)
            df_chain = df_burnin[df_burnin['Chain'] == chain]
            for lag in range(0, max_lag, 2):
                ips = 1
                autocorr_pairs = df_chain['AUTOCORR'].iloc[lag:lag + 2].sum()     

                if autocorr_pairs > 0:
                    autocorr += autocorr_pairs
                elif lag == max_lag - 2:
                    ips = 1
                    break
                else:
                    ips = (self.nr_samples) / (-1 + 2 * autocorr)
                    break
            ips_chains.append(ips)

        return(ips_chains)
    
    #TODO: IPS Total

      
    def rhat_burnin_plot(self):
        plt.cla()
        for burnin in range(0, self.max_burnin, self.burnin_step):
            
            self.RHAT['Burn-in'].extend([burnin] * self.nr_params)
            self.RHAT['Split-Rhat'].extend(self.split_rhat(burnin, rank_normalized=False).flatten().tolist())
            self.RHAT['rnSplit-Rhat'].extend(self.split_rhat(burnin, rank_normalized=True).flatten().tolist())
        rhat_burnin = sns.lineplot(data = self.RHAT, x= 'Burn-in', y = 'Split-Rhat')
        rhat_burnin = sns.lineplot(data = self.RHAT, x= 'Burn-in', y = 'rnSplit-Rhat')
        return rhat_burnin
    
    def ips_burnin_plot(self):
        plt.cla()
        for burnin in range(0, self.max_burnin, self.burnin_step):
            for chain in range(self.nr_chains):
                self.IPS['Burn-in'].append(burnin)
                self.IPS['IPS'].append(self.ips_per_chain(burnin)[chain])
                self.IPS['Chain'] = chain
        ips_burnin = sns.lineplot(data = self.IPS, x= 'Burn-in', y = 'IPS', hue = 'Chain')
        return ips_burnin
    
    #def autocorrelation_plot:
    def autocorrelation_plot(self):
        plt.cla() 
        df = self.AC[self.AC['Burn-in'] == 0]
        autocorr_plot = sns.lineplot(data = df, x = 'LAG', y = 'AUTOCORR', hue = 'Chain')
        return autocorr_plot


    def trace_plot(self, nr_plotted_params):
        rd_ind = random.sample(range(0, len(self.params)), nr_plotted_params)
        params_selected = self.params[:, rd_ind]
        params_selected = pd.DataFrame(params_selected)

        for i in range(nr_plotted_params):
            return sns.lineplot(data = params_selected.reset_index(), x = 'index', y = i)

#TODO: Monte Carlo Standard Error?, plot Convergence, ESS for every layer/bias





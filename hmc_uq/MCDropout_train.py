import os
import numpy as np
import torch
import sparsechem as sc
import argparse
from scipy.special import expit
import glob


TargetID=1908
opt_metric_list= ['acc', 'rocauc', 'loss', 'ace']
sampling_type='uncontrolled'
proportion_sampled_list = ['1.0']

te_fold=0
va_fold=1

#load Datasets
X_singleTask=sc.load_sparse('/home/rosa/git/AIDDProject/data_chembl/files_data_folding_current/chembl_29_X.npy')
Y_singleTask=sc.load_sparse('/home/rosa/git/AIDDProject/data_chembl/files_data_folding_current/chembl_29_thresh.npy')[:,TargetID]
folding=np.load("/home/rosa/git/AIDDProject/data_chembl/files_data_folding_current/folding.npy")

#test Data
X_test_np=X_singleTask[folding==te_fold].todense()
Y_test_np=Y_singleTask[folding==te_fold].todense()

X_validation_np=X_singleTask[folding==va_fold].todense()
Y_validation_np=Y_singleTask[folding==va_fold].todense()

#filter for nonzero values in Validation data
nonzero=np.nonzero(Y_validation_np)[0]
X_validation_filtered=X_validation_np[nonzero]

#filter for nonzero values in test data
nonzero=np.nonzero(Y_test_np)[0]
X_test_filtered=X_test_np[nonzero]
Y_test_filtered=Y_test_np[nonzero]
Y_test_filtered[Y_test_filtered==-1]=0



#to Torch
X_test=torch.from_numpy(X_test_filtered).float()
Y_test=torch.from_numpy(Y_test_filtered)
X_val=torch.from_numpy(X_validation_filtered).float()
num_input_features=X_test.shape[1]

te_fold=0
val_fold=1
epoch_number=400
num_output_features=1
batch_size=200

class Net(torch.nn.Module):
        def __init__(self, input_features, hidden_sizes, output_features, dropout):
            super().__init__()
            self.input_features=input_features
            self.hidden_sizes=hidden_sizes
            self.output_features=output_features
            self.input=torch.nn.Linear(in_features=input_features, out_features=hidden_sizes)
            self.relu=torch.nn.ReLU()
            self.dropout=torch.nn.Dropout(p=0.8)
            self.output=torch.nn.Linear(in_features=hidden_sizes, out_features=output_features)
            
        def forward(self, x, return_hidden):
            fc=self.input(x)
            a=self.relu(fc)
            dr=self.dropout(a)
            out=self.output(dr)
            
            if return_hidden==1:
                return out, a
            elif return_hidden==0:
                return out
            

for proportion_sampled in proportion_sampled_list:
    for opt_metric in opt_metric_list:
        print('MODEL: ', 'propSampled: ', proportion_sampled, ', metric: ', opt_metric)
        i=0
        nr_models=100
        for i in range(10):
        
            models_path="/home/rosa/git/AIDDProject/SingleTask/subsampling/models/"+sampling_type+"_subsampling/"+str(TargetID)+"/hp_optim_"+str(opt_metric)+"/propSampled_"+str(proportion_sampled)+"/"
            file_path= glob.glob(models_path + '/SingleTask_rep'+str(i)+'_*')[0]
            dropout = float(file_path.split('do')[1].split('_')[0])
            hidden_sizes = int(file_path.split('hidden_sizes')[1].split('_')[0])
            net=Net(hidden_sizes=hidden_sizes, input_features=num_input_features, output_features=num_output_features, dropout=dropout)
            #load_model
            state_dict = torch.load(file_path)
            net.load_state_dict(state_dict, strict=False)
                
            net.train()
        
            predictions_sum=np.zeros(shape=Y_test_filtered.shape)
            for k in range(nr_models): 
                #predict MC Dropout
                pred_test=net(X_test, return_hidden=0)
                pred_cpu_test=pred_test.detach()
                #average
                predictions_sum+=expit(pred_cpu_test.numpy())
            
            prediction_MCD=predictions_sum/nr_models
            #prediction_file_test="/home/rosa/git/AIDDProject/SingleTask/subsampling/predictions/uncontrolled_subsampling/"+str(TargetID)+"/hp_optim_"+str(opt_metric)+"/propSampled_"+str(proportion_sampled)+"/MCDropout/averages/average_SingleTask_rep"+str(i)+"_ID"+str(TargetID)+"_hidden_sizes"+str(hidden_sizes)+"_do"+str(dropout)+"_wd"+str(weight_decay)+"_fold"+str(te_fold)+"_propsampled"+str(proportion_sampled)+"-MCdropout.npy"
            prediction_file_test= "/home/rosa/git/AIDDProject/SingleTask/subsampling/predictions/uncontrolled_subsampling/"+str(TargetID)+"/hp_optim_"+str(opt_metric)+"/propSampled_"+str(proportion_sampled)+"/MCDropout/averages/average_" + file_path.split('/')[-1][:-2] + 'npy'
            np.save(prediction_file_test, prediction_MCD)
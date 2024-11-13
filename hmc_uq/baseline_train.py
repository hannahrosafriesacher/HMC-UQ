import numpy as np
import torch
import scipy.sparse
import sparsechem as sc
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import pandas as pd
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Training a single-task model.')
parser.add_argument('--TargetID', type=int, default=None)
parser.add_argument('--proportion_sampled', type=str, default=None)
parser.add_argument('--weight_decay', type=float, default=None)
parser.add_argument('--hidden_sizes', type=int, default=None)
parser.add_argument('--dropout', type=float, default=None)
parser.add_argument('--sampling_type', type=str, default=None)
args = parser.parse_args()
TargetID=args.TargetID
proportion_sampled=args.proportion_sampled
sampling_type=args.sampling_type
print('Target_ID: ' +str(TargetID)+'.................................................................................................')
print('proportion_sampled: ' +proportion_sampled+'.................................................................................................')

number_models_std=240

hidden_sizes=args.hidden_sizes
dropout=args.dropout
weight_decay=args.weight_decay
te_fold=0
val_fold=1
epoch_number=800
num_output_features=1
batch_size=200

#load Datasets
X_singleTask=sc.load_sparse('/home/rosa/git/AIDDProject/data_chembl/files_data_folding_current/chembl_29_X.npy')
Y_singleTask=sc.load_sparse('/home/rosa/git/AIDDProject/data_chembl/files_data_folding_current/chembl_29_thresh.npy')[:,TargetID]
folding=np.load('/home/rosa/git/AIDDProject/data_chembl/files_data_folding_current/folding.npy')

#Training Data
X_singleTask_train=np.load('/home/rosa/git/AIDDProject/data_chembl/subsampling_chembl/'+sampling_type+'_subsampling/TargetID_'+str(TargetID)+'/chembl_29_X_'+str(TargetID)+'_training_subsampled_'+proportion_sampled+'.npy', allow_pickle=True)
Y_singleTask_train=np.load('/home/rosa/git/AIDDProject/data_chembl/subsampling_chembl/'+sampling_type+'_subsampling/TargetID_'+str(TargetID)+'/chembl_29_thresh_'+str(TargetID)+'_training_subsampled_'+proportion_sampled+'.npy', allow_pickle=True)
Y_singleTask_train[Y_singleTask_train==-1]=0

#Validation Data
X_validation_np=X_singleTask[folding==val_fold].todense()
Y_validation_np=Y_singleTask[folding==val_fold].todense()

#test Data
X_test_np=X_singleTask[folding==te_fold].todense()
Y_test_np=Y_singleTask[folding==te_fold].todense()

#filter for nonzero values in validation data
nonzero=np.nonzero(Y_validation_np)[0]
X_validation_filtered=X_validation_np[nonzero]
Y_validation_filtered=Y_validation_np[nonzero]
Y_validation_filtered[Y_validation_filtered==-1]=0

#filter for nonzero values in test data
nonzero=np.nonzero(Y_test_np)[0]
X_test_filtered=X_test_np[nonzero]
Y_test_filtered=Y_test_np[nonzero]
Y_test_filtered[Y_test_filtered==-1]=0

#to Torch
X_train=torch.from_numpy(X_singleTask_train).float()
Y_train=torch.from_numpy(Y_singleTask_train)
X_validation=torch.from_numpy(X_validation_filtered).float()
Y_validation=torch.from_numpy(Y_validation_filtered)
X_test=torch.from_numpy(X_test_filtered).float()
Y_test=torch.from_numpy(Y_test_filtered)
num_input_features=X_singleTask_train.shape[1]

torch.set_num_threads(1)

singleModel_valLoss=[]
singleModel_testLoss=[]
for single_model in range(number_models_std):
    print("Single Model", single_model)

    #Model
    class Net(torch.nn.Module):
        def __init__(self, input_features, hidden_sizes, output_features, dropout):
            super(Net,self).__init__()
            self.input_features=input_features
            self.hidden_sizes=hidden_sizes
            self.output_features=output_features
            self.dropout=dropout
            self.net = torch.nn.Sequential(
                #SparseLinearLayer,
                torch.nn.Linear(in_features=input_features, out_features=hidden_sizes),
                #Relu,
                torch.nn.ReLU(),
                #Dropout
                torch.nn.Dropout(p=dropout),
                #Linear,
                torch.nn.Linear(hidden_sizes, output_features),
            )
            
        def forward(self, x):
            return self.net(x)
    #training loop
    net=Net(hidden_sizes=hidden_sizes, input_features=num_input_features, output_features=num_output_features, dropout=dropout)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=2e-4, weight_decay=weight_decay)
    valLoss_overEpochs=[]
    testLoss_overEpochs=[]
    

    for epoch in range(epoch_number):  # loop over the dataset multiple times 
        permutation = torch.randperm(X_train.size()[0])
        i=0
        for i in range(0,X_train.size()[0], batch_size):
            optimizer.zero_grad()
            # get the inputs; data is a list of [inputs, labels]
            indices = permutation[i:i+batch_size]
            inputs, labels = X_train[indices], Y_train[indices]        
            # forward + backward + optimizer
            net.eval()
            outputs = net(inputs)
            net.train()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        #get loss of each epoch for plotting convergence
        net.eval()

        #predict val
        pred_val=net(X_validation)
        valLoss=criterion(pred_val, Y_validation).detach().item()   
        valLoss_overEpochs.append(valLoss)

        #predict test
        pred_test=net(X_test)
        testLoss=criterion(pred_test, Y_test).detach().item()   
        testLoss_overEpochs.append(testLoss)
        
        #early stopping
        if len(valLoss_overEpochs)>2:
            if valLoss_overEpochs[epoch]>valLoss_overEpochs[epoch-1]:
                break
    #Save model
    '''model_file='/home/rosa/git/AIDDProject/SingleTask_subsampling/models/'+sampling_type+'_subsampling/'+str(TargetID)+'/propSampled_'+proportion_sampled+'_unregularized/hs_'+str(hidden_sizes)+'/SingleTask_rep'+str(single_model)+'_ID'+str(TargetID)+'_hidden_sizes'+str(hidden_sizes)+'_do'+str(dropout)+'_wd'+str(weight_decay)+'_valfold'+str(val_fold)+'_tefold'+str(te_fold)+'_propsampled'+proportion_sampled+'.pt'
    torch.save(net.state_dict(), model_file)'''

    #Save predictions
    '''pred_cpu_val=pred_val.detach()
    predictions_file_val='/home/rosa/git/AIDDProject/SingleTask_subsampling/predictions/'+sampling_type+'_subsampling/'+str(TargetID)+'/propSampled_'+proportion_sampled+'/valFold/SingleTask_rep'+str(10+single_model)+'_ID'+str(TargetID)+'_hidden_sizes'+str(hidden_sizes)+'_do'+str(dropout)+'_wd'+str(weight_decay)+'_fold'+str(val_fold)+'_propsampled'+proportion_sampled+'.npy'
    np.save(predictions_file_val, pred_cpu_val)'''

    '''pred_cpu_test=pred_test.detach()
    predictions_file_test='/home/rosa/git/AIDDProject/SingleTask_subsampling/predictions/'+sampling_type+'_subsampling/'+str(TargetID)+'/propSampled_'+proportion_sampled+'/teFold/SingleTask_rep'+str(10+single_model)+'_ID'+str(TargetID)+'_hidden_sizes'+str(hidden_sizes)+'_do'+str(dropout)+'_wd'+str(weight_decay)+'_fold'+str(te_fold)+'_propsampled'+proportion_sampled+'.npy'
    np.save(predictions_file_test, pred_cpu_test)'''

    pred_cpu_test=pred_test.detach()
    predictions_file_test='/home/rosa/git/AIDDProject/SingleTask/subsampling/predictions/'+sampling_type+'_subsampling/'+str(TargetID)+'/propSampled_'+proportion_sampled+'/ensemble/SingleTask_rep'+str(10+single_model)+'_ID'+str(TargetID)+'_hidden_sizes'+str(hidden_sizes)+'_do'+str(dropout)+'_wd'+str(weight_decay)+'_fold'+str(te_fold)+'_propsampled'+proportion_sampled+'.npy'
    np.save(predictions_file_test, pred_cpu_test)


    '''#getting list with losses of each params combination (take loss of last epoch)
    singleModel_valLoss.append(valLoss_overEpochs[-1])               
    singleModel_testLoss.append(testLoss_overEpochs[-1])
valLoss_average=np.mean(singleModel_valLoss)
valLoss_std=np.std(singleModel_valLoss)
testLoss_average=np.mean(singleModel_testLoss)
testLoss_std=np.std(singleModel_testLoss)

print('valLOSS average: ', valLoss_average)
print('valLOSS std: ', valLoss_std)
print('stdLOSS average: ', testLoss_average)
print('stdLOSS std: ', testLoss_std)'''

    

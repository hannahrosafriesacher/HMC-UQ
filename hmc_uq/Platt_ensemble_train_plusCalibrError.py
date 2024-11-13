import sys
sys.path.insert(1, '/home/rosa/git/KDE/ecekde')
from ece_kde import get_ece_kde
from ece_kde import get_bandwidth
import sparsechem as sc
import numpy as np
import scipy
import scipy.special
from sklearn.linear_model import LogisticRegression, BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from sklearn.utils.estimator_checks import check_estimator
from sklearn.metrics import mean_squared_error as mse
import argparse
import os
from scipy.special import logit
import torch


target_list = [1133, 532, 1908]
sampling_type='uncontrolled'
from_ensemble = False

def calcCalibrationErrors(y_true, y_score, num_bins):
    '''
    Args:
    y_true: matrix of observations ((type:numpy.ndarray in shape (n,)))
    y_score: matrix of logit scores (type:numpy.ndarray in shape (n,)))
    y_bin: number of bins to which datapoints are assigned
    
    Returns a list of the Expected Calibration Error (ECEperTarget) and the Adaptive Calbration Error (ACEperTarget) for every target for a classificatino task.
    
    '''        
    #split input array according to predictions in equally spaced bins
    def split_arrays_ECE(true_labels, predictions, bounds, index):
        true_labels_split=true_labels[np.logical_and(predictions>bounds[index], predictions<=bounds[index+1])]
        predictions_split=predictions[np.logical_and(predictions>bounds[index], predictions<=bounds[index+1])]
        bin_size=true_labels_split.shape[0]
        return true_labels_split, predictions_split, bin_size
    
    #split input array according to predictions in equally sized bins
    def split_arrays_ACE(true_labels, predictions, nr_bins, index):
        true_labels_split=np.array_split(true_labels, nr_bins)[index]
        predictions_split=np.array_split(predictions, nr_bins)[index]
        bin_size=true_labels_split.shape[0]
        return true_labels_split, predictions_split, bin_size

    #Calculate positive ratio (=accuracy)
    #if there are no measurements (=no predictions) in a split: add 0 to acc list
    #Note: if 0 is added to the list, the difference between acc and conf is the conf of this split
    def calculate_posRatio(input_arr):
        return (input_arr==1).sum()/input_arr.shape[0]

        
    #calculate Calibration Error (Nixon et al.: https://doi.org/10.48550/arXiv.1904.01685):
    def calculate_error(posRatio_bin, meanProb_bin, bin_size):
        return (np.abs(np.array(posRatio_bin)-np.array(meanProb_bin))*bin_size)
            #      |               acc(b)-         conf(n)| * nb



    #bounds for bins used for ECE caluculation
    bounds_ECE=list(np.linspace(0,1,num_bins+1))
    
    y_score_target=y_score
    y_true_target=y_true
    target_size=y_true.shape[0]
    
    #if the target has no measurements (target_size=0), the error of this target is nan.
    if target_size==0:
        ece_target=np.nan
        ace_target=np.nan
    
    
    #if there are measurements for the target (target_size>0): calculate ECE/ACE
    if target_size!=0:            
        #for ACE: sort y_true and y_score file by ascending probablity values in y_score file:
        index_sort_y_score=np.argsort(y_score_target, axis=0)
        y_score_sorted=y_score_target[index_sort_y_score]
        y_true_sorted=y_true_target[index_sort_y_score]    
        
        bin_ind=0
        ece=0
        ace=0

        #iterate over all bins
        for bin_ind in range(num_bins):

            #for current bin...
            #...split dataset and select split
            
            ECE_collector=split_arrays_ECE(y_true_target, y_score_target, bounds_ECE, bin_ind)
            ACE_collector=split_arrays_ACE(y_true_sorted, y_score_sorted, num_bins, bin_ind)

            if ECE_collector[2]!=0:
                #...obtain positive ratio (=acc calculated from true values) for current bin if bin has measurements
                posRatio_ECE=calculate_posRatio(ECE_collector[0])
                #...obtain probablity mean (=conf calculated from predictions) for each split for current bin if bin has measurements
                meanProb_ECE=np.mean(ECE_collector[1])
                #...calculate ECE for current target and sum error over all bins:
                ece += calculate_error(posRatio_bin=posRatio_ECE, meanProb_bin=meanProb_ECE, bin_size=ECE_collector[2])
            
                            
            if ACE_collector[2]!=0:
                #...obtain positive ratio (=acc calculated from true values) for current bin if bin has measurements
                posRatio_ACE=calculate_posRatio(ACE_collector[0])
                #...obtain probablity mean (=conf calculated from predictions) for each split for current bin if bin has measurements
                meanProb_ACE=np.mean(ACE_collector[1])
                #...calculate ACE for current bin and sum error over all bins with measurements:
                ace += calculate_error(posRatio_bin=posRatio_ACE, meanProb_bin=meanProb_ACE, bin_size=ACE_collector[2])

        #the final ECE/ACE is divided by number of datapoints in a target
        ece_target=ece/target_size
        ace_target=ace/target_size

    return ece_target, ace_target

def Brier(y_true, y_score):
    ''' Calculatin Brier-score: MSE between the prediction and the binary label
    '''
    return ((y_score - y_true)**2).mean()

for target in target_list:
    TargetID=target
    #-------------------calculate ACE/ACE for NN with Bayesian Layer------------------------------------
    #load y_file
    y_class=sc.load_sparse('/home/rosa/git/AIDDProject/data_chembl/files_data_folding_current/chembl_29_thresh.npy')
    folding=np.load('/home/rosa/git/AIDDProject/data_chembl/files_data_folding_current/folding.npy')
    y_class_val_TargetID=y_class[:,TargetID][folding==1].todense().A
    y_class_test_TargetID=y_class[:,TargetID][folding==0].todense().A

    nonzero_val=np.nonzero(y_class_val_TargetID)[0]
    y_class_val_nonzero=y_class_val_TargetID[nonzero_val]
    y_class_val_nonzero[y_class_val_nonzero==-1]=0
    y_class_val_nonzero=y_class_val_nonzero.flatten()

    nonzero_test=np.nonzero(y_class_test_TargetID)[0]
    y_class_test_nonzero=y_class_test_TargetID[nonzero_test]
    y_class_test_nonzero[y_class_test_nonzero==-1]=0
    y_class_test_nonzero=y_class_test_nonzero.flatten()


    subsampling_list=[1.0]
    metrics= ['acc', 'rocauc', 'loss', 'ace']
    for proportion_sampled in subsampling_list:
        for opt_metric in metrics:
            if from_ensemble == False:
                print('MODEL: ', 'proportion Sampled: ', proportion_sampled, ', metric: ', opt_metric)
                dirname_val="/home/rosa/git/AIDDProject/SingleTask/subsampling/predictions/"+sampling_type+"_subsampling/"+str(TargetID)+"/hp_optim_"+str(opt_metric)+"/propSampled_"+str(proportion_sampled)+"/ensemble/averages_val/"
                list_file_val=sorted(os.listdir(dirname_val))
                dirname_te="/home/rosa/git/AIDDProject/SingleTask/subsampling/predictions/"+sampling_type+"_subsampling/"+str(TargetID)+"/hp_optim_"+str(opt_metric)+"/propSampled_"+str(proportion_sampled)+"/ensemble/averages_test/"
                list_file_te=sorted(os.listdir(dirname_te))
            ECE_list=[]
            ACE_list=[]
            AUCROC_list=[]
            AUCPR_list=[]
            KDE_list=[]
            Brier_list = []
            for i in range(5):
                y_hat_vafold=logit(np.load(dirname_val+ list_file_val[i])).reshape(-1, 1)
                y_hat_tefold=logit(np.load(dirname_te+ list_file_te[i])).reshape(-1, 1)
                #Train model on Validation fold
                lr=LogisticRegression().fit(y_hat_vafold, y_class_val_nonzero)

                #################################################################
                #Predict Test fold
                y_platt_te=lr.predict_proba(y_hat_tefold)[:, 1]
                y_platt_te=y_platt_te.flatten()

                #AUC_ROC
                auc_roc_target_te=roc_auc_score(y_class_test_nonzero, y_platt_te)

                #PR_AUC
                precision, recall, thresholds = precision_recall_curve(y_class_test_nonzero, y_platt_te)
                auc_pr_te = auc(recall, precision)

                #ACE, ECE
                ECE_target_te, ACE_target_te= calcCalibrationErrors(y_class_test_nonzero, y_platt_te, 10)
                ECE_list.append(ECE_target_te)
                ACE_list.append(ACE_target_te)
                AUCROC_list.append(auc_roc_target_te)
                AUCPR_list.append(auc_pr_te)

                #Single_task_KDE
                y_hat_torch=torch.from_numpy(y_platt_te).unsqueeze(1)
                bw=get_bandwidth(y_hat_torch,  device='cpu')
                #bw=0.01
                y_class_torch=torch.from_numpy(y_class_test_nonzero)
                kde=get_ece_kde(y_hat_torch,y_class_torch, bw, p=1, mc_type='canonical', device='cpu')
                KDE_list.append(kde)

                brier = Brier(y_class_test_nonzero, y_platt_te)
                Brier_list.append(brier)

                np.save("/home/rosa/git/AIDDProject/SingleTask/subsampling/predictions/uncontrolled_subsampling/"+str(TargetID)+"/hp_optim_"+str(opt_metric)+"/propSampled_1.0/ensemble_platt/" + list_file_te[i], y_platt_te)

            np.save("/home/rosa/git/AIDDProject/SingleTask/subsampling/data_csv_files/"+sampling_type+"/"+str(TargetID)+"/hp_optim_"+str(opt_metric)+"/ensemble_platt/ECE/"+str(proportion_sampled)+"-ECE_"+list_file_te[i], ECE_list)
            np.save("/home/rosa/git/AIDDProject/SingleTask/subsampling/data_csv_files/"+sampling_type+"/"+str(TargetID)+"/hp_optim_"+str(opt_metric)+"/ensemble_platt/ACE/"+str(proportion_sampled)+"-ACE_"+list_file_te[i], ACE_list)
            np.save("/home/rosa/git/AIDDProject/SingleTask/subsampling/data_csv_files/"+sampling_type+"/"+str(TargetID)+"/hp_optim_"+str(opt_metric)+"/ensemble_platt/KDE/optimized/"+str(proportion_sampled)+"-KDE_"+list_file_te[i], KDE_list)
            np.save("/home/rosa/git/AIDDProject/SingleTask/subsampling/data_csv_files/"+sampling_type+"/"+str(TargetID)+"/hp_optim_"+str(opt_metric)+"/ensemble_platt/ROC_AUC/"+str(proportion_sampled)+"-ROC_AUC_"+list_file_te[i], AUCROC_list)
            np.save("/home/rosa/git/AIDDProject/SingleTask/subsampling/data_csv_files/"+sampling_type+"/"+str(TargetID)+"/hp_optim_"+str(opt_metric)+"/ensemble_platt/PR_AUC/"+str(proportion_sampled)+"-AUCPR_"+list_file_te[i], AUCPR_list)
            np.save("/home/rosa/git/AIDDProject/SingleTask/subsampling/data_csv_files/"+sampling_type+"/"+str(TargetID)+"/hp_optim_"+str(opt_metric)+"/ensemble_platt/BRIER/"+str(proportion_sampled)+"-BRIER_"+list_file_te[i], Brier_list)






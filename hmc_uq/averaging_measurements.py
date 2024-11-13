#This file obtains the average of multiple prediction files in a folder.
import sparsechem as sc
import numpy as np
import os
from scipy.special import expit

list_prop=[1.0, 0.7, 0.4, 0.1, 0.05, 0.01]
Type='MCDropout'
Target_ID=1133


#load list of file name of small models
#MC_Dropout
for prop_sampled in list_prop:
    for nr in range(10):
        list_files=os.listdir('/home/rosa/git/AIDDProject/SingleTask/subsampling/predictions/'+Type+'_subsampling/'+str(Target_ID)+'/propSampled_'+str(prop_sampled)+'/MCDropout/rep'+str(nr)+'_predictions/')
        print('list_files: ', len(list_files))
        list_prediction_files=[]

        #loading prediction files to list
        for i in range(0, len(list_files)):
            #print(list_files[i])
            current_file=expit(np.load('/home/rosa/git/AIDDProject/SingleTask/subsampling/predictions/'+Type+'_subsampling/'+str(Target_ID)+'/propSampled_'+str(prop_sampled)+'/MCDropout/rep'+str(nr)+'_predictions/'+str(list_files[i]), allow_pickle=True))
            list_prediction_files.append(current_file)
        print(len(list_prediction_files))

        #print('average predictions_AUC-------------')
        cumm_pred_now=np.zeros_like(list_prediction_files[0])
        for k in range(len(list_prediction_files)):
            cumm_pred_now+=list_prediction_files[k]
        print(cumm_pred_now.shape)
        print(list_prediction_files[0].shape)
        mean=cumm_pred_now/len(list_prediction_files)
        print(mean.shape)
        np.save('/home/rosa/git/AIDDProject/SingleTask/subsampling/predictions/'+Type+'_subsampling/'+str(Target_ID)+'/propSampled_'+str(prop_sampled)+'/MCDropout/averages/average_rep'+str(nr)+list_files[i][-63:], mean)




################################################################################################################################
################################################################################################################################
#Ensemble
#over multiple folders (e.g. for obtaining averaged of several folders)

proportion_sampled=[1.0, 0.7, 0.4, 0.1, 0.05, 0.01]
Type='uncontrolled'
Target_ID=1133
numbers= ["1", "2", "3", "4", "5"]
for proportion in proportion_sampled:
    for number in numbers:
        #load list of file name of small models
        list_files=os.listdir('/home/rosa/git/AIDDProject/SingleTask/subsampling/predictions/'+Type+'_subsampling/'+str(Target_ID)+'/propSampled_'+str(proportion)+'/ensemble/ensemble_50BaseEstimators/50BaseEstimators_'+number)
                                
        print('/home/rosa/git/AIDDProject/SingleTask/subsampling/predictions/'+Type+'_subsampling/'+str(Target_ID)+'/propSampled_'+str(proportion)+'/ensemble/ensemble_50BaseEstimators/50BaseEstimators_'+number)
        print(len(list_files))
        list_prediction_files=[]

        #loading prediction files to list
        for i in range(0, len(list_files)):
            current_file=expit(np.load('/home/rosa/git/AIDDProject/SingleTask/subsampling/predictions/'+Type+'_subsampling/'+str(Target_ID)+'/propSampled_'+str(proportion)+'/ensemble/ensemble_50BaseEstimators/50BaseEstimators_'+number+'/'+str(list_files[i]), allow_pickle=True))
            list_prediction_files.append(current_file)
        cumm_pred_now=np.zeros_like(list_prediction_files[0])
        for k in range(len(list_prediction_files)):
            cumm_pred_now+=list_prediction_files[k]
        mean=cumm_pred_now/len(list_prediction_files)
        np.save('/home/rosa/git/AIDDProject/SingleTask/subsampling/predictions/'+Type+'_subsampling/'+str(Target_ID)+'/propSampled_'+str(proportion)+'/ensemble/averages/50BBaseEstimators_average_'+number+'_'+str(list_files[i][-60:]), mean)
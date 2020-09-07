# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 13:23:34 2018

@author: asus
"""



from __future__ import division
import os
from sklearn.model_selection import  RepeatedKFold
from functions import load_matrices, extract_names, apply_threshold
from functions_2 import PSLMF
from functions import score_to_exact_rank

from functions import compute_evaluation_criteria_across_wholeMatrix
import numpy as np




data_folder = os.path.join(os.path.pardir, 'Datasets') 

observation_mat, cells_sim,cells_sim_2,observation_mat_IC50,drugMat =load_matrices(data_folder) 

seed = [80162,45929]

num_repeats=1 
model = PSLMF(c=1, K1=20, K2=20, r=23, lambda_p=0.6, lambda_l=0.6, alpha=0.4,beta=0.05, theta=1.3, max_iter=1000)

kf = RepeatedKFold(n_splits=10, n_repeats=num_repeats)
F1= 0.0
REC=0.0
Specificity=0.0
MCC=0.0
AUC=0.0
ACC=0.0
precision=0.0



for train_index, test_index, in kf.split(cells_sim, observation_mat):
   
        

    test_location_mat = np.array(observation_mat)
    
    test_location_mat[train_index] = 0
    train_location_mat = np.array(observation_mat - test_location_mat)
   
    true_result = np.array(test_location_mat[test_index])
    
    x = np.repeat(test_index, len(observation_mat[0]))
    y = np.arange(len(observation_mat[0]))
   
    y = np.tile(y, len(test_index))
  
    
    model.fix_model(train_location_mat, train_location_mat,drugMat, cells_sim,cells_sim_2, seed)
    scores = np.reshape(model.predict_scores(zip(x, y)), true_result.shape)
    pred_Ic50_values=np.reshape(model.predict_scores_2(zip(x, y)), true_result.shape)
    prediction = apply_threshold(scores, 0.41)   
   
    precision_new_o,fold_ACC ,fold_F1,REC_new_output,Specificity_new,MCC_new_output,AUC_new = compute_evaluation_criteria_across_wholeMatrix(true_result, prediction,scores)
    F1+=fold_F1
   
    ACC+=fold_ACC
    
    REC+=REC_new_output
    Specificity+=Specificity_new
    MCC+=MCC_new_output
    AUC+=AUC_new
    precision+=precision_new_o
    
   
    
   
    observation_mat_IC50_mat = np.array(observation_mat_IC50)
  
    RR=np.array(observation_mat_IC50_mat[test_index])
   
    pred=scores 
   
       
      
F1=round(F1/(10*num_repeats),2)
ACC=round(ACC/(10*num_repeats),2)

REC=round(REC/(10*num_repeats),2)
MCC=round(MCC/(10*num_repeats),2)
Specificity=round(Specificity/(10*num_repeats),2)
AUC=round(AUC/(10*num_repeats),2)
precision=round(precision/(10*num_repeats),2)



print('F1',F1)
print('ACC', ACC)

print('REC', REC)
print('Specificity', Specificity)
print('MCC', MCC)
print('AUC', AUC)
print('precision', precision)




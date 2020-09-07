# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 13:20:15 2018

@author: asus
"""


from __future__ import division
import os
import numpy as np
import pandas as pd
import scipy.stats
import math
from sklearn import metrics
from sklearn.metrics import confusion_matrix, precision_score, recall_score,roc_auc_score
from sklearn import metrics
def PMLPR_for_evaluation (pred,R):
    all_pre=[]
    minn=float(30.0)
    
    for u in range(len(R)):
        s_u = R[u]
        s_u=np.array(s_u)
        s_u_float=s_u.astype(np.float)
     
    
    for h in range (len(s_u)):
        if(s_u_float[h]<minn):
            minn =float(s_u[h])
                    
   
    
    for u in range(len(R)):
        
        
        s_u = R[u]
        s_u=np.array(s_u)
        s_u_float=s_u.astype(np.float)
        s_u_pred = pred[u,:]
        
        pred_sort=np.sort(s_u_pred)[::-1]
        s_u_sorted=np.sort(s_u_float)[::-1]
        
       
        
        
        total=0.0
        total_1=0.0
        total_2=0.0
        count=0
        for h in range (len(s_u)):
            
            count+=1
            
           # print((len(s_u)-(count-1)))
            item=pred_sort[h]
           # print('item', item)
            indexx=0
            for q in range (len(s_u)):
                if (item)==(s_u_pred[q]):
                    indexx=q
           
            
           # print('indexx', indexx)
            sorat=(float(s_u[indexx])-minn)*(len(s_u)-(count-1))
            makhraj=((float(s_u_sorted[h])-minn)*(len(s_u)-(count-1)))
            
            total_1 += (sorat)
            total_2 +=(makhraj)
           
        total=(total_1  / total_2)
       # print('total', total)
        all_pre.append(total)
    return np.mean(all_pre)



def score_to_exact_rank(s):
    #s=(-1*s)
    p=np.argsort(s)[::-1]
    
    return np.argsort(p)

def cal_exact_avg_ndcg(pred, R):
   
    all_ndcg = []
    
    for u in range(len(R)):
       
       
        
        
        s_u = R[u]
        #print(type(s_u))
        s_u=np.array(s_u)
        s_u_float=s_u.astype(np.float)
        r_u = score_to_exact_rank(s_u_float)
        s_u_pred = pred[u,:]
        r_u_pred = score_to_exact_rank(s_u_pred)
        total=0.0
       
        total_1=0.0
        total_2=0.0
        
        for h in range (len(s_u)):
                  
          
            G_u_max = (np.power(2,float( s_u[h]))) / np.log(r_u[h] + 2)
            G_u = (np.power(2,float( s_u[h]))) / np.log(r_u_pred[h] + 2)
            total_1 += (G_u)
            total_2 +=(G_u_max)
           
        total=(total_1  / total_2) 
       
        all_ndcg.append(total)
    return np.mean(all_ndcg)




def load_matrices(folder):
   
        
    with open(os.path.join(folder, "matrix of 0 and 1.txt"), "r") as raw:
        
        observation_mat = [line.strip("\n").split() for line in raw]
        
    with open(os.path.join(folder, "similarity matrix of drugs.txt"), "r") as raw:
        raw.next()
        
        drug_matt = [line.strip("\n").split()[1:] for line in raw]
        
     
          
        

    with open(os.path.join(folder, "Similarity matrix_based_on feature selection_rna.txt"), "r") as raw:
        #raw.next()
        cell_sim_1 = [line.strip("\n").split()[0:] for line in raw]
        
    with open(os.path.join(folder, "Similarity matrix_based_on_feature_selection_cnv.txt"), "r") as raw:
        #raw.next()
        cell_sim_2 = [line.strip("\n").split()[0:] for line in raw]
        
        
    with open(os.path.join(folder, "Similarity matrix_based_on_feature_selection_mut.txt"), "r") as raw:
        #raw.next()
        cell_sim_3 = [line.strip("\n").split()[0:] for line in raw]
        
        
    with open(os.path.join(folder, "Liu_similarity of IC50 values.txt"), "r") as raw:
        #raw.next()
        cell_sim_4 = [line.strip("\n").split()[0:] for line in raw]
        
        
    with open(os.path.join(folder, "Matrix of k-nearest neighbrs.txt"), "r") as raw:
        #raw.next()
        cell_sim_new = [line.strip("\n").split()[0:] for line in raw]
        


    with open(os.path.join(folder, "CCLE_similarity_based_on_tissue_type.txt"), "r") as raw:
        #raw.next()
        cell_sim_tissue = [line.strip("\n").split()[0:] for line in raw]      
        
        
        
    with open(os.path.join(folder, "matrix of IC50 values.txt"), "r") as raww:
        raww.next()
        print('****************************')
        observation_mat_IC50 = [line.strip("\n").split()[1:] for line in raww]
        
     
   

   
 

    observation_mat = np.array(observation_mat, dtype=np.float64)   
    cell_sim_1 = np.array(cell_sim_1, dtype=np.float64)      
    drug_mat=np.array(drug_matt, dtype=np.float64) 
    
    cell_sim_2 = np.array(cell_sim_2, dtype=np.float64)
    cell_sim_3 = np.array(cell_sim_3, dtype=np.float64) 
    cell_sim_4 = np.array(cell_sim_4, dtype=np.float64) 
    cell_sim_new=np.array(cell_sim_new, dtype=np.float64)
    cell_sim_tissue=np.array(cell_sim_tissue, dtype=np.float64)
   
    cell_lines_sim = ((1* cell_sim_1) + (5*cell_sim_4 )+(2* cell_sim_new)+(1* cell_sim_3)+(1* cell_sim_2)+(1*cell_sim_tissue)) / (11)
    
    
    # test:
    cell_lines_sim_2=((5*cell_sim_new)+(1* cell_sim_1)+(0* cell_sim_2)+(0* cell_sim_tissue)+(0* cell_sim_3))/(6)
    
    
    
   
    drug_mat_main=(drug_mat)
   # drug_mat_main=drug_matt_liu
    
    return observation_mat, cell_lines_sim,cell_lines_sim_2,observation_mat_IC50,drug_mat_main


def extract_names(dataset, folder):
    with open(os.path.join(folder, dataset+"_cell_line_locals.tab"), "r") as raw:
        locations = raw.next().strip("\n").split()
        cell_lines = [line.strip("\n").split()[0] for line in raw]
    return locations, cell_lines


def apply_threshold(scores, thre):
    
    prediction = np.zeros_like(scores)
    for i in range(len(scores)):
        
        for j in range(len(scores[i])):
            if (scores[i][j] <= thre):
                prediction[i][j] = 0
            else: 
                prediction[i][j] = 1
            
    
    return prediction


def compute_evaluation_criteria_across_wholeMatrix(true_result, prediction,scores):
    pres = np.zeros(len(true_result))
    recs = np.zeros(len(true_result))
    ACC=0.0
   
    F1_score=0.0
    REC=0.0
    Specificity=0.0
    MCC=0.0
    
    AUC_new_new=0.0
    precision=0.0
    co=0.0
    tp_final=0
    tn_final=0
    fp_final=0
    fn_final=0
    fmeas=[]
    true_auc=[]
    score_auc=[]
    
    
    for i in range(len(true_result[0])):
        
        
        fpr, tpr, thresholds = metrics.roc_curve(true_result[:,i], scores[:,i], pos_label=1)
      
        
        if (math.isnan(metrics.auc(fpr, tpr))!=True):
            co=co+1
            AUC_new_new+= metrics.auc(fpr, tpr)
       
        
        true_auc.append(true_result[:,i])
        score_auc.append(scores[:,i])
        
       
        
        returned= confusion_matrix(true_result[:,i], prediction[:,i]).ravel()
        if len(returned) == 4:
            tn, fp, fn, tp = returned
            
        if len(returned) == 1:
           tn= returned
           fp=0
           fn=0
           tp=0
        
        tp_final+=tp
        tn_final+=tn
        fp_final+=fp
        fn_final+=fn
        
  
      
    ACC=((tp_final+tn_final)/ (tp_final + fp_final + fn_final+tn_final))
    F1_score =((2*tp_final) / ((2*tp_final)+fp_final+ fn_final))
    REC=((tp_final)/(tp_final+fn_final))
       
    precision=((tp_final)/(tp_final+fp_final))
   
        
    MCC=(((tp_final*tn_final)-(fp_final*fn_final))/(np.sqrt((tp_final+fp_final)*(tp_final+fn_final)*(fp_final+tn_final)*(fn_final+tn_final))))
    
  
        
      
    Specificity=((tn_final)/(tn_final+fp_final))
    
    
    AUC_new_new=AUC_new_new/(co)
    
    return round(precision,2), round(ACC,2),round(F1_score,2),round(REC,2),round(Specificity,2),round(MCC,2),round(AUC_new_new,2)


    
    
 



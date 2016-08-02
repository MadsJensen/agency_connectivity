# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 09:17:15 2016

@author: au194693
"""
results = np.empty([2049])
for i in range(f2.shape[1]):
    results[i] = np.abs(np.mean(np.exp(
                        1j*(np.angle(f2[:,  i]) -
                        np.angle(f3[:, i])))))
                        
                       
result = np.empty([f1.shape[1], f1.shape[2]])
 
for ii in range(f1.shape[1]):
    for i in range(result.shape[-1]):
        result[ii, i] = np.abs(np.mean(np.exp(
        1j*(np.angle(f1[:, ii, i])))))


    result = result.mean(axis=0).squeeze()                    
    
    
lbl = []
for i in range(len(labels["labels"])):
    lbl.append(labels["labels"][i][0][0])

lbl_clean = []
for l in lbl:
    tmp = l.split()
    lbl_clean.append("_".join(tmp[:2]))
    



result = np.empty([f4.shape[0]])

for ii in range(f4.shape[1]):
    for i in range(f4.shape[0]):
        result[i] = np.abs(np.mean(np.exp(
            1j*(np.angle(f4[i, 52, :]) -
            np.angle(f4[i, 71, :])))))

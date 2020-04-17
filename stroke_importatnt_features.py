import math
import xlrd
import xlsxwriter
import pandas as pd
import numpy as np
import mixed
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid")

data = np.array(pd.read_fwf("processed.cleveland.data"))#loads data
#############################################################################
#Module for preprocessing
def preprocess(data):
    temp_arr = []
    for rows in range(0, len(data)):
        if rows == 0:
            acc = data[rows,:][0];acc = acc.split(',')
            for item in range(0,len(acc)):
                 if acc[item] == '?':
                     temp_arr.append(new_data[rows-1,item])
                 else:
                     temp_arr.append(float(acc[item]))
            new_data = np.array(temp_arr).reshape(1,len(temp_arr) );temp_arr = []
        
        else:
            acc = data[rows,:][0];acc = acc.split(',')
            for item in range(0,len(acc)):
                if acc[item] == '?':
                    temp_arr.append(new_data[rows-1,item])
                else:
                    temp_arr.append(float(acc[item]))
    
            new_data = np.vstack(( new_data,  np.array(temp_arr).reshape(1,len(temp_arr) )  ))
            temp_arr = []
    return new_data
        
 #######################################################################################
if __name__ == '__main__':
    data = preprocess(data);acc = [];plt.figure(figsize=(30,15))
    input = data[:,0:data.shape[1]-1]
    target_labels = data[:,data.shape[1]-1 ]
    names=['Age', 'Sex','Chest pain type','Resting blood pressure','Serum cholestoral in mg/dl ','Fasting blood sugar','Restecg (resting electrocardiographic results )','thatach (maximum heart rate achieved)','exang (exercise induced angina)','Oldpeak(depression induced by exercise relative to rest)','the slope of the peak exercise ST segment','ca(number of major vessels (0-3) colored by flourosopy)','thal']

    
    
    for col in range(0, input.shape[1]):
        MI = mixed.Mixed_KSG(input[:,col],target_labels)
        acc.append(MI)
        print('Mutual information for',names[col],'is', MI)
    sns.barplot(x=acc, y= names, dodge=False,
                label = names, color="b")
    plt.xlabel('Mutual information')
    plt.savefig('mutual info',format='eps')
  


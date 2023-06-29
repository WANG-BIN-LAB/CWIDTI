import pandas as pd
import numpy as np

dataset='drugbank'

path_to_string_nets = './data/'+dataset+'/'
string_nets = [
    'target_mismatch_kernel_3_1_similarity',
    'target_mismatch_kernel_3_2_similarity',
    'target_mismatch_kernel_4_1_similarity',
    'target_mismatch_kernel_4_2_similarity',
    'target_spectrum_kernel_3_similarity',
    'target_spectrum_kernel_4_similarity',
               ]

filenames = []
for net in string_nets:
        filenames.append(path_to_string_nets + net + '.txt')
for k in range(0,len(string_nets)):
      infile=filenames[k]
      m1 = pd.read_csv('./data/'+dataset+'/target.txt', names=['d1'])
      m1['id1'] = range(1, len(m1) + 1)
      m2 = pd.read_csv('./data/'+dataset+'/target.txt', names=['d2'])
      m2['id2'] = range(1, len(m2) + 1)
      m3 = pd.read_csv(infile, sep='\t',names=['d1', 'd2', 'score'])
      data = pd.merge(m1, m3, how='inner')
      data = pd.merge(m2, data, how='inner')
      data = data.iloc[:, [3, 1, 4]]
      data.to_csv('./data1/'+dataset+'/'+string_nets[k]+"_AllSimScores_T1.txt" , index=None, header=False, sep='\t')
      print(string_nets[k] + '------------table over')
      k=k+1



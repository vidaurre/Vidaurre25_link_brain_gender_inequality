# This script generates random brain data so that the scripts can be tested

import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

window_length = 10 # width of the age windows

# paths and names
directory = '/Users/au654912/CloudOneDrive/Work/data/main_volbrain_repo/'
directory_braindata = directory + 'volbrain_repo/preprocessed_data/'
datafile = directory_braindata  + 'vol2Brain_931.npz'
datafile_out = directory_braindata  + 'vol2Brain_931_synth.npz'

# load brain data
datvb = np.load(datafile, allow_pickle=True)
confounds = np.copy(datvb['confounds'])
braindata = np.copy(datvb['braindata'])
countries_vb_ = np.copy(datvb['countries'])
labels = np.copy(datvb['labels'])
(Nsubj,p) = braindata.shape

# get brain variables
index_str = []
for j in range(63,braindata.shape[1]):
    if ('cm3' in labels[j]) and ('Cerebellar' not in labels[j]) and \
        ('Temporal' not in labels[j]) and ('Parietal' not in labels[j]) and \
        ('Occipital' not in labels[j]) and ('Frontal' not in labels[j]) and \
        ('Limbic' not in labels[j]) and ('Basal Forebrain' not in labels[j]) and \
        ('Ventricle' not in labels[j]) and ('CSF' not in labels[j]) and \
        ('Insular' not in labels[j]) and ('total' not in labels[j]): 
        index_str.append(j)

braindata_synth = np.zeros(braindata.shape)
confounds_synth = np.zeros(confounds.shape)

cov = np.cov(braindata[:,index_str],rowvar=False) + 0.001 * np.eye(len(index_str))
m = np.mean(braindata[:,index_str],axis=0) 
braindata_synth[:,index_str] = np.random.multivariate_normal(m,cov,Nsubj)
confounds_synth[:,0:2] = confounds[:,0:2]

cov = np.cov(confounds[:,2:],rowvar=False) + 0.001 * np.eye(confounds.shape[1]-2)
m = np.mean(confounds[:,2:],axis=0) 
confounds_synth[:,2:] = np.random.multivariate_normal(m,cov,Nsubj)

print(datafile_out)
np.savez(datafile_out, confounds=confounds,confounds_labels=np.copy(datvb['confounds_labels']),
        braindata=braindata,countries=np.copy(datvb['countries']),labels=labels)

# All the analyses in this paper are done by age windows. 
# This script puts the data in the format that will then be used in the analysis. 
#
# Specifically, it computes Nsubjecs-by-Nsubjects matrices of distances between 
# each pair of subjects (one female vs one male), for each age window, separately for each country, and for each types of measurement: 
# - Grey matter volumes across cortical and subcortical regions: 
#   Each subject has a vector of anatomical measurements that we rescale to sum up to one. 
#   The Euclidean distance between these vectors define the entries of the "brain" distance matrices
# - Age: difference in age between each pair
# - ICV: difference in intracranial volume between each pair
# - SNR: difference in signal-to-noise ratio between each pair
# The age-specific, metric-specific distance matrices are vectorized into a NPairs-by-1 vector, one per country, where Npairs is the number of female-male pairs. 
# For each country, socioeconomical variables are gathered, also into NPairs-by-1 vectors, 
# which in this case contain the same values for all pairs within the country. 


import sys
import numpy as np
import country_converter
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

real_data = (len(sys.argv)==1) or int(sys.argv[1])

if not real_data: print('Running on synthetic data')

window_length = 10 # width of the age windows

# paths and names
directory = '/Users/au654912/CloudOneDrive/Work/data/main_volbrain_repo/'
directory_out = directory + 'volBrain_repo/preprocessed_data/'
directory_socioecon = directory + 'socioeconomic_data_repo/preprocessed_data/'
directory_braindata = directory + 'volbrain_repo/preprocessed_data/'
if real_data: datafile_out = directory_out + 'sexDist_vol2Brain_931.npz'
else: datafile_out = directory_out + 'sexDist_vol2Brain_931_synth.npz'

# load socioeconomical data
datafile = directory_socioecon + 'wid.npz'
datawid = np.load(datafile, allow_pickle=True)
X = datawid['X']
labels_vars = np.copy(datawid['labels_vars'])
countries_wid = np.copy(datawid['country_codes'])

GI = datawid['X'][:,labels_vars == 'GI_i'] # gender inequality
II = datawid['X'][:,labels_vars == 'II1_i'] # income inequality
WI = datawid['X'][:,labels_vars == 'WI1_i'] # wealth inequality 
GII = datawid['X'][:,labels_vars == 'GII'] # gender inequality i`ndex
Gini = datawid['X'][:,labels_vars == 'Gini'] # gender inequality index

# GDP stuff, get the first PC over the three related variables, which are very correlated anyway
pca = PCA()
pca.fit(X[:,(0,2,4)])
E = pca.transform(X[:,(0,2,4)])[:,0] 
E = np.expand_dims(E,1)

# load volBrain data
if real_data: datafile = directory_braindata + 'vol2Brain_931.npz'
else: datafile = directory_braindata + 'vol2Brain_931_synth.npz'
datvb = np.load(datafile, allow_pickle=True)
confounds = np.copy(datvb['confounds'])
braindata = np.copy(datvb['braindata'])
countries_vb_ = np.copy(datvb['countries'])
labels = np.copy(datvb['labels'])
(Nsubj,p) = braindata.shape

# Retrieve subjects' countries and deal with inconsistencies in the naming.
ucountries_vb = np.unique(countries_vb_)
countries_vb = np.empty(countries_vb_.shape,dtype=object)
countries_vb[:] = '-'
for ic in range(len(ucountries_vb)):
    c = ucountries_vb[ic]
    c0 = c
    if c == '-': 
        countries_vb[countries_vb_ == c] = '-'
        continue
    if c[-4:] == 'kiye': 
        c = 'Turkey'
    elif c[-4:] == 'Xico':
        c = 'Mexico'
    elif c == 'Brasil':
        c = 'Brazil'
    elif c[0:3] == 'Esp':
        c = 'Spain'
    elif c == 'Ltaly':
        c = 'Italy'
    elif c == 'Polska': 
        c = 'Poland'
    elif c == 'Valencia':
        c = 'Spain'
    cc = country_converter.convert(names=c, to='ISO2')
    countries_vb[countries_vb_ == c] = cc

ucountries_vb = np.unique(countries_vb)
Ncountries = len(ucountries_vb)

D = np.zeros((Nsubj,3)) # expanded matrix with the socioeconomical info
C = np.empty(Nsubj,dtype=object) # country label per subject
N_per_country = np.zeros(Ncountries)

# populate D and C
for ic in range(Ncountries):
    c = ucountries_vb[ic]
    idc_vb = (countries_vb == c)
    idc_wid = np.where(countries_wid == c)[0]
    if len(idc_wid) == 0:
        C[idc_vb] = ''
        continue    
    D[idc_vb,0] = GII[idc_wid[0],:]
    D[idc_vb,1] = Gini[idc_wid[0],:]
    D[idc_vb,2] = E[idc_wid[0],:]
    C[idc_vb] = c
    N_per_country[ic] = np.sum(idc_vb)

# remove countries with no valid subjects
ucountries_vb = ucountries_vb[N_per_country>0]
Ncountries = len(ucountries_vb)
take = ((C != '') & (C != '-'))
countries_vb = countries_vb[take]
D = D[take,:]
C = C[take] # actually the same than countries_vb
braindata = braindata[take,:]
confounds = confounds[take,:]

# socioeconomical info per country
MD = np.zeros((Ncountries,3))
for ic in range(Ncountries):
    c = ucountries_vb[ic]
    idc_vb = (countries_vb == c)
    idc_wid = np.where(countries_wid == c)[0]
    if len(idc_wid) == 0:
        C[idc_vb] = ''
        continue
    MD[ic,0] = GII[idc_wid[0],0]
    MD[ic,1] = Gini[idc_wid[0],0]
    MD[ic,2] = np.sum(idc_vb)
# standardize across countries
MD[:,0] = (MD[:,0] - np.mean(MD[:,0])) / np.std(MD[:,0])
MD[:,1] = (MD[:,1] - np.mean(MD[:,1])) / np.std(MD[:,1])

# As detailed in the paper, we quantify the gap between Gini and GII for those countries where these don't agree; 
# with a positive value indicating low gender equality, but also low salaries
# and a negative value indicating poor gender equality but good income equality
anticorr_metric = np.zeros((Ncountries,1)) 
# quantifies the degree of agreement between Gini and GII
GI_Gini_agree_metric = np.zeros((Ncountries,1)) 

for ic in range(Ncountries):
    if (MD[ic,0]<0) and (MD[ic,1]>0):
        anticorr_metric[ic] = (-MD[ic,0]+MD[ic,1]) 
    elif (MD[ic,0]>0) and (MD[ic,1]<0):
        anticorr_metric[ic] = - (MD[ic,0]-MD[ic,1]) 
    GI_Gini_agree_metric[ic] = MD[ic,0]*MD[ic,1]

# extract sex, age, ICV and SNR
sex = confounds[:,0]
age = confounds[:,1]
ICV = confounds[:,-1]
SNR = confounds[:,-2]

confounds = confounds[:,2:]
q = confounds.shape[1]

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
braindata_str = np.copy(braindata[:,index_str])
braindata_str = (braindata_str.T / np.sum(braindata_str,1)).T # scale up to 1 for each subject
braindata = braindata_str          
    
# standardize
mb = np.nanmean(braindata,axis=0)
sb = np.nanstd(braindata,axis=0)
braindata = (braindata - mb) / sb
mc = np.nanmean(confounds,axis=0)
sc = np.nanstd(confounds,axis=0)
confounds = (confounds - mc) / sc
confounds[np.isnan(confounds)] = 0

# denoise brain data using PCA
pca = PCA()
pca.fit(braindata)
ev = pca.explained_variance_ratio_
ncomp = np.where(np.cumsum(ev)>0.95)[0][0] + 1
print('Explained variance: ' + str(np.sum(ev[0:ncomp])) + ' with ' + str(ncomp) + '\\' + str(len(ev)) + ' PCs ' '')
braindata = pca.transform(braindata)[:,0:ncomp]
p = ncomp

# establish the age grid
age_grid = np.linspace(10, 80, 71)
T = len(age_grid)

braindata = braindata.astype(np.float32)
D = D.astype(np.float32)

# collecting information per age window:

M_Brain = []
M_WID = []
M_age = [] 
M_ICV = [] 
M_SNR = []

for t in range(T): # cycle through windows

    a = age_grid[t]
    choose_t = np.logical_and(age > (a-window_length/2), age < (a+window_length/2))

    # first all subjects
    Npairs = 0
    for ic in range(Ncountries): 
        c = ucountries_vb[ic]
        idc_vb = (countries_vb == c)
        choose = np.logical_and(choose_t, idc_vb)
        is_f = sex[choose] == -1
        is_m = sex[choose] == +1
        N_f = np.sum(is_f)
        N_m = np.sum(is_m)
        Npairs += N_f * N_m

    M_Brain_t = np.zeros((Npairs,1))
    M_WID_t = np.zeros((Npairs,3))
    M_age_t = np.zeros((Npairs,1)) 
    M_ICV_t = np.zeros((Npairs,1)) 
    M_SNR_t = np.zeros((Npairs,1))

    j = 0

    for ic in range(Ncountries):  # cycle through windows for the subjects in the window

        c = ucountries_vb[ic]
        idc_vb = (countries_vb == c)
        choose = np.logical_and(choose_t, idc_vb)
        braindata_t = braindata[choose,:]
        D_t = D[choose,:]
        age_t = age[choose]
        ICV_t = ICV[choose]
        SNR_t = SNR[choose]
        
        # sex of the people from this country and age window
        is_f = sex[choose] == -1
        is_m = sex[choose] == +1
        
        # separate male from females
        braindata_t_f = braindata_t[is_f,:] 
        braindata_t_m = braindata_t[is_m,:] 
        age_t_f = age_t[is_f]
        age_t_m = age_t[is_m]
        ICV_t_f = ICV_t[is_f]
        ICV_t_m = ICV_t[is_m]     
        SNR_t_f = SNR_t[is_f]
        SNR_t_m = SNR_t[is_m]         

        N_f = age_t_f.shape[0]
        N_m = age_t_m.shape[0]
        N = N_f * N_m # the number of male-female pairs
        if (N_f == 0) or (N_m == 0): continue 

        M_Brain_t[j:j+N,0] = cdist(braindata_t_f,braindata_t_m).astype(np.float32).flatten()
        M_age_t[j:j+N,0] = cdist(age_t_f[:,np.newaxis],age_t_m[:,np.newaxis]).astype(np.float32).flatten()
        M_ICV_t[j:j+N,0] = cdist(ICV_t_f[:,np.newaxis],ICV_t_m[:,np.newaxis]).astype(np.float32).flatten()
        M_SNR_t[j:j+N,0] = cdist(SNR_t_f[:,np.newaxis],SNR_t_m[:,np.newaxis]).astype(np.float32).flatten()
        M_WID_t[j:j+N,:] = (D[idc_vb,0:3][0,:] * np.ones((N,1)))
  
        j += N

    M_Brain.append(M_Brain_t)
    M_WID.append(M_WID_t)
    M_age.append(M_age_t)
    M_ICV.append(M_ICV_t)
    M_SNR.append(M_SNR_t)

    
print(datafile_out)

# save for later use
np.savez(datafile_out,   
        M_Brain=np.array(M_Brain, dtype=object),
        M_WID=np.array(M_WID, dtype=object),
        M_age=np.array(M_age, dtype=object),
        M_ICV=np.array(M_ICV, dtype=object),
        M_SNR=np.array(M_SNR, dtype=object),
        age_grid=age_grid, 
        allow_pickle=True)


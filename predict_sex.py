# This script generates the cross-validated, cross-age-window predictions 
# of sex given the brain data - which results are shown in Fig2.
# confounds are regressed out of the brain data. 
# Specifically, for each age-window a predictive model (based on linear regression) is trained, and tested
# in unseen subjects from each other window
# This scripts produce the data for Fig 2AEF

import sys
import numpy as np
import country_converter
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge, RidgeClassifierCV
from sklearn import metrics
import func_testing

real_data = (len(sys.argv)==1) or (int(sys.argv[1])) # real or synthetic data
fast_run = (len(sys.argv)==3) and (int(sys.argv[2])) # 1 this for a quick sanity-check run

if not real_data: print('Running on synthetic data')
if fast_run: print('Quick sanity check run')

N_subj = 1000 # minimum number of subjects in the window to run the analysis.
N_subj_per_sex = 400 # minimum number of subjects per sex in the window to run the analysis.
window_length = 10 # width of the age windows
if fast_run: nboot = 1 # number of bootstrap interations
else: nboot = 100

# paths and names
directory = '/Users/au654912/CloudOneDrive/Work/data/main_volbrain_repo/'
directory_out = directory + 'results/'
directory_braindata = directory + 'volbrain_repo/preprocessed_data/'
if real_data: datafile_braindata = directory_braindata + 'vol2Brain_931.npz'
else: datafile_braindata = directory_braindata + 'vol2Brain_931_synth.npz'
directory_socioecon = directory + 'socioeconomic_data_repo/preprocessed_data/'
datafile_socioecon = directory_socioecon + 'wid.npz'
if real_data: datafile_out = directory_out + 'predict_sex_vol2Brain_931.npz'
else: datafile_out = directory_out + 'predict_sex_vol2Brain_931_synth.npz'

# load countries and labels
datawid = np.load(datafile_socioecon, allow_pickle=True)
X = np.copy(datawid['X'])
countries_wid = np.copy(datawid['country_codes'])
labels_vars = np.copy(datawid['labels_vars'])
q = X.shape[1]
ucountries_wid = np.unique(countries_wid)

# load brain data
datvb = np.load(datafile_braindata, allow_pickle=True)
confounds = np.copy(datvb['confounds'])
braindata = np.copy(datvb['braindata'])
countries_vb_ = np.copy(datvb['countries'])
labels = np.copy(datvb['labels'])
(Nsubj,p) = braindata.shape

ucountries_vb = np.unique(countries_vb_)
countries_vb = np.empty(countries_vb_.shape,dtype=object)
countries_vb[:] = '-'

# check country name inconsistencies
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

# define key variables
D = np.zeros((Nsubj,q)) # expanded matrix with the socioeconomical info
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
    D[idc_vb,:] = X[idc_wid[0],:]
    C[idc_vb] = c
    N_per_country[ic] = np.sum(idc_vb)

# remove countries with no valid subjects
ucountries_vb = ucountries_vb[N_per_country>0]
Ncountries = len(ucountries_vb)

take = ((C != '') & (C != '-'))
countries_vb = countries_vb[take]
D = D[take,:]
C = C[take]
braindata = braindata[take,:]
confounds = confounds[take,:]

sex = confounds[:,0] 
age = confounds[:,1]
confounds = confounds[:,2:] # 2: scale factor, 3: SNR, 4: ICV
q = confounds.shape[1]

# some ICV values are extreme (particularly at old age), which can completely
# distort the analysis. Because of this we quantile-normalise to a Gaussian distribution
confounds = func_testing.quantile_normalize_to_gaussian(confounds)

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
braindata_str = (braindata_str.T / np.sum(braindata_str,1)).T
braindata = braindata_str

# establish the age grid
age_grid = np.linspace(10, 80, 71)
T = len(age_grid)

# define output variables 
accuracy = np.zeros((T,T,nboot))
f1 = np.zeros((T,T,nboot))
choose = np.zeros((T,T),dtype=bool)

for t in range(T): # cycle through windows for training 

    a = age_grid[t]
    within_range = np.logical_and(age > (a-window_length/2), age < (a+window_length/2))
    index_1 = np.where(within_range)[0]

    braindata_orig = braindata[within_range,:]
    confounds_orig = confounds[within_range,:]
    sex_orig = sex[within_range]
    age_orig = age[within_range]

    which_fe = np.where(sex_orig == -1)[0] # who are females
    which_ma = np.where(sex_orig == +1)[0] # who are males
    N_t_fe = len(which_fe)
    N_t_ma = len(which_ma)
    
    if len(sex_orig) < N_subj: continue
    if (len(which_fe) < N_subj_per_sex) or (len(which_ma) < N_subj_per_sex): continue

    for t2 in range(T): # cycle through windows for testing 

        a2 = age_grid[t2]
        within_range2 = np.logical_and(age > (a2-window_length/2), age < (a2+window_length/2))
        index_2 = np.where(within_range2)[0]

        choose[t,t2] = True

        for b in range(nboot): # bootstrap iterations, where we sample subjects randomly

            # sample train
            sample_fe_tr = np.random.choice(np.arange(N_t_fe), size=round(0.8*N_subj_per_sex), replace=True)
            sample_fe_tr = which_fe[sample_fe_tr]
            sample_ma_tr = np.random.choice(np.arange(N_t_ma), size=round(0.8*N_subj_per_sex), replace=True)
            sample_ma_tr = which_ma[sample_ma_tr]
            sample_tr = np.concatenate((sample_fe_tr,sample_ma_tr))
            index_tr = index_1[sample_tr]

            # exclude these subjects from the candidate subjects for test
            index_2_available = np.setdiff1d(index_2, index_tr)          
            braindata_output = braindata[index_2_available,:]
            confounds_output = confounds[index_2_available,:]
            sex_output = sex[index_2_available]
            age_output = age[index_2_available]
            which_fe2 = np.where(sex_output == -1)[0]
            which_ma2 = np.where(sex_output == +1)[0]
            N_t_fe2 = len(which_fe2)
            N_t_ma2 = len(which_ma2)

            # sample test
            sample_fe_te = np.random.choice(np.arange(len(which_fe2)),size=round(0.2*N_subj_per_sex), replace=True)
            sample_ma_te = np.random.choice(np.arange(len(which_ma2)),size=round(0.2*N_subj_per_sex), replace=True)
            sample_fe_te = which_fe2[sample_fe_te]
            sample_ma_te = which_ma2[sample_ma_te]
            sample_te = np.concatenate((sample_fe_te,sample_ma_te))

            confounds_tr = np.copy(confounds_orig[sample_tr,:])
            braindata_tr = np.copy(braindata_orig[sample_tr,:])
            sex_tr = np.copy(sex_orig[sample_tr])
            age_tr = np.copy(age_orig[sample_tr])

            confounds_te = np.copy(confounds_output[sample_te,:])
            braindata_te = np.copy(braindata_output[sample_te,:])
            sex_te = np.copy(sex_output[sample_te])
            age_te = np.copy(age_output[sample_te])

            # demeaning, etc 
            mb = np.nanmean(braindata_tr,axis=0)
            sb = np.nanstd(braindata_tr,axis=0)
            braindata_tr = (braindata_tr - mb) / sb
            braindata_te = (braindata_te - mb) / sb
            braindata_tr[np.isnan(braindata_tr)] = 0
            braindata_te[np.isnan(braindata_te)] = 0         

            mc = np.nanmean(confounds_tr,axis=0)
            sc = np.nanstd(confounds_tr,axis=0)
            nozero = sc != 0
            confounds_tr = (confounds_tr[:,nozero] - mc[nozero]) / sc[nozero]
            confounds_te = (confounds_te[:,nozero] - mc[nozero]) / sc[nozero]
            confounds_tr[np.isnan(confounds_tr)] = 0
            confounds_te[np.isnan(confounds_te)] = 0

            # deconfounding 
            regr = MultiOutputRegressor(Ridge(alpha = 0.01)).fit(confounds_tr, braindata_tr)
            braindata_te_dec = braindata_te - regr.predict(confounds_te) 
            braindata_tr_dec = braindata_tr - regr.predict(confounds_tr)

            # Use a ridge-based classifier (quicker than logistic and not that different in practice in this case)
            regr = RidgeClassifierCV().fit(braindata_tr_dec, sex_tr)
            sex_hat = regr.predict(braindata_te_dec)  

            accuracy[t,t2,b] = np.mean(sex_hat==sex_te) 
            f1[t,t2,b] = metrics.f1_score(sex_te,sex_hat)
        

print(datafile_out)

np.savez(datafile_out, 
        accuracy=accuracy,f1=f1,choose=choose,
        age_grid=age_grid)

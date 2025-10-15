# The script predict_sex.py shows that we can predict sex using deconfounded brain data.
# This one uses methods from explainable machine learning to interrogate these predictions.
# Here we however did not deconfound; instead, we included the confounds as additional regressors,
# which is more appropiate in this case because we want the SHAP value calculation to take into account these. 
# The questions we asked are: 
# How cortical vs subcortical areas contributed to the prediction of sex? (Fig 2BC)
# How the predictions change over time, for which we will apply hierarchical clustering on the computed SHAP values? (Fig 2D) 

import sys
import numpy as np
import country_converter
from sklearn.linear_model import LogisticRegression
import shap
import func_testing

real_data = (len(sys.argv)==1) or (int(sys.argv[1])) # real or synthetic data
fast_run = (len(sys.argv)==3) and (int(sys.argv[2])) # 1 this for a quick sanity-check run

if not real_data: print('Running on synthetic data')
if fast_run: print('Quick sanity check run')

N_subj = 1000 # minimum number of subjects in the window to run the analysis.
N_subj_per_sex = 400 # minimum number of subjects per sex in the window to run the analysis.
window_length = 10 # width of the age windows
if fast_run: nboot = 1 # number of bootstrap interations
else: nboot = 10

# paths and names
directory = '/Users/au654912/CloudOneDrive/Work/data/main_volbrain_repo/'
directory_out = directory + 'results/'
directory_braindata = directory + 'volbrain_repo/preprocessed_data/'
if real_data: datafile_braindata = directory_braindata + 'vol2Brain_931.npz'
else: datafile_braindata = directory_braindata + 'vol2Brain_931_synth.npz'
directory_socioecon = directory + 'socioeconomic_data_repo/preprocessed_data/'
datafile_socioecon = directory_socioecon + 'wid.npz'
if real_data: datafile_out = directory_out + 'shap_values_vol2Brain_931.npz'
else: datafile_out = directory_out + 'shap_values_vol2Brain_931_synth.npz'

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
ICV = confounds[:,-1]
confounds = confounds[:,2:]
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
pq = braindata.shape[1] + confounds.shape[1] + 1 # +age

# establish the age grid
age_grid = np.linspace(10, 80, 71)
T = len(age_grid)

# define output variables 
shap_values = np.zeros((T,pq,nboot))
beta_values = np.zeros((T,pq,nboot))
accuracy_insample = np.zeros((T,nboot))
choose = np.zeros(T,dtype=bool)

for t in range(T): # cycle through windows  

    a = age_grid[t]
    within_range = np.logical_and(age > (a-window_length/2), age < (a+window_length/2))

    braindata_orig = braindata[within_range,:]
    confounds_orig = confounds[within_range,:]
    sex_orig = sex[within_range]
    age_orig = age[within_range]
    C_orig = C[within_range]

    if len(sex_orig) < N_subj: continue
    which_fe = np.where(sex_orig == -1)[0]
    which_ma = np.where(sex_orig == +1)[0]

    if (len(which_fe) < N_subj_per_sex) or (len(which_ma) < N_subj_per_sex): continue

    choose[t] = True

    for b in range(nboot): # bootstrap iterations, where we sample subjects randomly

        sample_fe = np.random.choice(np.arange(len(which_fe)), size=N_subj_per_sex, replace=True)
        sample_fe = which_fe[sample_fe]
        sample_ma = np.random.choice(np.arange(len(which_ma)), size=N_subj_per_sex, replace=True)
        sample_ma = which_ma[sample_ma]
        sample = np.concatenate((sample_fe,sample_ma))

        braindata_t = braindata_orig[sample,:]
        confounds_t = confounds_orig[sample,:]
        sex_t = sex_orig[sample]
        age_t = age_orig[sample]
        C_t = C_orig[sample]            

        # here, instead of deconfounding, we add the confounds as regressors. 
        X = np.concatenate((braindata_t,confounds_t,np.expand_dims(age_t,axis=1)),axis=1)
        y = sex_t

        # demeaning, etc 
        mX = np.mean(X,axis=0)
        sX = np.std(X,axis=0)
        X = (X - mX) / sX

        # use logistic regression as a base model
        model = LogisticRegression(max_iter=10000)
        model.fit(X, y)

        # compute SHAP values 
        background_X = shap.maskers.Independent(X, max_samples=100)
        explainer = shap.Explainer(model.predict, background_X)
        shap_values_t = explainer(X)
        shap_values[t,:,b] = np.mean(np.abs(shap_values_t.values),axis=0)
        beta_values[t,:,b] = np.abs(model.coef_[0,:])

        sex_hat_t = model.predict(X)   
        accuracy_insample[t,b] = np.mean(sex_hat_t==sex_t) 
        
    print('Age: ' + str(a))

accuracy_insample = accuracy_insample[choose,:]
shap_values = shap_values[choose,:,:]
beta_values = beta_values[choose,:,:]
age_grid = age_grid[choose]


print(datafile_out)

np.savez(datafile_out, 
        accuracy_insample=accuracy_insample,shap_values=shap_values,
        beta_values=beta_values,age_grid=age_grid)

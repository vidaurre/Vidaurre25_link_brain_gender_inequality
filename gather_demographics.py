# This script collects statistics about demographics in order to create Fig1. 

import numpy as np
import country_converter

# paths and names
directory_base = '/Users/au654912/'
directory = directory_base + 'CloudOneDrive/Work/data/volBrain/'
directory_out = directory_base + 'CloudOneDrive/Work/Python/volBrain/out/'
name_data = 'vol2Brain_931'

# load socioeconomical data
datafile = directory + 'wid.npz'
datawid = np.load(datafile, allow_pickle=True)
X = np.copy(datawid['X'])
countries_wid = np.copy(datawid['country_codes'])
labels_vars = np.copy(datawid['labels_vars'])
q = X.shape[1]
ucountries_wid = np.unique(countries_wid)

# load volbrain data
datafile = directory + 'preprocessed_data/' + name_data + '.npz'
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
take = ((C != '') & (C != '-'))
countries_vb = countries_vb[take]
D = D[take,:]
C = C[take]
braindata = braindata[take,:]
confounds = confounds[take,:]

sex = confounds[:,0]
age = confounds[:,1]
confounds = confounds[:,2:]
ICV = confounds[:,-1]

ucountries_vb = np.unique(countries_vb)
Ncountries = len(ucountries_vb)

# define output variables 
N_country_sex = np.zeros((Ncountries,2))
mean_age_country_sex = np.zeros((Ncountries,3)) # male, female and all
mean_ICV = np.zeros((Ncountries))

for j in range(Ncountries):
    print(j)
    c = ucountries_vb[j]
    idc_vb = (countries_vb == c)
    idc_vb_m =  np.logical_and(idc_vb, sex == +1)
    idc_vb_f =  np.logical_and(idc_vb, sex == -1)
    N_country_sex[j,0] = np.sum(idc_vb_m)
    N_country_sex[j,1] = np.sum(idc_vb_f)
    mean_age_country_sex[j,0] = np.mean(age[idc_vb_m])
    mean_age_country_sex[j,1] = np.mean(age[idc_vb_f])
    mean_age_country_sex[j,2] = np.mean(age[idc_vb])
    mean_ICV[j] = np.mean(ICV[idc_vb])

datafile = directory_out + 'demographics_' + name_data + '.npz'

print(datafile)

np.savez(datafile, age=age,sex=sex,countries=ucountries_vb,
         N_country_sex=N_country_sex,mean_ICV=mean_ICV,
         mean_age_country_sex=mean_age_country_sex)
  


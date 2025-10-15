# Do some initial preprocessing of the country socioeconomic data
# It assumes the world_inequiality_dataset was downloaded at the 'directory_socioeconomic_data' location
# In the Github repository, this is 

import pandas as pd
import numpy as np
import country_converter


labels_vars = ('GDP_i','GDP_b','NI_i','NI_b','NW_i','NW_b','WIR_i','WIR_b','CI_i','CI_b',
               'GI_i','GI_b','II1_i','II1_b','II2_i','II2_b','WI1_i','WI1_b','WI2_i','WI2_b','Gini','GII')

# paths and names
directory = '/Users/au654912/CloudOneDrive/Work/data/main_volbrain_repo/'
directory_socioeconomic_data =  directory + '/socioeconomic_data_repo/'
directory_socioeconomic_data_raw = directory_socioeconomic_data + '/raw_data/'
directory_out = directory_socioeconomic_data + 'preprocessed_data/'
datafile_out = directory_out  + '/wid.npz'
datafile = directory + 'volbrain_repo/preprocessed_data/vol2Brain_931.npz'

# retrieve countries present in volbrain - we are only concerned about these
dat = np.load(datafile, allow_pickle=True)
countries_vb = dat['countries']
ucountries = np.unique(countries_vb)

# check inconsistencies in country names
ucountries_vb_longname = np.copy(ucountries)
for j in range(len(ucountries_vb_longname)):
    c = ucountries[j]
    if c == '-': continue
    if len(c)>4 and c[-4:] == 'kiye': c = 'Turkey'
    elif len(c)>4 and c[-4:] == 'Xico': c = 'Mexico'
    elif c == 'Brasil': c = 'Brazil'
    elif c[0:3] == 'Esp': c = 'Spain'
    elif c == 'Ltaly': c = 'Italy'
    elif c == 'Polska': c = 'Poland'
    elif c == 'Valencia': c = 'Spain'
    ucountries[j] = country_converter.convert(c, to='ISO2')
ucountries = np.unique(ucountries)
nc = len(ucountries)

R = 0.01 * np.eye(2)
R[0,0] = 0
J = 0

file = directory_socioeconomic_data_raw + '/gross_domestic_product/WID_Data_21102024-122223.xls'
ichar0 = 9 
dat = pd.read_excel(file,keep_default_na=False,header=None)
headers = dat.iloc[0,2:].values
country_codes_long = np.zeros(len(headers),dtype=object)
for j in range(len(headers)): country_codes_long[j] = headers[j][ichar0:ichar0+2]
ucountries_, ii1, ii2 = np.intersect1d(ucountries,country_codes_long, return_indices=True)
country_codes = country_codes_long[ii2]
nc = len(country_codes)
X = np.zeros((nc,10*2 + 2))

# GINI index
file = directory_socioeconomic_data_raw + '/P_Data_Extract_From_World_Development_Indicators/0912e9d4-2c12-4e3d-bb7f-a5a10d78d091_Data.csv'
dat = pd.read_csv(file,sep=',')
dat = dat.iloc[:217,:].values
gini_all = dat[:,4:]
gini = np.zeros(gini_all.shape[0])
countries_g = np.array(country_converter.convert(dat[:,3], to='ISO2') )
for j in range(len(gini)):
    g = gini_all[j,:]
    gf = g[g != '..']
    if len(gf)==0: 
        gini[j] = np.nan
        continue
    gini[j] = float(gf[-1])
Xg = np.zeros(nc)
for j in range(nc):
    jj = np.where(country_codes[j] == countries_g)[0][0]
    Xg[j] = gini[jj]
X[:,-2] = Xg

# Gender Inequality index
file = directory_socioeconomic_data_raw + '/GII.xlsx'
dat = pd.read_excel(file)
GII_all = dat.iloc[:,1].values
countries_gii = np.array(country_converter.convert(dat.iloc[:,0].values, to='ISO2') )
Xgii = np.zeros(nc)
for j in range(nc):
    jj = np.where(country_codes[j] == countries_gii)[0][0]
    Xgii[j] = GII_all[jj]
X[:,-1] = Xgii


constructs = ['gross_domestic_product/WID_Data_21102024-122223.xls',
              'national_income/WID_Data_21102024-161333.xls',
              'national_wealth/WID_Data_22102024-155358.xls',
              'wealth_income_ratio/WID_Data_22102024-155434.xls',
              'carbon_inequality/WID_Data_22102024-155636.xls',
              'gender_inequality/WID_Data_22102024-155727.xls',
              'income_inequality/WID_Data_21102024-162413.xls',
              'wealth_inequality/WID_Data_22102024-155509.xls']
ichar0v = [9,13,13,13,9,13,9,9]


for iv in range(len(constructs)): # WID files/variables

    file = directory_socioeconomic_data_raw + constructs[iv]
    ichar0 = ichar0v[iv]        
    print(file)

    dat = pd.read_excel(file,keep_default_na=False,header=None)
    qu = dat.iloc[1:,0].values
    yr = dat.iloc[1:,1].values
    d = dat.iloc[1:,2:].values # years by countries 
    headers = dat.iloc[0,2:].values
    country_codes_long = np.zeros(len(headers),dtype=object)
    for j in range(len(headers)): country_codes_long[j] = headers[j][ichar0:ichar0+2]

    ucountries_, ii1, ii2 = np.intersect1d(ucountries,country_codes_long, return_indices=True)
    country_codes = country_codes_long[ii2]
    d = d[:,ii2]

    if nc != d.shape[1]: raise Exception('Something is wrong')

    if iv < 6:

        for i in range(nc):
            x = np.ones((d.shape[0],2))
            x[:,1] = yr - np.mean(yr)
            di = d[:,i]
            iii = di != ''
            if np.sum(iii) < 2:
                X[i,J:(J+2)] = np.nan
                continue
            di = di[iii].astype(np.float32)
            y = np.expand_dims(di,1)
            x = x[iii,:]
            ab = np.linalg.inv(x.T @ x + R) @ (x.T @ y) 
            X[i,J:(J+2)] = ab[:,0]
        J += 2

    else:

        ij = (qu == 'p0p50')
        yr_p0p50 = yr[ij]
        d_p0p50 = d[ij,:] 
        ij = (qu == 'p90p100')
        yr_p90p100 = yr[ij]
        d_p90p100 = d[ij,:] 
        ij = (qu == 'p99p100')
        yr_p99p100 = yr[ij]
        d_p99p100 = d[ij,:]

        for k in range(2):
            if k==0: 
                d_ = d_p90p100 # / d_p0p50  # done below
                yr_ = yr_p0p50
            elif k==1:
                d_ = d_p99p100 # / d_p0p50 
                yr_ = yr_p0p50       
            for i in range(nc):
                iii = (d_[:,i] != '') & (d_p0p50[:,i] != '')
                if np.sum(iii) < 2:
                    X[i,J:(J+2)] = np.nan
                    continue
                d_i = d_[iii,i].astype(np.float32) / d_p0p50[iii,i].astype(np.float32)
                x = np.ones((len(d_i),2))
                yr_i = yr_[iii]
                x[:,1] = yr_i - np.mean(yr_i)
                y = np.expand_dims(d_i,1)
                ab = np.linalg.inv(x.T @ x + R) @ (x.T @ y) 
                X[i,J:(J+2)] = ab[:,0]
            J += 2


X = (X - np.nanmean(X, axis=0)) / np.nanstd(X, axis=0)
X[np.isnan(X)] = 0

print(datafile_out)
np.savez(datafile_out, country_codes=country_codes,labels_vars=labels_vars,X=X)


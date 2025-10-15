# this script runs permutation-testing-based t-tests for predicting pairwise brain distances using
# sex, age, confounds, GII, and, optionally, the Gini index or the GDP.
# It outputs p-values for each regressor. 
# This produces the results shown in Fig3BCD (for combination=0,1,2 respectively)

import sys
import numpy as np
import func_testing
import subset_optimisation


real_data = (len(sys.argv)<3) or (int(sys.argv[2])) # real or synthetic data
fast_run = (len(sys.argv)==4) and (int(sys.argv[3])) # 1 this for a quick sanity-check run

if not real_data: print('Running on synthetic data')
if fast_run: print('Quick sanity check run')

# 0 for analysis on GII, 1 for analysis on GII and Gini index, 2 for analysis on GII and GDP
combination = sys.argv[1]  

if not fast_run:
    repetitions = 5 # number of times we run the analysis
    Nperm = 10001 # number or permutations for the permutation testing. 
else:
    repetitions = 1 # number of times we run the analysis
    Nperm = 5 # number or permutations for the permutation testing.     
n_sampled_pairs = 5000 # how many pairs of subjects we are going to sample 

# paths and names
directory = '/Users/au654912/CloudOneDrive/Work/data/main_volbrain_repo/'
directory_out = directory + 'results/'
directory_braindata = directory + 'volbrain_repo/preprocessed_data/'
if real_data: datafile_braindata = directory_braindata + 'sexDist_vol2Brain_931.npz'
else: datafile_braindata = directory_braindata + 'sexDist_vol2Brain_931_synth.npz'

if combination == 0: suffix = '_GII' 
elif combination == 1: suffix = '_GIIGini' 
elif combination == 2: suffix = '_GIIGDP' 

if combination == 0: ind_wid = (0,)
elif combination == 1: ind_wid = (0,1)
elif combination == 2: ind_wid = (0,2)

if real_data: datafile_out = directory_out + 'ttests_vol2Brain_931' + suffix + '.npz'
else: datafile_out = directory_out + 'ttests_vol2Brain_931' + suffix + '_synth.npz'

# load distance matrices computed in computer_between_sexes_distances.py
data = np.load(datafile_braindata, allow_pickle=True)
M_Brain = data['M_Brain'] 
M_WID = data['M_WID'] 
M_age = data['M_age'] 
M_ICV = data['M_ICV'] 
M_SNR = data['M_SNR'] 

# establish the age grid
age_grid = np.linspace(10, 80, 71)
T = len(age_grid)

if combination==0:
    nmacro = 1 # GII
    p = nmacro + 3 # age and ICV and SNR diffs
else: 
    nmacro = 2 # Gini+GII (no diff, 3rd)
    p = nmacro + 3 # age and ICV and SNR diffs

# define output variables 
choose = np.zeros(T,dtype=bool)
r2 = np.zeros((T,Nperm,repetitions))
base_stat_f = np.zeros((T,Nperm,repetitions))
pval_f = np.zeros((T,repetitions))
pval_t = np.zeros((T,p,repetitions))
base_stat_t = np.zeros((T,Nperm,p,repetitions))
beta = np.zeros((T,Nperm,p,repetitions))
subsample_corr = np.zeros((T,repetitions))
original_corr = np.zeros((T,repetitions))
N_countries_involved = np.zeros((T,repetitions))

N_pairs = np.zeros(T)
for t in range(T): 
    N_pairs[t] = M_Brain[t].shape[0]

for t in range(T): 

    if combination>0:
        M_Brain_t = M_Brain[t]
        M_WID_t = M_WID[t]
        M_age_t = M_age[t]
        M_ICV_t = M_ICV[t]    
        M_SNR_t = M_SNR[t] 
        M_ICV_t = func_testing.quantile_normalize_to_gaussian(M_ICV_t)
        M_SNR_t = func_testing.quantile_normalize_to_gaussian(M_SNR_t)
        Xorig = np.concatenate((M_WID_t[:,ind_wid],M_age_t,M_ICV_t,M_SNR_t),axis=1) 
        # the next things are needed for the subset selection
        y1 = M_WID_t[:,ind_wid[0]]
        y2 = M_WID_t[:,ind_wid[1]]
        y1y2 = y1 * y2
        ucp = np.unique(y1y2)
        N_countries_represented = len(ucp)
        y1_ = np.zeros((N_countries_represented,1))
        y2_ = np.zeros((N_countries_represented,1))
        N_people_per_country = np.zeros(N_countries_represented)
        indices_countries = np.zeros((M_WID_t.shape[0]))
        for j in range(N_countries_represented):
            ind_j = y1y2 == ucp[j]
            indices_countries[ind_j] = j
            y1_[j,0] = y1[ind_j][0]
            y2_[j,0] = y2[ind_j][0]
            N_people_per_country[j] = np.sum(ind_j)    
    else:
        M_Brain_t = M_Brain[t]
        M_WID_t = M_WID[t]
        M_age_t = M_age[t]
        M_ICV_t = M_ICV[t]    
        M_SNR_t = M_SNR[t] 
        M_ICV_t = func_testing.quantile_normalize_to_gaussian(M_ICV_t)
        M_SNR_t = func_testing.quantile_normalize_to_gaussian(M_SNR_t)
        Xorig = np.concatenate((np.expand_dims(M_WID_t[:,0],axis=1),M_age_t,M_ICV_t,M_SNR_t),axis=1) 

    Yorig = M_Brain_t
            
    for r in range(repetitions):

        if combination>0: # we run our algorithm to minimise correlation between GII and GDP/Gini in the subsample
            k = n_sampled_pairs # round(0.1 * N_pairs[t])
            sample_c, corr_sample = subset_optimisation.heuristic_max_corr_quick(np.concatenate((y1_,y2_),1), 
                                                                        N_people_per_country.astype(int),
                                                    k=k, max_iter=50000, repetitions=10,
                                                    cooling_rate = 0.99999)
            orig_corr = subset_optimisation.correlation_quick(y1_[:,0],y2_[:,0],N_people_per_country.astype(int))
            # print('Correlation of selection: ' + str(corr_sample) + '; original:' + str(orig_corr))
            subsample_corr[t,r] = corr_sample
            original_corr[t,r] = orig_corr
            N_countries_involved[t,r] = np.sum(sample_c>0)
            sample = subset_optimisation.count_to_indices(sample_c,indices_countries)
        else: # if there's only GII, then we just do an ordinary random sample
            sample = np.random.choice(np.arange(N_pairs[t]).astype(int), size=round(0.95 * np.min(N_pairs)), replace=False) 

        X = np.copy(Xorig[sample,:]) # non-brain-imaging pairwise distances
        Y = np.copy(Yorig[sample]) # brain-imaging pairwise distances

        # standardising 
        mX = np.mean(X,axis=0)
        sX = np.std(X,axis=0)
        X = (X - mX) / sX

        mY = np.mean(Y)
        Y -= mY

        # run the ttests
        (pvaluef,fstat,pvaluet,tstat,r2_tr,beta_tr) = func_testing.standard_tftest(X,Y,Nperm)
        base_stat_f[t,:,r] = fstat
        pval_f[t,r] = pvaluef
        base_stat_t[t,:,:,r] = tstat[:,1:] # take out intercept
        beta[t,:,:,r] = beta_tr[:,1:]
        pval_t[t,:,r] = pvaluet[1:]
        r2[t,:,r] = r2_tr

    print('time: ' + str(t))


print(datafile_out)

if combination>0:
    np.savez(datafile_out, r2=r2, subsample_corr = subsample_corr, original_corr=original_corr,
            N_countries_involved=N_countries_involved,
            base_stat_f=base_stat_f,pval_f=pval_f,pval_t=pval_t,
            base_stat_t=base_stat_t,beta=beta,age_grid=age_grid)
else:
    np.savez(datafile_out, r2=r2, base_stat_f=base_stat_f,pval_f=pval_f,pval_t=pval_t,
            base_stat_t=base_stat_t,beta=beta,age_grid=age_grid)



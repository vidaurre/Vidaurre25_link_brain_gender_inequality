# Do some initial preprocessing of the brain data

import pandas as pd
import numpy as np

directory_base = '/Users/au654912/CloudOneDrive/'
df = pd.read_csv(directory_base + '/Work/data/volBrain/cleaned_volbrain_Diego/cleaned_fusion_vol2Brain_931.csv')
name_data = 'vol2Brain_931'

 
def correct_age(age):
    if isinstance(age, str) and (age == "UNKNOWN" or age == "Unknown"):
        return np.nan
    elif isinstance(age, str) and age != "UNKNOWN":
        age = float(age)
        if age > 150:
            return "UNKNOWN"
        else:
            return int(age)
    elif isinstance(age, (int, float, str)) and age > 150:
        age = "UNKNOWN"
    return age

def correct_sex(sex):
    if isinstance(sex, str) and (sex == "UNKNOWN" or sex == "Unknown"):
        return np.nan
    elif isinstance(sex, str) and sex != "UNKNOWN":
        if sex=='Female' or sex=='female': 
            return -1
        elif sex=='Male' or sex=='male':
            return +1
        else:
            return np.nan
    else:
        return np.nan

# Replace 'UNKNOWN' values with NaN
original_shape = df.shape[0]
print("Total number of counts:", original_shape)

# UNKOWN values are as NaN
df = df.replace('UNKNOWN', pd.NA)
# Drop rows where NaNs are in
df = df.dropna(subset=['Sex', 'Age'])
df['Age'] = df['Age'].apply(correct_age)
df['Sex'] = df['Sex'].apply(correct_sex)

print("Number of counts where Age and Sex missing:", original_shape - df.shape[0])
print(f"Lossed {100*(original_shape - df.shape[0])/original_shape:.2f}% of the data")
print(f"Number of NaNs in Age: {df['Age'].isna().sum()}")
print(f"Number of NaNs in Sex : {df['Sex'].isna().sum()}")
print(f"Number of counts in the final dataframe: {df.shape[0]}")

sex = df['Sex'].to_numpy()
age = df['Age'].to_numpy()

scale_factor = df['Scale factor'].to_numpy()
SNR = df['SNR'].to_numpy()
ICV = df['Intracranial Cavity (IC) volume cm3'].to_numpy()
confounds = np.concatenate((
    np.expand_dims(sex,1),
    np.expand_dims(age,1),        
    np.expand_dims(scale_factor,1),
    np.expand_dims(SNR,1),
    np.expand_dims(ICV,1)),
    axis=1)
braindata = df.iloc[:,5:-2].values
labels = df.columns.values[5:-2]
confounds_labels = np.array(('sex','age','scale_factor','SNR','ICV'))

var_ht_0 = (np.nanstd(braindata, axis=0) > 0)
braindata = braindata[:,var_ht_0]
labels = labels[var_ht_0]


countries = df['country'].to_numpy()

datafile = directory_base + '/Work/data/volBrain/preprocessed_data/' + name_data + '.npz'
print(datafile)
np.savez(datafile, confounds=confounds,confounds_labels=confounds_labels,
        braindata=braindata,countries=countries,labels=labels)





# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Basic-Info-and-Remove-Unneceassry-Features" data-toc-modified-id="Basic-Info-and-Remove-Unneceassry-Features-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Basic Info and Remove Unneceassry Features</a></span></li><li><span><a href="#Create-New-features-for-Data" data-toc-modified-id="Create-New-features-for-Data-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Create New features for Data</a></span></li><li><span><a href="#Fill-Missing-Values-with-mode-and-median" data-toc-modified-id="Fill-Missing-Values-with-mode-and-median-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Fill Missing Values with mode and median</a></span></li></ul></div>

# ## Basic Info and Remove Unneceassry Features

# In[1]:


import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# data visualization
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().magic('matplotlib inline')


# In[2]:


dtypes = {'pid' :'category',
 'f_male_bl':'category',
 'race_bl':'category',
 'ethnic_bl':'category',
 'f_onset_bulbar_bl':'category',
 'f_onset_limb_bl':'category',
 'f_onset_spine_bl':'category',
 'f_riluzole_bl':'category',
 'f_study_drug_bl':'category',
 'visit':'category'}


# In[3]:


path = './input'
ep_train = pd.read_csv(path+'/ur_adep_proact.csv', index_col=0, dtype = dtypes)
bl_train=pd.read_csv(path+'/ur_adbl_proact.csv', index_col=0,dtype = dtypes)

ep_test = pd.read_csv(path+'/topi_adep_proact.csv', index_col=0, dtype = dtypes)
bl_test=pd.read_csv(path+'/topi_adbl_proact.csv', index_col=0,dtype = dtypes)

train = pd.merge(ep_train, bl_train, how='left')
test = pd.merge(ep_test, bl_test, how='left')
label = 'r_alsfrs_r_total'


# In[4]:


# remove outliers for test data
test = test[test.pid!='609374']
test = test[test.pid!='164690']
test = test[test.pid!='674622']

for idx in ['r1_dyspnea_bl', 'r2_orthopnea_bl',  'r3_respiratory_insufficiency_bl']:
    test[idx] = test.q10_respiratory_bl
    
test.r_alsfrs_r_total = test.r_alsfrs_r_total.fillna(method='bfill')
pid_error =[]
for i in set(test.pid):
    try:
        test.loc[test['pid'] == i, 'alsfrs_r_total_bl'] =test[(test.pid==i )& (test.t==0)]['r_alsfrs_r_total'].values[0]
    except:
        pid_error.append(i)
        
for i in pid_error:
    test.loc[test['pid'] == i, 'alsfrs_r_total_bl'] =test[(test.pid==i )]['r_alsfrs_r_total'].values[0]


# In[5]:


# remove outlier for train data
## remove values without r_alsfrs_r_total 
remove_pid = train.groupby('pid').sum()['r_alsfrs_r_total'][train.groupby('pid').sum()['r_alsfrs_r_total'].isnull()].index.tolist()
train = train[~train.pid.isin(remove_pid)]
# assign values for train when t is less than 50 and r_alsfrs_r_total is missing values 
train.loc[(train.t<50)&(train.r_alsfrs_r_total.isnull()), 'r_alsfrs_r_total'] =  train.loc[(train.t<50)&(train.r_alsfrs_r_total.isnull()), 'alsfrs_r_total_bl'].values
train = train[~(train.r_alsfrs_r_total.isnull())]
train = train[~train.alsfrs_r_total_bl.isnull()]
train = train[~train.q5_cutting_bl.isnull()]

# set(train[train.alsfrs_r_total_bl.isnull()][['pid', 't', 'r_alsfrs_r_total', 'alsfrs_r_total_bl']].pid.values)


# In[6]:


# combine test and train dataset
train['set']=1
test['set']=0

df = pd.concat([train, test])


# In[7]:


# remove unnecessary features  from list(df).index("f_visit_vc") to list(df).index("r_alsfrs_r_total_derived")

cols = range(3, 37)
df.drop(df.columns[cols],axis=1,inplace=True) # remove unrelated features

# remove features in list columns
columns = ['alsfrs_total_comb_bl','alsfrs_r_total_comb_bl','alsfrs_r_total_derived_bl']
df.drop(columns, inplace=True, axis=1)

#remove features in list columns
columns=['alsfrs_total_bl','resp_preslope_bl','q10_respiratory_bl','ethnic_bl',
'uric_acid_bl', 'phosphorus_bl','resp_rate_bl','temp_C_bl','vc_preslope_bl',
'pexp_vc_bl','bmi_bl','height_cm_bl','exp_vc_bl']  
df.drop(columns, inplace=True, axis=1)


# In[13]:


df.isnull().sum()[df.isnull().sum()!=0].index.tolist()


# ## Create New features for Data

# In[8]:


# create feature month
l = df.t/30
month = [int(x) for x in l]
df['month'] = month

# remove  missing values which have over 20 missing values for each sample
df.drop(df[(df.isnull().sum(axis=1)>20)].index, inplace=True)


# In[9]:


# combine feature spine, limb and onset into a new feature and convert it into a value 
for i in ['f_onset_spine_bl', 'f_onset_limb_bl', 'f_onset_bulbar_bl']:
    df[i] = df[i].astype(str)

df['Type_spine_limb_bulbar'] = df['f_onset_spine_bl'].apply(lambda x: str(int(x))) + df['f_onset_limb_bl'].apply(lambda x: str(int(x)))+  df['f_onset_bulbar_bl'].apply(lambda x: str(int(x)))
df.Type_spine_limb_bulbar = df.Type_spine_limb_bulbar.astype('category')
for i in ['f_onset_spine_bl', 'f_onset_limb_bl', 'f_onset_bulbar_bl']:
    df[i] = df[i].astype('category')

#  convert the new feature 'Type_spine_limb_bulbar' into limb bulbar  or combination
# new feature is 'onest'
def transf(x):
    x = x
    if x[1]=='1' and x[2] =='0':
        return 'limb'
    elif x[1]=='0' and x[2] =='1':
        return 'bulbar'
    else:
        return 'combination' 


df['onset'] = df.Type_spine_limb_bulbar.apply(transf)
df['onset'] = df['onset'].astype('category')


# remove related four features 
df.drop(['f_onset_bulbar_bl', 'f_onset_limb_bl', 'f_onset_spine_bl', 'Type_spine_limb_bulbar'], axis=1, inplace=True)


# In[10]:


# create some features related with t
df['days_disease_tillNow'] = df['t'] - df['onset_days_bl'] #  how many days they got the disease 已经得了多少天的病了

df['years_disease'] = (df.days_disease_tillNow/365).round(2) # how many years they got the disease 得了多少天病转化为年
df['month_disease'] = (df.days_disease_tillNow/30).round(1)  # how many months they got the disease 得了多少天病转化为月

df['age_disease_year'] = (df.age_yr_bl - df.days_disease_tillNow/365).round() # age when they got the disease 得病时多少岁

df['age_bin'] = pd.cut(df['age_disease_year'], [0,40, 70,100]) # 


# In[11]:


# change data type 
df.race_bl = df.race_bl.astype('category')
df.f_riluzole_bl = df.f_riluzole_bl.astype('category')


# In[12]:


# transform OHE for feature race_bl, race_bl and rop
dummies = pd.get_dummies(df['race_bl']).rename(columns=lambda x: 'race_bl' + str(x))
df = pd.concat([df, dummies], axis=1)
df = df.drop(['race_bl'], axis=1)
dummies = pd.get_dummies(df['onset']).rename(columns=lambda x: 'onset' + str(x))
df = pd.concat([df, dummies], axis=1)
df = df.drop(['onset'], axis=1)


# ## Fill Missing Values with mode and median

# In[13]:


# fill in missing values with mode and median values for category and continuous variables separately 
columns_category = df.dtypes[df.dtypes=='category'].index.tolist()
columns_continuous = df.dtypes[df.dtypes !='category'].index.tolist()
for i in columns_category:
    df[i] = df[i].fillna(df[i].mode()[0])
for i in columns_continuous:
    df[i] = df[i].fillna(df[i].median())
    

# convert category features into one hot encoding and remove related features
categoryColumns_processed = [ 'f_male_bl', 'f_riluzole_bl', 'f_study_drug_bl', 'age_bin']
df_pro = df.copy()


df_pro = pd.concat([df_pro, pd.get_dummies(df_pro[categoryColumns_processed])], axis=1)
df_pro.drop(categoryColumns_processed, axis=1, inplace=True)
df_pro.rename(columns={'age_bin_(0, 40]': 'age_0040',  'age_bin_(40, 70]':'age_4070', 'age_bin_(70, 100]':'age_00100'},inplace=True)

label = 'r_alsfrs_r_total'
feature_selected = df_pro.columns.tolist()
feature_selected.remove('pid')
feature_selected.remove('r_alsfrs_r_total')
feature_selected.remove('set')
# df_pro['diff'] = df_pro['']


# In[14]:


# output  data
df_pro.to_csv('./data_processed_0411.csv', index=None) # save processed data


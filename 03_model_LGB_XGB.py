
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Load-Data-and-Packages" data-toc-modified-id="Load-Data-and-Packages-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Load Data and Packages</a></span></li><li><span><a href="#Model" data-toc-modified-id="Model-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Model</a></span><ul class="toc-item"><li><span><a href="#Data-Selection-and-Data-Split" data-toc-modified-id="Data-Selection-and-Data-Split-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Data Selection and Data Split</a></span></li><li><span><a href="#XGB-&amp;-LGB" data-toc-modified-id="XGB-&amp;-LGB-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>XGB &amp; LGB</a></span></li></ul></li><li><span><a href="#output-Train-and-Test-Files" data-toc-modified-id="output-Train-and-Test-Files-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>output Train and Test Files</a></span><ul class="toc-item"><li><span><a href="#output-Train-predicted-result" data-toc-modified-id="output-Train-predicted-result-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>output Train predicted result</a></span></li><li><span><a href="#output-Test-predicted-result" data-toc-modified-id="output-Test-predicted-result-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>output Test predicted result</a></span></li></ul></li></ul></div>

# # Load Data and Packages

# In[1]:


import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from math import sqrt
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression


# In[2]:


df = pd.read_csv('./data_processed_0411.csv',index_col=None)
# df=df[df.t<457]
df_pro = df[df.set==1]
df_test = df[df.set==0]

label = 'r_alsfrs_r_total'
feature_selected = df_pro.columns.tolist()
feature_selected.remove('pid')
feature_selected.remove(label)
feature_selected.remove('set')


# # Model

# ## Data Selection and Data Split

# In[3]:


# used to split data based on patient id
pid_list = list(set(df_pro['pid']))
mylist = []
np.random.seed(10)
for k in range(0, (len(pid_list))):
    x = np.random.randint(0, 10)
    mylist.append(x)     
columns = ['pid', 'cv_cohort']
cohort = pd.DataFrame(columns=columns)
cohort['pid'] = pid_list
cohort['cv_cohort'] = mylist


# ## XGB & LGB

# In[4]:


predicted_value_XGB = [ ]
predicted_value_LGB = [ ]
test_predicted_LGB = []
test_predicted_XGB = []

table = []
# temp_train = pd.DataFrame()
temp_test =pd.DataFrame()
for i in range(10):
    train_pid = cohort['pid'][cohort['cv_cohort'] != i]
    test_pid = cohort['pid'][cohort['cv_cohort'] == i]
    train = df_pro[df_pro['pid'].isin(train_pid)]
    test = df_pro[df_pro['pid'].isin(test_pid)]
    
    train_x = train[feature_selected][:]
    test_x = test[feature_selected][:]
    train_y = train[label][:]
    test_y = test[label][:]
    temp_test = pd.concat([temp_test, test[['pid', 't', 'month', label]]])
#     temp_train = pd.concat([temp_train, train[['pid', 't', 'month', label]]])
    # XGBoost
    dtrain=xgb.DMatrix(train_x,train_y)
    dtest=xgb.DMatrix(test_x, test_y)
    dvalid = xgb.DMatrix(df_test[feature_selected], df_test[label])
    params = {
    'booster': 'gbtree',
    'objective':'reg:linear',
    'max_depth':3, 
    'subsample': 0.6,
    'colsample_bytree': 0.6,
    'reg_alpha': 0.001,
    'min_child_weight': 11,
    'eta': 0.05,
    'seed': 42,
    'nthread': 4,
    'metris':'rmse'
    }

    watchlist = [(dtrain,'train'),(dtest,'test')]
    bst=xgb.train(params,dtrain,num_boost_round=10000, evals=watchlist, early_stopping_rounds=100, verbose_eval=50) 
    ypred_xgb=bst.predict(dtest)
    test_predicted_XGB.append(bst.predict(dvalid))
    # LGBRegressor
    lgb_model = lgb.LGBMRegressor(random_state=42, max_depth=7,
                                                            n_estimators=30000,
                                                            learning_rate=0.05,
                                                            num_leaves=7,
                                                            colsample_bytree=0.9,
                                                            subsample = 0.8,
                                                            reg_alpha = 0.5,
                                                            reg_lambda = 0.3,
                                                            n_jobs=-1)

    lgb_model.fit(train_x, train_y,
                  eval_metric='rmse', 
                  eval_set=[(train_x, train_y), (test_x, test_y)], 
                  verbose=100, early_stopping_rounds=100)

    ypred_lgb_sklearn = lgb_model.predict(test_x)
    test_predicted_LGB.append(lgb_model.predict(df_test[feature_selected]))
    
    predicted_value_XGB.extend(ypred_xgb)
    predicted_value_LGB.extend(ypred_lgb_sklearn)


# # output Train and Test Files

# ## output Train predicted result

# In[5]:


# obtain predicted train value table 
DNN_result = pd.read_csv('./train_predicted_value_DNN.csv')
DNN_result = DNN_result.sort_values(by=['pid','t']).reset_index(drop=True)
temp_test['mod1_XGB'] = predicted_value_XGB
temp_test['mod2_LGB'] = predicted_value_LGB
temp_test_reset = temp_test.sort_values(by=['pid','t']).reset_index(drop=True)
temp_test_reset['mod3_DNN'] = DNN_result['test_value']
temp_test_reset = temp_test_reset.rename(columns={'r_alsfrs_r_total':'true'})


# In[6]:


# calculate best ratio for final result 
# output predicted data 
a = temp_test_reset.mod1_XGB
b = temp_test_reset.mod2_LGB
c = temp_test_reset.mod3_DNN

best_ensemble_value =[10,0,0,0,0,0]
best_ensemble_value ={'Best_RMSE':10}

for i in range(11):
    for j in range(11):
        k = 10-(i+j)
        if k>=0:
            ensemble_value = 0.1*(a*i+b*j+c*k)
            r2_temp = r2_score(temp_test_reset["true"], pd.DataFrame(ensemble_value))
            rmse_temp = np.sqrt(mean_squared_error(temp_test_reset["true"], pd.DataFrame(ensemble_value)))
            if rmse_temp<best_ensemble_value['Best_RMSE']:
                best_ensemble_value['Best_RMSE'] = rmse_temp
                best_ensemble_value['R_2_Score'] = r2_temp
                best_ensemble_value['DNN_ratio'] = k
                best_ensemble_value['XGB_ratio'] = i
                best_ensemble_value['LGB_ratio'] = j
                best_ensemble_value['mod4_Ensemble'] = ensemble_value
                
temp_test_reset['mod4_Ensemble'] = best_ensemble_value['mod4_Ensemble']
temp_test_reset.to_csv('./train_predicted_value_full.csv', index=None)
print('Best Ratio for LGB {}, Best Ratio for XGB {}, Best Ratio for DNN {}, '.format(
    best_ensemble_value['LGB_ratio'], best_ensemble_value['XGB_ratio'], best_ensemble_value['DNN_ratio']))


# In[7]:


def prediction_summary(df, filter_feature=None, lb=0, ub=10^10):
    if filter_feature is not None:
        df = df[(df[filter_feature]>=lb)&(df[filter_feature]<=ub)]
    
    cols  = [i for i in df.columns if 'mod' in i]

    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    from scipy.stats import skew

    row_name = ['R^2', 'RMSE', 'Slope',  'Intercept' ,'Skewness']

    label = 'true'
    temp_value =[ [] for i in (cols)]
    for ind, feature in enumerate(cols):
        temp_value[ind].append(r2_score(df[label], df[feature]))
        temp_value[ind].append(np.sqrt(mean_squared_error(df[label], df[feature])))
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression().fit(df[label].values.reshape(-1, 1), df[feature])
        temp_value[ind].append(reg.coef_[0])
        temp_value[ind].append(reg.intercept_ )
        temp_value[ind].append(skew(df[feature]))
    temp_table = pd.DataFrame(temp_value).T
    temp_table.columns=cols
    temp_table.index=row_name
    return temp_table


# In[11]:


prediction_summary(temp_test_reset)
prediction_summary(temp_test_reset,'t', 0,388)


# ## output Test predicted result

# In[12]:


test_result = pd.read_csv('./test_predicted_value_DNN.csv')
test_result = test_result.rename(columns={'TRUE':'true'})


# In[13]:


# output test data for model XGBoost, LGBoost and DNN

test_table = df_test[['pid', 't', 'month', label]].copy()
test_table.reset_index(drop=True, inplace=True)
test_table.rename(columns={'r_alsfrs_r_total':'true'},inplace=True)

test_table['mod1_XGB'] = pd.DataFrame(test_predicted_XGB).T.mean(axis=1).values
test_table['mod2_LGB'] = pd.DataFrame(test_predicted_LGB).T.mean(axis=1).values
test_table['mod3_DNN'] = test_result['mod3_DNN']
test_table['mod4_Ensemble'] = test_table['mod1_XGB']*best_ensemble_value['XGB_ratio']+test_table['mod2_LGB']*best_ensemble_value['LGB_ratio']+test_table['mod3_DNN']*best_ensemble_value['DNN_ratio']

test_table.to_csv('./test_predicted_value_full.csv', index=None)


# In[14]:


prediction_summary(test_table)
prediction_summary(test_table,'t', 0,388)


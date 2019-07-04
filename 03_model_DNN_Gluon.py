
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Load-Data-and-Packages" data-toc-modified-id="Load-Data-and-Packages-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Load Data and Packages</a></span></li><li><span><a href="#Gluon" data-toc-modified-id="Gluon-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Gluon</a></span></li><li><span><a href="#output-Train-and-Test-Files" data-toc-modified-id="output-Train-and-Test-Files-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>output Train and Test Files</a></span><ul class="toc-item"><li><span><a href="#output-DNN-predicted-train-value" data-toc-modified-id="output-DNN-predicted-train-value-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>output DNN predicted train value</a></span></li><li><span><a href="#output-DNN-predicted-test-data" data-toc-modified-id="output-DNN-predicted-test-data-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>output DNN predicted test data</a></span></li></ul></li></ul></div>

# # Load Data and Packages

# In[1]:


import numpy as np
import pandas as pd
# data visualization
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from math import sqrt
from sklearn.metrics import r2_score

import mxnet as mx
from mxnet import gluon

from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon


# In[2]:


df = pd.read_csv('./data_processed_0411.csv',index_col=None)
label = 'r_alsfrs_r_total'
feature_selected = df.columns.tolist()
feature_selected.remove('pid')
feature_selected.remove(label)
feature_selected.remove('set')
df.loc[:,feature_selected] = df[feature_selected].apply(lambda x: (x-x.mean())/x.std())
df_pro = df[df.set==1]
df_test = df[df.set==0]


# # Gluon

# In[7]:


# Loss function
square_loss = gluon.loss.L2Loss()
def get_rmse(net, X_train,y_train):
    clipped_preds = nd.clip(net(X_train), 1, 48)
    return nd.sqrt(2*square_loss(clipped_preds, y_train).mean()).asscalar()

# define network structure
def get_net():
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(256, activation='relu'))
        net.add(gluon.nn.Dropout(0.5))
        net.add(gluon.nn.Dense(128, activation='relu'))
        net.add(gluon.nn.Dropout(0.5))
        net.add(gluon.nn.Dense(64, activation='relu'))
        net.add(gluon.nn.Dropout(0.3))
        net.add(gluon.nn.Dense(32, activation='relu'))
        net.add(gluon.nn.Dropout(0.1))
        net.add(gluon.nn.Dense(1))
    net.initialize()
    return net


# In[8]:


# used to split data based on patient id
np.random.seed(10)
pid_list = list(set(df_pro['pid']))
mylist = []
for k in range(0, (len(pid_list))):
    x = np.random.randint(0, 5)
    mylist.append(x)     
columns = ['pid', 'cv_cohort']
cohort = pd.DataFrame(columns=columns)
cohort['pid'] = pid_list
cohort['cv_cohort'] = mylist


# In[9]:


def train_model(k,net, X_train, y_train, X_test, y_test, df_test, epochs, verbose_epoch, learning_rate, weight_decay):
    print('training....')
    
    train_loss = []
    test_loss = []
    return_predicted_value = []
    return_predicted_value_true = []
    
    batch_size = 16
    # train data loaded
    dataset_train = gluon.data.ArrayDataset(X_train, y_train)
    data_iter_train = gluon.data.DataLoader(dataset_train, batch_size, shuffle=True)
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': learning_rate, 'wd': weight_decay})
    # parameter initialize firstly
    net.collect_params().initialize(force_reinit=True)
    for epoch in range(epochs):
        for data, label in data_iter_train:
            with autograd.record():
                output = net(data)
                loss = square_loss(output, label)
            loss.backward()
            trainer.step(batch_size)
            cur_train_loss = get_rmse(net, X_train, y_train)

        print("Epoch %d, train loss: %f" % (epoch, cur_train_loss))        
        cur_test_loss = get_rmse(net, X_test, y_test)
        print("Epoch %d, test_loss loss: %f" % (epoch, cur_test_loss))

        train_loss.append(cur_train_loss)
        if X_test is not None:
            cur_test_loss = get_rmse(net, X_test, y_test)
            test_loss.append(cur_test_loss)
    return_predicted_value.extend(net(X_test))
    return_predicted_value_true.extend(net(nd.array(df_test[feature_selected].values)))
    plt.plot(train_loss)
    plt.legend(['train'])
    if X_test is not None:
        plt.plot(test_loss)
        plt.legend(['train','test'])
    plt.savefig('./train_test_loss_'+str(k)+'.png')
    plt.show()
    if X_test is not None:
        return cur_train_loss, cur_test_loss, return_predicted_value, return_predicted_value_true
    else:
        return cur_train_loss


# In[10]:


def k_fold_cross_valid(k, epochs, verbose_epoch, df_pro, df_test, learning_rate, weight_decay, feature_selected, label, cohort):
    import time
    train_loss_sum = 0.0
    test_loss_sum = 0.0
    temp_test =pd.DataFrame()
    temp_test_real =  []
    predicted_value = []
    true_test_value = []
    for i in range(k):
        a = time.time()
        train_pid = cohort['pid'][cohort['cv_cohort'] != i]
        test_pid = cohort['pid'][cohort['cv_cohort'] == i]
        train = df_pro[df_pro['pid'].isin(train_pid)]
        test = df_pro[df_pro['pid'].isin(test_pid)]

        train_x = nd.array(train[feature_selected][:])
        train_y = nd.array(train[label][:])

        test_x = nd.array(test[feature_selected][:])
        test_y = nd.array(test[label][:])
        temp_test = pd.concat([temp_test, test[['pid', 't', 'month', label]]])
    
        net = get_net()
        train_loss, test_loss, return_predicted_value, return_predicted_value_true = train_model(i,net, train_x, train_y, test_x, test_y, df_test, epochs, verbose_epoch, learning_rate, weight_decay)
        predicted_value.extend(return_predicted_value)
        temp_test_real.append(return_predicted_value_true)
        true_test_value.extend(test_y)
        
        train_loss_sum += train_loss
        print("Test loss: %f" % test_loss)
        print('time pass per fold: ', time.time()-a)
        test_loss_sum += test_loss
        
    return train_loss_sum / k, test_loss_sum / k, predicted_value, temp_test, temp_test_real


# In[11]:


# huper-parameters
weight_decay = 0.05 #5 10 20
k = 5
epochs =25 #2 20 40
verbose_epoch = 10 
learning_rate = 0.0001 #0.001 


# In[ ]:


train_loss, test_loss, predicted_train, temp_test, temp_test_real = k_fold_cross_valid(k, epochs, verbose_epoch, df_pro, df_test, learning_rate, weight_decay, feature_selected, label, cohort)
print("%d-fold validation: Avg train loss: %f, Avg test loss: %f" % (k, train_loss, test_loss))


# # output Train and Test Files

# ## output DNN predicted train value 

# In[ ]:


predicted_tra = [predicted_train[i].asnumpy()[0] for i in range(len(predicted_train))]
temp_test['test_value'] = predicted_tra
temp_test.to_csv('./train_predicted_value_DNN.csv', index=None)


# ## output DNN predicted test data

# In[ ]:


test_predicted_all = []
for j in range(len(temp_test_real)):
    test_predicted_all.append([temp_test_real[j][i].asnumpy()[0] for i in range(len(temp_test_real[j]))])
    
test_table = df_test[['pid', 't', 'month', label]].copy()
test_table.reset_index(drop=True, inplace=True)
test_table.rename(columns={'r_alsfrs_r_total':'true'},inplace=True)
test_table['mod3_DNN'] = np.array(test_predicted_all).mean(axis=0)
test_table.to_csv('test_predicted_value_DNN.csv',index=None)


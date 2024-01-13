# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import warnings
import pickle
warnings.filterwarnings('ignore')
import scipy.stats
from matplotlib.ticker import MaxNLocator

def save_dict(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_dict(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def smooth(signal, window_len=5):
    if not window_len % 2:
        if window_len == 0:
            raise Exception("Window length needs to be a positive odd number.")
        else:
            window_len = window_len - 1
    if window_len > len(signal):
        raise Exception("Input signal length needs to be bigger than window size.")
 
    half_win = window_len //2
    sig_len = len(signal)
    smoothed_signal = np.zeros(sig_len)
    start_critical_index = half_win
    end_critical_index = sig_len - half_win - 1
    count = 0
    for index in range(0, sig_len):
        if index < start_critical_index:
            smoothed_signal[index] = np.mean(signal[0:2*index+1])
        elif index > end_critical_index:
            count += 1
            smoothed_signal[index] = np.mean(signal[-window_len+2*count:])
        elif index >=start_critical_index and index <= end_critical_index:
            startInd = max([0, index-half_win])
            endInd = min(index+half_win+1, sig_len)
            smoothed_signal[index] = np.mean(signal[startInd:endInd])
 
    return smoothed_signal 

def fit(x, y, coef = 1):
    x=np.array(x)
    y=np.array(y)
    
    x=x[~np.isnan(y)]
    y=y[~np.isnan(y)]
    x=x[~np.isinf(y)]
    y=y[~np.isinf(y)]

    y=y[~np.isnan(x)]
    x=x[~np.isnan(x)]
    y=y[~np.isinf(x)]
    x=x[~np.isinf(x)]
    x_fit=np.linspace(np.min(x),np.max(x),100)
    coef = np.polyfit(x, y, coef)
    
    xs_ori=np.linspace(np.min(x),np.max(x),10)
    xs=list()
    ys=list()
    for i in range(0,len(xs_ori)-1):
        indexs1=[i for i in np.where(x>xs_ori[i])[0]]
        indexs2=[i for i in np.where(x<=xs_ori[i+1])[0] if i in indexs1]
        if(len(indexs2)>0):
            xs.append(xs_ori[i])
            ys.append(np.mean(np.array(y)[indexs2]))
    
    xx=0
    xy=0
    for i in range(len(xs)):
        xx+=pow(xs[i],2)
        xy+=xs[i]*ys[i]
    k=xy/xx
    y_fit=k*x_fit
    return x_fit, y_fit,k

def goodness_of_fit(y_fitting, y_no_fitting):
    def __sst(y_no_fitting):
        y_mean = np.mean(y_no_fitting)
        sst = np.sum([(y - y_mean)**2 for y in y_no_fitting])
        return sst
    
    def __ssr(y_fitting, y_no_fitting):
        y_mean = np.mean(y_no_fitting)
        ssr = np.sum([(y - y_mean)**2 for y in y_fitting])
        return ssr
    
    def __sse(y_fitting, y_no_fitting):
        sse = np.sum([(y_fitting[i] - y_no_fitting[i])**2 for i in range(len(y_fitting))])
        return sse
    
    SSR = __ssr(y_fitting, y_no_fitting)
    SST = __sst(y_no_fitting)
    SSE = __sse(y_fitting, y_no_fitting)
    print(SSR,SST,SSE)
    rr = 1 - SSE / (SSR+SSE)
    return rr

def pearson(x,y):
    x=np.array(x)
    y=np.array(y)
    
    x=x[~np.isnan(y)]
    y=y[~np.isnan(y)]
    x=x[~np.isinf(y)]
    y=y[~np.isinf(y)]

    y=y[~np.isnan(x)]
    x=x[~np.isnan(x)]
    y=y[~np.isinf(x)]
    x=x[~np.isinf(x)]
    return str(round(scipy.stats.pearsonr(x,y)[0],2))

def spearman(x,y):
    x=np.array(x)
    y=np.array(y)
    
    x=x[~np.isnan(y)]
    y=y[~np.isnan(y)]
    x=x[~np.isinf(y)]
    y=y[~np.isinf(y)]

    y=y[~np.isnan(x)]
    x=x[~np.isnan(x)]
    y=y[~np.isinf(x)]
    x=x[~np.isinf(x)]
    return str(round(scipy.stats.spearmanr(x,y)[0],2))

B=1
C=2
eta = 0.3
source = 657

times = load_dict('B_PPI2_global_time_'+str(int(100*B))+'_'+str(int(100*C))).tolist()[0]
weights_hens = load_dict('B_PPI2_global_degree_hens_'+str(int(100*B))+'_'+str(int(100*C))).tolist()[0]
weights_L = load_dict('B_PPI2_global_degree_L_'+str(int(100*B))+'_'+str(int(100*C))).tolist()[0]
weights_multi_hens = load_dict('B_PPI2_global_degree_multi_hens_'+str(int(100*B))+'_'+str(int(100*C))).tolist()[0]
weights_multi = load_dict('B_PPI2_global_degree_multi_'+str(int(100*B))+'_'+str(int(100*C))).tolist()[0]

times=[i for (k,i) in enumerate(times) if k!=source and k<len(weights_multi)]
weights_hens=[i for (k,i) in enumerate(weights_hens) if k!=source and k<len(weights_multi)]
weights_L=[i for (k,i) in enumerate(weights_L) if k!=source]
weights_multi_hens=[i for (k,i) in enumerate(weights_multi_hens) if k!=source]
weights_multi=[i for (k,i) in enumerate(weights_multi) if k!=source]

colors = ['#ABBAE1', '#526D92', '#1B1464']

fig=plt.figure(figsize=(13,6))
ax=fig.add_subplot(111)

ax.plot(times,times,c=colors[2],linewidth=4,linestyle='-', alpha = 0.7)

fit_weights_hens, fit_times_hens, fit_k_hens=fit(weights_hens,times)
pearsonr_hens=pearson(weights_hens, times)
spearmanr_hens=spearman(weights_hens, times)
rr_hens=str(round(goodness_of_fit([fit_k_hens * i for i in weights_hens],times),2))
label_hens=r'$\mathcal{L}_d:\rho_s='+spearmanr_hens+'$'+'\n'+r'$\quad\  ,\rho_p='+pearsonr_hens+'$'
ax.scatter([fit_k_hens * i for i in weights_hens],times,s=70,marker = 's',c='none', edgecolors='#CCCCCC',label=label_hens,alpha =0.7)

fit_weights_multi_hens, fit_times_multi_hens, fit_k_multi_hens=fit(weights_multi_hens,times)
pearsonr_multi_hens=pearson(weights_multi_hens, times)
spearmanr_multi_hens=spearman(weights_multi_hens, times)
rr_multi_hens=str(round(goodness_of_fit([fit_k_multi_hens * i for i in weights_multi_hens],times),2))
label_multi_hens=r'$\mathcal{L}_{mp}:\rho_s='+spearmanr_multi_hens+'$'+'\n'+r'$\qquad\  \rho_p='+pearsonr_multi_hens+'$'
ax.scatter([fit_k_multi_hens * i for i in weights_multi_hens],times,s=70,marker = '^',c='none',edgecolors=colors[2],label=label_multi_hens,alpha =0.7)

fit_weights_multi, fit_times_multi, fit_k_multi=fit(weights_multi,times)
pearsonr_multi=pearson(weights_multi, times)
spearmanr_multi=spearman(weights_multi, times)
rr_multi=str(round(goodness_of_fit([fit_weights_multi * i for i in weights_multi],times),2))
label_multi=r'$\mathcal{L}_{itp}:\rho_s='+spearmanr_multi+'$'+'\n'+r'$\qquad\quad \rho_p='+pearsonr_multi+'$'
ax.scatter([fit_k_multi * i for i in weights_multi],times,s=70,marker = 'o',c='none',edgecolors=colors[1],label = label_multi,alpha =0.7)

plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=2))
plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=2))

ax.tick_params(axis='both',which='both',direction='out',width=1,length=10, labelsize=25)
bwith=1
ax.spines['bottom'].set_linewidth(bwith)\
    
ax.spines['left'].set_linewidth(bwith)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xlim([0,2])
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.xlabel(r"$\mathcal{L}(m\to i)$",fontsize=45)
plt.ylabel(r"$T(m\to i)$",fontsize=45)
plt.legend(fontsize=20,loc=1,bbox_to_anchor=(1.6,1))
plt.tight_layout()
# plt.axis('equal')
plt.savefig('B_PPI2_large_global_multi_'+str(int(100*B))+'.pdf',dpi=300)
plt.show()
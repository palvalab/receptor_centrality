# -*- coding: utf-8 -*-

"""
@author: Felix Siebenhühner

Main script for analyses in:
 "Linking the microarchitecture of neurotransmitter systems to large-scale MEG resting state networks"
by Siebenhühner et al.

Required libraries:
os, sys, numpy, scipy, random, pandas, matplotlib, bct, netneurotools
networkx, pickle, statsmodels, nibabel, nilearn, sklearn, pyvista

"""   
    


#%% Set project directory

base_dir  = ''            # insert project path here
fsav_dir  = f'{base_dir}/data/fsaverage'



#%% Import modules


import os
import numpy as np
from scipy.stats import zscore, pearsonr, spearmanr
import pandas as pd
import random

from sklearn.decomposition import PCA
from netneurotools.stats import gen_spinsamples

import sys
sys.path.insert(0,f'{base_dir}functions/')

import parcellation_functions as parcfun 
import plot_functions as plotfun 
from support_functions import get_bc, get_rec_densities, corr_perm, get_sigs        

plotfun.set_mpl_defs()



#%% Plot settings


colorcycle = np.array(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                       '#8c564b', '#d370b2', '#7f7f7f', '#bcbd22', '#17becf'])

colors = [np.array([(251,33,21),(255,102,188),(48, 30,234),( 26,136,154),(52,152,70),(135,195,10),(244,193,30),(247,130,17)]),
          np.array([(251,33,21),(255,102,188),(48, 30,234),( 26,136,154),(52,152,70),(135,195,10),(244,193,30),(247,130,17)]),
          np.array([(251,33,21),( 48, 30,234),(52,152, 70),(247,130,17)])]


parc = parcfun.Parc('parc2018yeo17_200',dir1=f'{base_dir}/data/')

n_parc = 200


#%% Frequency settings


freqs = np.array([ 
        1.05,  1.38,  1.63,  1.85,  2.15,  2.49,  2.88,  3.31,  3.73,
        4.15,  4.75,  5.35,  5.93,  6.62,  7.39,  8.14,  9.02,  9.83,
       10.92, 11.89, 13.11, 14.78, 16.3 , 17.8 , 19.7 , 21.6 , 23.73,
       26.55, 28.7 , 31.8 , 34.5 , 37.9 , 42.5 , 46.9 , 52.1 , 59.3 ,
       65.3 , 71.  , 78.  , 85.92, 95.6 ]) 

freq_strings = [str(f) for f in freqs]

n_freq    = 41
fb_inds_L = [range(0,8),range(8,12),range(12,17),range(17,21),range(21,30),range(30,41)]
f_bands_L = ['delta','low-theta','theta-alpha','high-alpha','beta','gamma']
n_fb      = len(fb_inds_L)



#%% Load connectivity data


dtype    = 'iPLV'              # 'iPLV' or 'oCC'
N_subj   = 67
conn     = np.load(f'{base_dir}data/conn_{dtype}_{N_subj}.npy') 
coh_inds = np.arange(N_subj)

mf_inds     = np.genfromtxt(f'{base_dir}data/mf_inds_{N_subj}.csv')
male_inds   = np.where(mf_inds == 1)[0]
female_inds = np.where(mf_inds == 2)[0]



#%%  Brain plots of node strength in frequency bands - Supp Figure 3


fig_type = 'png'

meg_values    = np.nanmean(conn,(0,3)) 
meg_values_fb = np.array([np.nanmean(meg_values[ix],0) for ix in fb_inds_L])

if dtype == 'iPLV':
    vranges = [[0.024,0.030], [0.017,0.021], [0.019,0.025], [0.021,0.029], [0.009,0.013], [0.0054,0.0086]]
else:
    vranges = [[0.056,0.068], [0.040,0.060], [0.060,0.120], [0.060,0.140], [0.030,0.070], [0.014,0.048]]
     
plot_dir =  f'{base_dir}/plots/conn_fb/'
os.makedirs(plot_dir,exist_ok=True)
    
for fb in range(n_fb):
    filename =  None if fig_type == 'none' else f'{plot_dir}/{dtype}_{f_bands_L[fb]}.{fig_type}'
    plotfun.plot_multi_view_pyvista(meg_values_fb[fb],parc,'plasma',fsav_dir ,figsize=[4,4],
                                    filename=filename, vrange=vranges[fb], nticks=3, title='', label=dtype)
       

#%% Prepare random permutation indices


n_perm = 10000
perms = np.zeros([n_parc,n_perm],'int')


for i in range(10000):
    lx = np.arange(100)
    rx = np.arange(100,200)
    random.shuffle(lx)
    random.shuffle(rx)
    perms[:,i] = np.concatenate((lx,rx))



#%% Prepare 'spin' indices (permutations with spatial autocorrelation preserved) 


hemiid  = np.zeros((parc.N, ))
hemiid[:int(parc.N/2)] = 1

coords_x = np.genfromtxt(f'{base_dir}/data/parcel_coords_parc2018yeo17_200.csv',delimiter=';')

nspins = 10000
spins  = gen_spinsamples(coords_x, hemiid, n_rotate=nspins, seed=1534)



#%% Load receptor maps


all_rec, rec_map_names, NR = get_rec_densities(f'{base_dir}/data/receptor_maps_17_200/',parc.N)

rec_names = np.array(['MU','H3','a4b2','CB1','VAChT','NAT',
                          '5HT1a','5HT1b','5HT2a','5HT4','5HT6','5HTT',
                          'D1','D2','DAT','GABAa','NMDA','M1','mGluR5'])

# z-score maps
all_rec = np.array([zscore(x) for x in all_rec])

# average maps with weighting
rec_comb = [[0,1],[2],[3],[4,5],[6,7,8],[9,10],[11,12],[13,14,15],[16,17,18],
            [19],[20],[21,22],[23],[24,25,26],[27,28],[29,30],[31],[32],[33,34,35]]

rec_weight = [204,39,1,1,22,77,18,4,5,10,77,8,36,36,22,65,19,29,3,59,30,100,30,13,49,37,55,6,174,16,6,29,24,22,28,73]

all_rec_weighted = np.array([all_rec[i]*rec_weight[i] for i in range(NR)])

rec_mean_weighted = np.array([np.mean(all_rec_weighted[comb],0) for comb in rec_comb])


# z-score the resulting maps
rec_mean_weighted = np.array([zscore(x) for x in rec_mean_weighted])


# rearrange
rec_idx    = [14, 11, 4, 5, 3, 0, 1, 17, 13, 12, 10, 9, 8, 7, 6, 2, 18, 16, 15]
rec_values = rec_mean_weighted[rec_idx]
rec_names  = rec_names[rec_idx]
n_rec      = len(rec_values)  



#%% Brain plots of density maps - Supp Figure 1

vrange=[-2,2]

fig_type = 'svg'
plot_dir = f'{base_dir}/plots/density_maps/'
os.makedirs(plot_dir,exist_ok=True)

for r in range(n_rec):
    
    filename = None if fig_type == 'none' else f'{plot_dir}{rec_names[r]}.{fig_type}'
    plotfun.plot_multi_view_pyvista(rec_values[r],parc,'plasma', fsav_dir,
                                    filename = filename, vrange=vrange, nticks=5, 
                                    title=rec_names[r])
  
       



#%%

'''   ###################################################################   '''
"""   ###############   PRINCIPAL COMPONENT ANALYSIS   ##################   """
'''   ###################################################################   '''



#%%  PCA on receptors


rec_values_z = np.array([zscore(va) for va in rec_values])

df1 = pd.DataFrame(rec_values_z.transpose())
df1.columns = rec_names
n_comp = 5

pca_rec  = PCA(n_components=n_comp)
pcas_rec = pca_rec.fit_transform(rec_values_z.transpose())


principal_rec_df = pd.DataFrame(data = pcas_rec,
              columns = ['PC '+str(i) for i in range(n_comp)])

principal_rec_df.tail()

print('\nExplained variation per principal component: {}'.format(pca_rec.explained_variance_ratio_))

pc_array = principal_rec_df.to_numpy()



#%% brain plots of PCs - Figure 1A


vrange=[-2.4,2.4]

fig_type = 'svg'
plot_dir = f'{base_dir}/plots/PC_brain_plots/'
os.makedirs(plot_dir,exist_ok=True)

for c in range(n_comp):
    filename =  None if fig_type == 'none' else f'{plot_dir}PC{c+1}.{fig_type}'
    f1 = plotfun.plot_multi_view_pyvista(pc_array[:,c],parc,'plasma', fsav_dir,
                                         filename = filename, vrange=vrange, nticks=5, 
                                         title='PC '+str(c+1))
  
           
    
#%% compute loadings of individual maps with PCs


PC_loadings = np.zeros([n_comp,n_rec])

for c in range(n_comp):
    for r in range(n_rec):        
        PC_loadings[c,r]  = pearsonr(rec_values[r],pc_array[:,c])[0]
    print(c)
        
    
    
#%% plot PC loadings - Figure 1B
    

plot_dir = f'{base_dir}/plots/receptor_loadings/'

fig_type = 'none'


fig1 = plotfun.plot_heatmap(PC_loadings.transpose(),figsize=[5.5,9],cbarf=0.0714,
                     yticks=range(n_rec), yticklabels=rec_names,
                     xticks=range(n_comp), xticklabels=['PC '+str(i+1) for i in range(n_comp)],
                     cmap='RdBu_r',zmin=-1,zmax=1,return_fig=1)

ax = fig1.get_axes()[0]

if fig_type != 'none':
    os.makedirs(plot_dir,exist_ok=True)
    filename = f'{plot_dir}Receptor_loadings.{fig_type}'
    fig1.savefig(filename)



#%%

'''   ###################################################################   '''
'''   ##########    COMPUTE COVARIANCE WITH RECEPTOR MAPS    ############   '''
'''   ###################################################################   '''



#%% Set metric, density, groups


metric = 'str'    # 'str' or 'bc' or 'deg'

density = 50      # percentage of edges to keep for sparse matrices

sub_dir = f'{dtype}_{metric}' if metric == 'str' else  f'{dtype}_{metric}{density}'
    
all_inds = [coh_inds,male_inds,female_inds]  
group_names = ['all','male','female']
      
n_gr = len(all_inds)

# set method for correlation and for multiple comparison correction        
method  = 'Spearman'
mc_meth = 'fdr_bh'


if metric == 'str':
    meg_values    = np.array([np.nanmean(conn[inds],(0,3)) for inds in all_inds])
    meg_values_fb = np.array([[np.nanmean(mv[ix],0) for ix in fb_inds_L] for mv in meg_values])
    
elif metric == 'bc':
    meg_values    = np.array([[get_bc(conn[inds,f].mean(0),density) for f in range(n_freq)] for inds in all_inds])
    meg_values_fb = np.array([[get_bc(conn[inds].mean(0)[fx].mean(0),density) for fx in fb_inds_L] for inds in all_inds])
    n_gr = 1
    
elif metric == 'deg':
    conn2         = [np.nanmean(conn[inds],(0)) for inds in all_inds]    
    thresh        = np.array([[np.percentile(cx[f],(100-density)) for f in range(n_freq)] for cx in conn2])
    meg_values    = np.array([[np.nanmean((conn2[g][f] * (conn2[g][f] > thresh[g][f])),0) for f in range(n_freq)] for g in range(n_gr)])
    conn2_fb      = np.array([[np.nanmean(conn2[g][ix],0) for ix in fb_inds_L] for g in range(n_gr)])
    thresh_fb     = np.array([[np.percentile(cx[fb],(100-density)) for fb in range(n_fb)] for cx in conn2_fb])
    meg_values_fb = np.array([[np.nanmean((conn2_fb[g][fb] * (conn2_fb[g][fb] > thresh_fb[g][fb])),0) for fb in range(n_fb)] for g in range(n_gr)])
    n_gr = 1



#%%
 
'''   ##########        COVARIANCE IN FREQUENCY BANDS        ############   '''
  

#%% Compute covariances in frequency bands     


method = 'Spearman'

n_perm = 10000

rec_covs_fb         = np.zeros([n_gr,n_rec,n_fb])
rec_pvals_c_fb      = np.zeros([n_gr,n_rec,n_fb])
rec_pvals_s_fb      = np.zeros([n_gr,n_rec,n_fb])
rec_pvals_p_fb      = np.zeros([n_gr,n_rec,n_fb])
rec_pvals_s_fb_mod  = np.zeros([n_gr,n_rec,n_fb])
rec_pvals_p_fb_mod  = np.zeros([n_gr,n_rec,n_fb])
rec_corrs_p_null_fb = np.zeros([n_gr,n_rec,n_fb,n_perm])
rec_corrs_s_null_fb = np.zeros([n_gr,n_rec,n_fb,n_perm])


for g in range(n_gr):
    for i in range(n_rec):
        for fb in range(n_fb):
                                 
            rho, pval_c, pval_p, null_p = corr_perm(meg_values_fb[g][fb],rec_values[i], perms, n_perm, method)
            rho, pval_c, pval_s, null_s = corr_perm(meg_values_fb[g][fb],rec_values[i], spins, n_perm, method)

            rec_covs_fb[g,i,fb]         = rho
            rec_pvals_c_fb[g,i,fb]      = pval_c
            rec_pvals_s_fb[g,i,fb]      = pval_s
            rec_pvals_p_fb[g,i,fb]      = pval_p
            rec_corrs_p_null_fb[g,i,fb] = null_p
            rec_corrs_s_null_fb[g,i,fb] = null_s

        print(i)

         
        
#%% plot heatmap of covariances per frequency band - Figure 3


fig_type = 'none'

plot_dir = f'{base_dir}plots/FB_maps_corr/{fig_type}/{sub_dir}'

for g in range(n_gr):    
        
    sigs, rec_pvals_p_fb_mod[g], rec_pvals_s_fb_mod[g], stars = get_sigs(rec_covs_fb[g],rec_pvals_p_fb[g],rec_pvals_s_fb[g])

    
    # make tiled heatmap plot
    plot_data = rec_covs_fb[g]
    fig1 = plotfun.plot_heatmap(plot_data,figsize=[6,9],cbarf=0.066,
                     yticks=range(n_rec), yticklabels=rec_names,
                     xticks=range(n_fb),  xticklabels=f_bands_L,
                     cmap='RdBu_r',zmin=-1,zmax=1,return_fig=1,title=group_names[g])
        
    ax = fig1.get_axes()[0]
    
    # add stars for significance
    for i in range(len(plot_data)):
        for j in range(len(plot_data[0])):
            text = ax.text(j, i, stars[i,j],fontsize=12,
                       ha="center", va="center", color="k")
    

    # save figure
    if fig_type != 'none':        
        os.makedirs(plot_dir,exist_ok=True)        
        filename = f'{plot_dir}/{dtype}_rec_corr_heatmap_{group_names[g]}.{fig_type}'
        fig1.savefig(filename,bbox_inches="tight" )



#%% Save r and p values for frequency bands to csv
      
stats_dir = f'{base_dir}stats/{sub_dir}/'
os.makedirs(stats_dir,exist_ok=True)
        
for g in range(n_gr):                       

    df_data = np.concatenate((rec_covs_fb[g],
                              rec_pvals_p_fb[g],rec_pvals_p_fb_mod[g],
                              rec_pvals_s_fb[g],rec_pvals_s_fb_mod[g]),1)

    index = np.concatenate((
              ['r_'+f_bands_L[fb] for fb in range(n_fb)],           
              ['p_perm_'+f_bands_L[fb] for fb in range(n_fb)],
              ['p_perm_adj_'+f_bands_L[fb] for fb in range(n_fb)],
              ['p_spin_'+f_bands_L[fb] for fb in range(n_fb)],
              ['p_spin_adj_'+f_bands_L[fb] for fb in range(n_fb)]))
    
    filename = f'{stats_dir}/{sub_dir}_cov_{group_names[g]}.csv'
    df1 = pd.DataFrame(df_data.T, columns = rec_names, index = index)
    df1.to_csv(filename,sep=';')



#%%

'''   ##########      COVARIANCE IN MORLET FREQUENCIES       ############   '''



#%% compute covariances for individual frequencies     


n_perm = 10000

rec_covs        = np.zeros([n_gr,n_rec,n_freq])
rec_pvals_c     = np.zeros([n_gr,n_rec,n_freq])
rec_pvals_s     = np.zeros([n_gr,n_rec,n_freq])
rec_pvals_p     = np.zeros([n_gr,n_rec,n_freq])
rec_pvals_s_mod = np.zeros([n_gr,n_rec,n_freq])
rec_pvals_p_mod = np.zeros([n_gr,n_rec,n_freq])
rec_covs_p_null = np.zeros([n_gr,n_rec,n_freq,n_perm])
rec_covs_s_null = np.zeros([n_gr,n_rec,n_freq,n_perm])

for g in range(1):
    for i in range(n_rec):
        for f in range(n_freq):
            
            rho, pval_c, pval_p, null_p = corr_perm(meg_values[g][f],rec_values[i], perms, n_perm, method)    
            rho, pval_c, pval_s, null_s = corr_perm(meg_values[g][f],rec_values[i], spins, n_perm, method)     

            rec_covs[g,i,f]        = rho
            rec_pvals_c[g,i,f]     = pval_c
            rec_pvals_s[g,i,f]     = pval_s
            rec_pvals_p[g,i,f]     = pval_p
            rec_covs_p_null[g,i,f] = null_p
            rec_covs_s_null[g,i,f] = null_s

        print(i)



#%%  plot lineplot of covariances per indiv. frequency - Supp. Fig 4
   

fig_type = 'none'

plot_dir = f'{base_dir}plots/lineplots_corr/{fig_type}/'

alpha = 0.05

if metric in ['bc','degree']:
    suffix  = '_' + metric + '_' + str(density)
else:
    suffix = ''


# choose basis for identifying significant interactions

sig_type = 'spin_p'              


plot_inds = [[18,17,16,15],[14,13,12,11,10,9,8], [7,6,5,4,3,2,1,0]]
figsize = [7,4.2]


#line plots grouped    
for g in range(1):
    
    sigs, rec_pvals_p_mod[g], rec_pvals_s_mod[g], stars = get_sigs(rec_covs[g],rec_pvals_c[g],rec_pvals_s[g],ind1='',ind2='◆',ind3='',ind4='◆')
    
    for px in range(len(plot_inds)):
        gr_inds   = plot_inds[px]
        cmap_line = plotfun.make_cmap(colors[px][:len(gr_inds)]/255)
        fig1      = plotfun.plot_corr(rec_covs[g,gr_inds],freqs,rec_names[gr_inds],ylim=[-1,1],nc=4,method=method,cmap=cmap_line,
                    xlim=[1,100],lpos='uc',figsize=figsize,markersize=6,return_fig = True,sig_corr=[sigs[1][gr_inds],sigs[3][gr_inds]])  
    
        # save figure
        if fig_type != 'none':
            os.makedirs(plot_dir,exist_ok=True)
            file1 = f'{plot_dir}{dtype}_covariance_{group_names[g]}_{px}_{sig_type}.{fig_type}'
            fig1.savefig(file1)
        
        

#%% Save r and p values for individual frequencies to csv
      

stats_dir = f'{base_dir}stats/{sub_dir}/'
os.makedirs(stats_dir,exist_ok=True)
        
for g in range(1):
                       
    df_data = np.concatenate((rec_covs[g],
                              rec_pvals_p[g],rec_pvals_p_mod[g],
                              rec_pvals_s[g],rec_pvals_s_mod[g]),1)

    index = np.concatenate((
              ['r_'+freq_strings[f] for f in range(n_freq)],           
              ['p_corr_'+freq_strings[f] for f in range(n_freq)],
              ['p_corr_adj_'+freq_strings[f] for f in range(n_freq)],
              ['p_spin_'+freq_strings[f] for f in range(n_freq)],
              ['p_spin_adj_'+freq_strings[f] for f in range(n_freq)]))
    
    filename = f'{stats_dir}/{sub_dir}_cov_{group_names[g]}_nb.csv'
    df1 = pd.DataFrame(df_data.T, columns = rec_names, index = index)
    df1.to_csv(filename,sep=';')




#%%


"""    ##########     COVARIANCE WITH PRINCIPAL COMPONENTS     ##########   """



#%% Compute covariance of frequency bands with PCs 


method = 'Spearman'

n_perm = 10000


PC_cov             = np.zeros([n_gr,n_comp,n_fb])
PC_cov_pvals_c     = np.zeros([n_gr,n_comp,n_fb])
PC_cov_pvals_p     = np.zeros([n_gr,n_comp,n_fb])
PC_cov_pvals_s     = np.zeros([n_gr,n_comp,n_fb])
PC_cov_pvals_p_mod = np.zeros([n_gr,n_comp,n_fb])
PC_cov_pvals_s_mod = np.zeros([n_gr,n_comp,n_fb])
PC_cov_p_null      = np.zeros([n_gr,n_comp,n_fb,n_perm])
PC_cov_s_null      = np.zeros([n_gr,n_comp,n_fb,n_perm])

for g in range(n_gr):
    for c in range(n_comp):
        for fb in range(n_fb):
            rho, pval_c, pval_p, null_p   = corr_perm(meg_values_fb[g][fb],pc_array[:,c],perms, n_perm,method=method)
            rho, pval_c, pval_s, null_s   = corr_perm(meg_values_fb[g][fb],pc_array[:,c],spins, n_perm,method=method)

            PC_cov[g,c,fb]          = rho
            PC_cov_pvals_c[g,c,fb]  = pval_c
            PC_cov_pvals_p[g,c,fb]  = pval_p
            PC_cov_pvals_s[g,c,fb]  = pval_s
            PC_cov_p_null[g,c,fb]   = null_p   
            PC_cov_s_null[g,c,fb]   = null_s    

        print(c)



#%% Plot covariance of PCs as heatmaps
    

sig_type = 'spin_p_fdr'

fig_type = 'none'

plot_dir = f'{base_dir}/plots/PC_fb_cov_{dtype}/'

tag = f'{dtype}_{metric}' if metric == 'str' else  f'{dtype}_{metric}{density}'

 
for g in range(n_gr):    
    
    sigs, PC_cov_pvals_p_mod[g], PC_cov_pvals_s_mod[g], stars = get_sigs(PC_cov[g],PC_cov_pvals_p[g],PC_cov_pvals_s[g])

    
    fig2 = plotfun.plot_heatmap(PC_cov[g][::-1],figsize=[5.8,2.5],cbarf=0.045,
                          xticks=range(n_fb), xticklabels=f_bands_L,xticklab_rot=45,
                          yticks=range(5), yticklabels=['PC '+str(5-i) for i in range(5)],
                          cmap='RdBu_r',zmin=-1,zmax=1,return_fig=1)
    
   
    ax = fig2.get_axes()[0]
    
    for i in range(n_comp):
        for j in range(n_fb):
            text = ax.text(j,i, stars[::-1][i,j],fontsize=12,
                       ha="center", va="center", color="k")
    
    if fig_type != 'none':
        os.makedirs(plot_dir,exist_ok=True)
        filename = f'{plot_dir}PC_cov_{group_names[g]}_{tag}_{sig_type}.{fig_type}'      
        fig2.savefig(filename,bbox_inches="tight")
    
    
    
#%% Save r and p values to csv


if metric in ['bc','deg']:
    sub_dir  = f'{dtype}_{metric}{density}_PC'
else:
    sub_dir = f'{dtype}_{metric}_PC'
    
stats_dir = f'{base_dir}stats/{sub_dir}/'
os.makedirs(stats_dir,exist_ok=True)

        
for g in range(n_gr):
                       
    df_data = np.concatenate((PC_cov[g],
                              PC_cov_pvals_p[g],PC_cov_pvals_p_mod[g],
                              PC_cov_pvals_s[g],PC_cov_pvals_s_mod[g]),1)
        
    index = np.concatenate((
              ['r_'+f_bands_L[fb] for fb in range(n_fb)],           
              ['p_corr_'+f_bands_L[fb] for fb in range(n_fb)],
              ['p_corr_adj_'+f_bands_L[fb] for fb in range(n_fb)],
              ['p_spin_'+f_bands_L[fb] for fb in range(n_fb)],
              ['p_spin_adj_'+f_bands_L[fb] for fb in range(n_fb)]))
    
    filename = f'{stats_dir}/{sub_dir}_cov_fb_{group_names[g]}.csv'
    df1 = pd.DataFrame(df_data.T, index = index, columns = ['PC'+str(i+1) for i in range(5)])
  
    df1.to_csv(filename,sep=';')




#%%








# -*- coding: utf-8 -*-
"""
Created on Fri May 31 13:55:48 2024

@author: Felix Siebenhühner

"""

import numpy as np
import networkx as nx

from scipy.stats import pearsonr, spearmanr



def get_rec_densities(dir15,n_parc,delimiter=','):
    
    rec_names = np.array(
    ['MU_carfentanil_hc204_kantonen',     'MU_carfentanil_hc39_turtonen',
     'H3_cban_hc8_gallezot',     'A4B2_flubatine_hc30_hillmer',
     'CB1_FMPEPd2_hc22_laurikainen',     'CB1_omar_hc77_normandin',
     'VAChT_feobv_hc18_aghourian_sum',     'VAChT_feobv_hc4_tuominen',
     'VAChT_feobv_hc5_bedard_sum',     'NAT_MRB_hc10_hesse',
     'NAT_MRB_hc77_ding',     '5HT1a_cumi_hc8_beliveau',
     '5HT1a_way_hc36_savli',     '5HT1b_az_hc36_beliveau',
     '5HT1b_p943_hc22_savli',     '5HT1b_p943_hc65_gallezot',
     '5HT2a_alt_hc19_savli',     '5HT2a_cimbi_hc29_beliveau',
     '5HT2a_mdl_hc3_talbot',     '5HT4_sb20_hc59_beliveau',
     '5HT6_gsk_hc30_radhakrishnan',     '5HTT_dasb_hc100_beliveau',
     '5HTT_dasb_hc30_savli',      'D1_SCH23390_hc13_kaller',
     'D2_fallypride_hc49_jaworska',     'D2_flb457_hc37_smith',
     'D2_flb457_hc55_sandiego',     'DAT_fepe2i_hc6_sasaki',
     'DAT_fpcit_hc174_dukart_spect',     'GABAa-bz_flumazenil_hc16_norgaard',
     'GABAa_flumazenil_hc6_dukart',      'NMDA_ge179_hc29_galovic',
     'M1_lsn_hc24_naganawa',     'mGluR5_abp_hc22_rosaneto',
     'mGluR5_abp_hc28_dubois',     'mGluR5_abp_hc73_smart'])
    
    NR = len(rec_names)
    
    all_rec = np.zeros([NR,n_parc])
        
    for i in range(NR):
              
         all_rec[i] = np.genfromtxt(f'{dir15}{rec_names[i]}.csv',delimiter=delimiter)
        
    return all_rec, rec_names, NR


def get_bc_bin(vals,density,k=None):
    thresh = np.percentile(vals,(100-density))
    vals *= (vals>thresh)
    gr = nx.from_numpy_array(vals)
    return np.array(list(nx.betweenness_centrality(gr,k).values()))


def get_bc(vals,density,k=None):
    thresh = np.percentile(vals,(100-density))
    vals *= (vals>thresh)
    gr = nx.from_numpy_array(vals)
    return np.array(list(nx.betweenness_centrality(gr,k).values()))




def corr_perm(x, y, perms, n_perm, method='Spearman'):
    """
    Function for computing correlations, as well as a p-value from permutations
    """
    
    null = np.zeros([n_perm])
    
    if method == 'Pearson':
        rho, pval_c = pearsonr(x, y)    
        for i in range(n_perm):
            null[i], _ = pearsonr(x[perms[:, i]], y)
    elif method == 'Spearman':
        rho, pval_c = spearmanr(x, y)    
        for i in range(n_perm):
            null[i], _ = spearmanr(x[perms[:, i]], y)

    pval_s = (1 + sum(abs((null - np.mean(null))) >
                    abs((rho - np.mean(null))))) / (n_perm + 1)
    return rho, pval_c, pval_s, null





def get_sigs(corrs,pvals_p,pvals_s,alpha=0.05,ind1='',ind2='◆',ind3='',ind4='◆'):
    """ 
    Get significant values before and after FDR-correction.
    """
       
    stars = np.empty(corrs.shape,dtype='object')

        
    level1     = pvals_p < alpha    
    pvals_sort = np.sort(pvals_p.flatten())
    n_sig      = np.sum(level1)
    ind        = n_sig - np.round(np.prod(corrs.shape)*alpha).astype(int)      
    ind        = 0 if ind < 0 else ind
    level2     = pvals_p <= pvals_sort[ind]
    pvals_p_m  = np.ones(pvals_p.shape)
    pvals_p_m[level2]=  pvals_p[level2] 
    
    level3     = pvals_s < alpha    
    pvals_sort = np.sort(pvals_s.flatten())
    n_sig      = np.sum(level3)
    ind        = n_sig - np.round(np.prod(corrs.shape)*alpha).astype(int)       
    ind        = 0 if ind < 0 else ind
    level4     = pvals_s <= pvals_sort[ind]
    pvals_s_m  = np.ones(pvals_s.shape)
    pvals_s_m[level4]=  pvals_s[level4] 
    
          
    stars[np.where(level1)]=ind1
    stars[np.where(level2)]=ind2
    stars[np.where(level3&level2)]=ind2+ind3
    stars[np.where(level4&level2)]=ind2+ind4


    return [level1,level2,level3,level4], pvals_p_m, pvals_s_m, stars




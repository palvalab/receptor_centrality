# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 14:59:28 2020
@author: Felix Siebenhühner
"""
import numpy as np


class Parc:
    def __init__(self,name,dir1='L:\\nttk_palva\\Utilities\\parcellations\\parc_info\\'
                 ,suffix='',split=False,LR_suffix='auto',old_abbr=False):
        self.name = name        
        all_names = np.genfromtxt(dir1 + name + suffix + '.csv',delimiter=';',dtype='str')
        if name == 'parc2009':
            self.N             = 148
            self.names         = all_names[:,0]
            self.abbr          = all_names[:,1]
            self.networks      = (all_names[:,2]).astype('int')
            self.netw_names    = ['Control','Default','Dorsal Attention','Limbic','Ventral Attention','Somatomotor','Visual']
            self.netw_abbr     = ['Con','DMN','DAN','Lim','VAN','SMN','Vis']
            self.netw_masks    = get_network_masks(self.networks,7,self.N)
            self.netw_indices  = [np.where(self.networks==i)[0] for i in range(7)]
            self.NN            = np.sum(self.netw_masks,(2,3))
            try:
                self.lobe     = all_names[:,3]
            except:
                pass
            
        elif name == 'AAL90':
            self.N = 90
            self.names = all_names[:,0]
            

        else:
            self.N = int(name[-3:])
            self.names = all_names[:,0]
            if LR_suffix == 'lhrh':
                self.names = modify_LR_suffix(self.names,LR_suffix='lhrh')
            if LR_suffix == 'LeftRight':
                self.names = modify_LR_suffix(self.names,LR_suffix='LeftRight')
        
            
            self.abbr  = all_names[:,1]
            
            if '2018yeo7' in name:        
                self.netw_names   = ['Control','Default','Dorsal Attention',
                                      'Limbic','Ventral Attention','Somatomotor','Visual']
                self.netw_abbr2   = ['Con','DMN','DAN','Lim','VAN','SMN','Vis']
                self.netw_abbr    = ['CT' ,'DM' ,'DA' ,'Lim','VA' ,'SM' ,'Vis']
                self.networks     = get_networks(self.abbr,self.netw_abbr)
                self.N_netw       = 7
                self.netw         = 'yeo7'
                self.netw_indices = [np.where(self.networks==i)[0] for i in range(7)]
                self.netw_masks   = get_network_masks(self.networks,7,self.N)
                self.NN           = np.sum(self.netw_masks,(2,3))
                
                #self.lobe         = np.concatenate((all_names[::2,4],all_names[1::2,4]))
                
            elif '2018yeo17' in name and not split:
                self.netw_names   = ['ContA','ContB','ContC','DefaultA','DefaultB','DefaultC','DorsAttnA',
                                      'DorsAttnB','LimbicA_TempPole','LimbicB_OFC','SalVentAttnA','SalVentAttnB',
                                      'SomMotA','SomMotB','TempPar','VisCent','VisPeri']
                if old_abbr:
                    self.netw_abbr    = ['ConA', 'ConB', 'ConC', 'DA_A', 'DA_B', 'DefA', 'DefB', 'DefC',
                     'LimA', 'LimB', 'SM_A', 'SM_B', 'TP', 'VA_A', 'VA_B', 'VisC', 'VisP']
                else:
                    self.netw_abbr    = ['CT-A','CT-B','CT-C','DM-A','DM-B','DM-C','DA-A',
                                      'DA-B','LimA','LimB','VA-A','VA-B',
                                      'SM-A','SM-B','TPar','VisA','VisB'] 
                self.networks     = get_networks(self.abbr,self.netw_abbr)
                self.netw         = 'yeo17'
                self.netw_indices = [np.where(self.networks==i)[0] for i in range(17)]
                self.N_netw       = 17
                self.netw_masks   = get_network_masks(self.networks,17,self.N)  
                self.NN           = np.sum(self.netw_masks,(2,3))

                try:
                    self.lobe     = all_names[:,2]
                except:
                    pass
                
            elif '2018yeo17' in name and split:
                self.netw_names   = ['ContA','ContB','ContC','DefaultA','DefaultB',
                                     'DefaultC','DorsAttnA','DorsAttnB','LimbicA_TempPole',
                                     'LimbicB_OFC','SalVentAttnA','SalVentAttnB',
                                     'SomMotA','SomMotB','TempPar','VisCent','VisPeri']

                
                self.netw_abbr    = ['CT-A','CT-B','CT-C','DM-A','DM-B','DM-C','DA-A',
                                      'DA-B','LimA','LimB','VA-A','VA-B',
                                      'SM-A','SM-B','TPar','VisA','VisB']
                self.networks     = get_networks_HS(self.names,self.netw_abbr)                
                
                self.netw_names   = [n + '_L' for n in self.netw_names] + \
                                    [n + '_R' for n in self.netw_names]
                self.netw_abbr    = [n + '_L' for n in self.netw_abbr] + \
                                    [n + '_R' for n in self.netw_abbr]
                self.netw         = 'yeo34'
                self.netw_indices = [np.where(self.networks==i)[0] for i in range(34)]
                self.N_netw       = 34
                self.netw_masks   = get_network_masks(self.networks,34,self.N)  
                self.NN           = np.sum(self.netw_masks,(2,3))
                self.name         = self.name + '_s' 

                try:
                    self.lobe     = all_names[:,2]
                except:
                    pass

             
    def __repr__(self):
        return ('\'' + self.name + '\'')
    
    

def get_networks(parcel_names,network_names):   
    ''' Get networks for parcels, independent of hemisphere '''
         
    net_ind = np.zeros(len(parcel_names))
    for s, system in enumerate(network_names):
        for p, parcel in enumerate(parcel_names):
            if system in parcel:
                net_ind[p] = s
    return net_ind.astype('int')


def get_networks_HS(parcel_names,network_names):      
    ''' Get networks for parcels, differing by hemisphere '''
      
    net_ind = np.zeros(len(parcel_names))
    for s, system in enumerate(network_names):
        for p, parcel in enumerate(parcel_names):
            if system in parcel and 'LH' in parcel:
                net_ind[p] = s
            elif system in parcel and 'RH' in parcel:
                net_ind[p] = s + len(network_names)
    return net_ind.astype('int')


def get_network_masks(network_indices,N_netw,N_parc):
    ''' Create binary masks for all networks '''
    
    network_masks = np.zeros([N_netw,N_netw,N_parc,N_parc])
    for i in range(N_netw):
        for j in range(N_netw):
            network_masks[i,j] = np.matmul(np.transpose([network_indices==i]),[network_indices==j]).astype('int')
    return network_masks    


def get_fidelity_and_cpp(project_dirs,set_sel,parc,suffix,parc_type=''):    
    ''' Load fidelity and cross-parcel PLV values for a list of sets '''
    
    N_sets = len(set_sel)
    fid = np.zeros([N_sets,parc.N])
    cpp = np.zeros([N_sets,parc.N,parc.N])

    
    for s in range(N_sets):            
        pd      = project_dirs[s]
        set1    = set_sel[s]
        subject = set1[:5]
        try:

            dir1    = pd + subject + '\\Fidelity' + parc_type + '\\' 
            file1   = dir1 + 'Fidelity_' + set1 + suffix + '_' + parc.name + '.csv'
            file2   = dir1 + 'CP-PLV_'   + set1 + suffix + '_' + parc.name + '.csv'
            fid[s]  = np.genfromtxt(file1,delimiter=';') 
            cpp[s]  = np.genfromtxt(file2,delimiter=';') 
        except:
            print('Not found: '+file1)
    ## get fid*fid product
    fidsq = np.zeros([len(fid),parc.N,parc.N])
    for s in range(len(fid)):
        fidsq[s] = np.outer(fid[s],fid[s])
    return fid, cpp, fidsq

   



def get_mean_netw(values,parc,axis=1):
    ''' get mean over an axis for each network
    INPUT: 
        values: 2D array [freqs x parcels]
        parc:   a Parcellation object
    '''
    
    a = [np.nanmean(values[:,parc.netw_indices[n]],axis) for n in range(parc.N_netw)]
    
    return a


def get_mean_NN(values,parc):
    ''' get mean K for each network pair
    INPUT: 
        values: 3D array [freqs x parcels x parcels]
        parc:   a Parcellation object
    '''
    
    N_freq = len(values)
    K = np.zeros([N_freq,parc.N_netw,parc.N_netw])
    
    for f in range(N_freq):
        for n1 in range(parc.N_netw):
            for n2 in range(parc.N_netw):
                K[f,n1,n2] = np.nansum(values[f] * parc.netw_masks[n1,n2])/parc.NN[n1,n2]
    
    return K


def get_K_NN(values,parc):
    ''' get mean K for each network pair
    INPUT: 
        values: 3D array [freqs x parcels x parcels]
        parc:   a Parcellation object
    '''
    
    N_freq = len(values)
    K = np.zeros([N_freq,parc.N_netw,parc.N_netw])
    
    for f in range(N_freq):
        for n1 in range(parc.N_netw):
            for n2 in range(parc.N_netw):
                K[f,n1,n2] = np.nansum((values[f] != 0) * parc.netw_masks[n1,n2])/parc.NN[n1,n2]
    
    return K







def get_indices_old_new(parc_name):
    ''' Load indices for mapping from between old and new Schaefer parcellations.
        Returns indices_old_to_new, indices_new_to_old.'''

    N = int(parc_name[-3:])
    
    dir_map     = 'L:\\nttk_palva\\Utilities\\parcellations\\changes_to_yeo\\names_old_and_new\\'    
    matrix      = np.genfromtxt(dir_map + parc_name + '.csv',delimiter=';',dtype='str')
    
    indices_new_to_old = matrix[:,3].astype('int') 
    indices_old_to_new = np.array([np.where(indices_new_to_old==i)[0][0] for i in range(N)])  
     
    return indices_old_to_new, indices_new_to_old



def get_indices_7_17(N,ptype='new'):
    ''' Load indices for mapping between 7 and 17 parcellation for N parcels and ptype 'old' or 'new' '''
    
    dir_map = 'L:\\nttk_palva\\Utilities\\parcellations\\mapping_7_17\\' + ptype + '_parcellations\\'    
    matrix  = np.genfromtxt(dir_map +  str(N) + '.csv',delimiter=';',dtype='str')
    
    indices_17_to_7  = matrix[:,3].astype('int') 
    indices_7_to_17 = np.array([np.where(indices_17_to_7==i)[0][0] for i in range(N)])  
    
    return indices_7_to_17, indices_17_to_7



def morph(IM0,indices):    

            
    IM1 = IM0[indices]
    IM2 = IM1[:,indices]
    
    return IM2



def morph_7_to_17(IM0,ptype='new'):
    
    N = len(IM0)
    indices_7_to_17 = get_indices_7_17(N,ptype)[0]
            
    IM1 = IM0[indices_7_to_17]
    IM2 = IM1[:,indices_7_to_17]
    
    return IM2


def morph_17_to_7(IM0,ptype='new'):
    
    N = len(IM0)
    indices_17_to_7 = get_indices_7_17(N,ptype)[1]

    IM1 = IM0[indices_17_to_7]
    IM2 = IM1[:,indices_17_to_7]
    
    return IM2


def morph_old_to_new(IM0,parc_name):
    
    indices_old_to_new = get_indices_old_new(parc_name)[0]
            
    IM1 = IM0[indices_old_to_new]
    IM2 = IM1[:,indices_old_to_new]
    
    return IM2


def morph_new_to_old(IM0,parc_name):
    
    indices_new_to_old = get_indices_old_new(parc_name)[1]
            
    IM1 = IM0[indices_new_to_old]
    IM2 = IM1[:,indices_new_to_old]
    
    return IM2



def modify_LR_suffix(parcel_names,LR_suffix='LeftRight'):
    ''' Ensures that parcel suffixes are either in -lh/-rh or __Left/__Right format.
    Parameter LR_suffix either as 'LeftRight' or 'lhrh' 
    '''    
    
    if LR_suffix == 'LeftRight':
        parcel_names = np.char.replace(parcel_names,'-lh','__Left')
        parcel_names = np.char.replace(parcel_names,'-rh','__Right')
    else:
        parcel_names = np.char.replace(parcel_names,'__Left','-lh')
        parcel_names = np.char.replace(parcel_names,'__Right','-rh')
    return parcel_names


def add_LR_suffix(parcel_names,LR_suffix='LeftRight'):
    ''' Ensures that parcel suffixes are either in -lh/-rh or __Left/__Right format.
    Parameter LR_suffix either as 'LeftRight' or 'lhrh' 
    '''    
        
    for i in range(len(parcel_names)):
        name = parcel_names[i]
        if LR_suffix == 'LeftRight':            
            if 'LH' in name:
                parcel_names[i] = name + '__Left'
            elif 'RH' in name:
                parcel_names[i] = name + '__Right'
        else:       
            if 'LH' in name:
                parcel_names[i] = name + '-lh'
            elif 'RH' in name:
                parcel_names[i] = name + '-rh'
        
    return parcel_names



def remove_LR_suffix(parcel_names,LR_suffix='LeftRight'):
    ''' Ensures that parcel suffixes are either in -lh/-rh or __Left/__Right format.
    Parameter LR_suffix either as 'LeftRight' or 'lhrh' 
    '''    
        
    for i in range(len(parcel_names)):
        name = parcel_names[i]
        if LR_suffix == 'LeftRight':            
            if 'LH' in name:
                parcel_names[i] = parcel_names[i].replace('__Left','')
            elif 'RH' in name:
                parcel_names[i] = parcel_names[i].replace('__Right','')
        else:       
            if 'LH' in name:
                parcel_names[i] = parcel_names[i].replace('-lh','')
            elif 'RH' in name:
                parcel_names[i] = parcel_names[i].replace('-rh','')
        
    return parcel_names





    
    
def match_parcels(parcel_names,parc_name,old=False,LR_suffix='LeftRight'):
    ''' Gets parcels of a particular parcellation into original order '''
    
    dirPN = 'L:\\nttk_palva\\Utilities\\parcellations\\parc_info\\'
    dirPO = 'L:\\nttk_palva\\Utilities\\parcellations\\parc_info_old\\'    
    
    
    if not np.any([x in parcel_names[3] for x in ['__Left','__Right','-lh','-rh']]):
        parcel_names2 = add_LR_suffix(parcel_names.astype('U100'),LR_suffix)
    else:
        parcel_names2 = modify_LR_suffix(parcel_names.astype('U100'),LR_suffix)
        
    N  = len(parcel_names2)
    if old:
        parcels_MNE = list(np.genfromtxt(dirPO + parc_name + '_info.csv',delimiter=';',dtype='str')[:,0])
    else:
        parcels_MNE = list(np.genfromtxt(dirPN + parc_name + '.csv',delimiter=';',dtype='str')[:,0])
        
    indices = np.full(N,np.nan)
    
    for i in range(N):
        indices[i] = parcels_MNE.index(parcel_names2[i])
    indices_inv = indices.astype('int')
        
    indices_fwd = np.array([np.where(indices==i)[0][0] for i in range(N)])  

    return parcels_MNE, indices_fwd, indices_inv
    
    
    
def test_label_equivalency_by_vertices(labelsA,labelsB,print_detailed=False):
    ''' Tests whether the labels in two collections are equivalent (and in same order) by comparing their vertices '''
    
    results =  [np.all(labelsA[i].vertices == labelsB[i].vertices) for i in range(len(labelsA))]
    
    if print_detailed:
        print(results)
    
    if np.all(results):
        print('Labels are equivalent.')
    else:
        print('Labels are not equivalent.')


def compare_two_labels(label1,label2):
    ''' Tests whether the labels in two collections are equivalent (and in same order) by comparing their vertices '''
    
    result =  label1.vertices == label2.vertices
    

    if result:
        print('Labels are equivalent.')
    else:
        print('Labels are not equivalent.')







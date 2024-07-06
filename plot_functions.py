# -*- coding: utf-8 -*-
"""
@author: Felix Siebenhühner
"""

#%% import external libraries

import os
import numpy as np
import scipy
import matplotlib as mpl
import matplotlib.colors as mpc
import matplotlib.pyplot as plt
import nibabel as nib
import pyvista as pv
import statsmodels.sandbox.stats.multicomp as mc

from matplotlib import cm
from scipy.stats import spearmanr, pearsonr

from numpy.ma import masked_invalid
from mpl_toolkits.axes_grid1 import make_axes_locatable




#%% helping functions

class Bunch():
    __init__ = lambda self, **kw: setattr(self, '__dict__', kw)
    
def get_colorcycle():
    return np.array(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                           '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])

def _depth(obj,c=0):
    ''' 
    Recursive function that returns depth/dimensionality of an array or list of lists.
    e.g. depth of a scalar is 0, of 1D-array 1, of 2D array 2, ...    
    '''    
    try:
         if type(obj) != str:
             obj1 = obj[0]
             c = _depth(obj1,c+1)
         else:
             return c
    except:
         return c
    return c


def _repeat_if_needed(param,N_plots,depth): 
    '''
    Recursive function that repeats parameters for plotting functions. Use with caution.
    INPUT:
        param: a single parameter or N-dimenional array
        N_plots: number of plots
        depth: target dimensionality for the return array
    '''    
    
    if _depth(param)==0:
        if param==None and depth>0:
            depth=1     
    if _depth(param) >= depth:
        param = param
    else:
        param = [param for n in range(N_plots)]
        param = _repeat_if_needed(param,N_plots,depth)
    return param
    


def correlate(values1, values2, method='Spearman', remove_nan = True, print_N = False):
    
    ''' 
    INPUT:
        values: 1D array
        values2: 1D array
    OUTPUT:
        corr, p_val: single values
    '''
    
    if remove_nan:
        inds1 = np.where(values1 != values1)[0]
        inds2 = np.where(values2 != values2)[0]
        inds  = [i for i in range(len(values1)) if not i in inds1 and i not in inds2]
        values1  = values1[inds]
        values2  = values2[inds]
        if print_N:
            print(len(values1))

    if method=='Pearson':
        corr, p_val = pearsonr(values1,values2)
    elif method=='Spearman':
        corr, p_val = spearmanr(values1,values2)  
        
    return corr, p_val   



def set_mpl_defs():
    mpl.style.use('default')
    mpl.rcParams['axes.grid'] = False
    mpl.rcParams['font.size'] =  12
    mpl.rcParams['axes.titlesize'] =  12
    mpl.rcParams['axes.labelsize'] =  12
    mpl.rcParams['legend.fontsize'] = 10
    mpl.rcParams['xtick.labelsize'] =  11
    mpl.rcParams['ytick.labelsize'] =  11
    mpl.rcParams['lines.linewidth'] =  1
    mpl.rcParams['axes.facecolor'] = '1'
    mpl.rcParams['figure.facecolor'] = '1'

def set_mpl_defs2():
    mpl.style.use('default')
    mpl.rcParams['axes.grid'] = False
    mpl.rcParams['font.size'] =  15
    mpl.rcParams['axes.titlesize'] =  18
    mpl.rcParams['axes.labelsize'] =  15
    mpl.rcParams['legend.fontsize'] = 10
    mpl.rcParams['xtick.labelsize'] =  13
    mpl.rcParams['ytick.labelsize'] =  13
    mpl.rcParams['lines.linewidth'] =  1
    mpl.rcParams['axes.facecolor'] = '1'
    mpl.rcParams['figure.facecolor'] = '1'



def make_cmap(colors, position=None, bit=False, len1=256):
    
    '''
    Takes a list of tuples which contain RGB values. 
    The RGB values may either be in 8-bit [0 to 255] (in which bit must be set to
    True when called) or arithmetic [0 to 1] (default). 
    Returns a cmap with equally spaced colors.
    Arrange your tuples so that the first color is the lowest value for the
    colorbar and the last is the highest. 
    Position contains values from 0 to 1 to dictate the location of each color.
    '''
    
    import numpy as np
    bit_rgb = np.linspace(0,1,len1)
    if position == None:
        position = np.linspace(0,1,len(colors))
    else:
        if len(position) != len(colors):
            print("position length must be the same as colors")
        elif position[0] != 0 or position[-1] != 1:
            print("position must start with 0 and end with 1")
    if bit:
        for i in range(len(colors)):
            colors[i] = (bit_rgb[colors[i][0]],
                         bit_rgb[colors[i][1]],
                         bit_rgb[colors[i][2]])
    cdict = {'red':[], 'green':[], 'blue':[]}
    for pos, color in zip(position, colors):
        cdict['red'].append((pos, color[0], color[0]))
        cdict['green'].append((pos, color[1], color[1]))
        cdict['blue'].append((pos, color[2], color[2]))

    cmap = mpl.colors.LinearSegmentedColormap('my_colormap',cdict,len1)
    
    return cmap


def make_cmap_from_hex(colors, position=None, bit=False):
    
    '''
    Takes a list of tuples which contain RGB values. 
    The RGB values may either be in 8-bit [0 to 255] (in which bit must be set to
    True when called) or arithmetic [0 to 1] (default). 
    Returns a cmap with equally spaced colors.
    Arrange your tuples so that the first color is the lowest value for the
    colorbar and the last is the highest. 
    Position contains values from 0 to 1 to dictate the location of each color.
    '''
    
    import numpy as np
    bit_rgb = np.linspace(0,1,256)
    colors2 = [[int(c[1:3],16)/255,int(c[3:5],16)/255,int(c[5:7],16)/255] for c in colors]
    if position == None:
        position = np.linspace(0,1,len(colors2))
    else:
        if len(position) != len(colors2):
            print("position length must be the same as colors")
        elif position[0] != 0 or position[-1] != 1:
            print("position must start with 0 and end with 1")
    if bit:
        for i in range(len(colors2)):
            colors2[i] = (bit_rgb[colors2[i][0]],
                          bit_rgb[colors2[i][1]],
                          bit_rgb[colors2[i][2]])
    cdict = {'red':[], 'green':[], 'blue':[]}
    for pos, color in zip(position, colors2):
        cdict['red'].append((pos, color[0], color[0]))
        cdict['green'].append((pos, color[1], color[1]))
        cdict['blue'].append((pos, color[2], color[2]))

    cmap = mpl.colors.LinearSegmentedColormap('my_colormap',cdict,256)
    
    return cmap






def make_better_subplots(NR,NC,figsize):
    
    ''' creates subplots, makes sure that axes object is 2D '''
    
    fig, axes = plt.subplots(NR,NC,figsize=figsize)
    
    if NR == 1 and NC == 1:
        axes = np.array([[axes]])    
    elif NR == 1 :
        axes = np.array([axes])
    elif NC == 1 :
        axes = np.transpose(np.array([axes]))

        
    return fig, axes



def get_4int_color_from_hex(x):
    return [int(x[1:3],16)/255,int(x[3:5],16)/255,int(x[5:7],16)/255,1]


def get_4int_color_from_str(x):
    y = mpl.colors.get_named_colors_mapping()[x]
    if '#' in y:
        y = get_4int_color_from_hex(y)
    else:
        y = list(y) + [1]
    return y

def get_colorspace(cmap,n_colors):   
    if type(cmap) is mpl.colors.LinearSegmentedColormap:
        colorspace = [cmap(i) for i in np.linspace(0, 1, n_colors)]               # get colors from colormap instance
    elif isinstance(cmap, str):
        colorspace = [plt.get_cmap(cmap)(i) for i in np.linspace(0, 1, n_colors)] # get colors from colormap by name  
    elif isinstance(cmap[0], str):  
        colorspace = []
        for c in cmap:
            if '#' in c:
                colorspace.append(get_4int_color_from_hex(c)) # get from hex values
            else:
                colorspace.append(get_4int_color_from_str(c))
    else:
        colorspace = cmap[:n_colors]                                              # get colors from array or list
    return colorspace


def mc_array(p_vals,mc_meth,alpha=0.05):
    ''' 
    Multiple comparison correction with established methods.
    INPUT:
        p_vals:  Array of any dimensionality
        mc_meth: Method, can be:
                bonferroni : one-step correction
                sidak : one-step correction
                holm-sidak : step down method using Sidak adjustments
                holm : step-down method using Bonferroni adjustments
                simes-hochberg : step-up method (independent)
                hommel : closed method based on Simes tests (non-negative)
                fdr_bh : Benjamini/Hochberg (non-negative)
                fdr_by : Benjamini/Yekutieli (negative)
                fdr_tsbh : two stage fdr correction (non-negative)
                fdr_tsbky : two stage fdr correction (non-negative) 
    OUTPUT:
        Array of 0/1 values, same shape as p_vals input.
    '''
    p_vals1 = np.reshape(p_vals, np.prod(p_vals.shape))
    sign_corr  = 1*mc.multipletests(p_vals1, method=mc_meth, alpha=alpha)[0]    
    return np.reshape(sign_corr,p_vals.shape)




#%%  histogram plots


def plot_histogram(data, N_bins=20, width=0.7,figsize=[8,6],return_fig=False):

    fig = plt.figure(figsize=figsize)
    hist, bins = np.histogram(data, bins=N_bins)
    width = width * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) // 2
    plt.bar(center, hist, align='center', width=width)
    plt.show()
    
    if return_fig:
        return fig
    
   



#%% line plots


def plot_lines(data, names=None, figsize=[13,4], cmap='jet', marker=None,
               xlabel ='', ylabel = '', xticks = None, yticks = None, 
               xticklabels = None, xticklab_rot = 0, yticklabels = None,
               title = None, less_spines = True, zero_line=False,xvals=None,
               outfile = None, xlim = None, ylim = None, ylim2 = None, ylabel2 = '',
               fontsize=12, return_fig=False):
    ''' INPUT:
        data:    1D array or list, or 2D array or list
        names:   list of names of data series for legend
        figsize: Figure size
        cmap:    Colormap. Can be:
                 - the name of a library cmap
                 - an instance of mpc.LinearSegmentedColormap    
                 - a list of colors as characters or RGB tuples (in 0-1 range)
        xlabel, ylabel: axis labels
        xticks, yticks: axis ticks, list of int
        less_spines: no axes on right and top 
        outfile: file to save the plot to 
        xlim, ylim: x and y limits, e.g. xlim=[-4,4]
        return_fig: whether to return the figure to caller as an object
    '''  
    try:                            # if 1D, force to 2D
        d54 = data[0][0]
    except IndexError:
        data = [data]
        
    fig = plt.figure(figsize=figsize)
    ax  = plt.subplot(111)    
    

    
    if type(cmap) is list:
        colors = cmap
    elif type(cmap) is mpc.LinearSegmentedColormap:
        colors = [cmap(i) for i in np.linspace(0, 1, len(data))] 
    else:
        colors = [plt.get_cmap(cmap)(i) for i in np.linspace(0, 1, len(data))]         
                
    if ylim2 != None:
        ax2 = ax.twinx()
        ax2.set_ylim(ylim2)
        ax2.set_ylabel(ylabel2,rotation=270,labelpad=11,color=colors[1])
        ax.set_ylabel(ylabel, fontsize=fontsize,color=colors[0])
        ax2.spines[['left', 'top']].set_visible(False)
    else:
        ax.set_ylabel(ylabel, fontsize=fontsize)
        ax2=ax
        
    if zero_line:
        ax.plot(np.zeros(len(data[0])),color=(0.5,0.5,0.5),label='_nolegend_')        

    if np.any(xvals == None):
        ax.plot(data[0],color=colors[0],marker=marker)
        for i in range(1,len(data)):
            ax2.plot(data[i],color=colors[i],marker=marker)
    else:
        ax.plot(xvals,data[0],color=colors[0],marker=marker)
        for i in range(1,len(data)):
            ax2.plot(xvals,data[i],color=colors[i],marker=marker)
        
    ax.tick_params(labelsize=fontsize-2)
    ax.set_xlabel(xlabel,fontsize=fontsize)
    if np.all(names != None):
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(names,loc='center left', bbox_to_anchor=(1, 0.5),fontsize=fontsize,frameon=False)

    if np.all(xticks != None):        
        ax.set_xticks(xticks)  
    if np.all(xticklabels != None):
        ax.set_xticklabels(xticklabels,fontsize=fontsize,rotation=xticklab_rot)
    if np.all(yticks != None):        
        ax.set_yticks(yticks)
    if np.all(yticklabels != None):
        ax.set_yticklabels(yticklabels,fontsize=fontsize)
    if np.all(xlim != None):
        ax.set_xlim(xlim)
    if ylim != None:
        ax.set_ylim(ylim)
    if title != None:
        fig.suptitle(title)

    if less_spines:        
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    
    if outfile != None:
        plt.savefig(outfile) 
       
    if return_fig:
        return fig


    
    
def semi_log_plot(data1, freqs, figsize=[7,3],
                  xlim=[1,100], ylabel='variable', ylim=None, CI=None,
                  legend=None, legend_pos='best', cmap='gist_rainbow',
                  ncols=3, xticks=None, bgcolor=None, title=None,yticks='none',
                  fontsize=11, markersize=0, show=True, outfile='none',
                  xlabel = 'Frequency [Hz]',  sig_id="none", sig_fac=1.05, sig_style='*', 
                  plot_zero=True, return_fig=False,ylim2=None,ylabel2=None,
                  Q=0): 
    
    '''
    plots data over log-spaced frequencies; 
    with or without confidence intervals (or standard deviation)
    INPUT:
      figsize:       figure size [width, height]
      data:          2D or 3D array/list (without or with CI/SD, resp.)
                     either [groups x freqs]
                     or [groups x (mean, lower bound, upper bound) x freqs]
      freqs:         1D or 2D array/list of frequencies (float)
                     use 2D if different frequencies for different groups                         
      xlim:          [xmin,xmax]
      ylim:          [ymin,ymax]
      ylabel:        label for the y axis    
      CI:            None if no CI/SDs, else alpha value (0 to 1) for CI/SD areas
      legend:        array of strings, 1 for each group
      legend_pos:    position of legend ('uc','br' or 'ur'); no legend if None
      cmap:          either name of a standard colormap 
                     or an instance of matplotlib.colors.LinearSegmentedColormap
      ncols:         number of columns in the plot legend
      xticks:        custom values for xticks. if None, standard value are used
      bgcolor:       background color
      fontsize:      fontsize
      markersize:    size of data point marker, default = 0
      show:          if False, the figure is not shown in console/window
      outfile:       if not None, figure will be exported to this file        
      sig_id:        indices where significance is to be indicated
                      can be 2D as [sig,sig_c], then sig in black, sig_c in red
      sig_fac:       controls the height at which significance indicators shown
      sig_style:     if '*' or '-' or '-*': indicated by stars above ploted lines
                     if 's' or 'o': indicated as markers on the plotted lines
      plot_zero:     whether to draw the x-axis  
      return_fig:    whether to return figure as object to caller
      ylim2:         [ymin2,ymax2]: If not None, a second y-axis will be added 
                     and used for all plots from 2nd onwards
      ylabel2:         label for 2nd y-axis

  
                   
    '''
    depth = lambda L: isinstance(L, list) and max(map(depth, L))+1
    fig,ax=plt.subplots(figsize=figsize) 
    if sig_id != 'none':
        if _depth(sig_id)==1:
            sig_id[0]=sig_id[0]-sig_id[1]
        sig_id=np.array(sig_id)
        sig_id[np.where(sig_id==0)]=np.nan    
    colorspace = get_colorspace(cmap,len(data1))
    if CI != None:
        colorspace_CI = np.array(colorspace)*np.array([1,1,1,CI])                              # colors for confidence intervals
    ax.set_prop_cycle(color=colorspace)                                                          # set different colors for different plots
    if ylim2 != None:
        ax2 = ax.twinx()
        ax2.set_ylim(ylim2)
        ax2.set_ylabel(ylabel2,rotation=270,labelpad=11,color=colorspace[1])
        ax.set_ylabel(ylabel, fontsize=fontsize,color=colorspace[0])
        ax2.spines[['left', 'top']].set_visible(False)
    else:
        ax.set_ylabel(ylabel, fontsize=fontsize)

    for i in range(len(data1)):                                                              # for each plot i
        if depth(freqs)==2:    
            freqs2=freqs[i]
        else:
            freqs2=freqs
        if CI != None: 
            N_F = len(data1[i][0])
            if ylim2 != None and i>0:
                ax2.plot(freqs2[:N_F],data1[i][0],'o-',markersize=markersize,color=colorspace[i])                                     # if CI, data for each plot i comes as [mean,CI_low, CI_high]
                ax2.fill_between(freqs2,data1[i][1],data1[i][2],color=colorspace_CI[i],label='_nolegend_')             # fill between CI_low and CI_high
            else:
                ax.plot(freqs2[:N_F],data1[i][0],'o-',markersize=markersize,color=colorspace[i])                                     # if CI, data for each plot i comes as [mean,CI_low, CI_high]
                ax.fill_between(freqs2,data1[i][1],data1[i][2],color=colorspace_CI[i],label='_nolegend_')             # fill between CI_low and CI_high
        else:
            N_F = len(data1[i])
            if ylim2 != None and i>0:
                ax2.plot(freqs2[:N_F],data1[i],'o-',markersize=markersize)   
            else:
                ax.plot(freqs2[:N_F],data1[i],'o-',markersize=markersize)   

    if plot_zero:
        ax.plot(freqs2,np.zeros(N_F),'k')      
    if Q>0:
        ax.plot(freqs2,np.full(N_F, Q),'gray')  
        ax.plot(freqs2,np.full(N_F,-Q),'gray')   

    if xticks==None:
        xticks=[1,2,3,5,10,20,30,50,100,200,300]    
    if yticks!='none':
        ax.set_yticks(yticks)
    ax.set_xscale('log')
    ax.set_xticks(xticks)
    if bgcolor != None:
#        ax.set_axis_bgcolor(bgcolor)
        ax.set_facecolor(bgcolor)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_xlim(xlim) 
    ax.axis('on')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_color('k')
    ax.spines['bottom'].set_color('k')
    if title !=None:
        plt.title(title,fontsize=fontsize+2)
    if type(sig_id)!=str:
        if _depth(sig_id)==1:
            plt.plot(freqs[:len(sig_id)],sig_id*np.nanmax(data1)*sig_fac,
                       sig_style,color='k')
        else:
            sig_id = np.array(sig_id)
            plt.plot(freqs[:len(sig_id[0])],(sig_id[0])*np.nanmax(data1)*sig_fac,
                       sig_style,color='k')
            plt.plot(freqs[:len(sig_id[0])],(sig_id[1])*np.nanmax(data1)*sig_fac,
                       sig_style,color='r')
    if np.any(ylim!=None):
        ax.set_ylim(ylim) 
        
    if not legend_pos == None:    
        loc_dict = {'uc': 'upper center', 'ur': 'upper right', 'ul':'upper left', 
                    'bc': 'lower center', 'br': 'lower right', 'bl':'lower left',
                    'best': 'best'}  
        if np.any(legend!=None):
                plt.legend(legend, loc=loc_dict.get(legend_pos), ncol=ncols, fontsize=fontsize-2,frameon=False) 
    if outfile!='none':    
        plt.savefig(outfile) 
    if show:
        plt.show()
        plt.close()
    if return_fig:
        return fig    
           
        
        
def semi_log_plot2(figsize,data1,freqs,xlim=[1,100],ylabel='variable',legend=None,outfile=None,
                  legend_pos='best',ylim=None,show=True,cmap='gist_rainbow',
                  ncols=3,CI=None,xticks=None,bgcolor=None,
                  fontsize=11, markersize=0,title=None,
                  sig_id="none",sig_fac=1.05,sig_style='*',plot_zero=True,return_fig=False):   
    
    '''
    plots data over log-spaced frequencies with sig. indices on the plot lines
    
    figsize:       figure size [width, height]
    data:          2D  array/list [groups x frequencies]
    freqs:         1D or 2D array/list of frequencies (float)
                   use 2D if different frequencies for different groups                         
    xlim:          [xmin,xmax]
    ylabel:        label for the y axis
    legend:        array of strings, 1 for each group
    outfile:       if not None, figure will be exported to this file
    legend_pos:    position of legend ('uc','br' or 'ur'); no legend if None
    ylim:          [ymin,ymax]
    show:          if False, the figure is not shown in console/window
    cmap:          either name of a standard colormap 
                   or an instance of matplotlib.colors.LinearSegmentedColormap
    ncols:         number of columns in the plot legend
    xticks:        custom values for xticks. if None, standard value are used
    bgcolor:       background color
    fontsize:      fontsize
    markersize:    size of data point marker, default = 0
    sig_style:     None or 1D or 2D e.g. 's' or 'o', or ['s','o']
    sig_id:        None or array of indices where significance is to be indicated
                    can be 2D as [group x freq]
                        or 3D as [2x group x freq] with 1st = [sig,sig_c]
    sig_fac:       controls the height at which significance indicators shown
    return_fig:    whether to return figure as object to caller

    '''
    
    
    fig,ax=plt.subplots(figsize=figsize) 
    N_groups = len(data1)
    colorspace = get_colorspace(cmap, len(data1))
    if sig_style!=None:
        if _depth(sig_style)==0:
            sig_data1 = data1*sig_id
            sig_data1[np.where(sig_data1==0)]=np.nan
            sig_style1 = sig_style
            colorspace = colorspace+colorspace
        else:
            sig_data1  = data1*sig_id[0]
            sig_data1C = data1*sig_id[1]
            sig_data1[np.where(sig_data1==0)]=np.nan
            sig_data1C[np.where(sig_data1C==0)]=np.nan
            sig_style1 = sig_style[0]
            sig_style2 = sig_style[1]        
            colorspace = colorspace+colorspace+colorspace
        
    ax.set_prop_cycle(color=colorspace)                                                          # set different colors for different plots
    for i in range(N_groups):                                                              # for each plot i
        if _depth(freqs)==2:    
            freqs2=freqs[i]
        else:
            freqs2=freqs
        N_F = len(data1[i])
        ax.plot(freqs2[:N_F],data1[i],'-',markersize=markersize)     
        
    if (sig_style != None) :
        for i in range(N_groups):                                                              # for each plot i
            if _depth(freqs)==2:    
                freqs2=freqs[i]
            else:
                freqs2=freqs
            N_F = len(data1[i]) 
            ax.plot(freqs2[:N_F],sig_data1[i],linestyle='',
                    marker=sig_style1,markersize=markersize) 
    if (sig_style != None) & (_depth(sig_style)==1):
        for i in range(N_groups):                                                              # for each plot i
            if _depth(freqs)==2:    
                freqs2=freqs[i]
            else:
                freqs2=freqs
            N_F = len(data1[i])   
            ax.plot(freqs2[:N_F],sig_data1C[i],linestyle='',
                    marker=sig_style2,markersize=markersize) 
    if plot_zero:
        ax.plot(freqs2,np.zeros(N_F),'k')
         
    if xticks==None:
        xticks=[1,2,3,5,10,20,30,50,100,200,300]
    ax.set_xscale('log')
    ax.set_xticks(xticks)
    if bgcolor != None:
#        ax.set_axis_bgcolor(bgcolor)
        ax.set_facecolor(bgcolor)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_xlim(xlim) 
    ax.axis('on')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_xlabel('Frequency [Hz]', fontsize=fontsize)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_color('k')
    ax.spines['bottom'].set_color('k')
    if title !=None:
        plt.title(title,fontsize=fontsize+2)
    
    if ylim!=None:
        ax.set_ylim(ylim) 
        
    loc_dict = {'uc': 'upper center', 'ur': 'upper right', 'ul':'upper left', 
                'bc': 'lower center', 'br': 'lower right', 'bl':'lower left',
                'best': 'best'}  
    if list(legend)!=None:
        plt.legend(legend, loc=loc_dict.get(legend_pos), ncol=ncols, fontsize=fontsize-2,frameon=False) 
    if outfile!=None:    
        plt.savefig(outfile) 
    if show:
        plt.show()
    plt.close()
    
    if return_fig:
        return fig            

    
    
    

def semi_log_plot_multi(rows,cols,dataL,freqs,xlimA,ylabA,titlesA,cmapA,figsize=[7,3],
                        legendA=None,outfile=None,legend_posA=None,
                        ylimA=None,show=False,ncols=3,CI=None,red_xa=False,red_ya=False,
                        xlabA=None,Ryt=None,xticks='auto',fontsize=8,plot_zero=False,
                        markersize=0,sig_idA=None,sig_fac=1,sig_style='*',return_fig = False): 
    '''
    multiple subplots of data over log-spaced frequencies
    with or without confidence intervals/standard deviation 
    
    figsize:       figure size [width, height]
    rows:          number of rows for subplots
    cols:          number of columns for subplots
    dataL:         3D or 4D array/list (without or with CI/SD resp.)
                   1st dim: datasets, 1 per subplot
                   2nd dim: groups within subplot 
                   optional dim: [mean, lower bound, upper bound] for CI or SD
                   last dim: frequencies   
                   The numbers of groups and frequencies can vary between 
                   subplots, if your dataL object is a list on the 1st dim.
    freqs:         1D, 2D or 3D array/list of frequencies (float) 
                   2D if every group uses different frequencies
                   3D if every dataset and every group uses different freqs 
                   Dimensions must match the data!
    xlimA:         2D array of [xmin,xmax] for each subplot
    ylabA:         2D array of labels for the y axis in each subplot
    titlesA:       2D array of subplots titles    
    cmapA:         array of colormaps, either names of standard colormaps 
                   or instances of matplotlib.colors.LinearSegmentedColormap
    legendA:       2D array of legends (strings); or None for no legends 
    outfile:       if not None, figure will be exported to this file
    legend_posA:   position of the legend ('uc' or 'ur') in each subplot; 
                   or None for no legends
    ylimA:         2D array of [ymin,ymax] for each subplot; or None for auto
    show:          if False, the figure is not shown in console/window

    ncols:         number of columns in the plot legend
    CI:            None if no CI/SDs, else alpha value (0 to 1) for CI/SD areas 
    xticks:        custom values for xticks. If auto, standard values are used
    xlabA:         array of booleans, whether to show the x label; 
                   or None for all True
    Ryt:           if not None, reduces the number of y ticks
    fontsize:      fontsize in plot
    markersize:    size of markers in plot
    return_fig:    whether to return figure as object to caller

    '''
    
    depth = lambda L: isinstance(L, list) and max(map(depth, L))+1
    fig,axes=plt.subplots(rows,cols,figsize=figsize)
    N_datasets = range(len(dataL)) 
    if CI == None:
        CI = np.array([None for i in N_datasets])
    elif _depth(CI)==0:
        CI = np.array([CI   for i in N_datasets])
            
    if ylimA==None:
        ylimA = [False for i in N_datasets]
    if xlabA==None:
        xlabA = [True for i in N_datasets]
    if legend_posA==None:
        legend_posA = [None for i in N_datasets]  
    if sig_idA==None:
        sig_idA = ['none' for i in N_datasets]  

    for d,data in enumerate(dataL):         # each dataset in one subplot 
        if (rows==1) or (cols ==1):
            ax = axes[d]
            yi = d//cols
            xi = d%cols
        else:
            yi = d//cols
            xi = d%cols
            ax = axes[yi,xi]
#        ax.hold(True)
        ax.set_title(titlesA[d],fontsize=fontsize)
        
        if plot_zero:
            ax.plot(np.zeros(len(data[0])),color=(0.5,0.5,0.5),label='_nolegend_')        

        colorspace = get_colorspace(cmapA[d], len(data))
        if np.any(CI!=None):
            colorspace_CI = np.array(colorspace)*np.array([1,1,1,CI[d]])
        #ax.set_color_cycle(colorspace)
        ax.set_prop_cycle(color=colorspace) 
        for i in range(len(data)):
            if depth(freqs)==3:
                freqs2=freqs[d][i]
            elif depth(freqs)==2:    
                freqs2=freqs[i]
            else:
                freqs2=freqs
                
            if CI[d]!=None:
                fr = freqs2[:len(data[i][0])]
                ax.plot(fr,data[i][0],'o-',markersize=markersize)
                ax.fill_between(fr,data[i][1],data[i][2],color=colorspace_CI[i],label='_nolegend_')    
            else:
                fr = freqs2[:len(data[i])]
                ax.plot(fr,data[i],'o-',markersize=markersize,color=colorspace[i])
            
            sig_id = sig_idA[d]
            if type(sig_id)!=str:
                if _depth(sig_id)==1:
                    ax.plot(freqs[:len(sig_id)],sig_id*np.nanmax(data)*sig_fac,
                               sig_style,color='k')
                else:
                    sig_id = np.array(sig_id)
                    ax.plot(freqs[:len(sig_id[0])],(sig_id[0])*np.nanmax(data)*sig_fac,
                               sig_style,color='k')
                    ax.plot(freqs[:len(sig_id[0])],(sig_id[1])*np.nanmax(data)*sig_fac,
                               sig_style,color='r')
        if Ryt != None:
            if Ryt[d] ==1:
                for label in ax.get_yticklabels()[::2]:
                    label.set_visible(False)  
                 
        ax.set_xscale('log')
        ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter()) 
        if xticks=='auto':
            xticks=[1,2,3,5,10,20,30,50,100,200,300]
        if red_xa and yi <rows-1:
            ax.set_xticks(xticks)
            ax.set_xticklabels([])
        else:
            ax.set_xticks(xticks)
            xticklabels = [str(i) for i in xticks]
            ax.set_xticklabels(xticklabels,fontsize=fontsize)
            if xlabA[d]==True:
                ax.set_xlabel('Frequency [Hz]',fontsize=fontsize)  
        if red_ya and xi >0:
            ax.set_yticks([])
        else:
            ax.set_ylabel(ylabA[d],fontsize=fontsize)
        


                
        ax.set_xlim(xlimA[d]) 
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if np.any(ylimA[d]!=None):
            ax.set_ylim(ylimA[d]) 
            
        loc_dict = {'uc': 'upper center', 'ur': 'upper right', 'ul': 'upper left',
                'br': 'lower right', 'best': 'best','bl':'lower left'}  
        if legend_posA[d] is not None:
            if legendA[d] is not None:
                ax.legend(legendA[d],loc=loc_dict.get(legend_posA[d]), ncol=ncols,frameon=0)
                        #  bbox_to_anchor=(0.5, 1.05), ncol=ncols)
        # if legendA[d] is not None:
        #     ax.legend(legendA[d], loc='upper right', ncol=ncols, frameon=0,fontsize=fontsize)  
   
    plt.tight_layout()   
    if outfile!=None:    
        plt.savefig(outfile)
    if show:
        plt.show()
  
    plt.clf()
    
    if return_fig:
        return fig
    




    
def plot_corr(corrs,freqs,gr_names,alpha=0.05,mc_meth='fdr_bh',title=None,
              figsize=[8,3],ylim=[-100,100],xlim=[1,100],lpos='best',nc=2,
              cmap='rainbow',markersize=8,fontsize=11,sig_corr=None,return_fig=False,
              method='Spearman'):
    '''
    values, p_vals: 2D [groups x frequencies]
    
    '''

   
    sig_id = sig_corr
    sig_style=['s','d']
    
    fig1 = semi_log_plot2(figsize,corrs,freqs,xlim=xlim,ylabel=f'{method}\'s r',
                        legend=gr_names,legend_pos=lpos,ylim=ylim,cmap=cmap,
                        ncols=nc,title=title,sig_id=sig_id,sig_style=sig_style,
                        markersize=markersize,fontsize=fontsize,return_fig=return_fig)
    if return_fig:
       return fig1











#%% heatmap and matrix plots
 

    
def plot_matrix(matrix,aspect=1,title='',interpol='none',zmin=-1,zmax=1,cmap='bwr',
                xlabel=None,ylabel=None): 
    
     plt.imshow(matrix,aspect=aspect,vmin=zmin,vmax=zmax,interpolation=interpol,cmap=cmap)
     if ylabel != None:
         plt.ylabel(ylabel)
     if xlabel != None:
        plt.xlabel(xlabel)
     plt.colorbar()
     plt.title(title)
     plt.show()      
    
    
 
        
    
def plot_heatmap(data1,figsize=[9,7],
                 cbar='right',zmax=None,zmin=None,cmap=plt.cm.YlOrRd,  
                 xlabel='',ylabel='',zlabel='',fontsize = 18,zlloc='center',
                 xticks=None,yticks=None,zticks=None,showticks=[1,1,0,0],
                 xticklabels=None, yticklabels='none',interpol=None,
                 zticklabels=None, xticklab_rot=45,title=None,aspect=None,cbarf=0.03,
                 masking='False',bad='white',under='None', topaxis=False, return_fig=False):
    ''' INPUT:
        data1:                 2D array
        figsize:              figure size             
        cbar:                 can be 'right', 'lower' or None
        zmax:                 max value for z axis (auto if None)
        zmin:                 min value for z axis (epsilon if None)
        cmap:                 colormap for colorbar; either:
                                - Instance of mpc.LinearSegmentedColormap 
                                - name of library cmap                            
        xlabel,ylabel,zlabel: labels for x, y, z axes
        fontsize:             fontsize for major labels
        xticks,yticks,zticks: tick values for x, y, z axes
        showticks:            whether to show ticks on lower,left,top,right
        xticklabels:          labels for x ticks
        yticklabels:          labels for y ticks
        xticklab_rot:         degrees by which to rotate xticklabels
        interpol:             interpolation, can be e.g. 'antialiased', 'nearest' 
        title:                title for the plot
        aspect:               sets ratio of x and y axes, automatic by default
        cbarf:                regulates size of colorbar
        masking:              whether to mask invalid values (inf,nan)
        bad:                  color for masked values
        under:                color for low out-of range values
        topaxis:              whether to have additional x axis on top
        return_fig:           whether to return the figure to caller as an object
    
    '''
    
    eps = np.spacing(0.0)
    fig, ax = plt.subplots(1,figsize=figsize)
    if aspect == None:
        aspect = data1.shape[1]/data1.shape[0] * figsize[1]/figsize[0]

    if type(cmap) != mpc.LinearSegmentedColormap:
        cmap = mpl.cm.get_cmap(cmap)
    cmap.set_bad(bad)
    cmap.set_under(under)        
    
    if masking:
        data1 = masked_invalid(data1)
    if topaxis:
        ax.tick_params(labeltop=True)        
    if zmax == None:
        zmax = max(np.reshape(data1,-1))  
    else:
        mask1 = data1>zmax
        data1  = data1*(np.invert(mask1)) + zmax*0.999*mask1
    if zmin != None:
        mask2 = data1<zmin
        data1  = data1*(np.invert(mask2)) + zmin*0.999*mask2   
        
    if zmin==None or zmin==0:    
       # PCM = ax.pcolormesh(data1,vmin=eps, vmax=zmax,cmap=cmap,shading=shading)   
        PCM = ax.imshow(data1,vmin=eps, vmax=zmax,cmap=cmap,aspect=aspect,interpolation=interpol) 
    else:
       # PCM = ax.pcolormesh(data1,vmin=zmin,vmax=zmax,cmap=cmap,shading=shading)   
        PCM = ax.imshow(data1,vmin=zmin,vmax=zmax,cmap=cmap,aspect=aspect,interpolation=interpol)   

    
    ax.set_xlim([-0.5,len(data1[0])-0.5])
    ax.set_ylim([-0.5,len(data1)-0.5])
    
    if np.all(xticks != None): 
        if _depth(xticks) == 0:    
            Nx=len(xticklabels)
            ax.set_xticks(np.arange(Nx)+0.5)
            ax.set_xlim([0,Nx]) 
        else:
            ax.set_xticks(xticks)
        if np.all(xticklabels!=None):
            ax.set_xticklabels(xticklabels,rotation=xticklab_rot)
            
    if np.all(yticks != None): 
        if _depth(yticks) == 0:    
            Ny=len(yticklabels)
            ax.set_yticks(np.arange(Ny)+0.5)
            ax.set_ylim([0,Ny]) 
        else:
            ax.set_yticks(yticks)
        if np.all(yticklabels !='none'):
            ax.set_yticklabels(yticklabels)

        
    ax.set_xlabel(xlabel,fontsize=18)
    ax.set_ylabel(ylabel,fontsize=18)
    ax.tick_params(axis='both',which='both',labelsize=fontsize-2)
    ax.tick_params(bottom=showticks[0], left=showticks[1], top=showticks[2], right=showticks[3])
    if cbar == 'lower':
        orient = 'horizontal'
    else:
        orient = 'vertical'
    if cbar != None:    
        if zticks !=None:
            cb  = plt.colorbar(PCM, ax=ax, ticks = zticks, orientation = orient,
                               fraction=cbarf)
            if zticklabels  != None:
                if orient == 'vertical':
                    cb.ax.set_yticklabels(zticklabels)
                else:
                    cb.ax.set_xticklabels(zticklabels)
        else:
            cb  = plt.colorbar(PCM, ax=ax, orientation = orient,fraction=cbarf)
        # cb.set_label(zlabel,fontsize=18, loc = zlloc)        # 
        cb.set_label(zlabel,fontsize=18)

        cb.ax.tick_params(labelsize=14) 

    if title != None:
        plt.title(title,fontsize=18)
        
    if return_fig:
        return fig


        
    
def plot_heatmap_with_stats(data,data2,data3,alpha,Q,linecol='k',figsize=[9,7],
                 zmax=None,zmin=None,cmap=plt.cm.YlOrRd,  
                 xlabel='',ylabel='',zlabel='',fontsize = None,
                 xticks=None,yticks=None,zticks=None,
                 xticklabels=None, yticklabels='none',xmax_n=None,
                 zticklabels=None, xticklab_rot=45,title=None,ymax_f=None,zero_to_nan=False,
                 masking='False',bad='white',under='None', topaxis=False,return_fig=False):
    ''' 
        PLots heatmap with additional line plots showing the means of x and y axes.
    
        INPUT:
        data:                 2D array
        data2, data3:         1D arrays
        figsize:              figure size             
        cbar:                 can be 'right', 'lower' or None
        zmax:                 max value for z axis (auto if None)
        zmin:                 min value for z axis (epsilon if None)
        cmap:                 colormap for colorbar; either:
                                - Instance of mpc.LinearSegmentedColormap 
                                - name of library cmap                            
        xlabel,ylabel,zlabel: labels for x, y, z axes
        fontsize:             fontsize for major labels
        xticks,yticks,zticks: tick values for x, y, z axes
        xticklabels:          labels for x ticks
        yticklabels:          labels for y ticks
        xticklab_rot:         degrees by which to rotate xticklabels
        masking:              whether to mask invalid values (inf,nan)
        bad:                  color for masked values
        under:                color for low out-of range values
        topaxis:              whether to have additional x axis on top
        return_fig:           whether to return the figure to caller as an object

    
    '''
    
    
    data  = data  - alpha*100
    data2 = data2 - alpha*100
    data3 = data3 - alpha*100

    
    fig = plt.figure(figsize=figsize)
    ax  = fig.add_axes([0.0,  0.27, 0.8, 0.64])
    ax2 = fig.add_axes([0.87, 0.27, 0.1, 0.64])
    ax3 = fig.add_axes([0.0,  0.0,  0.8, 0.15])
    ax4 = fig.add_axes([0.87, 0.06, 0.1, 0.04])
    
       
    # ax  = fig.add_axes([0.2, 0.25, 0.9,  0.7])
    # ax2 = fig.add_axes([0.0, 0.25, 0.1, 0.7])
    # ax3 = fig.add_axes([0.2, 0.0, 0.7, 0.15])
    
    # 
    eps = np.spacing(0.0)
    
    
    if fontsize==None:
        fontsize=figsize[1]+figsize[0]
    
    if cmap == 'Blues':
        linecol = 'b'
    elif cmap == 'Reds':
        linecol = 'r'
        
    if type(cmap) != mpc.LinearSegmentedColormap:
        cmap = mpl.cm.get_cmap(cmap)
    # cmap.set_bad(bad)
    # cmap.set_under(under)     
    
    if zero_to_nan:
        data[np.where(data==0)]=np.nan 

    if masking:
        data = masked_invalid(data)
    if topaxis:
        ax.tick_params(labeltop=True)        
    if zmax == None:
        zmax = max(np.reshape(data,-1))     
    if zmin==None or zmin==0:    
        PCM = ax.pcolormesh(data,vmin=eps,vmax=zmax,cmap=cmap)   
    else:
        PCM = ax.pcolormesh(data,vmin=zmin,vmax=zmax,cmap=cmap)   

    
    ax.set_xlim([0,len(data[0])])
    ax.set_ylim([0,len(data)])
    
    if np.all(xticks != None): 
        if _depth(xticks) == 0:    
            Nx=len(xticklabels)
            ax.set_xticks(np.arange(Nx)+0.5)
            ax.set_xlim([0,Nx]) 
        else:
            ax.set_xticks(xticks)
        if xticklabels !=None:
            ax.set_xticklabels(xticklabels,rotation=xticklab_rot)
            
    if np.all(yticks != None):         
        ax.tick_params(axis='y',left=True,right=True,labelright=True)
        if _depth(yticks) == 0:    
            Ny=len(yticklabels)
            ax.set_yticks(np.arange(Ny)+0.5)
            ax.set_ylim([0,Ny]) 
        else:
            ax.set_yticks(yticks)
        if yticklabels !='none':
            ax.set_yticklabels(yticklabels)

        
    ax.set_xlabel(xlabel,fontsize=fontsize)
    ax.set_ylabel(ylabel,fontsize=fontsize)
    ax.tick_params(axis='both',which='both',length=0,labelsize=fontsize-2)

    orient = 'horizontal'

       
    if zticks !=None:
        cb  = plt.colorbar(PCM,cax=ax4,ticks=zticks,orientation = orient)                               
        if zticklabels  != None:
            if orient == 'vertical':
                cb.ax.set_yticklabels(zticklabels)
            else:
                cb.ax.set_xticklabels(zticklabels)
    else:
        cb  = plt.colorbar(PCM, cax=ax4, orientation = orient)
    cb.set_label(zlabel,fontsize=fontsize)
    cb.ax.tick_params(labelsize=fontsize*0.7) 

        
       
    # simple plot of mean K over freqs AFO network
    
    ax2.plot(data2,np.arange(0,len(data2)),linecol,marker='o')
    if xmax_n == None:
        xmax_n = 1+np.ceil(zmax/3)
    ax2.set_xlim([0,xmax_n])
    ax2.set_ylim([-.5,len(data2)-0.5])
    ax2.get_yaxis().set_visible(False)
    ax2.tick_params(labelsize=fontsize-3)

    # simple plot of mean K over parcels AFO freq.   
     
    ax3.plot(data3,linecol,marker='o')
    ax3.plot(np.full(len(data3),(Q)*100),'k')
    ax3.set_xlim([0,len(data3)-1])
    if ymax_f == None:
        ymax_f = 1 + np.ceil(np.max(zmax)/3)
    ax3.set_ylim([0,ymax_f])
    ax3.get_xaxis().set_visible(False)
    ax3.tick_params(labelsize=fontsize-3)
 
    if title != None:
        fig.tight_layout(rect=[0, 0.0, 1, 0.94])
        fig.suptitle(title,fontsize=fontsize)
        
    if return_fig:
        return fig



def plot_heatmaps(data, titles=None, N_cols=3, figsize=None, fontsizeT=13, fontsizeL=11, 
                  ylabel=None, xlabel=None, zlabel= None, cmap='jet',zmax=None, zmin=0,
                  xticks = None, yticks = None, zticks = None, N_rows = 'auto',cbarf=None,
                  xticklabels = None, yticklabels=None, zticklabels=None,aspect='auto',red_titles=False,
                  suptitle=None,xticklab_rot=0,
                  red_xa=True,red_ya=True,red_cb=False,adjust=None,
                  return_fig=False):
    
    ''' Plots several heatmaps.
    
    Input:
        data:               3D array or list of 2D arrays
        titles:             array of titles, empty by default 
        N_cols:             number of columns, default 3
        figsize:            fixed figure size, will be determined automatically if None
        fontsizeT:          fontsize for title
        fontsizeL:          fontsize for labels
        zmin,zmax:          can be single values or lists/arrays (if you want different limits for each plot)
        xticks:             can be given as single list/array (for all plots) or as list of lists (different for each plot)
                               same applies to labels, and x and z axes.
        xlab:               axis label, empty by default, can be single string or list of strings 
        cmap:               name of a library cmap, or instance of mpc.LinearSegmentedColormap, or a list of either of these.
        xticklab_rot:       degrees by which to rotate xticklabels
        masking:            whether to mask invalid values (inf,nan)
        bad:                color for masked values
        red_xa:             whether to show ticks and label only for lowest x-axis
        red_ya:             whether to show ticks and label only for leftmost y-axis
        red_cb:             whether to show only one colorbar
        return_fig:         whether to return figure as object to caller
    '''    
    
    data = np.array(data)
    N_plots = len(data)
    if N_rows == 'auto':
        N_rows  = int(np.ceil(1.*N_plots/N_cols) )
    
    N_plots2 = N_rows*N_cols
    # data2 = np.zeros([N_plots2,len(data[0]),len(data[0][0])])
    # data2[:N_plots] = data
    data2 = [[]]*N_plots2
    for i in range(N_plots):
        data2[i] = data[i]

    if figsize==None:
        figsize =[N_cols*4.8,N_rows*3.5]
        
    if zlabel == None:
        zlabel=''

        
    cmaps       = _repeat_if_needed(cmap, N_plots2, 1)    
    zmax        = _repeat_if_needed(zmax, N_plots2, 1)   
    zmin        = _repeat_if_needed(zmin, N_plots2, 1)   
    xticks      = _repeat_if_needed(xticks, N_plots2, 2)   
    yticks      = _repeat_if_needed(yticks, N_plots2, 2) 
    zticks      = _repeat_if_needed(zticks, N_plots2, 2)   
    xticklabels = _repeat_if_needed(xticklabels, N_plots2, 2)   
    yticklabels = _repeat_if_needed(yticklabels, N_plots2, 2)  
    zticklabels = _repeat_if_needed(zticklabels, N_plots2, 2)  

    fig,axes=plt.subplots(N_rows,N_cols,figsize=figsize)     
    # plt.subplots_adjust(wspace=.2,hspace=.3)
    
    if type(xlabel) == str:
        xlabel = [xlabel] * N_plots2
    if type(ylabel) == str:
        ylabel = [ylabel] * N_plots2    
    if type(zlabel) == str:
        zlabel = [zlabel] * N_plots2  

    
    for i in range(N_plots2):    
        if (N_rows==1) or (N_cols ==1):
            ax = axes[i]
        else:
            ax = axes[i//N_cols,i%N_cols]      
        yi = i//N_cols
        xi = i%N_cols
        
        if i <N_plots:
           # ax.hold(True)
            ax.grid(False)
            if zmax[i] == None:
                zmax[i] = np.around(np.nanmax(data2[i])*32,3)/25
            p = ax.imshow(data2[i],origin='lower',interpolation='none',cmap=cmaps[i],
                          vmax=zmax[i],vmin=zmin[i],aspect=aspect)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="8%", pad="8%")
            if cbarf != None:
                cax.set_box_aspect(cbarf)
          
            if np.all(titles!=None):
                if not (red_titles and yi >0):
                    ax.set_title(titles[i],fontsize=fontsizeT)
            
        if red_xa and yi <N_rows-1:
            if np.all(xticks[i] !=None):
                ax.set_xticks(xticks[i])
                ax.set_xticklabels([])
        else:
            if np.all(xticks[i] !=None):
                ax.set_xticks(xticks[i])
            if np.all(xticklabels[i] !=None):
                ax.set_xticklabels(xticklabels[i],rotation=xticklab_rot)  
            if np.all(xlabel!=None):    
                ax.set_xlabel(xlabel[i], fontsize=fontsizeL) 
        if red_ya and xi >0:
            ax.set_yticks(yticks[i])
            ax.set_yticklabels([])
        else:
            
            if np.all(ylabel!=None):    
                ax.set_ylabel(ylabel[i], fontsize=fontsizeL)   
            if np.all(yticks[i] !=None):
                ax.set_yticks(yticks[i])

            if np.all(yticklabels[i] !=None):
                ax.set_yticklabels(yticklabels[i])         
        if red_ya and xi<N_cols-1:
            cb = plt.colorbar(p, cax=cax, ticks = [])  
            if zticklabels[i] != None:
                    cb.ax.set_yticklabels(zticklabels)
       
                
        if red_cb and xi<N_cols-1:
            cb = plt.colorbar(p, cax=cax, ticks = zticks[i]) 
            cb.remove() 
        else: 
            if np.all(zticks[i]==None):
                zticks[i] = zmin[i] + (zmax[i]-zmin[i])*np.arange(0,1.1,.25)
                cb = plt.colorbar(p, cax=cax, ticks = zticks[i]) 
            else:
                cb = plt.colorbar(p, cax=cax, ticks = zticks[i]) 
               
            # if not red_ya or xi==N_cols-1:
            #     cb.set_label(zlabel[i], fontsize=fontsizeL)  
            
    plt.tight_layout()
    if np.any(adjust)!=None:
        fig.subplots_adjust(hspace=adjust[0], wspace=adjust[1])

    if suptitle != None:
        fig.tight_layout(rect=[0, 0.0, 1, 0.96])
        fig.suptitle(suptitle,fontsize=16)
         
    if return_fig:
        return fig     





    
   

    
    
#%% scatter plots
    

def scatter_corr(values1,values2,xlabel,ylabel,xlim,ylim,figsize=[6,4],
                 method='Spearman',degree=1,dotcolor='k',linecolor='r',lw=1,
                 title='',zero_line=False,remove_nans=True,pround=2,msize=4,
                 return_fig=0,fontsize=12):
    
    if remove_nans: 
        indsk1  = np.where(values1==values1)[0]
        indsk2  = np.where(values2==values2)[0]
        indsk   = [i for i in range(len(values1)) if (i in indsk1) and (i in indsk2)]
        x = values1[indsk]
        y = values2[indsk]
    else:
        x     = values1
        y     = values2

    f1= plt.figure(figsize=figsize)

    xvals = np.arange(xlim[0],xlim[1]*1.001,(xlim[1]-xlim[0])/5)
      
    p1 = np.poly1d(np.polyfit(x,y,degree)  )
        
    if zero_line:
        plt.plot(xvals,np.zeros(len(xvals)),color='#888888', label='_nolegend_')       
    plt.plot(x,y,'o',color=dotcolor, label='_nolegend_',markersize=msize)
    plt.plot(xvals,p1(xvals),color=linecolor,linewidth=lw)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.title(title,fontsize=fontsize)
    plt.ylabel(ylabel,fontsize=fontsize)
    plt.xlabel(xlabel,fontsize=fontsize)    
    plt.xticks(fontsize=fontsize-1)
    plt.yticks(fontsize=fontsize-1)
    if degree == 1:
        r,p   = correlate(x,y,method=method)
        legend = ['r='+str(np.around(r,2))+', p= '+str(np.around(p,pround))]
        plt.legend(legend,fontsize=fontsize)
        
    if return_fig:
        return f1
 
    
 

def rand_jitter(arr,fac=0.01):
    stdev = fac * (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev

    
 
    
def scatter_corr_multi(valueA1,valueA2,NR,NC,xlabel,ylabel,xlim,ylim,figsize=[6,4],
                 method='Spearman',degree=1,dotcolorA=['r','g','b'],linecolor='r',lw=1,
                 titleA=None,zero_line=False,remove_nans=True,plot_identity=False,
                 jitter=False,markersize=4,show_corr=True,xlog=False,ylog=False):
    
    '''
    INPUT:
        valueA1,a2: value arrays, shape [N_plots x N_groups x ]
        
    '''
       

    N_plots = len(valueA1)
    
    fig, axes = make_better_subplots(NR,NC,figsize=figsize)
   

    for k in range(N_plots):
        
            values1 = np.array(valueA1[k])     
            values2 = np.array(valueA2[k])
            
            j = k%NC
            i = k//NC
            ax = axes[i,j]
    
        
            if remove_nans: 
                indsk1  = np.where(values1==values1)[0]
                indsk2  = np.where(values2==values2)[0]
                indsk   = [i for i in range(len(values1)) if (i in indsk1) and (i in indsk2)]
                x = values1[indsk]
                y = values2[indsk]
            else:
                x     = values1
                y     = values2
                
        
            xvals = np.arange(xlim[0],xlim[1]*1.001,(xlim[1]-xlim[0])/5)
              
            p1 = np.poly1d(np.polyfit(x,y,degree)  )
                
            if zero_line:
                ax.plot(xvals,np.zeros(len(xvals)),color='#888888', label='_nolegend_')   
            if jitter:
                x1 = 1*rand_jitter(x)
                y1 = 1*rand_jitter(y)
            else:
                x1 = 1*x
                y1 = 1*y

                
            ax.plot(x1,y1,'o',color=dotcolorA[k], label='_nolegend_',markersize=markersize)
            if show_corr:
                ax.plot(xvals,p1(xvals),color=linecolor,linewidth=lw)
            if plot_identity:
                ax.plot(xvals,xvals,'--',color='grey',linewidth=lw)

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            if not np.all(titleA==None):
                ax.set_title(titleA[k])
            
            if xlog:
                ax.set_xscale('log')

            if ylog:
                ax.set_yscale('log')

            # print(i)
            if i== NR-1:
                ax.set_xlabel(xlabel)    
            else:
                ax.set_xticks([])
            if j==0:
                ax.set_ylabel(ylabel)    
            else:
                ax.set_yticks([])
        
            if degree == 1 and show_corr:
                r,p   = correlate(x,y,method=method)
                legend = ['r='+str(np.around(r,2))+', p= '+str(np.around(p,3))]
                facecolor = 'y' if p < 0.05 else 'white'
                ax.legend(legend,facecolor=facecolor)
            
           








def scatter_corr_groups(values1,values2,xlabel,ylabel,xlim,ylim,figsize=[6,4],method='Spearman',
                 dotcolor='b',linecolor='r',title='',zero_line=False,remove_nans=True):
    
    if remove_nans: 
        indsk1  = np.where(values1==values1)[0]
        indsk2  = np.where(values2==values2)[0]
        indsk   = [i for i in range(len(values1)) if (i in indsk1) and (i in indsk2)]
        values1 = values1[indsk]
        values2 = values2[indsk]


    plt.figure(figsize=figsize)
    x     = values1
    y     = values2
    m,res = np.polyfit(x,y,1)
    r,p   = correlate(x,y,method=method)
    xvals = np.arange(xlim[0],xlim[1]*1.001,(xlim[1]-xlim[0])/5)
    yvals = xvals*m + res      
    if zero_line:
        plt.plot(xvals,np.zeros(len(xvals)),color='#888888', label='_nolegend_')       
    plt.plot(x,y,'o',color=dotcolor, label='_nolegend_')
    plt.plot(xvals,yvals,color=linecolor)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    legend = ['r='+str(np.around(r,2))+', p= '+str(np.around(p,3))]

    plt.legend(legend)





def scatter_corr_grid(values1,values2,freq_strings,name1,name2,
                      xlim,ylim,nrows,ncols):
    
    
    
    fig, axes = plt.subplots(nrows,ncols,figsize=[19,13])
    k=0
    for i in range(nrows):
        for j in range(ncols):
            k+=1
            f= k*3            
            ax    = axes[i,j]
            x     = np.mean(values1[:,f],1)
            y     = np.mean(values2[:,f],1)
            m,res = np.polyfit(x,y,1)
            r,p   = correlate(x,y)
            xvals = np.arange(xlim[0],xlim[1],(xlim[1]-xlim[0])/5)
            yvals = xvals*m + res            
            ax.plot(x,y,'o',color='b')
            ax.plot(xvals,yvals,'k')
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_title(freq_strings[f] + ' Hz')
            legend = ['','r='+str(np.around(r,2))+', p= '+str(np.around(p,2))]
            if p < 0.05:
                facecolor = 'y'
            else:
                facecolor = None
            ax.legend(legend,facecolor=facecolor)
            
            if i==2:
                ax.set_xlabel(name1)
            else:
                 ax.set_xticklabels([])
            if j==0:
                ax.set_ylabel(name2)
            else:
                ax.set_yticklabels([])











def scatter_corr_grid_N_gr(values1,values2,freq_strings,var_names,group_names,all_inds,
                      xlim,ylim,gr_corr=True,all_corr=False,figsize=[20,14],
                      nr=3,nc=4,order=1,subj_dim=0,markersize=5,corr_method='Spearman',
                      freq_sel=np.arange(3,41,3),lloc=None,colors=['b','g','r'],
                      title=None,return_fig=False,hide_dot_legend=False,hide_all_legend=False,
                      less_xticks=True,less_yticks=True):
    
    '''   
    Makes a grid of scatterplots, with subjects markers color-coded for groups.
    Polynomial fits can be computed for individual groups and/or all subjects.
    If order=1 (linear fit), r and p are computed with Pearson or Spearman (def)
    
    INPUT:
        values1: 3D array [subjects x freqs x parcels] or [parcels x freqs x subjects]
                 or 2D array [subjects x freqs]
                 or 1D array [subjects]
        values2: 3D array or 2D array or 1D array, as values1
        
        Only 1 of the value arrays can be 1D. If an array is 3D, average over parcel dimension is taken.
        
        freq_strings: frequency strings
        var_names:    names of the variables
        group_names:  names of the subject groups
        all_inds:     list of arrays with the subjects indexes per group 
        xlim:         either a single value pair or a list of value pairs
        ylim:         either a single value pair or a list of value pairs
        gr_corr:      whether to show correlations for individual groups
        all_corr:     whether to show correlation for all groups pooled
        figsize:      Figure size
        nr, nc:       numbers of rown, columns
        order:        order of polynomial fit (default=1)
        subj_dim:     dimension of subjects (default=0, can also be 2). Correlation is done across 1st dim.
        markersize:   size of the markers in scatterplot
        corr_method:  'Spearman' or 'Pearson' for computing fit if order == 1
        freq_sel:     indices of frequencies to be plotted
        lloc:         location of legend in plots, e.g. "upper right"
        colors:       list or array of colors (as characters or rgb values)
        return_fig:   if yes, function returns a matplotlib figure 
            
    '''

    
    fig, axes = plt.subplots(nr,nc,figsize=figsize)
    k=-1
    N_gr = len(all_inds)
    all_inds_cc = np.concatenate(all_inds)
    
    if _depth(xlim)==1:
        xlimA = np.tile(xlim,(nr*nc,1))
    else:
        xlimA = xlim
    if _depth(ylim)==1:
        ylimA = np.tile(ylim,(nr*nc,1))
    else:
        ylimA = ylim
    
    if _depth(values1) == 2: 
        values1 = values1[:,:,np.newaxis]
    if _depth(values2) == 2: 
        values2 = values2[:,:,np.newaxis]
        
    if _depth(values2) == 1:    
        dim1      = len(values1[0])      
        dim2      = len(values1[0][0])
        values2 = np.swapaxes(np.tile(values2,(dim2,dim1,1)),0,2)
        
    if _depth(values1) == 1:    
        dim1      = len(values2[0])      
        dim2      = len(values2[0][0])
        values1 = np.swapaxes(np.tile(values1,(dim2,dim1,1)),0,2)
        
    if _depth(order) == 0:    
        order = [order]
        
    for i in range(nr):
        for j in range(nc):
            k += 1
            f      = freq_sel[k]  
            ax     = axes[i,j]
            legend = []
            xlim   = xlimA[k]
            ylim   = ylimA[k]
            facecolor = None
            
            if subj_dim == 0:
                x     = np.mean(values1[:,f],1)                  # mean across parcels
                y     = np.mean(values2[:,f],1)
                x_gr  = [np.mean(values1[inds,f],1) for inds in all_inds]
                y_gr  = [np.mean(values2[inds,f],1) for inds in all_inds]
            elif subj_dim == 2:
                x     = np.mean(values1[:,f,all_inds_cc],1)               # mean across subjects
                y     = np.mean(values2[:,f,all_inds_cc],1)
                x_gr  = [np.mean(values1[:,f,inds],1) for inds in all_inds]
                y_gr  = [np.mean(values2[:,f,inds],1) for inds in all_inds]

                 
            if gr_corr:
                for g in range(N_gr):          
                    for o in order:
                        pol1    = np.polyfit(x_gr[g],y_gr[g],o)
                        xvals1  = np.arange(xlim[0],xlim[1]*1.01,(xlim[1]-xlim[0])/20)
                        yvals1  = np.polyval(pol1,xvals1)
                        ax.plot(xvals1,yvals1,color=colors[g])                    
                    if 1 in order:
                        r1,p1   = correlate(x_gr[g],y_gr[g],method=corr_method)    
                        legend.append('r='+str(np.around(r1,2))+', p= '+str(np.around(p1,2)))
                        if p1 < 0.05:
                            facecolor = 'y'   
                    if 2 in order:
                        r1,p1   = correlate(x_gr[g],y_gr[g],method='QuadPart')    
                        legend.append('sqrt(R2)='+str(np.around(r1,2))+', p= '+str(np.around(p1,2)))
                        if p1 < 0.05:
                            facecolor = 'y'  
                    # if order == 'both':
                    #     pol1    = np.polyfit(x_gr[g],y_gr[g],order)
                    #     xvals1  = np.arange(xlim[0],xlim[1]*1.01,(xlim[1]-xlim[0])/20)
                    #     yvals1  = np.polyval(pol1,xvals1)
                    #     ax.plot(xvals1,yvals1,color=colors[g])             
                    #     r1,p1   = correlate(x_gr[g],y_gr[g],method=corr_method)    
                    #     r2,p2   = correlate(x_gr[g],y_gr[g],method='QuadPart')   
                    #     legend.append('r='+str(np.around(r1,2))+', p= '+str(np.around(p1,2)))
                    #     legend.append('sqrt(R2)='+str(np.around(r2,2))+', p= '+str(np.around(p2,2)))
                    #     if (p1 < 0.05) or (p2 < 0.05):
                    #         facecolor = 'y'   
                    
            if all_corr:
                for o in order:
                    pol  = np.polyfit(x,y,o)
                    xvals = np.arange(xlim[0],xlim[1]*1.01,(xlim[1]-xlim[0])/20)
                    yvals = np.polyval(pol,xvals)                     
                    ax.plot(xvals,yvals,['k','grey'][o-1])

                if 1 in order:
                    r,p   = correlate(x,y,method=corr_method)      
                    legend.append('r='+str(np.around(r,2))+', p= '+str(np.around(p,2)))            
                    if p < 0.05:
                        facecolor = 'y'
                if 2 in order:
                    r,p   = correlate(x,y,method='QuadPart')      
                    legend.append('sqrt(R2)='+str(np.around(r,2))+', p= '+str(np.around(p,2)))            
                    if p < 0.05:
                        facecolor = 'y'
                    
           
            for g in range(N_gr):  
                ax.plot(x_gr[g],y_gr[g],'o',color=colors[g],markersize=markersize)                
                if not hide_dot_legend:
                    legend.append(group_names[g])   
               
            
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_title(freq_strings[f] + ' Hz')            
                        
            if not hide_all_legend:
                ax.legend(legend,facecolor=facecolor,loc=lloc)

            
            if i==nr-1:
                ax.set_xlabel(var_names[0])
            elif less_xticks:
                  ax.set_xticklabels([])
            if j==0:
                ax.set_ylabel(var_names[1])
            elif less_yticks:
                ax.set_yticklabels([])
                
    if title is not None:
        fig.suptitle(title)
    if return_fig:
        return fig
   

    
    
    

#%%


'''   ###############      BRAIN SURFACE PLOTS     ###############  '''




#%% 

def array_scalar(arr):
    if type(arr) is np.ndarray:
        return np.isscalar(next(iter(arr)))
    else:
        return False

def get_face_label(labels):
    n_parcels = len(set(labels))
    res = labels[0] if (n_parcels == 1) else -1
    return res

def get_triangle_stats(vertex_stats, triangles, func=np.mean):
    res = np.zeros(len(triangles), dtype=vertex_stats.dtype)
    for face_idx, face_stats in enumerate(vertex_stats[triangles]):
        res[face_idx] = func(face_stats)
    return res

class BrainSurface:
    def __init__(self, subject_path, parcellation='Schaefer2018_100Parcels_17Networks', hemis=None, surface='pial',
                  anat_colors = [(0.95,0.95,0.95),(0.82,0.82,0.82)]):
        self.subject_path = subject_path
        self.parcellation = parcellation
        self.surface = surface
        self.anat_colors = anat_colors
        if hemis is None:
            self.hemis = ['lh', 'rh']
        else:
            self.hemis = list(hemis)
        self._load_hemis()
        self._load_annotations()
        self.data = dict()
        self.plotter = None
        
    def _load_hemis(self):
        self.surfaces = dict()
        index_offset = 0
        coords = list()
        self.triangles = list()
        curviture = list()
        for hemi in self.hemis:
            surf_path = os.path.join(self.subject_path, 'surf', f'{hemi}.{self.surface}')
            surf_coords, surf_triangles = nib.freesurfer.io.read_geometry(surf_path)
            surf_triangles += index_offset
            surf_triangles = np.hstack([np.full_like(surf_triangles[:,:1], 3), surf_triangles])
            curv_path = os.path.join(self.subject_path, 'surf', f'{hemi}.curv')
            surf_curv = nib.freesurfer.read_morph_data(curv_path)
            coords.append(surf_coords)
            self.triangles.append(surf_triangles)
            curviture.append(surf_curv)
            index_offset += surf_coords.shape[0]
        coords = np.vstack(coords)
        self.triangles = np.vstack(self.triangles)
        self.vertex_curviture = np.hstack(curviture)
        self.face_curviture = get_triangle_stats(self.vertex_curviture, self.triangles[:, 1:])
        self.brain_mesh = pv.PolyData(coords, self.triangles)
        
    def _load_annotations(self):
        self.annotations = dict()
        self.parcel_names = list()
        vertex_labels = list()
        for hemi in self.hemis:
            annot_path = os.path.join(self.subject_path, 'label', f'{hemi}.{self.parcellation}.annot')
            labels_orig, _, annot_ch_names = nib.freesurfer.io.read_annot(annot_path)   
            labels_orig += len(self.parcel_names)
            annot_ch_names = [n.decode() for n in annot_ch_names]
            vertex_labels += labels_orig.tolist()
            self.parcel_names += annot_ch_names
        self.vertex_labels = np.array(vertex_labels)
        self.face_labels = get_triangle_stats(self.vertex_labels, self.triangles[:, 1:], func=get_face_label)
        
    def _is_scalar(self, data):
        return np.isscalar(next(iter(data.values())))
    
    def set_data(self, data):
        self.data = {key:value for (key,value) in data.items() if key in self.parcel_names}
        
    def plot(self, colormap='viridis', camera_position=None, zoom=1.0, show=True):
        try:
            is_scalar = self._is_scalar(self.data)
        except:
            is_scalar = False    
        scalars = np.full(len(self.face_labels), np.nan)
        colormap = 'viridis' if is_scalar else list()
        for item_index, (key, value) in enumerate(self.data.items()):
            label = self.parcel_names.index(key)
            mask = (self.face_labels == label)
            if is_scalar:
                scalars[mask] = value
            else:
                scalars[mask] = item_index
                colormap.append(value)
        curviture_bin = (self.face_curviture > 0)
        no_data_mask = np.isnan(scalars)
        scalars[no_data_mask] = curviture_bin[no_data_mask] + len(self.data)
        if no_data_mask.sum() > 0:
            colormap.append(self.anat_colors[0])
            colormap.append(self.anat_colors[1])
        if not(is_scalar):
            colormap = mpl.colors.ListedColormap(colormap)
        self.plotter = pv.Plotter(off_screen=True)
        self.plotter.add_mesh(self.brain_mesh, scalars=scalars, cmap=colormap, categories=not(is_scalar))
        if not(camera_position is None):
            self.plotter.camera_position = camera_position
        self.plotter.set_background('white')
        self.plotter.camera.Zoom(zoom)
        self.plotter.remove_scalar_bar()
        try:
            if pv.__version__ in ['0.36.1','0.32.1']:
                self.plotter.store_image = True
        except:
            pass

        
        if show:
             self.plotter.show()
        
    def save_to_image(self, img_path, **kwargs):
        fig, ax = plt.subplots(**kwargs)
        ax.imshow(self.plotter.image)
        ax.set_axis_off()
        fig.savefig(img_path, **kwargs)
        plt.close(fig)
            
            
#%%

def plot_multi_view_pyvista(data1,parc,colors,subj_path = 'L:\\nttk_palva\\Utilities\\fsaverage\\',
                            filename=None,figsize=[12,12],vrange=None,title=None,
                            views = ['lat-med'],z_score=False,zero_mean=False,use_sys_colors=False,
                            threshold=-50,nticks=11,label='',cbar_off=False,surface='inflated',
                            return_fig = False,ticklabels=[], legend= None):
    ''' 
    INPUT:  
        data1 :   Input data. Can be either:
                    - value array of size N, where N is the number of parcels. colors are picked on a linear scale from 
                            colormap provided in colors. Parcels with value below threshold will not be shown. 
                            A colorbar is added.
                    - color array of size [N x 3]. Parcels with color (-1,-1,-1) not shown.
                    - string 'sys', in which case parcels are colored by subsystem with subsys colors are given in colors
        
        parc:     An instance of the class Parc from parc_functions module.
        colors:   Either: 
                   - a colormap object, or name of a predefined colormap if data is given as a value array
                   - ignored when data1 is an array of colors 
                   - list of N colors for N functional systems when data1 == 'sys'
        filename: If not None, figure(s) will be saved in the file format specified by the filename.
        views:    List of views, can contain 'lat-med', 'top-down' and 'ant-post'. Each view produces a figure with 4 plots.
        threshold: minimum value that is displayed. color range starts from max(threshold, minimum value).
        nticks:   how many ticks in colorbar, if shown.
        '''
        
    left_dict  = {}
    right_dict = {}
    
    
    if array_scalar(data1):
        
        if use_sys_colors:
            cmap = make_cmap(colors,len1=len(parc.netw_names))
            vrange = [0,max(parc.networks)]

        else:
            if type(colors) not in [mpl.colors.ListedColormap,mpl.colors.LinearSegmentedColormap]:
                if type(colors)==str:
                    cmap = cm.get_cmap(colors)
                else:
                    cmap =  make_cmap(colors)  
            else:
                cmap = colors
            colors = [cmap(i) for i in range(256)]
            if vrange == None:            
                min1   = np.max([np.min(data1),threshold])
                max1   = np.nanmax(data1)
                vrange = [min1,max1]
            else:
                min1,max1 = vrange
            dx     = 255/(max1 - min1)     
    else:               
        cmap=None 
        vrange = [0,max(parc.networks)]
        
    if z_score:
        data1 = scipy.stats.zscore(data1)
        
    if zero_mean:
        data1 = data1 - data1.mean()
      
    for p in range(parc.N): 
        name = parc.names[p]
        if name[-6:] == '__Left':
            name = name[:-6] + '-lh'
        if name[-7:] == '__Right':
            name = name[:-7] + '-rh'
            
        if array_scalar(data1):                                # map values to colormap
        
            if use_sys_colors:
                
                if data1[p] > threshold:
                    network = parc.networks[p]
                    color   = colors[network]
                    if name[-2:] == 'lh':               
                        left_dict[name[:-3]]  = color
                    else:
                        right_dict[name[:-3]] = color  
        
                
            else:
        
                if data1[p] > threshold:         
                    ind    = int(((data1[p] - min1) * dx))
                    ind    = sorted((0,ind,255))[1]
                    color  = colors[ind]
                    if name[-2:] == 'lh':               
                        left_dict[name[:-3]]  = color
                    else:
                        right_dict[name[:-3]] = color    
                       
        elif data1 == 'sys':                                   # plot systems
            network = parc.networks[p]
            color   = colors[network]                        
            if name[-2:] == 'lh':               
                left_dict[name[:-3]]  = color
            else:
                right_dict[name[:-3]] = color
                
            cmap = make_cmap(colors,len1=len(colors))
                
                
        elif len(data1[0]) == 3:                               # plot given colors
            if not data1[p][0] == -1:
                if name[-2:] == 'lh':               
                    left_dict[name[:-3]]  = data1[p]
                else:
                    right_dict[name[:-3]] = data1[p]

                
    pyvista_views(left_dict,right_dict,subj_path,parc,figsize,views,filename,cmap,vrange,
                  nticks,title,label,cbar_off,surface,ticklabels,legend)
    
    
    
#%% 

def pyvista_views(left_dict,right_dict,subj_path,parc,figsize,views=['lat-med'],filename=None,
                  cmap=None,vrange=None,nticks=11,title=None,label='',cbar_off = False,surface='inflated',
                  ticklabels=[], legend=None):
    
    ''' Creates a 4-brain figure for each entry in "views" (can be 'lat-med','ant_post','top-down').
        Saves figure(s) if a filename is given. Filename must end in a legal file format.
        If "views" is not just 'lat-med', view description is added to filename for each figure.
    
    ''' 
    if filename != None and views != ['lat-med']:
        filenames = {'lat-med' : filename[:-4] + '_lat-med'  + filename[-4:],                     
                     'ant-post': filename[:-4] + '_ant-post' + filename[-4:],
                     'top-down': filename[:-4] + '_top-down' + filename[-4:]}                    
    else:
        filenames = {'lat-med': filename,'ant-post':None,'top-down':None}
            
    
        
    if 'lat-med' in views:
        pyvista_multi(left_dict,right_dict,subj_path,parc,figsize=figsize,cmap=cmap,vrange=vrange,nticks=nticks,
                      hemis  = ['lh','rh','lh','rh'],
                      cpos   = [(-1,0,0),(1,0,0),(1,0,0),(-1,0,0)], 
                      zooms  = [1.7,1.7,1.6,1.6],
                      filename = filenames['lat-med'],surface=surface,
                      title = title, label=label, cbar_off = cbar_off,
                      ticklabels=ticklabels, legend=legend
                      )
        
    if 'ant-post' in views:
        pyvista_multi(left_dict,right_dict,subj_path,parc,figsize=figsize,cmap=cmap,vrange=vrange,nticks=nticks,
                      hemis  = ['rh','lh','lh','rh'],
                      cpos   = [(0,1,0),(0,1,0),(0,-1,0),(0,-1,0)], 
                      zooms  = [1.7]*8,
                      filename = filenames['ant-post'],surface=surface,
                      title = title, label=label, cbar_off = cbar_off,
                      ticklabels=ticklabels, legend=legend
                      )
        
    if 'top-down' in views:    
        pyvista_multi(left_dict,right_dict,subj_path,parc,figsize=figsize,cmap=cmap,vrange=vrange,nticks=nticks,
                      hemis  = ['lh','lh','rh','rh'],
                      cpos   = [(0,0,1),(0,0,-1),(0,0,1),(0,0,-1)], 
                      zooms  = [1.6]*4,
                      filename = filenames['top-down'],surface=surface,
                      title = title, label=label, cbar_off = cbar_off,
                      ticklabels=ticklabels, legend=legend
                      )
        
                
#%%

def pyvista_multi(left_dict,right_dict,subj_path,parc,figsize=[12,12],filename=None,cmap=None,
                  vrange=None,nticks=11,title=None,label='',cbar_off=False,surface='inflated',
                  hemis  = ['lh','lh','rh','rh'],
                  cpos   = [(-1,0,0),(1,0,0),(1,0,0),(-1,0,0)], 
                  zooms  = [1.75,1.65,1.75,1.65],
                  ticklabels=[],
                  legend=None
                  ):
        
    '''
    A colorbar is added at the bottom, unless cbar_off is True
    '''
    
    npl = len(hemis)
    
    dict_list = []
    for h in hemis:
        if h == 'lh':
            dict_list.append(left_dict)
        else:
            dict_list.append(right_dict)
            
            
    brains = []    
    
    for i in range(npl):
        b = BrainSurface(subj_path,parcellation=parc.name,hemis=[hemis[i]],surface=surface)  
        b.set_data(dict_list[i])
        b.plot(camera_position=cpos[i],zoom=zooms[i],show=True)
        brains.append(b)
        

    nc = len(hemis)//2

    fig, axes = plt.subplots(2,nc,figsize=figsize)
    for i in range(nc):
        for j in range(2):
            k = j*2+i
            ax = axes[j,i]
            px = ax.imshow(brains[k].plotter.image)
            ax.set_axis_off()
    plt.tight_layout()
    
    if not cbar_off:
        fig.subplots_adjust(bottom=0.1)
        axc = fig.add_axes([0, 0, 1, 1])
        tick_arr = np.linspace(vrange[0],vrange[1],nticks,endpoint=True)
        px = axc.imshow(np.array([vrange]),cmap=cmap)
        axc.set_visible(False)
        cbar = plt.colorbar(px,orientation ='horizontal',aspect=36,ticks=tick_arr)
        cbar.ax.set_xlabel(label,fontsize=figsize[0]*3.5)
        if len(ticklabels)>9:
            cbar.ax.tick_params(labelsize=figsize[0]*2.5,rotation=45)
        else:
            cbar.ax.tick_params(labelsize=figsize[0]*3.5)
        if len(ticklabels)>0:
            cbar.ax.set_xticklabels(ticklabels)

    if legend != None: 
        fig.text(x = 1, y= 0.87, s = legend,fontsize = figsize[0]*2, ha = "left",va = "top")
    plt.suptitle(title,fontsize=figsize[0]*3.5)
    plt.show()

    if filename != None:   
        fig.savefig(filename,bbox_inches='tight')
    plt.close(fig)












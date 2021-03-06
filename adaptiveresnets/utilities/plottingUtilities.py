# Copyright (c) 2021, The University of Texas at Austin 
# & University of Michigan
#
# All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the aerolearn package. For more information see
# https://github.com/aerolearn/aerolearn/
#
# aerolearn is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free
# Software Foundation) version 2.0 dated June 1991.

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rc
from pylab import *
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{amsfonts}')


def plot_accs_vs_data(data_dictionary,std_data_dictionary = {},\
            axis_label = ['Training Data','Accuracy','Testing Accuracy vs Training Data'],\
                     out_name = 'max_accs',show_plot = True,full_ylim = False,legend_loc = 'best'):
    """
    This plotting function plots accuracy data vs number of training data used during training
        - :code:`data_dictionary` - dictionary of data used for plotting, keys are label names,
            values are n_data, accuracies
        - :code:`std_data_dictionary` - a dictionary of standard deviations used for error bars in plotting
            keys are label names, values are n_data, stds
        - :code:`axis_label` - python list of three strings: [x label, y label, title]
        - :code:`out_name` - string path for where to save the plot
        - :code:`show_plot` - Boolean for showing the plotting
        - :code:`full_ylim` - Boolean to show full y axis, or to truncate based on data
        - :code:`legend_loc` - string for legend_loc passed into matplotlib legend function
    """
    fig = plt.figure(figsize=(10,5))
    plt.gcf().subplots_adjust(bottom=0.15)
    ax = fig.add_subplot(111)
    
    min_y = 1.
    max_y = 1.
    
    for i,key in enumerate(data_dictionary.keys()):
        n_datas,max_accs = data_dictionary[key]
        label = ''
        for piece in key.split('_'):
            label+=piece+' '
        ax.plot(n_datas,max_accs,label = label,color = 'C'+str(i))
        if key in std_data_dictionary.keys():
            _,std = std_data_dictionary[key]
            ax.fill_between(n_datas,max_accs+std,max_accs - std,alpha = 0.2)
        min_y = min(min_y,min(max_accs))
        max_y = max(max_y,max(max_accs))
    
    ax.set_xlabel(axis_label[0],fontsize = 25)

    ax.set_ylabel(axis_label[1],fontsize = 25)
    ax.set_title(axis_label[2],loc = 'center',fontsize = 25)
    if full_ylim:
        ax.set_ylim(bottom = 0.0,top = 1.0+1e-2)
    else:
        ax.set_ylim(bottom = min_y,top = max_y)
    
    
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=15)
    ax.legend(loc = legend_loc,fontsize = 20)
    plt.savefig(out_name+'.pdf', bbox_inches = 'tight')
    if show_plot:
        plt.show()



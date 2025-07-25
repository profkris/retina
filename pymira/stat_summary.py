import pickle
import os 
import numpy as np
from matplotlib import pyplot as plt
join = os.path.join
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use('Agg')

dirs = '/mnt/data2/retinas - GIULIA/vessel_stats'

group1 = ['control_2.1',
          'control_2.2',
          'control_2.3',
          'control_3.1',
          'control_3.2',
          'control_4.1',
          'control_4.2',
          'control_4.3',
          'control_5.1',
          'control_5.2',
          'control_5.3' ]
          
group2 = ['diabetic_6.1',
          'diabetic_6.2',
          'diabetic_6.3', 
          'diabetic_7.1', 
          'diabetic_7.2',
          'diabetic_7.3',  
          'diabetic_8.1', 
          'diabetic_8.2',
          'diabetic_8.3',
          'diabetic_9.1',
          'diabetic_9.2',
          'diabetic_9.3',
          'diabetic_10.1', 
          'diabetic_10.2',
          'diabetic_10.3',
          'diabetic_11.1',
          'diabetic_11.2',
          'diabetic_11.3' ]

group1_prefs = [join('control',x) for x in group1]
group2_prefs = [join('diabetic',x) for x in group2]
group3_prefs = None

def load_data(prefs,pname='radii'):
    param = []
    for pref in prefs:
        path = join(dirs,pref)
        with open(join(path,'{}.p'.format(pname)),'rb') as fo:
            param.extend(pickle.load(fo))
    return np.asarray(param)

nbins = 30
ylog = False
pname, param_label,range,ylog = 'radii','Vessel radius (um)',[0.,25.],True
#pname, param_label,range,ylog = 'vessel_length','Vessel length (um)',[0.,500.],True
#pname, param_label,range,nbins = 'branching_angle','Branching angle',[0.,180.],20
#pname, param_label,range = 'vessel_volume','Vessel volume (um3)',[0.,500.]
#pname, param_label,range = 'nconn','Branch node connections',[2.,5.]

group1_param = load_data(group1_prefs,pname=pname)
group2_param = load_data(group2_prefs,pname=pname)
if group3_prefs is not None:
    group3_param = load_data(group3_prefs,pname=pname)
else:
    group3_param = None

def histogram(v, nbins=30,range=None,xlabel=None,color='green',clear=True,labels=None,density=True,ylog=False):
    if clear:
        plt.clf()
    #range = [v.min(),v.max()]
        
    # the histogram of the data
    n, bins, patches = plt.hist(v, nbins, density=density,label=labels,range=range)#, facecolor=color, alpha=0.5)
    plt.legend(prop={'size': 10})
    if xlabel is not None:
        plt.xlabel(xlabel)
    
    if ylog:
        ax = plt.gca()    
        ax.set_yscale('log')
        
    #plt.legend(handles, labels)

#histogram(ko_param, nbins=30,range=None,color='red',clear=True)
#histogram(ctrl_param, nbins=30,range=None,color='red',clear=False)        
#histogram(het_param, nbins=30,range=None,color='blue',clear=False)  
labels = ['Control','Diabetic']              
histogram([group1_param,group2_param], nbins=nbins,range=range,color='red',clear=True,labels=labels,xlabel=param_label,density=True,ylog=ylog)

U1, p = mannwhitneyu(group1_param, group2_param) #, method="exact")
print(U1,p)
pf = '{:g}'.format(float('{:.{p}g}'.format(p, p=2)))
plt.figtext(0.5, 0.7, f'Mann-Whitney: p={pf}')

if False:
    #plt.show()
    plt.savefig("stats_summary.png", dpi=300)
    plt.close()
else:
    ofile = join(dirs,f'{pname}_histogram.png')
    fig = plt.savefig(ofile)
    
#tots = np.asarray([np.sum(x) for x in [group1_param,group2_param]])
#nseg = np.asarray([x.shape[0] for x in [group1_param,group2_param]])


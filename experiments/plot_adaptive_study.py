"""Plots for the fixed v6 intervention experiment, at the instance level."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'results/v6';OUT=ROOT/'figures/v6'
OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.5,'axes.labelsize':10.5,
 'axes.titlesize':11,'legend.fontsize':9.5,'pdf.fonttype':42,'ps.fonttype':42,
 'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})
METHODS=['Guided','Crowding','Gated','Adaptive-class','Adaptive-slot','ILS-descent']
COLORS=['#20639B','#C05A00','#8064A2','#00866A','#A75375','#444444']


def save(fig,name):
    fig.savefig(OUT/(name+'.pdf'))
    fig.savefig(OUT/(name+'.png'),dpi=600)
    plt.close(fig)


def main():
    df=pd.read_csv(DATA/'performance_summary.csv');long=df[df.horizon==250]
    fig,ax=plt.subplots(figsize=(7.4,3.5),layout='constrained')
    vals=[long[long.method==m].mean_excess.to_numpy() for m in METHODS]
    ax.boxplot(vals,positions=np.arange(6),widths=.5,showfliers=False,
        medianprops={'color':'black','linewidth':1.2},boxprops={'color':'#666666'})
    rng=np.random.default_rng(914)
    for i,m in enumerate(METHODS):
        for stratum,marker,col in [('original TSPLIB','o','#20639B'),('added TSPLIB','s','#00866A'),('generated','^','#C05A00')]:
            s=long[(long.method==m)&(long.stratum==stratum)]
            ax.scatter(i+rng.uniform(-.16,.16,len(s)),s.mean_excess,s=15,alpha=.65,
                marker=marker,color=col,label=stratum if i==0 else None)
    ax.set_xticks(range(6),['Guided','Crowding','Gated','Adaptive\nclass','Adaptive\nslot','ILS\ndescent'])
    ax.set_yscale('symlog',linthresh=1)
    ax.set_ylabel('Instance mean reference excess (%)')
    ax.set_ylim(bottom=min(0.,float(long.mean_excess.min())))
    ax.set_yticks([0,.5,1,2,5,10])
    ax.set_yticklabels(['0','0.5','1','2','5','10'])
    ax.grid(axis='y',alpha=.2)
    handles,labels=ax.get_legend_handles_labels()
    fig.legend(handles,labels,frameon=False,ncol=3,loc='outside upper center',fontsize=9.5)
    save(fig,'figure3_extension_performance')

    fig,axes=plt.subplots(1,2,figsize=(7.4,3.15),layout='constrained')
    base=long[long.method=='Guided']
    for stratum,marker,col in [('original TSPLIB','o','#20639B'),('added TSPLIB','s','#00866A'),('generated','^','#C05A00')]:
        s=base[base.stratum==stratum]
        axes[0].scatter(s.neff_mean,s.clone_neff_mean,s=24,marker=marker,color=col,alpha=.8,label=stratum)
    axes[0].plot([0,16],[0,16],color='#999999',lw=.8)
    axes[0].axvline(4,color='#777777',ls='--',lw=.8);axes[0].axhline(4,color='#777777',ls='--',lw=.8)
    axes[0].set(xlabel='Mean slot influence count',ylabel='Mean class influence count',xlim=(0,16.5),ylim=(0,16.5))
    axes[0].legend(frameon=False,fontsize=9.5,loc='upper left')
    original=df[df.original]
    differences=[]
    for h in [50,250]:
        w=original[original.horizon==h].pivot(index='instance',columns='method',values='mean_excess')
        differences.append(w['Adaptive-class']-w['Adaptive-slot'])
    aligned=pd.concat(differences,axis=1).dropna()
    for _,row in aligned.iterrows():axes[1].plot([0,1],row,color='#AAAAAA',lw=.6,alpha=.7)
    for i in [0,1]:
        values=aligned.iloc[:,i].to_numpy()
        axes[1].scatter(np.full(len(values),i),values,s=16,color='#00866A',alpha=.7)
        rng=np.random.default_rng(92614)
        boot=np.median(values[rng.integers(len(values),size=(10000,len(values)))],axis=1)
        lo,hi=np.quantile(boot,[.025,.975]);med=np.median(values)
        axes[1].errorbar(i,med,yerr=[[med-lo],[hi-med]],fmt='s',color='black',capsize=4,ms=5)
    axes[1].axhline(0,color='#555555',ls='--',lw=.8)
    axes[1].set(xticks=[0,1],xticklabels=['50 iterations','250 iterations'],
        ylabel='Class minus slot\npolicy excess (pp)',xlim=(-.25,1.25))
    for ax in axes:ax.grid(alpha=.15)
    save(fig,'figure4_control_diagnostics')
    print('Two vector and 600 dpi figure pairs created')


if __name__=='__main__':main()

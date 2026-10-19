"""Publication plots from instance-level v5 summaries only."""
from pathlib import Path
from io import BytesIO
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'results'/'v5'
OUT=ROOT/'figures'/'v5'
OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.5,'axes.labelsize':10.5,
 'axes.titlesize':11,'legend.fontsize':9.5,'pdf.fonttype':42,'ps.fonttype':42,
 'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})
POP=['Guided','Neutral','Guided-crowding','Neutral-crowding']
COLORS=['#20639B','#4D4D4D','#C05A00','#7E57A3']
STYLES=['-','--','-','--']


def save(fig,name):
    fig.savefig(OUT/f'{name}.pdf',dpi=600)
    buffer=BytesIO()
    fig.savefig(buffer,format='png',dpi=600)
    payload=buffer.getvalue()
    with Image.open(BytesIO(payload)) as check:
        check.verify()
    target=OUT/f'{name}.png'
    temporary=target.with_suffix('.png.tmp')
    temporary.write_bytes(payload)
    temporary.replace(target)
    plt.close(fig)


trace_path=DATA/'diagnostic_traces.csv'
trace=pd.read_csv(trace_path if trace_path.exists() else DATA/'diagnostic_traces.csv.gz')
trace=trace[trace.method.isin(POP)]
mean=trace.groupby(['instance','method','iteration']).mean(numeric_only=True).reset_index()
mean.to_csv(DATA/'instance_diagnostic_traces.csv',index=False)
fig,axes=plt.subplots(1,3,figsize=(7.4,2.8),layout='constrained')
for metric,ax,label in zip(['neff','clone_neff','edge_diversity'],axes,
                           ['Slot influence count','Clone-resolved\ninfluence count','Mean edge\ndisagreement']):
    for method,color,style in zip(POP,COLORS,STYLES):
        s=mean[mean.method==method].groupby('iteration')[metric]
        m=s.mean();lo=s.quantile(.25);hi=s.quantile(.75)
        ax.plot(m.index,m.values,label=method,color=color,linestyle=style,lw=1.6)
        ax.fill_between(m.index,lo,hi,color=color,alpha=.08,linewidth=0)
    ax.set(xlabel='Completed iterations',ylabel=label,xlim=(0,50))
    ax.grid(alpha=.2)
axes[0].set_ylim(0,16.5);axes[1].set_ylim(0,16.5);axes[2].set_ylim(0,.8)
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False)
save(fig,'figure1_influence_traces')

summary=pd.read_csv(DATA/'performance_summary.csv')
methods=POP+['ILS-800','ILS-work']
fig,ax=plt.subplots(figsize=(7.4,3.4),layout='constrained')
values=[summary[summary.method==m].mean_excess.to_numpy() for m in methods]
ax.boxplot(values,positions=np.arange(6),widths=.5,showfliers=False,
           medianprops={'color':'black','linewidth':1.2},
           boxprops={'color':'#555555'},whiskerprops={'color':'#777777'},capprops={'color':'#777777'})
rng=np.random.default_rng(182)
for i,m in enumerate(methods):
    for suite,marker,col in [('TSPLIB','o','#20639B'),('synthetic','^','#C05A00')]:
        s=summary[(summary.method==m)&(summary.suite==suite)]
        ax.scatter(i+rng.uniform(-.16,.16,len(s)),s.mean_excess,s=16,marker=marker,
                   color=col,alpha=.7,label=suite if i==0 else None)
ax.set_xticks(np.arange(6),['Guided','Neutral','Guided\ncrowding','Neutral\ncrowding','ILS\n800 candidates','ILS\nmatched trials'])
ax.set_ylabel('Instance mean excess to reference (%)')
ax.grid(axis='y',alpha=.2);ax.legend(frameon=False,loc='upper left')
save(fig,'figure2_performance')

fig,axes=plt.subplots(1,2,figsize=(7.4,3.35),layout='constrained')
for ax,normal,crowd in zip(axes,['Guided','Neutral'],['Guided-crowding','Neutral-crowding']):
    a=summary[summary.method==normal].set_index('instance')
    b=summary[summary.method==crowd].set_index('instance')
    for idx in a.index:
        ax.plot([a.loc[idx,'edge_diversity_mean'],b.loc[idx,'edge_diversity_mean']],
                [a.loc[idx,'mean_excess'],b.loc[idx,'mean_excess']],color='#AAAAAA',lw=.6,zorder=0)
    ax.scatter(a.edge_diversity_mean,a.mean_excess,s=22,color='#20639B',label='Indexed survival')
    ax.scatter(b.edge_diversity_mean,b.mean_excess,s=25,marker='^',color='#C05A00',label='Structural survival')
    ax.set(title=normal+' candidate guidance',xlabel='Final mean edge disagreement',ylabel='Mean reference excess (%)')
    ax.grid(alpha=.2)
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False)
save(fig,'figure3_diversity_quality')

fig,axes=plt.subplots(1,2,figsize=(7.4,3),layout='constrained')
for ax,metric,label in zip(axes,['edge_diversity','count'],['Edge disagreement\ninside band','Tours inside band']):
    for method,color,style in zip(POP,COLORS,STYLES):
        v=[summary[summary.method==method][f'q{pct:02d}_{metric}_mean'].mean() for pct in (5,10,20)]
        ax.plot([5,10,20],v,'o',color=color,linestyle=style,label=method)
    ax.set(xlabel='Maximum excess to current\npopulation best (%)',ylabel=label,xticks=[5,10,20])
    ax.grid(alpha=.2)
axes[1].set_ylim(0,16.5)
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False)
save(fig,'figureS1_quality_bands')
print('Four figure pairs created')

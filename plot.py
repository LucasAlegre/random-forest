import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import brewer2mpl
import seaborn as sns

sns.set(rc={'figure.figsize':(12,9)})
sns.set(font_scale = 2)

bmap = brewer2mpl.get_map('Set2', 'qualitative', 7)
colors = bmap.mpl_colors

ntrees = [1,5,10,25,50]

boxplots = []
for i in [1, 5, 10, 25, 50]:
    df = pd.read_csv('results/pima.tsv_n_{}.csv'.format(i))
    boxplots.append(df['f1'])

ax = sns.boxplot(x=ntrees, y=boxplots, linewidth=2.5)
#ax = sns.swarmplot(x=ntrees, y=boxplots, color=".25")

ax.set(xlabel='Number of Trees', ylabel='Average F1 score')

plt.show()

ax.get_figure().savefig('results/pima'+'.pdf', bbox_inches='tight')

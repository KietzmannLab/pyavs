"""Plot number of scenes per semantic cluster for NSD and AVS.

One point per cluster, two datasets:
  - NSD: count of NSD scenes per cluster (error bar from bootstrapped CI across clusters)
  - AVS: always 60 scenes per cluster by design (no visible error bar)
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pyavs import get_data_path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_DATA_ROOT = get_data_path()
if _DATA_ROOT is None:
    raise FileNotFoundError(
        "No data path configured. Run: pyavs configure --data-path /path/to/data"
    )
EMBEDDINGS_CSV = os.path.join(_DATA_ROOT, 'input', 'scene_sampling_MEG', 'df_mean_embeddings_clustered_60.csv')
AVS_SCENES_CSV = os.path.join(_DATA_ROOT, 'input', 'scene_sampling_MEG', 'experiment_cocoIDs.csv')

OUTPUT_DIR = os.path.dirname(__file__)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df_emb = pd.read_csv(EMBEDDINGS_CSV, usecols=['cocoID', 'cluster'])
df_avs = pd.read_csv(AVS_SCENES_CSV, usecols=['cocoID', 'clusterID'])

# Count scenes per cluster
nsd_counts = df_emb.groupby('cluster').size().reset_index(name='count')
nsd_counts['dataset'] = 'NSD'
nsd_counts['fraction'] = nsd_counts['count'] / nsd_counts['count'].sum()

avs_counts = df_avs.groupby('clusterID').size().reset_index(name='count')
avs_counts = avs_counts.rename(columns={'clusterID': 'cluster'})
avs_counts['dataset'] = 'AVS'
avs_counts['fraction'] = avs_counts['count'] / avs_counts['count'].sum()

# compute the fraction of each clster in NSD and AVS (for sanity check, should be similar across datasets)

# make a df that hold 60 clsuters, their count in NSD and AVS, and the fraction of each cluster in NSD and AVS (for sanity check, should be similar across datasets)



plot_df = pd.concat([nsd_counts, avs_counts], ignore_index=True)
# make fraction percentage for better readability
plot_df['fraction'] = plot_df['fraction'] * 100
# round to no decimal places for better readability
#plot_df['fraction'] = plot_df['fraction'].round(0)
# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
sns.set_context("poster")
plt.figure(figsize=(7, 5))


sns.stripplot(
    data=plot_df,
    x='dataset',
    y='fraction',
    #errorbar=('ci', 95),
    palette={'NSD': 'darkgrey', 'AVS': 'forestgreen'},
    order=['NSD', 'AVS'],
    edgecolor='k',
    size=10, jitter=0.2, linewidth=0.8,  
    #capsize=0.15,
)
# add the mean as a point on top of the stripplot
# sns.pointplot(
#     data=plot_df,
#     x='dataset',
#     y='fraction',
#     estimator='mean',
#     ci=95,
#     color='darkgrey',
#     markers='_',
#     #scale=1.5, 
#     errorbar=('ci', 95),
#     order=['NSD', 'AVS'], join=False, zorder=100
# )


#plt.xlabel('dataset')
plt.ylabel('share of scenes \nin dataset per cluster\n[count]')
plt.legend(["Number of semantic\nclusters k = 60"], frameon=False)
sns.despine()
plt.tight_layout()

out_path = os.path.join(OUTPUT_DIR, 'scenes_per_cluster.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight')

# save as pdf as well
out_path = os.path.join(OUTPUT_DIR, 'scenes_per_cluster.pdf')
plt.savefig(out_path, format='pdf', bbox_inches='tight')
print(f'saved {out_path} in {OUTPUT_DIR}')

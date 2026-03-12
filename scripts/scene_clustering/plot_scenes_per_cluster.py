"""Plot number of scenes per semantic cluster for NSD and AVS.

One point per cluster, two datasets:
  - NSD: count of NSD scenes per cluster (error bar from bootstrapped CI across clusters)
  - AVS: always 60 scenes per cluster by design (no visible error bar)
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EMBEDDINGS_CSV = '/share/klab/datasets/avs/input/scene_sampling_MEG/df_mean_embeddings_clustered_60.csv'
AVS_SCENES_CSV = '/share/klab/datasets/avs/input/scene_sampling_MEG/experiment_cocoIDs.csv'

OUTPUT_DIR = os.path.dirname(__file__)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df_emb = pd.read_csv(EMBEDDINGS_CSV, usecols=['cocoID', 'cluster'])
df_avs = pd.read_csv(AVS_SCENES_CSV, usecols=['cocoID', 'clusterID'])

# Count scenes per cluster
nsd_counts = df_emb.groupby('cluster').size().reset_index(name='count')
nsd_counts['dataset'] = 'NSD'

avs_counts = df_avs.groupby('clusterID').size().reset_index(name='count')
avs_counts = avs_counts.rename(columns={'clusterID': 'cluster'})
avs_counts['dataset'] = 'AVS (k=60)'

plot_df = pd.concat([nsd_counts, avs_counts], ignore_index=True)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
sns.set_context("poster")
plt.figure(figsize=(8, 6))

sns.pointplot(
    data=plot_df,
    x='dataset',
    y='count',
    errorbar=('ci', 95),
    palette={'NSD': 'cornflowerblue', 'AVS (k=60)': 'salmon'},
    order=['NSD', 'AVS (k=60)'],
    capsize=0.15,
)

plt.xlabel('dataset')
plt.ylabel('scenes per cluster [count]')
plt.legend([], frameon=False)
sns.despine()
plt.tight_layout()

out_path = os.path.join(OUTPUT_DIR, 'scenes_per_cluster.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close()
print(f'saved {out_path}')

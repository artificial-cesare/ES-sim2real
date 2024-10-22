import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

seeds = [100, 200, 300, 400, 500]
train_classes = ["SetParams", "Rand", "AutoRand"]
train_env_mass = 6
train_env_friction = 2

global_min = 0
global_max = None

data = {}
for train_class in train_classes:
    df = pd.DataFrame(columns=["mass", "friction", "reward"])
    for seed in seeds:
        file_path = f'/content/perf_{train_env_mass}_{train_env_friction}_{seed}_{train_class}.csv'
        tmp_df = pd.read_csv(file_path)
        if global_max is None or global_max < np.max(tmp_df["reward"]):
            global_max = np.max(tmp_df["reward"])
        df = pd.concat([df, tmp_df], ignore_index=True)

    df = df.groupby(['friction', 'mass']).agg({'reward': 'mean'}).reset_index()
    pivot_table = df.pivot('friction', 'mass', 'reward')  # Correct pivot_table creation

    data[train_class] = pivot_table  # Store the pivot_table instead of df

fig, axs = plt.subplots(1, len(train_classes), figsize=(15, 5))  # Adjusted for dynamic number of classes

for index, (key, pivot_table) in enumerate(data.items()):
    sns.heatmap(pivot_table, ax=axs[index], annot=False, vmin=global_min, vmax=global_max, cmap='viridis')
    axs[index].set_title(f'Heatmap {key}')  # Use key for the title

fig.suptitle('Agent Performances')

plt.tight_layout()
#plt.savefig(f'/content/heatmap_{train_env_mass}_{train_env_friction}.png')
plt.show()
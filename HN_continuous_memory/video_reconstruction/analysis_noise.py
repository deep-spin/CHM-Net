import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define colors
tikz_blue = '#009ADE'  # normalized_logprob

# Apply seaborn theme for gridlines
sns.set_theme(style="darkgrid")
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Palatino']

# Function to load data
def load_data(filename):
    with open(filename, 'r') as f:
        return json.load(f)

# Function to aggregate data
def aggregate_data(data):
    discrete_values = np.array([entry["discrete"] for entry in data.values()])
    discrete_agg = np.mean(discrete_values, axis=0)
    discrete_std = np.std(discrete_values, axis=0)
    
    continuous_values = np.array([entry["continuous"] for entry in data.values()])
    continuous_mean = np.mean(continuous_values, axis=0)
    continuous_std = np.std(continuous_values, axis=0)
    
    return discrete_agg, discrete_std, continuous_mean, continuous_std

# Load first dataset
filename1 = "/mnt/data-poseidon/saul/continuous_HN/video_reconstruction/results/beta_10.0_nb_basis_128_num_iters_1_N_1024_resolution_224_mask_[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0].json"
data1 = load_data(filename1)
discrete_agg1, discrete_std1, continuous_mean1, continuous_std1 = aggregate_data(data1)

# Load second dataset
filename2 = "/mnt/data-poseidon/saul/continuous_HN/embedding_reconstruction/results/beta_10.0_nb_basis_1024_num_iters_1_N_2048_resolution_224_std_[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0].json"
data2 = load_data(filename2)
discrete_agg2, discrete_std2, continuous_mean2, continuous_std2 = aggregate_data(data2)

# Define x values
x = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

# Plotting side by side
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# First plot
axs[0].plot(x, continuous_mean1, marker='o', linestyle='-', color=tikz_blue, label="Continuous HN", linewidth=2)
axs[0].fill_between(x, continuous_mean1 - continuous_std1, continuous_mean1 + continuous_std1, color=tikz_blue, alpha=0.2)
axs[0].plot(x, discrete_agg1, color="red", linestyle='--', label="Discrete HN", linewidth=2)
axs[0].fill_between(x, discrete_agg1 - discrete_std1, discrete_agg1 + discrete_std1, color="red", alpha=0.2)
axs[0].set_xlabel('Mask Percentage', fontsize=18)
axs[0].set_ylim(0.3, 0.9)
axs[0].set_title("Video", fontsize=18)
axs[0].legend(fontsize=18)
axs[0].tick_params(axis='y', labelsize=15)
axs[0].tick_params(axis='x', labelsize=15)
axs[0].grid(True)
x = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0])
# Second plot
axs[1].plot(x, continuous_mean2, marker='o', linestyle='-', color=tikz_blue, label="Continuous HN", linewidth=2)
axs[1].fill_between(x, continuous_mean2 - continuous_std2, continuous_mean2 + continuous_std2, color=tikz_blue, alpha=0.2)
axs[1].plot(x, discrete_agg2, color="red", linestyle='--', label="Discrete HN", linewidth=2)
axs[1].fill_between(x, discrete_agg2 - discrete_std2, discrete_agg2 + discrete_std2, color="red", alpha=0.2)
axs[1].set_xlabel('$\sigma$', fontsize=18)
axs[1].set_ylim(0.4, 1.05)
axs[1].set_title("Embedding", fontsize=18)
axs[1].tick_params(axis='y', labelsize=15)
axs[1].tick_params(axis='x', labelsize=15)
axs[1].grid(True)

# Save and show plot
plt.tight_layout()
plt.savefig('eff3.pdf')
plt.show()

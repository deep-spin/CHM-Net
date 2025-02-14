import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define colors
tikz_blue = '#009ADE'  # normalized_logprob
tikz_orange = '#F28522'  # length and -length

# Apply seaborn theme for gridlines
sns.set_theme(style="darkgrid")
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Palatino']

# Function to load data
def load_data(N, basis, mask):
    filename = f"/mnt/data-poseidon/saul/continuous_HN/video_reconstruction/results/beta_10.0_nb_basis_{basis}_num_iters_1_N_{N}_resolution_224_mask_{mask}.json"
    with open(filename, 'r') as f:
        return json.load(f)
# Function to aggregate data
def aggregate_data(data):
    discrete_values = np.array([entry["discrete"] for entry in data.values()])
    continuous_values = np.array([entry["continuous"] for entry in data.values()])
    return np.mean(discrete_values, axis=0), np.std(discrete_values, axis=0), np.mean(continuous_values, axis=0), np.std(continuous_values, axis=0)

# Load and aggregate data
# Load and aggregate data
Ns = [512, 1024,2048,  4096]
bases = [[8, 16, 32, 64, 128, 256, 512], [8, 16, 32, 64, 128, 256, 512, 1024], [8, 16, 32, 64, 128, 256, 512, 1024, 2048], [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]]
data = []
for mask in ["0.5"]:
    for N, basis in zip(Ns, bases):
        raw_data = load_data(N, basis, mask)
        data.append((*aggregate_data(raw_data), basis, f"$L={N}$"))
# Plotting
fig, axs = plt.subplots(2, 2, figsize=(26.6, 16))
for i, (discrete_agg, discrete_std, continuous_mean, continuous_std, basis, title) in enumerate(data):
    row = i // 2
    col = i % 2  
    axs[row, col].plot(basis, continuous_mean, marker='o', markersize=14, linestyle='-', color=tikz_blue, label="Continuous HN", linewidth=6)
    axs[row, col].axhline(discrete_agg[-1], color="black", linestyle='dotted', label="Discrete HN, $L_{sub}=L$", linewidth=6)
    axs[row, col].fill_between(basis, continuous_mean - continuous_std, continuous_mean + continuous_std, color=tikz_blue, alpha=0.2)
    axs[row, col].plot(basis, discrete_agg, color="red", linestyle='--', label="Discrete HN, $L_{sub}=N$", linewidth=6)
    axs[row, col].fill_between(basis, discrete_agg - discrete_std, discrete_agg + discrete_std, color="red", alpha=0.2)
    axs[row, col].set_xscale('log', base=2)
    axs[row, col].set_ylim(0.2, 0.9)  # Set y-axis limits
    axs[row, col].set_title(title, fontsize=35)  # Set title
    axs[1, col].set_xlabel('N', fontsize=32)  # Set x-axis label
    axs[row, col].set_xticks(basis)  # Set the positions of the ticks

    axs[row, col].set_xticklabels(basis, fontsize=32)
    axs[row, col].tick_params(axis='y', labelsize=32)  # Adjust y-axis tick label size
    axs[row, col].grid(True)  # Enable grid

    # Optional: Add legend only for the first plot in each row
    if col == 0:
        axs[0, col].legend(fontsize=32)


# Set gridlines, ticks, and limits
# Adjust layout and show plot
plt.subplots_adjust(hspace=5)
plt.tight_layout()

plt.savefig('eff.pdf')
plt.show()
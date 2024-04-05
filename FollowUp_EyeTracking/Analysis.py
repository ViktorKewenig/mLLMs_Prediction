
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load the CSV file
categorical_data = pd.read_csv("categorical_data_1.csv")
categorical_data = categorical_data.loc[:, ~categorical_data.columns.str.contains('^Unnamed')]

# load correlations by word and segment
correlations_by_word_segment = np.load("correlations_by_word_segment.npy",allow_pickle=True).item()

# Convert the dictionary to a DataFrame for merging
correlations_df = pd.DataFrame([
    {'word': key[1], 'segment': key[0], 'pearson_corr': value[0], 'spearman_corr': value[1]}
    for key, value in correlations_by_word_segment.items()
])


# Convert 'word' to string in both DataFrames
categorical_data['word'] = categorical_data['word'].astype(str)
correlations_df['word'] = correlations_df['word'].astype(str)

# Now merge the data
merged_data = pd.merge(categorical_data, correlations_df, on=['word', 'segment'])


# Linear Regression Models to test if 'present' can predict correlation values
# Model for Pearson correlation
model_pearson = smf.ols('pearson_corr ~ C(present)', data=merged_data).fit()
print("Model for Pearson Correlation:")
print(model_pearson.summary())

# Model for Spearman correlation
model_spearman = smf.ols('spearman_corr ~ C(present)', data=merged_data).fit()
print("\nModel for Spearman Correlation:")
print(model_spearman.summary())

#####################
# Plot Violin Plots #
#####################

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



# Colors and styles
BG_WHITE = '#FFFFFF'
GREY_LIGHT = "#b4aea9"
GREY50 = "#7F7F7F"
BLUE_DARK = "#1B2838"
BLUE = "#2a475e"
BLACK = "#282724"
GREY_DARK = "#747473"
RED_DARK = "#850e00"
COLOR_SCALE = ['#FF1105','#FF1105']

# Assuming 'merged_data' is your DataFrame with 'spearman_corr' and 'present' columns
# Example:
# merged_data = pd.DataFrame({
#     'spearman_corr': np.random.randn(100),
#     'present': np.random.randint(0, 2, size=100)
# })

# Create positions for the violin plots
positions = sorted(merged_data['present'].unique())

# Create horizontal lines for reference
hlines = [0.2, 0, -0.2]

fig, ax = plt.subplots(figsize=(14, 10))

# Set background color
fig.patch.set_facecolor(BG_WHITE)
ax.set_facecolor(BG_WHITE)

# Add reference lines
for h in hlines:
    ax.axhline(h, color=GREY50, ls=(0, (5, 5)), alpha=0.8, zorder=0)

# Add violin plots
violins = ax.violinplot(
    [merged_data[merged_data['present'] == x]['spearman_corr'] for x in positions],
    positions=positions,
    widths=0.45,
    bw_method="silverman",
    showmeans=False, 
    showmedians=False,
    showextrema=False
)

# Customize violins
for pc in violins["bodies"]:
    pc.set_facecolor("none")
    pc.set_edgecolor(BLACK)
    pc.set_linewidth(1.4)
    pc.set_alpha(1)

# Add boxplots
medianprops = dict(linewidth=4, color=GREY_DARK, solid_capstyle="butt")
boxprops = dict(linewidth=2, color=GREY_DARK)

ax.boxplot(
    [merged_data[merged_data['present'] == x]['spearman_corr'] for x in positions],
    positions=positions, 
    showfliers=False,
    showcaps=False,
    medianprops=medianprops,
    whiskerprops=boxprops,
    boxprops=boxprops
)

# Add jittered dots for individual data points
for pos in positions:
    data = merged_data[merged_data['present'] == pos]['spearman_corr']
    x = np.random.normal(pos, 0.05, size=len(data))
    ax.scatter(x, data, s=100, color=COLOR_SCALE[pos], alpha=0.4)

# Add mean value labels
for pos in positions:
    mean = merged_data[merged_data['present'] == pos]['spearman_corr'].mean()
    ax.scatter(pos, mean, s=250, color=RED_DARK, zorder=3)
    ax.text(
        pos, mean + 0.02,
        f"Mean: {round(mean, 2)}",
        fontsize=13, ha="center", va="bottom",
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.15")
    )

# Set labels and title
plt.xticks(positions, ['Absent', 'Present'])
plt.title('Spearman Correlation Distribution by Condition Presence', fontsize=16)
plt.xlabel('Condition Presence', fontsize=14)
plt.ylabel('Spearman Correlation', fontsize=14)

# Clean up plot
sns.despine(left=True, bottom=True)
plt.grid(False)
plt.tight_layout()
plt.show()

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import re
from matplotlib.ticker import FuncFormatter

textwidth_pt = 455.24408
inches_per_pt = 1.0 / 72.27
subplot_width_fraction = 0.32

fig_width_in = textwidth_pt * inches_per_pt
fig_height_in = fig_width_in / 1.618
fontsize_in_latex = 8 * 1 / subplot_width_fraction

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["TeX Gyre Heros"],
    "figure.figsize": [fig_width_in, fig_height_in],
    "font.size": fontsize_in_latex,
    "axes.labelsize": fontsize_in_latex - 6,
    "legend.fontsize": fontsize_in_latex - 7,
    "legend.labelcolor": "black",
    "xtick.labelsize": fontsize_in_latex - 6,
    "ytick.labelsize": fontsize_in_latex - 6,
    "axes.linewidth": 0.5,
    "axes.edgecolor": "gray",
    "text.latex.preamble": r'\usepackage{amsmath}'
})
# --- End of Configuration ---

# <<< CHANGED: Update glob pattern to match the new embedding file names
folder_path = r"/home/rasmus.jensen/Documents/HiFi-KPI/Final_reports"
csv_files = glob.glob(os.path.join(folder_path, "presentation_Level_*_clean_train_gemma_embeddings.csv"))


# <<< CHANGED: Update regex to find the number immediately following "Level_"
def extract_number(filename):
    match = re.search(r'Level_(\d+)', filename)
    return int(match.group(1)) if match else 0


def make_rolling_macro_f1(csv_files, title, output_filename, legend_location="lower left", yaxis_label=True,
                          bbox_to_anchor=None):
    # Sort files based on the extracted number so n=1, n=3, n=5 appear in order
    csv_files.sort(key=extract_number)

    fig, ax = plt.subplots()
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X']
    linestyles = ['-', '--', '-.', ':']

    for i, file in enumerate(csv_files):
        df = pd.read_csv(file)
        df['support'] = pd.to_numeric(df['support'], errors='coerce')
        df['f1-score'] = pd.to_numeric(df['f1-score'], errors='coerce')
        df['cumulative_support'] = df['support'].cumsum()
        df['cumulative_macro_f1'] = df['f1-score'].expanding(min_periods=1).mean()

        # <<< CHANGED: Update regex here as well for the plot legend labels
        match = re.search(r'Level_(\d+)', file)
        number = match.group(1) if match else "Unknown"
        label = f"$n={number}$"

        ax.plot(df['cumulative_support'], df['cumulative_macro_f1'],
                marker=markers[i % len(markers)],
                linestyle='None',
                markersize=3,
                label=label,
                alpha=0.75)

    def thousands_formatter(x, pos):
        return f'{int(x / 1000)}k'

    ax.set_xlabel(r'Cumulative Support')
    if yaxis_label:
        ax.set_ylabel(r'Rolling Macro F$_1$')
    ax.set_xlim(0, 175000)
    ax.set_ylim(0, 1)
    ax.legend(loc=legend_location, bbox_to_anchor=bbox_to_anchor, handlelength=0.8, handletextpad=0.4, borderpad=0.4,
              labelspacing=0.2)
    ax.grid(False)

    ax.xaxis.set_major_formatter(FuncFormatter(thousands_formatter))

    ax.text(0.95, 0.95, title, transform=ax.transAxes,
            fontsize=fontsize_in_latex, fontweight='bold', ha='right', va='top')

    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.show()


make_rolling_macro_f1(csv_files, "(a) Presentation (TC)", "presentation_TC.png", yaxis_label=False, bbox_to_anchor=(0, -0.025))
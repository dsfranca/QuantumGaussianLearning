import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# --- Data Definitions ---
data_10e5 = {
    'l': [2, 4, 6, 8, 10],
    'Error': [0.13646366775844276, 0.045442938936133936, 0.04475055220885116, 0.045705644963831116, 0.0478910149319256]
}
data_10e4 = {
    'l': [2, 4, 6, 8, 10],
    'Error': [0.2362797825586731, 0.2114592090059657, 0.29234335574237336, 0.3067729129354426, 0.36553700053862914]
}
# this one is the one with exact covariance matrices
data_10e00 = {
    'l': [2, 4, 6, 8, 10],
    'Error': [0.11653511141870965, 2.3904821522258146e-7, 4.9713788641270185e-11, 1.354472090042691e-14, 9.769962616701378e-15]
}

# --- Baselines ---
# Sampled baselines (for main plot)
global_naive_error = 0.09187648542918603
global_linear_error = 1.8128421760574922

# Exact baseline (for inset)
classical_exact_error = 1.07e+00

def plot_with_upper_left_inset(data1, data2, data3):
    df1 = pd.DataFrame(data1) # Green
    df2 = pd.DataFrame(data2) # Red
    df3 = pd.DataFrame(data3) # Purple (Exact)

    with plt.style.context('fast'):
        fig, ax = plt.subplots(figsize=(10, 7))

        # --- MAIN PLOT ---
        
        # 1. Plot Red (Low Sampling)
        ax.plot(df2['l'], df2['Error'], marker='o', linestyle='-', linewidth=2, 
                color='#d62728', label='Local Rec. (Sampled, $N=10^{4}$)')

        # 2. Plot Green (High Sampling)
        ax.plot(df1['l'], df1['Error'], marker='o', linestyle='-', linewidth=2, 
                color='#2ca02c', label='Local Rec. (Sampled, $N=10^5$)')

        # 3. Baselines (Sampled)
        ax.axhline(y=global_naive_error, color='#1f77b4', linestyle='--', linewidth=1.5, alpha=0.7,
                   label='Global Rec. (Sampled, $N=10^5$)')
        

        # Formatting
        ax.set_title('Convergence of Local Reconstruction vs Locality Parameter $l$\n(Fixed System Size m=100)', fontsize=14, pad=15)
        ax.set_xlabel('Locality Parameter ($l$)', fontsize=12)
        ax.set_ylabel('Reconstruction Error', fontsize=12)
        ax.set_yscale('log')
        
        # Limits for Main Plot
        ax.set_ylim(bottom=0.02, top=4)
        
        # Grid and Ticks
        ax.grid(True, which="major", linestyle='-', color='0.9')
        ax.set_xticks(df1['l'])
        
        # Legend (Main)
        ax.legend(loc='lower right', frameon=True, framealpha=1, edgecolor='0.8')

        # --- INSET PLOT (Upper Left) ---
        
        # [left, bottom, width, height]
        ax_ins = ax.inset_axes([0.08, 0.55, 0.35, 0.35]) 
        
        # 1. Plot Local Exact (Purple)
        ax_ins.plot(df3['l'], df3['Error'], marker='o', markersize=4, linestyle='-', linewidth=1.5, 
                    color="#893093", label='Local rec. (Exact)')

        # 2. Plot Classical Exact (Black Dotted) - The new value 1.07
        ax_ins.axhline(y=classical_exact_error, color='black', linestyle=':', linewidth=1.5, alpha=0.8,
                       label='Classical rec. (Exact)')
        
        # Inset Formatting
        ax_ins.set_title("Exact Covariance Limit", fontsize=9)
        ax_ins.set_yscale('log')
        ax_ins.set_xlabel('$l$', fontsize=8)
        ax_ins.set_ylabel('Error', fontsize=8)
        
        # Set Y-lims for inset to show the gap (10^-16 to 10^1)
        ax_ins.set_ylim(bottom=10**-16, top=10)
        
        # Inset Grid and Ticks
        ax_ins.grid(True, which="major", linestyle='-', color='0.9', alpha=0.5)
        ax_ins.tick_params(axis='both', which='major', labelsize=8)
        ax_ins.set_xticks([2, 4, 6, 8, 10])
        
        # Inset Legend
        ax_ins.legend(fontsize=7, loc='lower left')

        plt.tight_layout()
        plt.savefig('inset_exact_comparison.png', dpi=300)
        plt.show()

if __name__ == "__main__":
    plot_with_upper_left_inset(data_10e5, data_10e4, data_10e00)
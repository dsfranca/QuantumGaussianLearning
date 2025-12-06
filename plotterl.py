import pandas as pd
import matplotlib.pyplot as plt

# 1. Define the data
# Dataset 1: High sampling (100,000 samples) - The "good" convergence
data_10e5 = {
    'l': [2, 4, 6, 8, 10],
    'Error': [
        0.13646366775844276,
        0.045442938936133936,
        0.04475055220885116,
        0.045705644963831116,
        0.0478910149319256
    ]
}

# Dataset 2: Low sampling (10,000 samples) - The "noisy" convergence
data_10e4 = {
    'l': [2, 4, 6, 8, 10],
    'Error': [
        0.2362797825586731,
        0.2114592090059657,
        0.29234335574237336,
        0.3067729129354426,
        0.36553700053862914
    ]
}
global_naive_error =0.09187648542918603  # Standard Global Inversion (With Log)
global_linear_error = 1.8128421760574922 # Classical Inversion (Without Log)
def plot_l_comparison(data1, data2):
    """
    Plots the reconstruction error as a function of l for two different sample sizes.
    """
    try:
        # Create DataFrames
        df1 = pd.DataFrame(data1)
        df2 = pd.DataFrame(data2)
        
        # Create the figure and axis
        plt.figure(figsize=(8, 6))
        
        # Plot Dataset 1 (10^5 Samples)
        plt.plot(
            df1['l'], 
            df1['Error'], 
            marker='o', 
            linestyle='-', 
            linewidth=2,
            color='tab:green',
            label='Local Reconstruction ($S=10^5$)'
        )

        # Plot Dataset 2 (10^4 Samples)
        plt.plot(
            df2['l'], 
            df2['Error'], 
            marker='s', 
            linestyle='-', 
            linewidth=2,
            color='tab:red',
            label='Local Reconstruction ($S=10^4$)'
        )
        # --- Plot Baselines ---
        
        # Global Naive (With Log)
        plt.axhline(
            y=naive_err, 
            color='tab:blue', 
            linestyle='-.', 
            linewidth=2, 
            label=f'Global inversion $S=10^{-5}$'
        )

        # Global Linear (Without Log)
        plt.axhline(
            y=linear_err, 
            color='black', 
            linestyle=':', 
            linewidth=2, 
            label=f'Global classical inversion (no log) $S=10^{-5}$'
        )
        # Add titles and labels
        plt.title('Convergence of Local Reconstruction vs Locality Parameter $l$\n(Fixed System Size m=100)', fontsize=14)
        plt.xlabel('Locality Parameter ($l$)', fontsize=12)
        plt.ylabel('Reconstruction Error (Log Scale)', fontsize=12)
        
        # Add a grid for easier reading
        plt.grid(True, which="both", ls=":", alpha=0.7)
        
        # Use Log Scale for Y axis to see the noise floor difference clearly
        plt.yscale('log')
        
        # Force integer ticks on X axis since l is discrete
        plt.xticks(df1['l'])
        
        # Add a legend
        plt.legend()
        
        # Adjust layout
        plt.tight_layout()
        
        # Show the plot
        print("Displaying plot...")
        output_filename = 'error_vs_l_comparison.png'
        plt.savefig(output_filename, dpi=300)
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    plot_l_comparison(data_10e5, data_10e4)
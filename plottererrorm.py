import pandas as pd
import matplotlib.pyplot as plt
import io


csv_data = """SystemSize_m,Avg_Naive_Error,Avg_Local_Error,Avg_Naive_Time_sec,Avg_Local_Time_sec
100.0,0.4010972029436672,0.2631239762907575,2.400156587,3.396325023333333
150.0,0.3928258789044998,0.2748748531673381,3.9310780783333334,7.532880201666667
200.0,0.35826884076290716,0.2337704908283255,7.7327406970000006,14.665984107
250.0,0.3972735892843282,0.2743049774475094,10.733624034,26.26754447233333
300.0,0.3711741059882267,0.4203029908480333,16.299742551666665,42.89746546166666
350.0,0.4274396113926449,0.29250774376483274,20.900993286,70.60134130133333
400.0,0.4357873274388382,0.26940144802738447,29.833314852666664,101.399556959
450.0,0.4331616119309301,0.3100299390573451,35.16779048633333,123.40153188699999
500.0,0.7585688466610149,0.34835006243051136,43.19673302333334,167.50661666933334
550.0,1.1310695177195065,0.29508586342544313,57.43090550466667,266.870392432
600.0,0.6208069626686499,0.3249324499348633,61.101498217,326.4959266453333
650.0,0.975680786042548,0.34648153479404115,74.33219965466667,407.70215670933334
700.0,2.3174495930532166,0.28310782666633477,92.06555236933333,497.03597990599997
750.0,2.286723046920183,0.2998280227799168,103.19488614199999,624.8410402056667
800.0,1.0395880803061852,0.3573651154071505,114.56099054066668,713.238369111
850.0,0.8598503437775632,0.32807269203488554,133.395484973,840.632176182
900.0,1.177811070972692,0.3630584450489603,140.63008168000002,943.5083508563333
950.0,2.7012240989545298,0.3595928106425783,161.49451646833333,1053.145244494
1000.0,1.133486292895195,0.3553531697554541,190.1686131063333,1168.433412371
1050.0,1.0682937282757805,0.3338623486107253,251.40017381533335,1386.7390368626666
1100.0,1.1312066720340683,0.29996120492653966,274.19389307733337,1847.9073768636665
1150.0,1.0951515303942891,0.35256753083656406,301.36816995099997,1884.167629747"""

def plot_results(data_string):
    """
    Reads simulation results from a string and plots the errors vs system size.
    """
    try:
        # Load the dataset using io.StringIO to treat the string like a file
        df = pd.read_csv(io.StringIO(data_string))
        
        # Create the figure and axis
        plt.figure(figsize=(10, 6))
        
        # Plot Avg_Naive_Error vs SystemSize
        plt.plot(
            df['SystemSize_m'], 
            df['Avg_Naive_Error'], 
            marker='o', 
            linestyle='-', 
            linewidth=2,
            color='tab:blue',
            label='Average Error Global Reconstruction'
        )
        
        # Plot Avg_Local_Error vs SystemSize
        plt.plot(
            df['SystemSize_m'], 
            df['Avg_Local_Error'], 
            marker='s', 
            linestyle='--', 
            linewidth=2,
            color='tab:orange',
            label='Average Error Local Reconstruction'
        )
        
        # Add titles and labels
        plt.title('Reconstruction Errors vs Number of Modes for $1D$, ill-conditioned Hamiltonian', fontsize=16)
        plt.xlabel('Number of modes (m)', fontsize=12)
        plt.ylabel('Average ReconstructionError', fontsize=12)
        
        # Add a grid for easier reading
        plt.grid(True, linestyle=':', alpha=0.7)
        
        # Add a legend
        plt.legend()
        plt.ylim(0, 3)
        # Adjust layout
        plt.tight_layout()
        
        # Show the plot
        print("Displaying plot...")
        output_filename = 'simulation_errors_plot_ill.png'
        plt.savefig(output_filename, dpi=300)
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    plot_results(csv_data)
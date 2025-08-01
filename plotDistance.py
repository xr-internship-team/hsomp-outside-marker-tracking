import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Define the name of your log file
log_file = 'debug_frame_log.csv'

try:
    # Read the CSV file, parsing the 'Time' column as dates
    df = pd.read_csv(log_file, parse_dates=['Time'])

    if df.empty:
        print(f"The file '{log_file}' is empty. No data to plot.")
    else:
        print("Successfully loaded data. Generating plot...")

        # Calculate elapsed time in seconds
        start_time = df['Time'].iloc[0]
        df['ElapsedSeconds'] = (df['Time'] - start_time).dt.total_seconds()


        # Custom function to format the x-axis labels to M:SS
        def format_seconds_to_mmss(seconds, pos):
            minutes = int(seconds // 60)
            remaining_seconds = int(seconds % 60)
            return f'{minutes}:{remaining_seconds:02d}'


        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot ElapsedSeconds vs. Distance
        ax.plot(df['ElapsedSeconds'], df['Distance'], marker='.', linestyle='-', markersize=4)

        # --- NEW: Set major ticks to appear every 30 seconds ---
        ax.xaxis.set_major_locator(mticker.MultipleLocator(30))

        # Apply the custom formatter to the x-axis
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(format_seconds_to_mmss))

        # Add titles and labels for clarity
        ax.set_title('Tag Distance Over Time')
        ax.set_xlabel('Elapsed Time (Minutes:Seconds)')
        ax.set_ylabel('Distance (meters)')
        ax.grid(True)

        plt.tight_layout()
        plt.show()

except FileNotFoundError:
    print(f"Error: The file '{log_file}' was not found.")
except (KeyError, IndexError):
    print(f"Error: The file '{log_file}' is likely empty or missing the required 'Time'/'Distance' columns.")
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker  # We need this for custom tick formatting

# Define the name of your log file
log_file = 'debug_frame_log.csv'

try:
    # Read the CSV file, parsing the 'Time' column as dates
    df = pd.read_csv(log_file, parse_dates=['Time'])

    # Ensure the DataFrame is not empty
    if df.empty:
        print(f"The file '{log_file}' is empty. No data to plot.")
    else:
        print("Successfully loaded data. Generating plot...")

        # --- NEW: Calculate elapsed time ---
        # Get the first timestamp as the starting point
        start_time = df['Time'].iloc[0]
        # Subtract the start time from all timestamps to get the duration in seconds
        df['ElapsedSeconds'] = (df['Time'] - start_time).dt.total_seconds()


        # --- NEW: Custom function to format the x-axis labels ---
        def format_seconds_to_mmss(seconds, pos):
            """Converts seconds into a M:SS format string."""
            minutes = int(seconds // 60)
            remaining_seconds = int(seconds % 60)
            # Formats the string like "2:05"
            return f'{minutes}:{remaining_seconds:02d}'


        # Create the plot using subplots for more control
        fig, ax = plt.subplots(figsize=(12, 6))

        # --- MODIFIED: Plot ElapsedSeconds vs. Distance ---
        ax.plot(df['ElapsedSeconds'], df['Distance'], marker='.', linestyle='-', markersize=4)

        # --- NEW: Apply the custom formatter to the x-axis ---
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(format_seconds_to_mmss))

        # Add titles and labels for clarity
        ax.set_title('Tag Distance Over Time')
        ax.set_xlabel('Elapsed Time (Minutes:Seconds)')
        ax.set_ylabel('Distance (meters)')
        ax.grid(True)

        # Adjust plot to prevent labels from overlapping
        plt.tight_layout()

        # Show the plot
        plt.show()

except FileNotFoundError:
    print(f"Error: The file '{log_file}' was not found.")
except (KeyError, IndexError):
    print(f"Error: The file '{log_file}' is likely empty or missing the required 'Time'/'Distance' columns.")

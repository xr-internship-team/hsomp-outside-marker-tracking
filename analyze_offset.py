import csv
import numpy as np

def calculate_y_offset(csv_file='logfile.csv'):
    """
    Calculate the Y-axis offset between tripod-mounted marker (collection_number=1) 
    and actual HoloLens center (collection_number=2)
    """
    
    # Read the CSV file
    tripod_y_values = []
    hololens_y_values = []
    
    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)  # Skip header
        
        # Find the indices of required columns
        ty_index = header.index('Ty')
        collection_index = header.index('collection_number')
        
        for row in csv_reader:
            try:
                ty_value = float(row[ty_index])
                collection_num = int(row[collection_index])
                
                if collection_num == 1:  # Tripod scenario
                    tripod_y_values.append(ty_value)
                elif collection_num == 2:  # HoloLens center scenario
                    hololens_y_values.append(ty_value)
            except (ValueError, IndexError):
                continue  # Skip invalid rows
    
    if len(tripod_y_values) == 0 or len(hololens_y_values) == 0:
        print("Error: Missing data for one or both collections")
        return None
    
    # Calculate average Y positions
    avg_y_tripod = np.mean(tripod_y_values)
    avg_y_hololens = np.mean(hololens_y_values)
    
    # Calculate the offset (difference)
    y_offset = avg_y_hololens - avg_y_tripod
    
    print("=== Y-Axis Offset Analysis ===")
    print(f"Tripod Scenario (Collection 1) - Count: {len(tripod_y_values)} samples")
    print(f"  Average Y: {avg_y_tripod:.6f}")
    print(f"  Y Range: {np.min(tripod_y_values):.6f} to {np.max(tripod_y_values):.6f}")
    print(f"  Y Std Dev: {np.std(tripod_y_values):.6f}")
    
    print(f"\nHoloLens Center Scenario (Collection 2) - Count: {len(hololens_y_values)} samples")
    print(f"  Average Y: {avg_y_hololens:.6f}")
    print(f"  Y Range: {np.min(hololens_y_values):.6f} to {np.max(hololens_y_values):.6f}")
    print(f"  Y Std Dev: {np.std(hololens_y_values):.6f}")
    
    print(f"\n=== Calculated Offset ===")
    print(f"Y Offset (HoloLens - Tripod): {y_offset:.6f}")
    print(f"This means the HoloLens center is {y_offset:.6f}m {'above' if y_offset > 0 else 'below'} the tripod marker")
    
    return y_offset

if __name__ == "__main__":
    offset = calculate_y_offset()
    if offset is not None:
        print(f"\nTo apply in markerTracking.py:")
        print(f"Add {offset:.6f} to the Y coordinate before sending to Unity")
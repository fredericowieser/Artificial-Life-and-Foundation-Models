import pandas as pd
import sys
import os
import math

def append_mean_std_to_csv(csv_path):
    # Check that the CSV file exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    

    df = pd.read_csv(csv_path)

    new_row = {}
    for col in df.columns:
        if col.lower() == "initial prompt":
            new_row[col] = "MEAN"
        elif pd.api.types.is_numeric_dtype(df[col]):
            mean_val = df[col].mean()
            std_val = df[col].std()
            n = len(df[col])
            # Compute the standard error: std / sqrt(n)
            se_val = std_val / math.sqrt(n) if n > 0 else 0
            # Format using three decimal places
            new_row[col] = f"{mean_val:.3f} $\\pm$ {se_val:.3f}"
        else:
            new_row[col] = ""
    
    # Append the new row to the DataFrame
    df_appended = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
    # Overwrite the CSV with the updated DataFrame
    df_appended.to_csv(csv_path, index=False)
    print(f"Updated CSV saved to {csv_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_csv>")
        sys.exit(1)
    
    csv_file_path = sys.argv[1]
    append_mean_std_to_csv(csv_file_path)

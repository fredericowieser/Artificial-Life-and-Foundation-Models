import pandas as pd

# Load the CSV file
df = pd.read_csv('data/non_temporal_large_run/high_level_summary.csv')

# Identify all columns containing "oe_score" (case insensitive)
oe_columns = [col for col in df.columns if 'oe_score' in col.lower()]

# Apply the transformation (1 - value) to each oe_score column except standard deviation columns.
for col in oe_columns:
    if 'std' not in col.lower():
        df[col] = 1 - df[col]
    else:
        print(f"Skipping conversion for standard deviation column: {col}")

# Calculate the difference between the final and initial oe_score.
# Adjust these column names if your CSV uses different names.
if 'initial_prompt_oe_score' in df.columns and 'final_prompt_oe_score' in df.columns:
    df['oe_prompt_score_diff'] = df['final_prompt_oe_score'] - df['initial_prompt_oe_score']
else:
    print("Warning: 'initial_oe_score' and/or 'final_oe_score' columns not found.")

# Drop the 'count_rows' column if it exists
if 'count_rows' in df.columns:
    df = df.drop(columns=['count_rows'])

# Round all numerical columns to 6 decimal places
df = df.round(6)

# Save the modified DataFrame to a new CSV file
df.to_csv('data/non_temporal_large_run/high_level_summary_modified.csv', index=False)

print("Transformation complete. Modified file saved as 'high_level_summary_modified.csv'")

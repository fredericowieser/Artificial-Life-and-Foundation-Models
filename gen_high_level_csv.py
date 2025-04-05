import os
import csv
import statistics

def generate_high_level_summary(root_dir,
                                summary_filename="summary.csv",
                                high_level_filename="high_level_summary.csv"):
    """
    Reads 'summary.csv' (containing multiple rows per subfolder) from `root_dir`,
    then aggregates them so that each subfolder becomes exactly one row in
    'high_level_summary.csv'.

    Columns in the final CSV:
      - subfolder
      - count_rows
      - image_oe_score_mean
      - image_oe_score_min
      - image_oe_score_max
      - image_oe_score_stdev
      - text_oe_score_mean
      - any config_<key> columns

    Parameters
    ----------
    root_dir : str
        Directory containing 'summary.csv'.
    summary_filename : str, optional
        Name of the summary CSV file. Defaults to "summary.csv".
    high_level_filename : str, optional
        Name of the output aggregated CSV file. Defaults to "high_level_summary.csv".

    Returns
    -------
    None
        Writes the aggregated CSV to `root_dir`.
    """

    summary_path = os.path.join(root_dir, summary_filename)
    out_path = os.path.join(root_dir, high_level_filename)

    if not os.path.isfile(summary_path):
        print(f"Error: '{summary_filename}' not found in {root_dir}")
        return

    # 1) Read the summary.csv into a list of dict rows
    with open(summary_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print(f"No rows found in {summary_filename}, nothing to aggregate.")
        return

    # Identify config columns (those starting with 'config_') plus known columns.
    all_columns = set(rows[0].keys())
    config_columns = [c for c in all_columns if c.startswith("config_")]

    # We'll group rows by subfolder
    # groups[subfolder] = {
    #   "rows": [row, row, ...],
    #   "image_scores": [],
    #   "text_scores": [],
    #   "config_values": {}
    # }
    groups = {}

    for row in rows:
        subfolder = row.get("subfolder", "")
        if subfolder not in groups:
            groups[subfolder] = {
                "rows": [],
                "image_scores": [],
                "text_scores": [],
                "config_values": {}
            }
        groups[subfolder]["rows"].append(row)

    # 2) Collect scores and config from each subfolder
    for subfolder, data_dict in groups.items():
        for row in data_dict["rows"]:
            # Parse image_oe_score if present
            img_str = row.get("image_oe_score", "")
            if img_str != "":
                try:
                    data_dict["image_scores"].append(float(img_str))
                except ValueError:
                    pass

            # Parse text_oe_score if present
            txt_str = row.get("text_oe_score", "")
            if txt_str != "":
                try:
                    data_dict["text_scores"].append(float(txt_str))
                except ValueError:
                    pass

            # Collect config columns (assuming consistent across the subfolder)
            for ccol in config_columns:
                val = row.get(ccol, "")
                # If there's a conflict, one approach is to keep the first non-empty
                if val != "":
                    data_dict["config_values"][ccol] = val

    # 3) Compute aggregated stats for each subfolder
    aggregated_rows = []
    for subfolder, data_dict in groups.items():
        image_vals = data_dict["image_scores"]
        text_vals = data_dict["text_scores"]
        config_vals = data_dict["config_values"]
        sub_rows = data_dict["rows"]

        row_count = len(sub_rows)

        # Summaries for image_oe_score
        if image_vals:
            img_mean = statistics.mean(image_vals)
            img_min = min(image_vals)
            img_max = max(image_vals)
            # stdev uses sample-based if you want sample STDEV. If you prefer population:
            #   stdev = statistics.pstdev(image_vals)
            if len(image_vals) > 1:
                img_stdev = statistics.stdev(image_vals)
            else:
                img_stdev = 0.0
        else:
            img_mean = img_min = img_max = img_stdev = None

        # Summaries for text_oe_score (likely all the same, but we average if there's variation)
        if text_vals:
            text_oe_score_mean = statistics.mean(text_vals)
        else:
            text_oe_score_mean = None

        # Build the single aggregated row for this subfolder
        aggregated_row = {
            "subfolder": subfolder,
            "count_rows": row_count,
            "image_oe_score_mean": img_mean,
            "image_oe_score_min": img_min,
            "image_oe_score_max": img_max,
            "image_oe_score_stdev": img_stdev,
            "text_oe_score_mean": text_oe_score_mean
        }

        # Include the config columns we found
        for ccol in config_columns:
            aggregated_row[ccol] = config_vals.get(ccol, "")

        aggregated_rows.append(aggregated_row)

    # 4) Write out 'high_level_summary.csv'
    # We'll define our base columns in a stable order, then add config columns
    base_cols = [
        "subfolder",
        "count_rows",
        "image_oe_score_mean",
        "image_oe_score_min",
        "image_oe_score_max",
        "image_oe_score_stdev",
        "text_oe_score_mean"
    ]
    config_cols_sorted = sorted(config_columns)
    final_cols = base_cols + config_cols_sorted

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=final_cols)
        writer.writeheader()

        for row_dict in aggregated_rows:
            # Ensure every column is present
            for col in final_cols:
                if col not in row_dict:
                    row_dict[col] = None
            writer.writerow(row_dict)

    print(f"✅ high_level_summary.csv written to: {out_path}")

if __name__ == "__main__":
    # Example usage:
    ROOT_DIR = "data/temporal_large_run"
    generate_high_level_summary(ROOT_DIR,
                                summary_filename="summary.csv",
                                high_level_filename="high_level_summary.csv")

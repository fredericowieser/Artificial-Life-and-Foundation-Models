import os
import json
import csv

def parse_config(config_path):
    """
    Reads a key:value config file (e.g., config.txt) into a dictionary.
    Each line must contain 'key: value'.
    Ignores lines without a colon or empty lines.
    """
    config = {}
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            for line in f:
                if ":" in line:
                    key, val = line.strip().split(":", 1)
                    config[key.strip()] = val.strip()
    return config

def load_prompts(prompts_path):
    """
    Loads prompts from a file where each non-empty line is one prompt.
    Returns a list of strings.
    """
    if not os.path.isfile(prompts_path):
        return []
    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts

def load_json(json_path):
    """
    Loads a JSON file if it exists. Returns None if not found.
    """
    if not os.path.isfile(json_path):
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def collect_subfolder_rows(subdir_path):
    """
    For a given subfolder, collects:
      - config entries from config.txt,
      - all prompts from prompts.txt,
      - image-based OE scores from oe_scores.json,
      - a single text-based OE score from oe_text_scores.json.

    Returns a list of dictionaries, where each dictionary is a row for the CSV.
    One row per iteration/prompt in the subfolder.
    """

    # --- FILE PATHS
    config_path     = os.path.join(subdir_path, "config.txt")
    prompts_path    = os.path.join(subdir_path, "prompts.txt")
    oe_image_json   = os.path.join(subdir_path, "oe_scores.json")      # image-based
    oe_text_json    = os.path.join(subdir_path, "oe_text_scores.json") # text-based

    # --- PARSE THE CONFIG
    config_dict = parse_config(config_path)

    # --- LOAD PROMPTS
    prompts = load_prompts(prompts_path)

    # --- LOAD IMAGE-BASED OE SCORES
    # Format from 'oe_scores.json' typically: { "prompt_text": float, ... }
    oe_img_data = load_json(oe_image_json) or {}

    # --- LOAD TEXT-BASED OE SCORE
    # Format from 'oe_text_scores.json' typically: { "text_oe_score": float, "prompts": [...], ... }
    oe_txt_data = load_json(oe_text_json) or {}
    text_oe_score = oe_txt_data.get("text_oe_score", "")

    # Create one row per prompt/iteration
    rows = []

    for i, prompt_text in enumerate(prompts):
        row_data = {}

        # Tag subfolder name
        row_data["subfolder"] = os.path.basename(subdir_path)

        # Add iteration index and prompt text
        row_data["iteration_index"] = i
        row_data["prompt_text"] = prompt_text

        # Add the matching image-based OE score if it exists
        # (or empty string if not found)
        image_oe_score = oe_img_data.get(prompt_text, "")
        row_data["image_oe_score"] = image_oe_score

        # Add the single text-based OE score for the entire subfolder
        row_data["text_oe_score"] = text_oe_score

        # Flatten config into "config_key": value
        for k, v in config_dict.items():
            row_data[f"config_{k}"] = v

        rows.append(row_data)

    return rows

def generate_summary_csv(root_dir, csv_filename="summary.csv"):
    """
    Iterates over each subdirectory of root_dir, collects config/prompt/OE data,
    and writes them to one CSV in root_dir.

    Each subfolder can produce multiple rows, one row per prompt in that folder.
    Columns will include:
      - subfolder
      - iteration_index
      - prompt_text
      - image_oe_score
      - text_oe_score
      - config_<key> for each config key found
    """

    subfolders = [
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ]

    all_rows = []
    all_cols = set(["subfolder", "iteration_index", "prompt_text", "image_oe_score", "text_oe_score"])

    for subfolder in subfolders:
        subdir_path = os.path.join(root_dir, subfolder)
        # Gather data from that subfolder
        rows = collect_subfolder_rows(subdir_path)

        # Keep track of columns
        for row in rows:
            all_cols.update(row.keys())

        all_rows.extend(rows)

    # Sort columns for consistent CSV output
    col_list = sorted(all_cols)

    # Write the CSV at the root_dir level
    csv_out_path = os.path.join(root_dir, csv_filename)
    with open(csv_out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=col_list)
        writer.writeheader()

        for row_dict in all_rows:
            # Ensure each column is present (fill with "")
            for col in col_list:
                if col not in row_dict:
                    row_dict[col] = ""
            writer.writerow(row_dict)

    print(f"✅ Summary CSV written to: {csv_out_path}")

if __name__ == "__main__":
    ROOT_DIR = "data/temporal_large_run"
    generate_summary_csv(ROOT_DIR, csv_filename="summary.csv")

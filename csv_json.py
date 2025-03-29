import csv
import json

def csv_to_tree(csv_path, json_path):
    # We'll group rows by the first prompt.
    tree = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            branch_idx = int(row["branch_index"])
            # Split the prompt chain on ";" – note: remove extra spaces.
            prompts = [p.strip() for p in row["prompt_chain"].split(";")]
            if len(prompts) < 2:
                continue  # We expect at least a root and one child.
            root_prompt = prompts[0]
            child_prompt = prompts[1]
            # If the root already exists, append the child.
            if root_prompt not in tree:
                tree[root_prompt] = {"prompt": root_prompt, "children": []}
            tree[root_prompt]["children"].append({
                "branch_index": branch_idx,
                "prompt": child_prompt
            })
    # Convert tree (a dict) to a list for JSON output.
    tree_list = list(tree.values())
    with open(json_path, "w", encoding="utf-8") as out_f:
        json.dump(tree_list, out_f, indent=2)
    print(f"Wrote tree JSON to {json_path}")

if __name__ == "__main__":
    csv_path = "/Users/baidn/Artificial-Life-and-Foundation-Models/graph_test/final_prompt_chains.csv"  # Path to your CSV file
    json_path = "/Users/baidn/Artificial-Life-and-Foundation-Models/graph_test/branches.json"           # Output JSON file
    csv_to_tree(csv_path, json_path)

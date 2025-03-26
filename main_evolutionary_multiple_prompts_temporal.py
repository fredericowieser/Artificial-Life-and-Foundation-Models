import os
import csv
import shutil
from main_evolutionary_prompts_temporal import main, parse_args
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

def run_multiple_initial_prompts():
    # Path to your CSV
    csv_path = "/workspace/Artificial-Life-and-Foundation-Models/resources/Generation_prompts.csv"
    
    # Parse the global arguments once (without specifying --prompts)
    base_args = parse_args()
    
    # Read CSV
    i=0
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)  # expects headers in first row
        for idx, row in enumerate(reader):
            init_prompt = row["prompt"].strip()
            
            # If you want to create a subfolder for each prompt:
            # e.g. 'results/prompt_0', 'results/prompt_1', etc.
            run_folder = os.path.join(
                base_args.save_dir or "results",
                f"prompt_{init_prompt}"
            )
            os.makedirs(run_folder, exist_ok=True)
            
            # Make a copy of args so we can override them
            run_args = base_args
            run_args.prompts = init_prompt  # The first (initial) prompt
            run_args.save_dir = run_folder  # Where we store results
            
       
            print(f"\n=== Running pipeline for initial prompt: {init_prompt} ===")
            print(f"Results will be saved in: {run_folder}")
            
      
            main(run_args)
            i+=1
            if i==10:
                break

if __name__ == "__main__":
    run_multiple_initial_prompts()

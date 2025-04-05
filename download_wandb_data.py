import os
import wandb

def download_wandb_data(entity: str, project: str, sweep_id: str):
    """
    For each run in the given W&B sweep:
      - Create a folder named after the run's prompt (run.config["prompts"]) or run.id if not present.
      - Scan all history steps (run.scan_history) to find '_type: video-file' entries.
      - Download each of those video files (e.g. GIF) to the local folder.
      - Append each video's caption to prompts.txt.
      - Save run.config in config.txt.
    """

    api = wandb.Api()
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")

    # Create a top-level directory named after the sweep if you like (optional):
    # os.makedirs(sweep_id, exist_ok=True)

    for run in sweep.runs:
        # -----------------------
        # 1) CREATE RUN FOLDER
        # -----------------------
        # Use the prompt string for folder naming (fallback to run.id)
        prompt_str = run.config.get("prompts", run.id)
        # Clean up folder name: replace spaces/slashes, etc.
        safe_prompt_name = prompt_str.replace(" ", "_").replace("/", "_")

        # If you worry about collisions, append run.id:
        # safe_prompt_name = f"{safe_prompt_name}_{run.id}"

        run_folder = os.path.join(sweep_id, safe_prompt_name)
        os.makedirs(run_folder, exist_ok=True)

        print(f"\n--- Processing run: {run.id} ---")
        print(f"Folder: {run_folder}")

        # -----------------------
        # 2) SAVE CONFIG.TXT
        # -----------------------
        config_txt = os.path.join(run_folder, "config.txt")
        with open(config_txt, "w", encoding="utf-8") as cf:
            for k, v in run.config.items():
                cf.write(f"{k}: {v}\n")

        # -----------------------
        # 3) COLLECT VIDEO ENTRIES
        # -----------------------
        # We'll gather (file_path, caption) pairs across *all steps*.
        video_entries = []

        # Go through each step in the run's history
        for row_idx, row in enumerate(run.scan_history()):
            # row is a dict with keys like "iteration_4" -> { "_type": "video-file", ... }
            for key, val in row.items():
                if isinstance(val, dict) and val.get("_type") == "video-file":
                    media_path = val.get("path", None)
                    caption = val.get("caption", "No caption")
                    if media_path:
                        video_entries.append((media_path, caption))

        # If you also want to parse run.summary, you can do so here:
        # (sometimes the final step is truncated, but if your logging pushes to summary, it might help)
        # for key, val in dict(run.summary).items():
        #     if isinstance(val, dict) and val.get("_type") == "video-file":
        #         media_path = val.get("path", None)
        #         caption = val.get("caption", "No caption")
        #         if media_path and (media_path, caption) not in video_entries:
        #             video_entries.append((media_path, caption))

        # -----------------------
        # 4) DOWNLOAD FILES & WRITE PROMPTS.TXT
        # -----------------------
        prompts_path = os.path.join(run_folder, "prompts.txt")
        downloaded_count = 0
        with open(prompts_path, "w", encoding="utf-8") as pf:
            # Retrieve list of run files up-front to help debug missing paths
            run_files = {f.name for f in run.files()}
            print(f"Found {len(run_files)} total files in run {run.id}. Checking for videos...")

            for media_path, caption in video_entries:
                # E.g. "media/videos/iteration_4_16003_8bac38969e7bd9247bc3.gif"
                if media_path in run_files:
                    # Attempt to download
                    try:
                        wb_file = run.file(media_path)
                        wb_file.download(root=run_folder, replace=True)
                        pf.write(f"{caption}\n")
                        downloaded_count += 1
                    except Exception as e:
                        print(f"  Could not download {media_path}: {e}")
                else:
                    # Not found in run.files() -> possibly in an artifact or not uploaded?
                    print(f"  WARNING: {media_path} not found among run.files().")

        print(f"Downloaded {downloaded_count} videos for run {run.id} (prompt: '{prompt_str}')")

    print("\nAll done!")


if __name__ == "__main__":
    # Replace with your actual W&B details:
    ENTITY = "ucl-asal"
    PROJECT = "alife-project"
    SWEEP_ID = "eo1uy4vj"

    download_wandb_data(ENTITY, PROJECT, SWEEP_ID)
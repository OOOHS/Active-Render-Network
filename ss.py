import os
import shutil

root = "/home/liushudong/ARN/outputs/rollouts"

folders = sorted([f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))])

keep_every = 10

for idx, folder in enumerate(folders):
    folder_path = os.path.join(root, folder)

    # keep only every 10th folder
    if (idx + 1) % keep_every != 0:
        print(f"Deleting: {folder_path}")
        shutil.rmtree(folder_path)
    else:
        print(f"KEEP: {folder_path}")

print("Done.")

from pathlib import Path
import torch

input_folder = Path("results_tampered")
output_folder = Path("tampered_latents")
output_folder.mkdir(parents=True, exist_ok=True)

for file in input_folder.iterdir():
    print(file.name)  # or file.path if you need full path
    # tensor = torch.load(file)
    output_file = output_folder / (file.stem + ".txt")

    # Write the result to the text file
    with open(output_file, "w") as f:
        f.write(f"File: {file.name}\n")

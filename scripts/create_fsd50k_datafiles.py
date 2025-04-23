import json
import os

# Base directories using environment variable
user = os.environ.get('USER')
base_dir = f"/scratch/{user}"

# Function to process FSD50K captions
def process_fsd50k_json(input_path, split_name, output_path):
    # Read input JSON
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # Update wav paths to use relative paths
    for item in data["data"]:
        item["wav"] = f"processed_data_files/fsd50k/{split_name}/{item['wav']}"
    
    # Write to output JSON
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"Created {output_path} with {len(data['data'])} entries")

# Process FSD50K captions
captions_dir = f"{base_dir}/fsd50k_captions"
output_dir = f"{base_dir}/LASS/datafiles"
os.makedirs(output_dir, exist_ok=True)

# Dev set
process_fsd50k_json(
    f"{captions_dir}/fsd50k_dev_auto_caption.json",
    "dev_audio",
    f"{output_dir}/fsd50k_dev.json"
)

# Eval set
process_fsd50k_json(
    f"{captions_dir}/fsd50k_eval_auto_caption.json",
    "eval_audio",
    f"{output_dir}/fsd50k_eval.json"
)

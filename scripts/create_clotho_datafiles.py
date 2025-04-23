import pandas as pd
import json
import os

user = os.environ.get('USER')
base_dir = f"/scratch/{user}"
processed_audio_dir = f"{base_dir}/LASS/processed_data_files/clotho"

# Function to convert CSV to JSON
def create_clotho_json(csv_path, split_name, output_path):
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Create data list
    data = []
    
    for _, row in df.iterrows():
        filename = row['file_name']
        # Use relative path from LASS directory for better portability
        wav_path = f"processed_data_files/clotho/{split_name}/{filename}"
        
        # Add an entry for each caption (1-5)
        for i in range(1, 6):
            caption_key = f'caption_{i}'
            if caption_key in row and pd.notna(row[caption_key]):
                data.append({
                    "wav": wav_path,
                    "caption": row[caption_key]
                })
    
    # Write to JSON file
    with open(output_path, 'w') as f:
        json.dump({"data": data}, f, indent=4)
    
    print(f"Created {output_path} with {len(data)} entries")

# Create JSON files for each split
csv_dir = f"{base_dir}/clotho_dataset"
output_dir = f"{base_dir}/LASS/datafiles"
os.makedirs(output_dir, exist_ok=True)

# Development set
create_clotho_json(
    f"{csv_dir}/clotho_captions_development.csv",
    "development",
    f"{output_dir}/clotho_development.json"
)

# Validation set
create_clotho_json(
    f"{csv_dir}/clotho_captions_validation.csv",
    "validation",
    f"{output_dir}/clotho_validation.json"
)

# Evaluation set
create_clotho_json(
    f"{csv_dir}/clotho_captions_evaluation.csv",
    "evaluation",
    f"{output_dir}/clotho_evaluation.json"
)

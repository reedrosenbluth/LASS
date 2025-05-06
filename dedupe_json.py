import json
import argparse
import pathlib
from collections import OrderedDict

def find_audio_path_key(entry):
    """Attempts to find the key used for the audio path in a JSON entry."""
    # Prioritize 'wav' based on previous examples, then try others
    common_keys = ['wav', 'audio_path', 'audiopath', 'audio', 'file', 'filename', 'path']
    for key in common_keys:
        if key in entry:
            return key
    return None

def deduplicate_json_files(input_files, output_file):
    """
    Reads a list of JSON files, de-duplicates entries based on audio path,
    and writes the unique entries to a new JSON file.

    Keeps the *first* encountered entry for each unique audio path.

    Args:
        input_files (list): List of input JSON file paths (str or Path).
        output_file (str or Path): Path to the output de-duplicated JSON file.
    """
    print(f"Processing input files: {', '.join(map(str, input_files))}")
    unique_entries = OrderedDict() # Use OrderedDict to keep the first entry encountered
    total_entries_processed = 0
    files_processed_count = 0
    files_failed_count = 0
    entries_missing_key_count = 0
    audio_key_name = None
    checked_first_entry = False

    for file_path_str in input_files:
        file_path = pathlib.Path(file_path_str)
        if not file_path.is_file():
            print(f"Warning: Input file not found: {file_path}. Skipping.")
            files_failed_count += 1
            continue

        try:
            with open(file_path, 'r') as f:
                loaded_json = json.load(f)
            files_processed_count += 1

            # Extract the list of entries
            entries_list = None
            if isinstance(loaded_json, list):
                entries_list = loaded_json
            elif isinstance(loaded_json, dict) and 'data' in loaded_json and isinstance(loaded_json['data'], list):
                entries_list = loaded_json['data']
            else:
                print(f"Warning: Expected a list or a dict with a 'data' key containing a list in {file_path}. Skipping file.")
                files_failed_count += 1
                continue

            if not entries_list:
                print(f"Info: File {file_path} has an empty 'data' list or is an empty list.")
                continue

            # Determine audio path key from the first file with entries
            if not checked_first_entry and entries_list:
                for entry in entries_list:
                    if isinstance(entry, dict):
                        detected_key = find_audio_path_key(entry)
                        if detected_key:
                            audio_key_name = detected_key
                            print(f"Detected audio path key: '{audio_key_name}' (from {file_path.name})")
                            checked_first_entry = True
                            break
                if not checked_first_entry:
                    print(f"Warning: Could not auto-detect audio path key in {file_path.name}. Will try common keys for each entry.")

            # Process entries and store unique ones
            for entry in entries_list:
                total_entries_processed += 1
                if not isinstance(entry, dict):
                    print(f"Warning: Skipping non-dictionary item in {file_path}: {entry}")
                    continue

                current_audio_key = audio_key_name
                if not current_audio_key:
                    current_audio_key = find_audio_path_key(entry)

                if current_audio_key and current_audio_key in entry:
                    audio_path = entry[current_audio_key]
                    # If this audio path hasn't been seen yet, add the entry
                    if audio_path not in unique_entries:
                        unique_entries[audio_path] = entry
                else:
                    key_tried = audio_key_name or 'auto-detected key'
                    print(f"Warning: Could not find audio path key ({key_tried}) in entry in {file_path.name}: {list(entry.keys())}")
                    entries_missing_key_count += 1

        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {file_path}. Skipping.")
            files_failed_count += 1
        except Exception as e:
            print(f"Error processing file {file_path}: {e}. Skipping.")
            files_failed_count += 1

    # --- Prepare and Write Output --- 
    final_unique_list = list(unique_entries.values())
    output_data = {"data": final_unique_list}
    num_unique = len(final_unique_list)

    print(f"\nFinished processing input files.")
    print(f"  Total entries processed: {total_entries_processed}")
    print(f"  Unique audio paths found: {num_unique}")
    if entries_missing_key_count > 0:
        print(f"  Entries skipped due to missing audio key: {entries_missing_key_count}")
    if files_failed_count > 0:
        print(f"  Input files skipped due to errors/not found: {files_failed_count}")

    try:
        output_path = pathlib.Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=4) # Use indent for readability
        print(f"Successfully wrote {num_unique} unique entries to: {output_path}")
    except Exception as e:
        print(f"Error writing output file {output_file}: {e}")

    return num_unique

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="De-duplicates entries in JSON data files based on unique audio paths, keeping the first entry found."
    )
    parser.add_argument(
        '--input_files', 
        nargs='+', 
        required=True, 
        help='List of input JSON files (e.g., train_file1.json train_file2.json).'
    )
    parser.add_argument(
        '--output_file', 
        type=str, 
        required=True, 
        help='Path for the output de-duplicated JSON file (e.g., train_unique.json).'
    )

    args = parser.parse_args()

    deduplicate_json_files(args.input_files, args.output_file)

    print("\nDe-duplication process complete.") 
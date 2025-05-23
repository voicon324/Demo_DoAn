import os
import json
import numpy as np
import multiprocessing
from functools import partial
import time

# Configuration
IMAGE_DIR = r"F:\keyframes"
EMBEDDING_DIR = r"C:\Users\hokha\Downloads\_output_\embeddings"
OUTPUT_DATA_DIR = "data" # Directory for output JSON files
OUTPUT_JSON_FILENAME_BASE = "milvus_import_data" # Base name for output files
MAX_RECORDS_PER_FILE = 50000 # Max number of records per JSON file
# Number of processes to use (default to CPU count - 1 to leave one core free)
NUM_PROCESSES = max(1, multiprocessing.cpu_count() - 1)

def process_image_batch(image_batch):
    """Process a batch of images in parallel"""
    results = []
    
    for video_name, frame_filename in image_batch:
        image_path_in_json = os.path.join(IMAGE_DIR, video_name, frame_filename).replace("\\", "/")
        frame_id = os.path.splitext(frame_filename)[0]  # Remove extension to get frame_id
        
        # Construct embedding filename as video_name_frame_id.npy
        embedding_filename = f"{video_name}_{frame_id}.npy"
        embedding_path = os.path.join(EMBEDDING_DIR, embedding_filename)
        
        if not os.path.exists(embedding_path):
            print(f"Warning: Embedding file not found for {frame_filename} at {embedding_path}. Skipping.")
            continue
            
        try:
            # Load the embedding from the .npy file
            embedding_vector = np.load(embedding_path)
            #normalize the embedding vector
            embedding_vector = embedding_vector / np.linalg.norm(embedding_vector)
            embedding_list = embedding_vector.astype(np.float32).tolist()
            
            results.append({
                "image_path": image_path_in_json, 
                "embedding": embedding_list
            })
        except Exception as e:
            print(f"Error processing embedding for {image_path_in_json}: {e}")
            
    return results

def load_existing_embeddings_and_json():
    start_time = time.time()
    data_for_milvus = []
    
    os.makedirs(OUTPUT_DATA_DIR, exist_ok=True) # Ensure output directory exists

    if not os.path.exists(IMAGE_DIR):
        print(f"Error: Image directory \'{IMAGE_DIR}\' not found.")
        return

    if not os.path.exists(EMBEDDING_DIR):
        print(f"Error: Embedding directory \'{EMBEDDING_DIR}\' not found.")
        return

    # Walk through the nested directory structure
    image_files = []
    video_folders = [f for f in os.listdir(IMAGE_DIR) if os.path.isdir(os.path.join(IMAGE_DIR, f))]
    
    if not video_folders:
        print(f"No video folders found in \'{IMAGE_DIR}\'.")
        return
    
    for video_folder in video_folders:
        folder_path = os.path.join(IMAGE_DIR, video_folder)
        frames = [
            (video_folder, f) for f in os.listdir(folder_path) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        image_files.extend(frames)

    if not image_files:
        print(f"No image files found in video folders within \'{IMAGE_DIR}\'.")
        return

    total_images = len(image_files)
    print(f"Found {total_images} images to process across {len(video_folders)} video folders.")
    print(f"Using {NUM_PROCESSES} processes for parallel processing")
    
    # Split the work among processes
    batch_size = (total_images + NUM_PROCESSES - 1) // NUM_PROCESSES
    batches = [image_files[i:i+batch_size] for i in range(0, total_images, batch_size)]
    
    # Process batches in parallel
    with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
        # Use imap to get results as they complete
        batch_results = list(pool.imap_unordered(process_image_batch, batches))
        
        # Flatten results
        for batch in batch_results:
            data_for_milvus.extend(batch)

    if not data_for_milvus:
        print("No data was generated for Milvus import.")
        return

    print(f"Successfully processed {len(data_for_milvus)} embeddings")

    # Split data_for_milvus into chunks and write to multiple files
    num_files = (len(data_for_milvus) + MAX_RECORDS_PER_FILE - 1) // MAX_RECORDS_PER_FILE
    file_extension = ".json"

    for i in range(num_files):
        start_index = i * MAX_RECORDS_PER_FILE
        end_index = start_index + MAX_RECORDS_PER_FILE
        chunk_data = data_for_milvus[start_index:end_index]
        
        if num_files > 1:
            current_output_filename = f"{OUTPUT_JSON_FILENAME_BASE}_part_{i+1}{file_extension}"
        else:
            current_output_filename = f"{OUTPUT_JSON_FILENAME_BASE}{file_extension}"
        
        output_file_path = os.path.join(OUTPUT_DATA_DIR, current_output_filename)

        try:
            with open(output_file_path, 'w') as f:
                json.dump(chunk_data, f, indent=4)
            print(f"Successfully generated Milvus import data at '{output_file_path}'")
            print(f"Total records in this file: {len(chunk_data)}")
        except Exception as e:
            print(f"Error writing JSON to file {output_file_path}: {e}")
    
    elapsed_time = time.time() - start_time
    print(f"Total processing time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    # Protect the entry point when using multiprocessing
    load_existing_embeddings_and_json()

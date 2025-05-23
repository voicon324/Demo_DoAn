import os
import numpy as np
from tqdm import tqdm
import multiprocessing

# Configuration (should match app.py)
IMAGE_DIR = r"F:\keyframes"  # Directory containing images
EMBEDDING_DIR = r"C:\Users\hokha\Downloads\_output_\embeddings"  # Directory containing .npy embeddings
OUTPUT_MANUAL_DATA_DIR = "manual_data"  # Directory for output NPZ files
OUTPUT_FILENAME_BASE = "manual_search_cache"  # Base name for output files
OUTPUT_FILENAME_EXT = ".npz"
MAX_RECORDS_PER_FILE = 50000  # Max number of records per .npz file
NUM_PROCESSES = max(1, multiprocessing.cpu_count() - 1)  # Use all CPUs except one

def process_batch(batch_data, file_index):
    """Process a batch of embedding files and create an NPZ file"""
    batch_embeddings = []
    batch_image_paths = []
    
    for emb_file in batch_data:
        embedding_path = os.path.join(EMBEDDING_DIR, emb_file)
        try:
            embedding = np.load(embedding_path)
        except Exception as e:
            print(f"Error loading embedding from {emb_file}: {e}. Skipping.")
            continue

        # Try to find the corresponding image file
        original_filename_stem = ""
        if emb_file.lower().endswith(('.jpeg.npy', '.png.npy')):
            if emb_file.lower().endswith('.jpeg.npy'):
                original_filename_stem = emb_file[:-len('.jpeg.npy')]
            elif emb_file.lower().endswith('.png.npy'):
                original_filename_stem = emb_file[:-len('.png.npy')]
        elif emb_file.lower().endswith('.npy'):
            original_filename_stem = emb_file[:-len('.npy')]
        else:
            print(f"Unrecognized embedding file pattern: {emb_file}. Skipping.")
            continue
        
        found_image_path = None
        possible_extensions = ['.jpg', '.jpeg', '.png']
        pivot = original_filename_stem.rfind('_')
        video_name = original_filename_stem[:pivot] if pivot != -1 else original_filename_stem
        frame_id = original_filename_stem[pivot + 1:] if pivot != -1 else original_filename_stem
        for img_ext in possible_extensions:
            potential_path = os.path.join(IMAGE_DIR, video_name, frame_id + img_ext)
            if os.path.exists(potential_path):
                found_image_path = potential_path
                break
        
        if not found_image_path and os.path.exists(os.path.join(IMAGE_DIR, original_filename_stem)):
            found_image_path = os.path.join(IMAGE_DIR, original_filename_stem)

        if found_image_path:
            batch_embeddings.append(embedding)
            batch_image_paths.append(os.path.normpath(found_image_path))

    # Save the batch if it has any items
    if batch_embeddings:
        output_file_name = f"{OUTPUT_FILENAME_BASE}_part_{file_index}{OUTPUT_FILENAME_EXT}"
        output_file_path = os.path.join(OUTPUT_MANUAL_DATA_DIR, output_file_name)
        try:
            np.savez_compressed(output_file_path, 
                               embeddings=np.array(batch_embeddings), 
                               image_paths=np.array(batch_image_paths))
            return len(batch_embeddings), 1
        except Exception as e:
            print(f"Error saving batch to {output_file_path}: {e}")
    
    return 0, 0

def process_func(args):
    """Wrapper to unpack arguments for multiprocessing"""
    return process_batch(args[0], args[1])

def main():
    if not os.path.exists(EMBEDDING_DIR):
        print(f"Error: Embedding directory '{EMBEDDING_DIR}' not found.")
        return

    if not os.path.exists(IMAGE_DIR):
        print(f"Error: Image directory '{IMAGE_DIR}' not found.")
        return

    os.makedirs(OUTPUT_MANUAL_DATA_DIR, exist_ok=True)

    print(f"Scanning embedding directory: {EMBEDDING_DIR}")
    all_npy_files = [f for f in os.listdir(EMBEDDING_DIR) if f.lower().endswith('.npy')]
    # Exclude embedding files ending with image extensions like .jpg.npy, .jpeg.npy, .png.npy
    embedding_files = [f for f in all_npy_files if not f.lower().endswith(('.jpg.npy', '.jpeg.npy', '.png.npy'))]

    if not embedding_files:
        print(f"No .npy embedding files (excluding those ending with .jpg.npy, .jpeg.npy, .png.npy) found in '{EMBEDDING_DIR}'.")
        return

    print(f"Found {len(embedding_files)} potential embedding files to process (excluding .jpg.npy, .jpeg.npy, .png.npy files).")
    
    # Split embedding files into batches
    num_batches = min(NUM_PROCESSES, (len(embedding_files) + MAX_RECORDS_PER_FILE - 1) // MAX_RECORDS_PER_FILE)
    batch_size = (len(embedding_files) + num_batches - 1) // num_batches
    batches = [embedding_files[i:i+batch_size] for i in range(0, len(embedding_files), batch_size)]
    
    print(f"Processing in parallel using {num_batches} batches with up to {NUM_PROCESSES} processes.")
    
    # Process batches in parallel
    with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
        results = list(tqdm(
            pool.imap(
                process_func, 
                [(batch, i+1) for i, batch in enumerate(batches)]
            ),
            total=len(batches),
            desc="Processing batches"
        ))
    
    # Aggregate results
    total_records_processed = sum(result[0] for result in results)
    files_saved_count = sum(result[1] for result in results)
    
    if total_records_processed == 0:
        print("No valid embeddings and image paths were processed to save.")
    else:
        print(f"Processing complete. Total records processed: {total_records_processed}. Total files saved: {files_saved_count}.")

if __name__ == "__main__":
    main()

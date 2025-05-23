\
import os
import json
from pymilvus import connections, Collection, FieldSchema, DataType, CollectionSchema, utility
from tqdm import tqdm

# Milvus connection details (local Docker Compose)
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "image_similarity_local"  # Or your desired collection name
DIMENSION = 512  # Dimension of CLIP embeddings (must match your model)
ID_FIELD_NAME = "id"
IMAGE_PATH_FIELD_NAME = "image_path"
EMBEDDING_FIELD_NAME = "embedding" # Changed from INDEX_FIELD_NAME for clarity with JSON
BATCH_INSERT_SIZE = 1000  # Adjust as needed

# Data directory
INPUT_DATA_DIR = "data"

def create_milvus_collection_if_not_exists():
    if not utility.has_collection(COLLECTION_NAME):
        fields = [
            FieldSchema(name=ID_FIELD_NAME, dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name=IMAGE_PATH_FIELD_NAME, dtype=DataType.VARCHAR, max_length=4096), # Increased max_length
            FieldSchema(name=EMBEDDING_FIELD_NAME, dtype=DataType.FLOAT_VECTOR, dim=DIMENSION)
        ]
        schema = CollectionSchema(fields, description="Local image similarity search collection")
        collection = Collection(COLLECTION_NAME, schema=schema)
        index_params = {
            "metric_type": "IP",  # Inner Product, equivalent to Cosine for normalized vectors
            "index_type": "HNSW", # Better for search performance than AUTOINDEX
            "params": {
                "M": 16,           # Number of bidirectional links for each node
                "efConstruction": 200  # Higher value means higher index quality but slower build
            }
        }
        collection.create_index(EMBEDDING_FIELD_NAME, index_params)
        print(f"Collection '{COLLECTION_NAME}' created and index built.")
        return collection
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists.")
        return Collection(COLLECTION_NAME)

def main():
    print(f"Attempting to connect to Milvus at {MILVUS_HOST}:{MILVUS_PORT}...")
    try:
        connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
        print("Connected to Milvus successfully!")
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")
        return

    collection = create_milvus_collection_if_not_exists()
    collection.load() # Load collection for searching/inserting

    json_files = [f for f in os.listdir(INPUT_DATA_DIR) if f.startswith("milvus_import_data") and f.endswith(".json")]
    
    if not json_files:
        print(f"No JSON data files found in '{INPUT_DATA_DIR}' starting with 'milvus_import_data'.")
        return

    print(f"Found {len(json_files)} JSON files to import: {json_files}")
    total_inserted_count = 0

    for json_file in json_files:
        file_path = os.path.join(INPUT_DATA_DIR, json_file)
        print(f"Processing file: {file_path}...")
        try:
            with open(file_path, 'r') as f:
                data_to_import = json.load(f)
        except Exception as e:
            print(f"Error reading or parsing JSON file {file_path}: {e}")
            continue

        if not isinstance(data_to_import, list) or not all(isinstance(item, dict) for item in data_to_import):
            print(f"Data in {file_path} is not a list of dictionaries. Skipping.")
            continue
        
        # Ensure data has the correct fields: image_path and embedding
        # The JSON from generate_milvus_import_json.py should have these.
        
        num_records_in_file = len(data_to_import)
        print(f"Found {num_records_in_file} records in {json_file}.")

        for i in tqdm(range(0, num_records_in_file, BATCH_INSERT_SIZE), desc=f"Importing from {json_file}"):
            batch = data_to_import[i:i + BATCH_INSERT_SIZE]
            
            # Prepare batch for Milvus: list of lists or list of dicts
            # Milvus SDK expects entities in a specific format.
            # If your JSON is [{"image_path": "...", "embedding": [...]}, ...],
            # you might need to transform it if SDK expects separate lists for each field.
            # However, pymilvus insert often takes list of dicts directly.
            
            # Example: entities = [ [path1, path2, ...], [emb1, emb2, ...] ]
            # Or: entities = [ {"image_path": path1, "embedding": emb1}, ... ] -> This is what our JSON is.

            # Check if image_path already exists to avoid duplicates (optional, can slow down import)
            # For bulk import, usually, you'd clear the collection or ensure data is new.
            # For simplicity, this example doesn't do a pre-check for existing paths during bulk load.

            try:
                insert_result = collection.insert(batch)
                total_inserted_count += len(insert_result.primary_keys) # Or len(batch) if auto_id=False
                # collection.flush() # Flush after each batch or at the end
            except Exception as e:
                print(f"Error inserting batch into Milvus: {e}")
                print(f"Problematic batch (first item): {batch[0] if batch else 'empty batch'}")


    if total_inserted_count > 0:
        print("Flushing collection...")
        collection.flush()
        print(f"Successfully imported {total_inserted_count} records into '{COLLECTION_NAME}'.")
        print(f"Collection entity count: {collection.num_entities}")
    else:
        print("No new records were imported.")

if __name__ == "__main__":
    main()

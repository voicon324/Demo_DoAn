import streamlit as st
import time
from pymilvus import connections, Collection, FieldSchema, DataType, CollectionSchema, utility
import open_clip
import numpy as np
import torch
from PIL import Image
from io import BytesIO
import os
import shutil

# Milvus connection details (local)
MILVUS_HOST = os.getenv("MILVUS_HOST", "milvus-standalone") # Service name from docker-compose
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")

COLLECTION_NAME = "image_similarity_local" 
DIMENSION = 512  # Dimension of CLIP embeddings
INDEX_FIELD_NAME = "embedding"
ID_FIELD_NAME = "id"
IMAGE_PATH_FIELD_NAME = "image_path"
BATCH_INSERT_SIZE = 5000 # Batch size for Milvus inserts

# Local folder paths (within the Docker container)
IMAGE_DIR = "/app/local_images"
EMBEDDING_DIR = "/app/local_embeddings"
MANUAL_SEARCH_DATA_DIR = "/app/manual_data" # Directory for manual search cache
MANUAL_SEARCH_CACHE_FILE_BASE = os.path.join(MANUAL_SEARCH_DATA_DIR, "manual_search_cache") # Base name for cache files
MANUAL_SEARCH_CACHE_EXT = ".npz"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Create local directories if they don't exist (mainly for local execution, Docker volumes handle this in container)
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(EMBEDDING_DIR, exist_ok=True)
os.makedirs(MANUAL_SEARCH_DATA_DIR, exist_ok=True)

# Connect to Local Milvus
try:
    st.sidebar.info(f"Attempting to connect to Local Milvus at {MILVUS_HOST}:{MILVUS_PORT}...")
    connections.connect(
        alias="default",
        host=MILVUS_HOST,
        port=MILVUS_PORT
    )
    st.sidebar.success(f"Connected to Local Milvus ({MILVUS_HOST}:{MILVUS_PORT})!")
except Exception as e:
    st.sidebar.error(f"Failed to connect to Local Milvus: {e}")
    st.error(f"Failed to connect to Local Milvus: {MILVUS_HOST}:{MILVUS_PORT}. Please ensure the Milvus service is running and accessible. Error: {e}")
    st.stop()

print(f"device: {DEVICE}")

# Load open_clip model instead of SentenceTransformer
@st.cache_resource
def load_model():   
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    model.eval()
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    return model, preprocess, tokenizer

model, preprocess, tokenizer = load_model()
model.to(DEVICE)

def create_milvus_collection():
    if not utility.has_collection(COLLECTION_NAME, using="default"):
        fields = [
            FieldSchema(name=ID_FIELD_NAME, dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name=IMAGE_PATH_FIELD_NAME, dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name=INDEX_FIELD_NAME, dtype=DataType.FLOAT_VECTOR, dim=DIMENSION)
        ]
        schema = CollectionSchema(fields, description="Image similarity search collection on Local Milvus")
        collection = Collection(COLLECTION_NAME, schema=schema, using="default")
        index_params = {
            "metric_type": "IP", 
            "index_type": "AUTOINDEX", # Using AUTOINDEX for simplicity with local Milvus
            "params": {},
        }
        collection.create_index(INDEX_FIELD_NAME, index_params)
        st.sidebar.info(f"Collection '{COLLECTION_NAME}' created on Local Milvus.")
        return collection
    else:
        st.sidebar.info(f"Collection '{COLLECTION_NAME}' already exists on Local Milvus.")
        return Collection(COLLECTION_NAME, using="default")

collection = create_milvus_collection()

collection.load()

# Function to get embedding, from file or by computing
def get_or_create_embedding(image_path):
    image_filename = os.path.basename(image_path)
    embedding_filename = f"{image_filename}.npy"
    embedding_path = os.path.join(EMBEDDING_DIR, embedding_filename)

    if os.path.exists(embedding_path):
        return np.load(embedding_path)
    else:
        try:
            img = Image.open(image_path).convert("RGB")
            processed_img = preprocess(img).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad(), torch.autocast(DEVICE):
                embedding = model.encode_image(processed_img)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            
            # Convert to numpy and save
            embedding_np = embedding.squeeze().cpu().numpy()
            embedding_np = np.array(embedding_np, dtype=np.float32)
            np.save(embedding_path, embedding_np)
            return embedding_np
        except Exception as e:
            st.warning(f"Could not process or embed image {image_path}: {e}")
            return None

@st.cache_data
def load_and_insert_initial_images():
    st.sidebar.info("Initial image loading from app.py is currently disabled. Data should be imported using the dedicated import script.")
    return

# --- UI for adding new images ---
st.sidebar.subheader("Add New Images")
uploaded_new_images = st.sidebar.file_uploader(
    "Upload images here", 
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_new_images:
    new_images_batch_data = []
    inserted_count = 0
    failed_count = 0
    already_exists_count = 0

    for uploaded_file in uploaded_new_images:
        image_path = os.path.join(IMAGE_DIR, uploaded_file.name)
        # Save the uploaded file to local_images
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        embedding = get_or_create_embedding(image_path) # This will also save embedding to local_embeddings
        if embedding is not None:
            normalized_image_path = os.path.normpath(image_path)
            # Check if path already exists in Milvus to avoid duplicates if re-uploading before a full rescan
            res = collection.query(expr=f"{IMAGE_PATH_FIELD_NAME} == \"{normalized_image_path}\"", limit=1)
            if not res:
                new_images_batch_data.append({IMAGE_PATH_FIELD_NAME: normalized_image_path, INDEX_FIELD_NAME: embedding})
            else:
                st.sidebar.info(f"{uploaded_file.name} already exists in Local Milvus with the same path.")
                already_exists_count +=1
        else:
            st.sidebar.warning(f"Could not embed {uploaded_file.name}. Skipped.")
            failed_count += 1
    
    if new_images_batch_data:
        try:
            collection.insert(new_images_batch_data)
            collection.flush()
            inserted_count = len(new_images_batch_data)
            st.sidebar.success(f"Successfully added and indexed {inserted_count} new image(s) in a batch.")
        except Exception as e:
            st.sidebar.error(f"Failed to insert batch of new images: {e}")
            failed_count += len(new_images_batch_data) # Assume all in batch failed if exception
            inserted_count = 0 

    if inserted_count == 0 and failed_count == 0 and already_exists_count == len(uploaded_new_images) and uploaded_new_images:
        st.sidebar.info("All uploaded images were already in the database or no new files were processed.")
    elif inserted_count == 0 and failed_count == 0 and not uploaded_new_images: # Should not happen if uploaded_new_images is true
        pass
    elif inserted_count == 0 and failed_count > 0 :
         st.sidebar.warning(f"Failed to add {failed_count} image(s).")
    
    if inserted_count > 0 or failed_count > 0 or already_exists_count > 0:
        st.experimental_rerun()


st.title("Image Similarity Search: Local Milvus vs. Manual")

# Add sample queries section
st.subheader("Sample Queries")
sample_queries = [
    "a person walking on the street",
    "beautiful landscape with mountains",
    "a car driving on the road",
    "children playing in a park",
    "sunset over the ocean",
    "food on a table",
    "people in a meeting",
    "animal in the wild",
    "sports event",
    "cityscape at night"
]

cols = st.columns(5)
for i, query in enumerate(sample_queries):
    col_idx = i % 5
    if cols[col_idx].button(query, key=f"sample_query_{i}"):
        st.session_state['query_text'] = query  # Lưu giá trị query vào session_state

# Khởi tạo nếu chưa có trong session_state
if 'query_text' not in st.session_state:
    st.session_state['query_text'] = ""

query_text = st.text_input("Enter a text description for an image search:", value=st.session_state['query_text'])

uploaded_query_image = st.file_uploader("Or upload an image for search (query image):", type=["jpg", "jpeg", "png"], key="query_uploader")

top_k = st.slider("Select Top K results:", 1, 10, 3)

if 'zilliz_results' not in st.session_state:
    st.session_state.zilliz_results = None
if 'zilliz_time' not in st.session_state:
    st.session_state.zilliz_time = None
if 'manual_results' not in st.session_state:
    st.session_state.manual_results = None
if 'manual_time' not in st.session_state:
    st.session_state.manual_time = None
if 'embedding_time' not in st.session_state:
    st.session_state.embedding_time = None

@st.cache_data
def load_all_local_embeddings():
    all_loaded_embeddings = []
    all_loaded_image_paths = []
    cache_files_found = []
    
    # Ensure the directory for cache files exists
    os.makedirs(os.path.dirname(MANUAL_SEARCH_CACHE_FILE_BASE), exist_ok=True)

    # Check for multipart cache files first
    i = 1
    while True:
        # MANUAL_SEARCH_CACHE_FILE_BASE already includes the directory path
        part_file_name = f"{MANUAL_SEARCH_CACHE_FILE_BASE}_part_{i}{MANUAL_SEARCH_CACHE_EXT}"
        if os.path.exists(part_file_name):
            cache_files_found.append(part_file_name)
            i += 1
        else:
            break
    
    # If no part files, check for a single cache file
    single_cache_file = MANUAL_SEARCH_CACHE_FILE_BASE + MANUAL_SEARCH_CACHE_EXT
    if not cache_files_found and os.path.exists(single_cache_file):
        cache_files_found.append(single_cache_file)

    if cache_files_found:
        st.sidebar.caption(f"Loading manual search data from cache file(s): {', '.join(cache_files_found)}...")
        for cache_file in cache_files_found:
            try:
                data = np.load(cache_file, allow_pickle=True)
                embeddings = data['embeddings']
                image_paths = data['image_paths']
                if len(embeddings) > 0:
                    all_loaded_embeddings.extend(embeddings)
                    all_loaded_image_paths.extend(list(image_paths))
                else:
                    st.sidebar.caption(f"Cache file {cache_file} was empty.")
            except Exception as e:
                st.sidebar.warning(f"Error loading from {cache_file}: {e}. Will try to rebuild if other caches fail.")
        
        if all_loaded_embeddings:
            st.sidebar.caption(f"Loaded {len(all_loaded_embeddings)} embeddings from cache.")
            return np.array(all_loaded_embeddings), all_loaded_image_paths
        else:
            st.sidebar.warning("All cache files were empty or failed to load. Rebuilding from individual files.")

    # If cache doesn't exist or fails, load manually
    st.sidebar.caption(f"Building manual search cache from individual files in {EMBEDDING_DIR}...")
    all_embeddings_manual = []
    all_image_paths_manual = []

    # Ensure EMBEDDING_DIR exists
    os.makedirs(EMBEDDING_DIR, exist_ok=True)
    embedding_files = [f for f in os.listdir(EMBEDDING_DIR)]
    st.sidebar.caption(f"Found {len(embedding_files)} embedding files in {EMBEDDING_DIR}.")
    
    # Ensure IMAGE_DIR exists
    os.makedirs(IMAGE_DIR, exist_ok=True)

    for emb_file in embedding_files:
        try:
            embedding = np.load(os.path.join(EMBEDDING_DIR, emb_file))
            # New logic to handle .jpg.npy, .png.npy etc.
            if emb_file.lower().endswith(('.jpg.npy', '.jpeg.npy', '.png.npy')):
                if emb_file.lower().endswith('.jpeg.npy'):
                    original_filename_stem = emb_file[:-len('.jpeg.npy')]
                elif emb_file.lower().endswith('.png.npy'):
                    original_filename_stem = emb_file[:-len('.png.npy')]
                else: # .jpg.npy
                    original_filename_stem = emb_file[:-len('.jpg.npy')]
            elif emb_file.lower().endswith('.npy'):
                 original_filename_stem = emb_file[:-len('.npy')]
            else:
                st.warning(f"Unrecognized embedding file pattern: {emb_file}. Skipping.")
                continue

            found_image_path = None
            possible_extensions = ['.jpg', '.jpeg', '.png']

            # Attempt 1: stem + extension (e.g. embedding 'img1.npy' for 'img1.jpg')
            for img_ext in possible_extensions:
                potential_path = os.path.join(IMAGE_DIR, original_filename_stem + img_ext)
                if os.path.exists(potential_path):
                    found_image_path = potential_path
                    break
            
            # Attempt 2: if stem might already include an extension (e.g. 'img1.jpg.npy' for 'img1.jpg')
            if not found_image_path:
                base_name_of_stem = os.path.splitext(original_filename_stem)[0]
                for img_ext in possible_extensions:
                    potential_path = os.path.join(IMAGE_DIR, base_name_of_stem + img_ext)
                    if os.path.exists(potential_path):
                        found_image_path = potential_path
                        break
            
            # Attempt 3: Check if original_filename_stem is the direct image name (e.g. 'img1.jpg' for 'img1.jpg.npy')
            if not found_image_path and os.path.exists(os.path.join(IMAGE_DIR, original_filename_stem)):
                 found_image_path = os.path.join(IMAGE_DIR, original_filename_stem)

            if found_image_path:
                all_embeddings_manual.append(embedding)
                all_image_paths_manual.append(os.path.normpath(found_image_path))
            else:
                st.warning(f"Could not find original image for embedding: {emb_file}. Skipping for manual search.")

        except Exception as e:
            st.warning(f"Error loading embedding {emb_file}: {e}")
    # nomalize embeddings
    all_embeddings_manual = np.array(all_embeddings_manual)
    all_embeddings_manual = all_embeddings_manual / np.linalg.norm(all_embeddings_manual, axis=1, keepdims=True)
    return all_embeddings_manual, all_image_paths_manual

def get_top_k_embeddings(embeddings, query_embedding, top_k):
    distances = []
    for embedding in embeddings:
        distance = np.dot(embedding, query_embedding)
        distances.append(distance)

    distances = np.array(distances)
    top_k_indices = np.argsort(-distances)[:top_k]
    return top_k_indices, distances

ALL_EMBEDDINGS_MANUAL, ALL_IMAGE_PATHS_MANUAL = load_all_local_embeddings()

#normalize embeddings
ALL_EMBEDDINGS_MANUAL = ALL_EMBEDDINGS_MANUAL / np.linalg.norm(ALL_EMBEDDINGS_MANUAL, axis=1, keepdims=True)

if len(ALL_EMBEDDINGS_MANUAL) > 0:
    st.sidebar.caption(f"Loaded {len(ALL_EMBEDDINGS_MANUAL)} embeddings for manual search comparison.")
else:
    st.sidebar.caption("No embeddings found in local_embeddings for manual search.")

if st.button("Search"):
    query_embedding = None

    if uploaded_query_image is not None:
        st.session_state.zilliz_results = "loading"
        st.session_state.manual_results = "loading"
        st.session_state.zilliz_time = None
        st.session_state.manual_time = None
        st.session_state.embedding_time = None

        embed_start_time = time.time()
        image = Image.open(uploaded_query_image).convert("RGB")
        st.image(image, caption="Your Query Image", width=150)
        
        # Process image with open_clip
        processed_img = preprocess(image).unsqueeze(0).to(DEVICE).float()
        with torch.no_grad(), torch.autocast(DEVICE):
            embedding = model.encode_image(processed_img)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        
        query_embedding = embedding.squeeze().cpu().numpy()
        
        embed_end_time = time.time()
        st.session_state.embedding_time = embed_end_time - embed_start_time

    elif query_text:
        st.session_state.zilliz_results = "loading"
        st.session_state.manual_results = "loading"
        st.session_state.zilliz_time = None
        st.session_state.manual_time = None
        st.session_state.embedding_time = None

        embed_start_time = time.time()
        
        # Process text with open_clip
        text = tokenizer([query_text]).to(DEVICE)
        with torch.no_grad(), torch.autocast(DEVICE):
            embedding = model.encode_text(text)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            
        query_embedding = embedding.squeeze().cpu().numpy()
        
        embed_end_time = time.time()
        st.session_state.embedding_time = embed_end_time - embed_start_time
    else:
        st.warning("Please enter a text query or upload an image.")
        st.stop()

    if query_embedding is not None:
        # --- Milvus Search (Local) ---
        zilliz_search_start_time = time.time() # Keep var name for simplicity, or change to milvus_search_start_time
        search_params = {
            "metric_type": "IP",
            "params": {}, # Adjust if specific index params are needed for local Milvus search
        }
        results_milvus = collection.search( # Changed from results_zilliz
            data=[query_embedding.tolist()],
            anns_field=INDEX_FIELD_NAME,
            param=search_params,
            limit=top_k,
            expr=None, # Add expressions if needed, e.g., filtering by metadata
            output_fields=[IMAGE_PATH_FIELD_NAME]
        )
        zilliz_search_end_time = time.time() # Keep var name
        st.session_state.zilliz_time = zilliz_search_end_time - zilliz_search_start_time
        st.session_state.zilliz_results = [(hit.entity.get(IMAGE_PATH_FIELD_NAME), hit.distance) for hit in results_milvus[0]] # Changed from results_zilliz

        # --- Manual Search (using pre-loaded embeddings) ---
        manual_search_start_time = time.time()
        if len(ALL_EMBEDDINGS_MANUAL) > 0:
            sorted_indices, similarities = get_top_k_embeddings(ALL_EMBEDDINGS_MANUAL, query_embedding, top_k)
            results_manual = [(ALL_IMAGE_PATHS_MANUAL[i], similarities[i]) for i in sorted_indices]
        else:
            results_manual = []
        manual_search_end_time = time.time()
        st.session_state.manual_time = manual_search_end_time - manual_search_start_time
        st.session_state.manual_results = results_manual
    else:
        st.warning("Could not generate embedding for the query.")

st.subheader("Search Results")

if st.session_state.embedding_time is not None:
    st.write(f"**Query Embedding Time:** {st.session_state.embedding_time:.4f} seconds")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Milvus Search (Local)") # Changed from Zilliz Cloud Search
    if st.session_state.zilliz_results == "loading":
        with st.spinner("Searching with Local Milvus..."): # Changed
            pass
    elif st.session_state.zilliz_results:
        if st.session_state.zilliz_time is not None:
            st.write(f"**Search Time:** {st.session_state.zilliz_time:.4f} seconds")
        for img_path, similarity in st.session_state.zilliz_results:
            split_path = img_path.replace("\\", "/").split("/")
            video_folder = split_path[-2] if len(split_path) > 1 else ""
            image_filename = split_path[-1] if len(split_path) > 0 else ""
            path_image = os.path.join(IMAGE_DIR, video_folder, image_filename)
            if os.path.exists(path_image):
                st.image(path_image, width=150, caption=f"Path: {os.path.basename(path_image)}\nSimilarity: {similarity:.4f}")
            else:
                st.warning(f"Image not found at path: {path_image}")
    elif st.session_state.zilliz_results is None and (query_text or uploaded_query_image):
        st.write("Waiting for search...")

with col2:
    st.subheader("Manual Search (Local Embeddings)")
    if st.session_state.manual_results == "loading":
        with st.spinner("Performing manual search..."):
            pass
    elif st.session_state.manual_results:
        if st.session_state.manual_time is not None:
            st.write(f"**Search Time:** {st.session_state.manual_time:.4f} seconds")
        for img_path, similarity in st.session_state.manual_results:
            split_path = img_path.replace("\\", "/").split("/")
            video_folder = split_path[-2] if len(split_path) > 1 else ""
            image_filename = split_path[-1] if len(split_path) > 0 else ""
            path_image = os.path.join(IMAGE_DIR, video_folder, image_filename)
            if os.path.exists(path_image):
                st.image(path_image, width=150, caption=f"Path: {os.path.basename(path_image)}\nSimilarity: {similarity:.4f}")
            else:
                st.warning(f"Image not found at path: {path_image}")
    elif st.session_state.manual_results is None and (query_text or uploaded_query_image):
        st.write("Waiting for search...")

st.sidebar.markdown("---")
st.sidebar.header("About")
st.sidebar.info(
    "This demo showcases local image similarity search using a local Milvus instance and compares it "
    "with a manual brute-force comparison against locally stored embeddings. "
    "Images (from local folder or upload) and text queries are embedded using open_clip (CLIP model)."
)
st.sidebar.markdown("Instructions:")
st.sidebar.caption(
    "1. Ensure your local Milvus instance is running.\n"
    "2. Run 'generate_milvus_import_json.py' to prepare data files.\n"
    "3. Run 'import_generated_data_to_milvus.py' to load data into Milvus.\n"
    "4. Run 'manual_data/prepare_manual_search_data.py' for manual search cache.\n"
    "5. Place initial images in the 'local_images' folder (mounted into the container).\n"
    "6. Embeddings will be saved in 'local_embeddings' (mounted).\n"
    "7. Use the 'Add New Images' uploader in the sidebar to add more images to Milvus and local storage."
)
st.sidebar.markdown("---")

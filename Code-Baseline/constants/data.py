import sys
import os
from dotenv import load_dotenv

load_dotenv()

hf_auth = os.getenv("HF_AUTH_KEY")

# Dictionary mapping version names to model IDs and paths
version_mapping = {
    "v3_70": {
        "model_id": "meta-llama/Meta-Llama-3-70B-Instruct",
        "model_path": "model3_70"
    }
}

# Default version
default_version = "v3_70"

llama_version = default_version
rewriter_llama_version = llama_version

#load from pre-embeded data or not
load_presist = sys.argv[1] if len(sys.argv) > 1 else "n"

# Get model ID and path for llama version
llama_config = version_mapping.get(llama_version, version_mapping[default_version])
model_id = llama_config["model_id"]
model_path = llama_config["model_path"]

# Get model ID and path for rewriter version
rewriter_config = version_mapping.get(rewriter_llama_version, version_mapping[default_version])
rewriter_model_id = rewriter_config["model_id"]
rewriter_model_path = rewriter_config["model_path"]

documen_corpus_type=[
    {"name":"None", "document_location":"REQuestA-Dataset-Final-SRS-PDF", "default":"Yes"},
]
loc = ''
persist_directory = f'{loc}embeddings_requesta-srs-pbp-basic-3000'
results_directory = 'results_requesta/'
     
#create the results directory if not exist
if not os.path.exists(results_directory):
    os.makedirs(results_directory)

query_rewting_times = []
release_mapping_times = []
document_retrieval_times = []
context_selection_times = []
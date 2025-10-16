# Global configuration variables
OLLAMA_BASE_URL = "http://localhost:11434"  # Change this to your Ollama server URL
DEFAULT_MODEL = "qwen2.5:72b"
EMBEDDING_MODEL = "bge-m3:latest"   
SIMILARITY_MODEL = "qwen2:7b"


# OpenAI Configuration
base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
OPENAI_API_KEY = ""  # Set your OpenAI API key here
# OPENAI_MODEL = "qwen-plus-2025-04-28"  # Default model
OPENAI_MODEL = "qwen-plus-2025-01-25"

OPENAI_EMBEDDING_MODEL = "text-embedding-v3"  # Default embedding model
# OPENAI_SIMILARITY_MODEL = "qwen-plus-2025-04-28"  # Model for similarity checks
OPENAI_SIMILARITY_MODEL = "qwen-plus-2025-01-25"

# Model Provider Selection
USE_OPENAI = True  # Set to True to use OpenAI, False to use Ollama
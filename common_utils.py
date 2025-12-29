import os
import re
import logging
import json
import subprocess
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Set, Dict, Optional, List, Union
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from langchain_text_splitters import MarkdownTextSplitter, RecursiveCharacterTextSplitter

import pytesseract
from pdf2image import convert_from_path
from unstructured.partition.auto import partition

load_dotenv()

tfidf_vectorizer = TfidfVectorizer(
  stop_words='english',
  ngram_range=(1, 3),
  max_features=1000,
  lowercase=True
)

qdrant: Optional[QdrantClient] = None
model: Optional[SentenceTransformer] = None
splitter: Optional[Union[MarkdownTextSplitter, RecursiveCharacterTextSplitter]] = None
logger: Optional[logging.Logger] = None
text_cache: Dict[str, str] = {}
processed_files_cache: Dict[str, float] = {}
COLLECTION_NAME = "resume_database"
SUPPORTED_EXTENSIONS = {'.doc', '.docx', '.pdf', '.txt'}

def get_logger(log_file: str) -> logging.Logger:
  global logger
  if logger is None:
    logger = logging.getLogger('resume_analyzer')
    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
  return logger

def validate_folder(path: str, folder_type: str = "input") -> Path:
  folder_path = Path(path.strip())
  try:
    if folder_type == "input":
      if not folder_path.exists() or not folder_path.is_dir():
        raise ValueError(f"Input folder does not exist or is not a directory: {folder_path}")
    elif folder_type == "output":
      folder_path.mkdir(parents=True, exist_ok=True)
      test_file = folder_path / ".test_write"
      test_file.touch()
      test_file.unlink()
    else:
      raise ValueError(f"Invalid folder_type: {folder_type}. Use 'input' or 'output'.")
    return folder_path
  except (PermissionError, OSError) as e:
    raise ValueError(f"Folder validation failed for {folder_path}: {e}")

def validate_environment():
  load_dotenv()
  required_vars = ["QDRANT_URL", "QDRANT_API_KEY"]
  missing = [var for var in required_vars if not os.getenv(var)]
  if missing:
    if logger:
      logger.error(f"Missing environment variables: {missing}")
    raise ValueError(f"Missing environment variables: {missing}")

def init(qdrant_url: str, qdrant_api_key: str, collection_name: str):
  global qdrant, model, splitter, logger
  
  try:
    if logger is None:
      get_logger("resume_analyzer.log")
      
    if logger:
      logger.info(f"Initializing Qdrant with URL: {qdrant_url}, Collection: {collection_name}")
    
    if qdrant is None:
      qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
      collections = qdrant.get_collections()
      if logger:
        logger.info(f"Connected to Qdrant. Available collections: {[c.name for c in collections.collections]}")
    
    if model is None:
      device = os.getenv("TRANSFORMER_DEVICE", "cpu")
      if logger:
        logger.info(f"Loading SentenceTransformer model on device: {device}")
      model = SentenceTransformer('intfloat/multilingual-e5-large-instruct', device=device)
      if logger:
        logger.info("SentenceTransformer model loaded successfully")
    
    if splitter is None:
      try:
        splitter = MarkdownTextSplitter(chunk_size=300, chunk_overlap=75)
        if logger:
          logger.info("MarkdownTextSplitter initialized successfully")
      except Exception as splitter_error:
        if logger:
          logger.warning(f"Failed to initialize MarkdownTextSplitter: {splitter_error}")
          logger.info("Falling back to RecursiveCharacterTextSplitter")
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=75)
        if logger:
          logger.info("RecursiveCharacterTextSplitter initialized successfully")
    
    ensure_collection_exists(collection_name)
    if logger:
      logger.info("Initialization completed successfully")
    
  except Exception as e:
    if logger:
      logger.error(f"Initialization failed: {e}")
    # Reset global variables on failure
    qdrant = None
    model = None
    splitter = None
    raise ValueError(f"Failed to initialize components: {e}")

def is_initialized() -> bool:
  return all([qdrant is not None, model is not None, splitter is not None])

def ensure_collection_exists(collection_name: str):
  if qdrant is None:
    raise ValueError("Qdrant client not initialized")
  
  if model is None:
    raise ValueError("Model not initialized")
  
  try:
    existing_collections = [c.name for c in qdrant.get_collections().collections]
    if collection_name not in existing_collections:
      # Fix the size parameter
      embedding_dim = model.get_sentence_embedding_dimension()
      if embedding_dim is None:
        raise ValueError("Model embedding dimension is None")
      
      qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
          size=embedding_dim, # Now guaranteed to be int
          distance=Distance.COSINE
        )
      )
      if logger:
        logger.info(f"Created Qdrant collection: {collection_name}")
    else:
      if logger:
        logger.info(f"Collection {collection_name} already exists")
  except Exception as e:
    if logger:
      logger.error(f"Failed to ensure collection {collection_name}: {e}")
    raise

def load_json_set(path: Path) -> Set[str]:
  try:
    if path.exists():
      with path.open("r", encoding='utf-8') as f:
        return set(json.load(f))
    return set()
  except Exception as e:
    if logger:
      logger.error(f"Failed to load {path}: {e}")
    return set()

def save_json_set(data: Set[str], path: Path):
  try:
    with path.open("w", encoding='utf-8') as f:
      json.dump(sorted(list(data)), f, indent=2)
    if logger:
      logger.info(f"Saved {len(data)} items to {path}")
  except Exception as e:
    if logger:
      logger.error(f"Failed to save {path}: {e}")

def safe_upsert_with_retry(client: QdrantClient, collection_name: str, points: list, max_retries: int = 3) -> bool:
  for attempt in range(max_retries):
    try:
      client.upsert(collection_name=collection_name, points=points)
      return True
    except Exception as e:
      wait = 2 ** attempt
      if logger:
        logger.warning(f"Qdrant upsert failed (attempt {attempt + 1}): {e}. Retrying in {wait}s...")
      import time
      time.sleep(wait)
  if logger:
    logger.error("Qdrant upsert failed after all retries.")
  return False

# In common_utils.py, find the convert_doc_to_pdf function and modify it:

def convert_doc_to_pdf(file_path: Path) -> Optional[Path]:
  output_dir = file_path.parent
  pdf_path = output_dir / f"{file_path.stem}.pdf"
  try:
    possible_paths = [
            r"C:\Program Files\LibreOffice\program\soffice.exe",
            r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
            "soffice",  # If in PATH
        ]
    soffice_path = None
    for path in possible_paths:
        if Path(path).exists() or path == "soffice":
            soffice_path = path
            break
    if not soffice_path:
        logger.error("LibreOffice not found. Please install it from libreoffice.org")#type: ignore
        return None
    subprocess.run(
            [
                soffice_path,
                "--headless",
                "--convert-to", "pdf",
                "--outdir", str(output_dir),
                str(file_path)
            ],
            check=True,
            timeout=30,
            capture_output=True
    )
    if pdf_path.exists():
      if logger:
        logger.info(f"Converted {file_path} to PDF: {pdf_path}")
      return pdf_path
    if logger:
      logger.error(f"Conversion failed: {pdf_path} not created")
    return None
  except subprocess.CalledProcessError as e:
    if logger:
      logger.error(f"LibreOffice conversion failed for {file_path}: {e}")
    return None


def is_text_pdf(file_path: Path) -> bool:
  try:
    elements = partition(filename=str(file_path))
    return any(element.text.strip() for element in elements)
  except Exception as e:
    if logger:
      logger.error(f"Error checking text in PDF {file_path}: {e}")
    return False

def ocr_pdf(file_path: Path) -> str:
  try:
    images = convert_from_path(str(file_path))
    text = ""
    for image in images:
      text += pytesseract.image_to_string(image) + " "
    text = re.sub(r'\s+', ' ', text.strip())
    if logger:
      logger.info(f"OCR completed for {file_path}")
    return text
  except Exception as e:
    if logger:
      logger.error(f"OCR failed for {file_path}: {e}")
    return ""

def unstructured_to_markdown(file_path: Path) -> str:
  try:
    elements = partition(filename=str(file_path))
    text = "\n".join(element.text for element in elements if element.text.strip())
    text = re.sub(r'\s+', ' ', text.strip())
    if logger:
      logger.info(f"Extracted markdown from {file_path}")
    return text
  except Exception as e:
    if logger:
      logger.error(f"Text extraction failed for {file_path}: {e}")
    return ""

def safe_create_documents(text: str, metadata: Optional[dict] = None) -> List[str]:
  """Split text into chunks using the configured splitter"""
  if not is_initialized():
    raise ValueError("Components not properly initialized. Call init() first.")
  
  if splitter is None:
    raise ValueError("Text splitter not initialized")
  
  try:
    chunks = splitter.split_text(text)
    if logger:
      logger.info(f"Created {len(chunks)} document chunks")
    return chunks
  except Exception as e:
    if logger:
      logger.error(f"Failed to create documents: {e}")
    return []
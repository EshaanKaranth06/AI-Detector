import os
import re
import phonenumbers
from collections import Counter
from pathlib import Path
from typing import Any, List, Dict, Optional, Union
from qdrant_client.models import ScoredPoint
import json
import pytesseract
from PIL import Image, ImageEnhance, ImageOps
from pdf2image import convert_from_path
import time
from rf_classifier import PlagiarismRFClassifier
from dotenv import load_dotenv
import pymupdf4llm
from langchain_text_splitters import MarkdownTextSplitter, RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from sklearn.feature_extraction.text import TfidfVectorizer
from unstructured.partition.auto import partition
from common_utils import (
  get_logger, SUPPORTED_EXTENSIONS, qdrant, convert_doc_to_pdf, ocr_pdf, is_text_pdf
)
from plagiarism_heuristics import (
  combined_ngram_similarity, tfidf_similarity, fuzzy_similarity, normalized_bleurt_score
)

load_dotenv()
logger = get_logger("logs/detector.log")

# Global variables
qdrant: Optional[QdrantClient] = None
model: Optional[SentenceTransformer] = None
splitter: Optional[Union[MarkdownTextSplitter, RecursiveCharacterTextSplitter]] = None
tfidf_vectorizer: Optional[TfidfVectorizer] = None
processed_files_cache: Dict[str, float] = {}
text_cache: Dict[str, str] = {}
COLLECTION_NAME: Optional[str] = None
INPUT_FOLDER: Optional[str] = None
OUTPUT_FOLDER: Optional[str] = None
REFERENCE_FOLDER: Optional[str] = None
rf_classifier: Optional[PlagiarismRFClassifier] = None

def resume_section_split(text: str) -> List[str]:
  sections = re.split(
    r'\n(?=(experience|skills|education|summary|profile|work|projects|certifications?|achievements?|qualifications?|objective|career\s+objective|professional\s+summary|work\s+experience|educational\s+background|technical\s+skills|personal\s+details|contact|references)\b)',
    text.lower()
  )
  
  meaningful_sections = []
  for s in sections:
    stripped = s.strip()
    if len(stripped) > 50: # Minimum meaningful content length
      meaningful_sections.append(stripped)
  
  return meaningful_sections


def chunk_within_sections(sections: List[str], chunk_size: int = 300, chunk_overlap: int = 75) -> List[str]:
  """
  Apply chunking within each section rather than globally.
  This maintains semantic coherence.
  """
  from langchain_text_splitters import MarkdownTextSplitter
  
  section_splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  all_chunks = []
  
  for section in sections:
    # Chunk each section independently
    section_docs = section_splitter.create_documents([section])
    section_chunks = [doc.page_content for doc in section_docs]
    all_chunks.extend(section_chunks)
  
  return all_chunks


def unstructured_to_markdown(file_path: Path) -> str:
  try:
    elements = partition(filename=str(file_path))
    md_lines = []
    for el in elements:
      text = el.text.strip()
      if not text:
        continue
      category = el.category or ""
      if category == "Title":
        md_lines.append(f"# {text}")
      elif category in {"Header", "Subheader"}:
        md_lines.append(f"## {text}")
      elif category == "ListItem":
        md_lines.append(f"- {text}")
      elif category == "NarrativeText":
        md_lines.append(text)
      elif ":" in text:
        md_lines.append(f"**{text}**")
      else:
        md_lines.append(text)
      md_lines.append("")
    return "\n".join(md_lines)
  except Exception as e:
    logger.error(f"Markdown extraction failed for {file_path}: {e}")
    return ""


def init(qdrant_url: str, qdrant_api_key: str, collection_name: str) -> None:
  global qdrant, model, splitter, tfidf_vectorizer, processed_files_cache, text_cache, COLLECTION_NAME, rf_classifier
  
  if not qdrant_url or not qdrant_api_key:
    raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in the environment variables")
  
  COLLECTION_NAME = collection_name
  
  # Always load the model during init to avoid Model not initialized errors
  load_model() 
  
  splitter = MarkdownTextSplitter(chunk_size=300, chunk_overlap=75)
  tfidf_vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 3),
    max_features=1000,
    lowercase=True
  )
  processed_files_cache = {}
  text_cache = {}
  initialize_qdrant(qdrant_url, qdrant_api_key)

  try:
    rf_model_path = Path("models/plagiarism_rf.pkl")
    if rf_model_path.exists():
      assert rf_model_path is not None
      rf_classifier = PlagiarismRFClassifier(rf_model_path)
      logger.info("Random Forest classifier loaded successfully")
    else:
        logger.warning("RF model not found at models/plagiarism_rf.pkl")
        logger.warning("Using fallback rule-based classification")
        rf_classifier = None
  except Exception as e:
        logger.error(f"Failed to load RF classifier: {e}")
        rf_classifier = None


def load_model() -> None:
  global model
  if model is None:
    try:
      device = os.getenv("TRANSFORMER_DEVICE","cpu") #"cuda" if torch.cuda.is_available() else
      logger.info(f"Loading SentenceTransformer model on device: {device}")
      model = SentenceTransformer(
        'intfloat/multilingual-e5-large-instruct',
        device=device
      )
      logger.info("Model loaded successfully")
    except Exception as e:
      logger.error(f"Failed to load model: {e}")
      raise


def initialize_qdrant(qdrant_url: str, qdrant_api_key: str) -> None:
  global qdrant
  try:
    qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    ensure_collection_exists()
    logger.info("Database connected successfully")
  except Exception as e:
    logger.error(f"Failed to connect to database: {e}")
    raise


def ensure_collection_exists() -> None:
  global qdrant, model, COLLECTION_NAME
  
  if qdrant is None:
    raise ValueError("Qdrant client not initialized")
  if COLLECTION_NAME is None:
    raise ValueError("Collection name not set")
  
  try:
    collections_response = qdrant.get_collections()
    collection_names = [col.name for col in collections_response.collections]
    if COLLECTION_NAME not in collection_names:
      load_model()
      if model is None:
        raise ValueError("Model not initialized")
      vector_size = model.get_sentence_embedding_dimension()
      if vector_size is None:
        raise ValueError("Could not get embedding dimension")
      qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
      )
      logger.info(f"Created collection: {COLLECTION_NAME}")
  except Exception as e:
    logger.error(f"Error managing collection: {e}")
    raise

def extract_text_from_file(file_path: Path) -> str:
  global text_cache
  if not isinstance(file_path, Path):
    logger.error(f"Invalid file_path type: {type(file_path)} for {file_path}")
    raise ValueError(f"Invalid file_path: {file_path}")
  if not file_path.exists() or not file_path.is_file():
    logger.error(f"File does not exist or is not a file: {file_path}")
    raise ValueError(f"File does not exist: {file_path}")

  cache_key = f"{file_path.name}_{file_path.stat().st_mtime}"
  if cache_key in text_cache:
    logger.debug(f"Cache hit for {file_path.name}")
    return text_cache[cache_key]

  filename = file_path.name
  file_ext = file_path.suffix.lower()
  was_converted = False
  converted_path: Optional[Path] = None

  try:
    if file_ext not in SUPPORTED_EXTENSIONS:
      logger.error(f"Unsupported file type: {filename}")
      raise ValueError(f"Unsupported file type: {file_ext}")

    original_text = ""
    if file_ext != ".pdf":
      logger.info(f"Converting {filename} to PDF")
      converted_path_result = convert_doc_to_pdf(file_path)
      converted_path = Path(converted_path_result) if converted_path_result else None
      if not converted_path or not converted_path.exists():
        logger.warning(f"LibreOffice conversion failed for {filename}")
        try:
          original_text = unstructured_to_markdown(file_path)
          if not original_text.strip():
            logger.warning(f"Unstructured method failed for {filename}. Trying OCR.")
            original_text = ocr_pdf(file_path)
        except Exception as e:
          logger.error(f"Fallback extraction failed for {filename}: {e}")
          original_text = ""
      else:
        file_path = converted_path
        was_converted = True

    try:
      result = pymupdf4llm.to_markdown(str(file_path))
      original_text = str(result) 
      md_text = original_text.lower()
      if not md_text.strip():
        logger.warning(f"Empty markdown from pymupdf4llm for {filename}")
        original_text = unstructured_to_markdown(file_path)
        md_text = original_text.lower()
    except Exception as e:
      logger.error(f"pymupdf4llm failed for {filename}: {e}")
      try:
        original_text = unstructured_to_markdown(file_path)
        md_text = original_text.lower()
      except Exception as e:
        logger.error(f"Unstructured method failed for {filename}: {e}")
        md_text = ""

    if not md_text.strip() and is_text_pdf(file_path):
      logger.info(f"Trying OCR fallback for {filename}")
      try:
        original_text = ocr_pdf(file_path)
        md_text = original_text.lower()
      except Exception as e:
        logger.error(f"OCR failed for {filename}: {e}")
        md_text = ""
        original_text = ""

    if was_converted and converted_path and converted_path.exists():
      try:
        os.remove(converted_path)
        logger.debug(f"Removed intermediate PDF: {converted_path}")
      except (PermissionError, OSError) as e:
        logger.error(f"Failed to delete {converted_path}: {e}")

    if not md_text.strip():
      logger.error(f"No text extracted for {filename}")
      raise ValueError(f"No text extracted for {filename}")

    text_cache[cache_key] = md_text
    return original_text

  except Exception as e:
    logger.error(f"Unexpected error extracting text from {filename}: {e}")
    raise


def get_resume_files(folder_path: Path) -> List[Path]:
  if not folder_path.exists():
    logger.error(f"Input folder does not exist: {folder_path}")
    return []
  try:
    files = [file_path for file_path in folder_path.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS]
    logger.info(f"Found {len(files)} supported files")
    return files
  except PermissionError as e:
    logger.error(f"Permission error accessing folder: {e}")
    return []


def classify_plagiarism_level(overall_score, flagged_score, sources, flagged_chunks):
  # If 4+ chunks come from 1 person, it's PLAGIARIZED regardless of the low % score
  if len(sources) == 1 and flagged_chunks >= 4:
    return True, "PLAGIARIZED (Identity Theft Pattern)"
  
  if len(sources) > 0:
    if flagged_score > 60 or overall_score > 45:
      return True, "HIGH"
    if flagged_chunks >= 2 and flagged_score > 45:
      return True, "MODERATE"
    if flagged_chunks >= 1:
      return True, "LOW"

  return False, "CLEAN"

def extract_contact_text(path: Path) -> str:
  try:
    ext = path.suffix.lower()

    # If already a PDF or an image  use OCR directly
    if ext == ".pdf":
      images = convert_from_path(str(path), dpi=400)
      text = ""
      for img in images:
        img = ImageOps.grayscale(img)
        img = ImageEnhance.Contrast(img).enhance(2.0)
        text += pytesseract.image_to_string(img, config='--psm 6 --oem 3')
      return text

    elif ext in [".jpg", ".jpeg", ".png"]:
      image = Image.open(path)
      image = ImageOps.grayscale(image)
      image = ImageEnhance.Contrast(image).enhance(2.0)
      return pytesseract.image_to_string(image, config='--psm 6 --oem 3')

    # If it's a DOC or DOCX  convert to PDF via LibreOffice, then OCR
    elif ext in [".doc", ".docx"]:
      converted_pdf_result = convert_doc_to_pdf(path)
      if converted_pdf_result:
        converted_pdf = Path(converted_pdf_result)
        if converted_pdf.exists():
          return extract_contact_text(converted_pdf)
      logger.warning(f"Could not convert {path.name} to PDF for OCR.")
      return ""

    else:
      logger.warning(f"Unsupported file type for contact extraction: {path.name}")
      return ""

  except Exception as e:
    logger.error(f"Failed to extract contact text from {path.name}: {e}")
    return ""


def extract_emails(text: str, fallback_path: Optional[Path] = None) -> List[str]:
  def run_email_regex(content: str) -> List[str]:
    # Extract emails
    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    found = re.findall(pattern, content, re.IGNORECASE)
    return list(set(email.lower() for email in found)) 

  emails = []

  #  First pass: try from clean text
  if text and isinstance(text, str):
    emails = run_email_regex(text)
    logger.debug(f"Emails from text for {fallback_path.name if fallback_path else 'input text'}: {emails}")

  # Fallback 1: OCR and retry regex
  if not emails and fallback_path:
    logger.warning(f"Trying OCR fallback for email extraction: {fallback_path.name}")
    ocr_text = extract_contact_text(fallback_path)
    emails = run_email_regex(ocr_text)
    logger.debug(f"Emails from OCR for {fallback_path.name}: {emails}")

  if not emails:
    logger.warning(f"No emails found in {fallback_path.name if fallback_path else 'input text'}")

  return emails


def extract_phones(text: str, region: str = "IN", fallback_path: Optional[Path] = None) -> List[str]:
  def run_phonenumbers(content: str) -> List[str]:
    numbers = []
    for match in phonenumbers.PhoneNumberMatcher(content, region):
      num = match.number
      if phonenumbers.is_valid_number(num):
        formatted = phonenumbers.format_number(num, phonenumbers.PhoneNumberFormat.E164)
        numbers.append(formatted)
    return list(set(numbers))

  def run_phone_regex(content: str) -> List[str]:
    phone_regex = re.compile(
      r'''(
        (?:\+|00)?91[\s\-\.()]*(\d{5})[\s\-\.()]*(\d{5})    # Indian numbers
        |
        \(?\+?\d{1,4}\)?[\s\-\.()]*(\d{2,5})[\s\-\.()]*(\d{6,8})
        |
        \(?\d{3}\)?[\s\-\.()]?\d{3}[\s\-\.()]?\d{4}       # US format
      )''',
      re.IGNORECASE | re.VERBOSE
    )
    matches = []
    for m in phone_regex.finditer(content):
      raw = m.group(0)
      try:
        num = phonenumbers.parse(raw.strip(), region)
        if phonenumbers.is_valid_number(num):
          formatted = phonenumbers.format_number(num, phonenumbers.PhoneNumberFormat.E164)
          matches.append(formatted)
      except Exception:
        continue
    return list(set(matches))

  phones = []
  
  #  Try 1: Smart matcher
  if text and isinstance(text, str):
    phones = run_phonenumbers(text)
    logger.debug(f"Phones from smart matcher for {fallback_path.name if fallback_path else 'input text'}: {phones}")
  
  # Fallback 1: Try regex if nothing found
  if not phones and text:
    phones = run_phone_regex(text)
    logger.debug(f"Phones from regex fallback for {fallback_path.name if fallback_path else 'input text'}: {phones}")
  
  # Fallback 2: OCR + rerun matcher and regex
  if not phones and fallback_path:
    logger.warning(f"Trying OCR fallback for phone extraction: {fallback_path.name}")
    ocr_text = extract_contact_text(fallback_path)
    phones = run_phonenumbers(ocr_text)
    if not phones:
      phones = run_phone_regex(ocr_text)
    logger.debug(f"Phones from OCR for {fallback_path.name}: {phones}")

  if not phones:
    logger.warning(f"No phones found in {fallback_path.name if fallback_path else 'input text'}")

  return phones


def search_similar_chunks(embedding: List[float], limit: int = 5) -> List[ScoredPoint]:
  global qdrant, COLLECTION_NAME
  
  if qdrant is None:
    raise ValueError("Qdrant client not initialized")
  if COLLECTION_NAME is None:
    raise ValueError("Collection name not set")
  
  try:
    if not embedding or not isinstance(embedding, list):
      raise ValueError(f"Invalid embedding provided for search in collection {COLLECTION_NAME}")
    logger.info(f"Searching with embedding length: {len(embedding)}")
    search_results = qdrant.query_points(
      collection_name=COLLECTION_NAME,
      query=embedding,
      limit=limit,
      with_payload=True,
      with_vectors=True
    ).points
    logger.debug(f"Found {len(search_results)} matches, point types: {[type(point).__name__ for point in search_results]}")
    if search_results:
      logger.debug(f"Sample point: {search_results[0]}")
    return search_results
  except Exception as e:
    logger.error(f"Error searching database: {e}")
    return []


def calculate_composite_score(cosine: float, tfidf: float, fuzzy: float, ngram: float) -> float:
  try:
    weights = {'cosine': 0.4, 'tfidf': 0.15, 'fuzzy': 0.05, 'ngram': 0.4}
    return (weights['cosine'] * cosine + weights['tfidf'] * tfidf +
        weights['fuzzy'] * fuzzy + weights['ngram'] * ngram)
  except Exception:
    return 0.0


def analyze_document(text: str, filename: str, file_path: Path) -> Dict:
  global qdrant, model, splitter, COLLECTION_NAME
  
  if qdrant is None:
    raise ValueError("Qdrant client not initialized")
  if model is None:
    raise ValueError("Model not initialized")
  if splitter is None:
    raise ValueError("Splitter not initialized")
  if COLLECTION_NAME is None:
    raise ValueError("Collection name not set")
  
  results: Dict = {
    'filename': filename,
    'status': 'success',
    'error': None,
    'overall_score': 0.0,
    'flagged_score': 0.0,
    'total_chunks': 0,
    'flagged_chunks': 0,
    'is_plagiarized': False,
    'plagiarized_sources': [],
    'file_mtime': '',
    'processed_at': '',
    'matched_chunks': [],
    'input_emails': [],
    'input_phones': [],
    'source_emails': [],
    'source_phones': [],
    'bleurt_scores': [] 
  }

  try:
    if not isinstance(file_path, Path) or not file_path.exists():
      results['error'] = f"Invalid or non-existent file path: {file_path}"
      results['status'] = 'error'
      logger.error(results['error'])
      return results

    results['file_mtime'] = file_path.stat().st_mtime
    results['processed_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"Processing file: {filename}")

    if text is None or not text.strip():
      results['error'] = "No text extracted from file"
      results['status'] = 'error'
      logger.error(f"No text extracted for {filename}")
      return results

    collection_info = qdrant.get_collection(COLLECTION_NAME)
    logger.info(f"Collection {COLLECTION_NAME} has {collection_info.points_count} points")
    
    if collection_info.points_count == 0:
      results['error'] = "Qdrant collection is empty; no reference documents to compare against"
      results['status'] = 'error'
      logger.warning(f"Collection {COLLECTION_NAME} is empty")
      return results

    logger.info(f"Extracting contact info from {filename}")
    try:
      input_emails = extract_emails(text, file_path)
      input_phones = extract_phones(text, region="IN", fallback_path=file_path)
      results['input_emails'] = input_emails
      results['input_phones'] = input_phones
      if input_emails or input_phones:
        print(f" Found: {len(input_emails)} emails, {len(input_phones)} phones")
      else:
        logger.warning(f"No contact information found for {filename}")
    except Exception as e:
      logger.error(f"Contact extraction failed for {filename}: {e}")
      results['input_emails'] = []
      results['input_phones'] = []

    try:
      logger.info(f"Applying resume-aware section splitting for {filename}")
      sections = resume_section_split(text)
      logger.info(f"Identified {len(sections)} semantic sections")
      
      # Chunk within sections to preserve semantic meaning
      texts = chunk_within_sections(sections, chunk_size=300, chunk_overlap=75)
      logger.info(f"Created {len(texts)} chunks from sections")

      def is_irrelevant_chunk(text: str) -> bool:
        lowered = text.lower().strip()
        
        # Skip empty or very short chunks
        if len(lowered) < 10:
          return True
        
        # Generic resume section headers
        generic_headers = [
          'languages', 'language', 'interests', 'hobbies', 'proficiency',
          'responsibilities', 'responsibility', 'skills', 'skill',
          'experience', 'education', 'objective', 'summary',
          'contact', 'personal', 'references', 'declaration',
          'career objective', 'professional summary', 'work experience',
          'educational background', 'technical skills', 'personal details','board', 'school', 'university', 'degree',
          'certified', 'certification', 'azure', 'aws', 'dp-', 'ai-', 'az-','nvidia'
        ]
        
        # Common single/few word entries
        common_words = [
          'public', 'private', 'protected', 'static', 'final', 'abstract',
          'class', 'interface', 'extends', 'implements',
          'male', 'female', 'married', 'single', 'hindi', 'english',
          'aws', 'ec2', 's3', 'vpc', 'iam', 'lambda', 'rds', 'dynamodb', 
        'java', 'python', 'javascript', 'html', 'css', 'react', 'node',
        'sql', 'docker', 'kubernetes', 'jenkins', 'git', 'agile', 'scrum',
        ]
        
        # Check if chunk is just a generic header or common word
        if any(header in lowered for header in generic_headers):
          # Additional check: if it's mostly just the header with minimal content
          words = lowered.split()
          if len(words) <= 5: # Very short chunks are likely just headers
            return True
        
        # Check if chunk is just common programming/resume keywords
        if lowered in common_words:
          return True
        
        # Check if chunk is mostly formatting (asterisks, colons, etc.)
        content_chars = sum(1 for c in lowered if c.isalnum())
        if content_chars < len(lowered) * 0.5: # Less than 50% alphanumeric
          return True
        
        # Check for very generic phrases
        generic_phrases = [
          'click here', 'see more', 'read more', 'view all',
          'download', 'upload', 'submit', 'cancel', 'ok', 'yes', 'no'
        ]
        
        if any(phrase in lowered for phrase in generic_phrases):
          return True
        
        return False

      filtered_texts = [t for t in texts if not is_irrelevant_chunk(t)]
      if not filtered_texts:
        results['error'] = "Only irrelevant sections found in text"
        results['status'] = 'error'
        logger.error("All chunks were filtered out")
        return results
      texts = filtered_texts
      results['total_chunks'] = len(texts)
    except Exception as e:
      results['error'] = f"Failed to split text: {e}"
      results['status'] = 'error'
      return results

    if not texts:
      results['error'] = "No chunks generated from text"
      results['status'] = 'error'
      logger.error("No chunks generated")
      return results

    load_model()
    try:
      
      logger.info(f"Encoding chunks with E5 'passage:' instruction prefix")
      texts_with_instruction = [f"query: {t}" for t in texts]
      
      start_time = time.time()
      embeddings = model.encode(texts_with_instruction, show_progress_bar=True, batch_size=16)
      logger.info(f"Encoded {len(texts)} chunks in {time.time() - start_time:.2f} seconds")
      if embeddings is None or len(embeddings) == 0:
        raise ValueError("No embeddings generated")
    except Exception as e:
      results['error'] = f"Failed to generate embeddings: {e}"
      results['status'] = 'error'
      logger.error(f"Embedding generation failed: {e}")
      return results

    all_scores = []
    max_scores_per_chunk = []
    plagiarized_sources = set()
    matched_chunks = []
    source_contacts: Dict[str, Dict[str, List[str]]] = {}
    source_bleurt_scores: Dict[str, List[float]] = {} # Track BLEURT scores for matched chunks per source file
    chunk_features_for_rf: List[Dict[str, Any]] = []
    
    
    source_hit_map = Counter()

    for i, emb in enumerate(embeddings):
      try:
        points = search_similar_chunks(emb.tolist(), limit=5)
        chunk_max_score = 0.0
        for scored_point in points:
          try:
            if hasattr(scored_point, 'payload') and scored_point.payload is not None:
              matched_text = scored_point.payload.get("text", "")
              source_file = scored_point.payload.get("source_file", "unknown")
              cosine_score = float(scored_point.score)
            elif isinstance(scored_point, tuple):
              if len(scored_point) >= 3:
                payload = scored_point[2] if scored_point[2] is not None else {}
                matched_text = payload.get("text", "")
                source_file = payload.get("source_file", "unknown")
                cosine_score = float(scored_point[1])
              else:
                logger.warning(f"Unexpected tuple format: {scored_point}")
                continue
            else:
              logger.warning(f"Unknown point type: {type(scored_point)}")
              continue

            if source_file == filename:
              logger.info(f"Same file detected for {filename}, skipping match")
              continue

            # Note: We compare against the original text (without "passage:") for these metrics
            tfidf_score = tfidf_similarity(texts[i], matched_text)
            fuzzy_score = fuzzy_similarity(texts[i], matched_text)
            ngram_score = combined_ngram_similarity(texts[i], matched_text)
            bleurt_score = normalized_bleurt_score(texts[i], matched_text)

            composite_score = calculate_composite_score(
              cosine_score, tfidf_score, fuzzy_score, ngram_score
            )

            def is_boilerplate_content(input_text: str, matched_text: str) -> bool:
                """Detect if content is standard resume boilerplate"""
                input_lower = input_text.lower()
                match_lower = matched_text.lower()
                
                # 1. Certification patterns
                cert_keywords = ['certified', 'certification', 'certificate', 'credential']
                cert_providers = ['microsoft', 'aws', 'azure', 'google cloud', 'oracle', 
                                'cisco', 'comptia', 'pmp', 'scrum', 'databricks', 'nptel']
                
                has_cert_keyword = any(kw in input_lower for kw in cert_keywords)
                has_cert_provider = any(prov in input_lower for prov in cert_providers)
                
                if has_cert_keyword and has_cert_provider:
                    # Both texts mention certifications - likely standard cert list
                    return True
                
                # 2. Education section patterns
                education_keywords = ['education', 'academic', 'degree', 'university', 
                                    'college', 'school', 'bachelor', 'master', 'b.tech',
                                    'm.tech', 'b.e', 'm.e', 'bca', 'mca', '10th', '12th',
                                    'sslc', 'hsc', 'board', 'examination']
                
                edu_count = sum(1 for kw in education_keywords if kw in input_lower)
                
                # Check for table-like structure (common in education sections)
                has_pipes = '|' in input_text or '|' in matched_text
                has_years = any(str(year) in input_text for year in range(2000, 2030))
                
                if edu_count >= 2 and (has_pipes or has_years):
                    return True
                
                # 3. Section headers (very short, mostly formatting)
                if len(input_text.split()) <= 5 and len(matched_text.split()) <= 5:
                    header_words = ['experience', 'education', 'skills', 'summary', 
                                    'objective', 'projects', 'details', 'information',
                                    'professional', 'academic', 'technical', 'certifications',
                                    'professional experience', 'work experience', 'employment history',
                                    'software engineer', 'data engineer', 'developer', 'analyst',
                                    'present', 'current', '–', 'pvt ltd', 'inc', 'technologies']
                    if any(hw in input_lower for hw in header_words):
                        return True
                
               
                date_patterns = ['present', 'current', '–', '-']
                location_patterns = ['india', 'bangalore', 'pune', 'mumbai', 'delhi', 
                                    'hyderabad', 'chennai', 'noida', 'gurgaon']
                
                has_dates = any(dp in input_lower for dp in date_patterns)
                has_location = any(lp in input_lower for lp in location_patterns)
                
                if has_dates and has_location and len(input_text.split('\n')) <= 4:
                    return True
                
                return False

            if is_boilerplate_content(texts[i], matched_text):
                logger.debug(f"Skipping boilerplate match: {texts[i][:50]}...")
                continue

            chunk_max_score = max(chunk_max_score, composite_score)
            is_semantic = (cosine_score > 0.82 and bleurt_score > 0.45)
            is_literal = (ngram_score > 0.40 or tfidf_score > 0.45)
            is_pattern = (cosine_score > 0.65 and source_hit_map[source_file] >= 2)

 
            if is_literal or is_semantic or is_pattern:
            
                chunk_features_for_rf.append({
                    "cosine": cosine_score,
                    "tfidf": tfidf_score,
                    "ngram": ngram_score,
                    "fuzzy": fuzzy_score,
                    "bleurt": bleurt_score,
                    "composite": composite_score,
                    "source_file": source_file,
                })
                
                source_hit_map[source_file] += 1
                plagiarized_sources.add(source_file)
                all_scores.append(composite_score)
                
                
                if source_file not in source_bleurt_scores:
                    source_bleurt_scores[source_file] = []
                    source_bleurt_scores[source_file].append(bleurt_score)

                    matched_chunks.append({
                    'similarity_score': composite_score * 100,
                    'source_url': source_file,
                    'input_chunk': texts[i],
                    'source_chunk': matched_text,
                    'bleurt_score': bleurt_score,
                    'cosine_score': cosine_score,
                    'tfidf_score': tfidf_score,
                    'ngram_score': ngram_score,
                    'fuzzy_score': fuzzy_score,
                    })
                    
                    # this one is to break to avoid matching the same input chunk to multiple chunks in the same source file 
                    break
          except Exception as e:
            logger.error(f"Error processing match {i}: {e}")
            continue
        max_scores_per_chunk.append(chunk_max_score)
      except Exception as e:
        logger.error(f"Error processing chunk {i}: {e}")
        max_scores_per_chunk.append(0.0)
        continue

    # Calculate average BLEURT score for matched chunks per source file
    avg_bleurt_scores: List[Union[float, str]] = []
    for source_file, scores in source_bleurt_scores.items():
      avg_score = round(sum(scores) / len(scores), 2) if scores else 0.0
      avg_bleurt_scores.extend([avg_score, source_file])

    results['bleurt_scores'] = avg_bleurt_scores

    if plagiarized_sources and (input_emails or input_phones):
      logger.info("Checking for same person by contact information...")
      sources_to_remove = set()

      for source_file in plagiarized_sources:
        possible_paths = [
          Path(REFERENCE_FOLDER) / source_file if REFERENCE_FOLDER else Path(source_file),
          Path(INPUT_FOLDER) / source_file if INPUT_FOLDER else Path(source_file),
          Path(".") / source_file
        ]
        for path in possible_paths:
          if path.exists():
            try:
              source_text = extract_text_from_file(path)
              if source_text:
                src_emails = set(extract_emails(source_text, path))
                src_phones = set(extract_phones(source_text, region="IN", fallback_path=path))
                source_contacts[source_file] = {
                  'emails': list(src_emails),
                  'phones': list(src_phones)
                }
                email_overlap = bool(set(input_emails) & src_emails)
                phone_overlap = bool(set(input_phones) & src_phones)
                if email_overlap or phone_overlap:
                  logger.info(f"Same person detected: {filename} and {source_file} share contact info")
                  sources_to_remove.add(source_file)
                break
            except Exception as e:
              logger.error(f"Error comparing contact info for {source_file}: {e}")
              continue

      if sources_to_remove:
        logger.info(f"Removing {len(sources_to_remove)} sources: {sources_to_remove}")
        plagiarized_sources -= sources_to_remove
        matched_chunks = [chunk for chunk in matched_chunks if chunk['source_url'] not in sources_to_remove]
        all_scores = [chunk['similarity_score']/100 for chunk in matched_chunks]
        # Update avg_bleurt_scores to exclude removed sources
        avg_bleurt_scores = []
        for source_file, scores in source_bleurt_scores.items():
          if source_file not in sources_to_remove:
            avg_score = round(sum(scores) / len(scores), 2) if scores else 0.0
            avg_bleurt_scores.extend([avg_score, source_file])
        results['bleurt_scores'] = avg_bleurt_scores

    if all_scores:
      results['flagged_score'] = round((sum(all_scores)/len(all_scores)*100), 2)
    else:
      results['flagged_score'] = 0.0
    if max_scores_per_chunk:
      results['overall_score'] = round((sum(max_scores_per_chunk)/len(max_scores_per_chunk)*100), 2)
    else:
      results['overall_score'] = 0.0

    results['flagged_chunks'] = len(all_scores)
    results['plagiarized_sources'] = list(plagiarized_sources)
    results['matched_chunks'] = matched_chunks

    # Use RF classifier if available, otherwise fallback to rules
    global rf_classifier
    
    if rf_classifier is not None and chunk_features_for_rf:
        try:
            logger.debug(f"Using RF with {len(chunk_features_for_rf)} features")
            
            # Build source map from collected features
            rf_source_map: Dict[str, int] = {}
            for feat in chunk_features_for_rf:
                src = feat['source_file']
                rf_source_map[src] = rf_source_map.get(src, 0) + 1
            
            # Get RF features and predict
            features = rf_classifier.extract_features(
                chunk_features_for_rf,
                total_chunks=len(texts),
                source_hit_map=rf_source_map
            )
            
            is_plagiarized, level, confidence = rf_classifier.predict(features)
            

            if level == "LOW":
                is_plagiarized = False

            if level == "HIGH" and len(results['matched_chunks']) < 3:
                level = "MODERATE"      

            if is_plagiarized and len(results['matched_chunks']) <= 1:
                is_plagiarized = False # Silence the alarm for single matches
                level = "CLEAN (Minor Overlap)"
                
            if confidence < 0.6:
                is_plagiarized = False
                level = "CLEAN"
            
            results['is_plagiarized'] = is_plagiarized
            results['plagiarism_level'] = level
            results['rf_confidence'] = confidence
            results['classification_method'] = 'random_forest'
            
            # Update sources based on RF decision
            if is_plagiarized:
                strong_sources = set()
                for feat in chunk_features_for_rf:
                    src = feat['source_file']
                    if (feat['composite'] > 0.50 or 
                        feat['cosine'] > 0.75 or
                        rf_source_map[src] >= 2):
                        strong_sources.add(src)
                results['plagiarized_sources'] = list(strong_sources)
            else:
                results['plagiarized_sources'] = []
            
            logger.info(f"RF: {level} (conf: {confidence:.3f})")
            
        except Exception as e:
            logger.error(f"RF failed: {e}, using fallback")
            is_plagiarized, level = classify_plagiarism_level(
                results['overall_score'], results['flagged_score'],
                list(plagiarized_sources), results['flagged_chunks']
            )
            results['is_plagiarized'] = is_plagiarized
            results['plagiarism_level'] = level
            results['rf_confidence'] = None
            results['classification_method'] = 'rule_based_fallback'
    else:
        # Fallback to rule-based
        is_plagiarized, level = classify_plagiarism_level(
            results['overall_score'], results['flagged_score'],
            list(plagiarized_sources), results['flagged_chunks']
        )
        results['is_plagiarized'] = is_plagiarized
        results['plagiarism_level'] = level
        results['rf_confidence'] = None
        results['classification_method'] = 'rule_based'

    if is_plagiarized and results['plagiarized_sources']:
      print(f"Extracting contact info from {len(results['plagiarized_sources'])} source files")
      source_phones = []
      source_emails = []
      for source_file in results['plagiarized_sources']:
        if source_file in source_contacts:
          src_emails = source_contacts[source_file]['emails']
          src_phones = source_contacts[source_file]['phones']
          source_emails.extend(src_emails)
          source_phones.extend(src_phones)
          print(f" {source_file}: {len(src_emails)} emails, {len(src_phones)} phones")
        else:
          possible_paths = [Path(REFERENCE_FOLDER) / source_file if REFERENCE_FOLDER else Path(source_file), 
                   Path(INPUT_FOLDER) / source_file if INPUT_FOLDER else Path(source_file), 
                   Path(".") / source_file]
          source_found = False
          for path in possible_paths:
            if path.exists():
              try:
                source_text = extract_text_from_file(path)
                if source_text:
                  src_emails = extract_emails(source_text, path)
                  src_phones = extract_phones(source_text, region="IN", fallback_path=path)
                  source_emails.extend(src_emails)
                  source_phones.extend(src_phones)
                  print(f" {source_file}: {len(src_emails)} emails, {len(src_phones)} phones")
                  source_found = True
                  break
              except Exception as e:
                print(f"Failed to extract from {source_file}: {e}")
          if not source_found:
            print(f"Source file not found: {source_file}")

      results['source_emails'] = list(set(source_emails))
      results['source_phones'] = list(set(source_phones))
      if source_emails or source_phones:
        logger.info(f"Total from sources: {len(results['source_emails'])} unique emails, {len(results['source_phones'])} unique phones")

  except Exception as e:
    results['error'] = f"Analysis failed: {e}"
    results['status'] = 'error'
    logger.error(f"Analysis error for {filename}: {e}")

  return results


def load_existing_results(output_folder: Path) -> Dict[str, Dict]:
  global processed_files_cache
  json_path = output_folder / "plagiarism_results.json"
  existing_results: Dict[str, Dict] = {}
  try:
    if json_path.exists():
      with open(json_path, 'r', encoding='utf-8') as f:
        results_list = json.load(f)
      for result in results_list:
        filename = result.get('filename')
        if filename:
          existing_results[filename] = result
          file_mtime = result.get('file_mtime')
          if file_mtime:
            processed_files_cache[filename] = file_mtime
      logger.info(f"Loaded {len(existing_results)} existing results")
    else:
      logger.info("No existing results found - starting fresh analysis")
  except Exception as e:
    logger.error(f"Failed to load existing results: {e}")
    logger.info("Starting fresh analysis")
  return existing_results


def should_skip_file(filename: str, file_path: Path) -> bool:
  global processed_files_cache
  if filename not in processed_files_cache:
    return False
  try:
    if not file_path.exists():
      del processed_files_cache[filename]
      return False
    current_mtime = file_path.stat().st_mtime
    cached_mtime = processed_files_cache[filename]
    if abs(current_mtime - cached_mtime) > 1:
      processed_files_cache[filename] = current_mtime
      return False
    return True
  except Exception:
    return False


def save_json_results(results: List[Dict], output_folder: Path) -> None:
  json_path = output_folder / "plagiarism_results.json"
  json_results = []
  for result in results:
    json_result = result.copy()
    if isinstance(json_result.get('plagiarized_sources'), set):
      json_result['plagiarized_sources'] = list(json_result['plagiarized_sources'])
    json_results.append(json_result)
  with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(json_results, f, indent=4, ensure_ascii=False)


def print_save_summary(output_folder: Path) -> None:
  json_path = output_folder / "plagiarism_results.json"
  logger.info(f"Results saved to {output_folder}")
  logger.info(f" - JSON data: {json_path}")


def save_results(results: List[Dict], output_folder: str) -> None:
  try:
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    save_json_results(results, output_path)
    print_save_summary(output_path)
  except PermissionError as e:
    logger.error(f"Permission error while saving results: {e}")
  except OSError as e:
    logger.error(f"IO error while saving results: {e}")
  except Exception as e:
    logger.error(f"Failed to save results: {e}")


def process_folder(input_folder: str, output_folder: str) -> List[Dict]:
  global processed_files_cache, INPUT_FOLDER
  input_path = Path(input_folder)
  output_path = Path(output_folder)
  INPUT_FOLDER = input_folder
  logger.info(f"Starting batch processing of folder: {input_path}")

  existing_results = load_existing_results(output_path)
  resume_files = get_resume_files(input_path)
  if not resume_files:
    logger.info("No supported files found in input folder")
    return list(existing_results.values()) if existing_results else []

  new_files = []
  skipped_files = []
  for file_path in resume_files:
    if should_skip_file(file_path.name, file_path):
      skipped_files.append(file_path)
    else:
      new_files.append(file_path)

  logger.info(f"Analysis: {len(resume_files)} total files -> {len(new_files)} new, {len(skipped_files)} skipped")

  valid_existing_results = [
    result for result in existing_results.values()
    if Path(input_path / result['filename']).exists()
  ]
  all_results = valid_existing_results

  if new_files:
    logger.info(f"Processing {len(new_files)} files")
    for i, file_path in enumerate(new_files, 1):
      logger.info(f" [{i}/{len(new_files)}] {file_path.name}")
      try:
        text = extract_text_from_file(file_path)
        result = analyze_document(text, file_path.name, file_path)
        all_results.append(result)

        if result['status'] == 'success' and result.get('file_mtime'):
          processed_files_cache[file_path.name] = result['file_mtime']

        if result['status'] == 'success':
          status = result.get('plagiarism_level', "PLAGIARIZED" if result['is_plagiarized'] else "CLEAN")
          logger.info(f"{status} ({result['overall_score']:.2f}%)")
          if result.get('plagiarized_sources'):
            logger.info(f" Sources: {', '.join(result['plagiarized_sources'])}")
        else:
          logger.error(f"Failed: {result.get('error', 'Unknown error')}")
      except Exception as e:
        logger.error(f"Unexpected error: {e}")
        result = {
          'filename': file_path.name,
          'status': 'error',
          'error': f'Unexpected error: {e}',
          'overall_score': 0.0,
          'flagged_score': 0.0,
          'is_plagiarized': False,
          'plagiarized_sources': [],
          'processed_at': None
        }
        all_results.append(result)

  save_results(all_results, output_folder)
  print_summary(all_results, len(new_files), len(skipped_files))
  return all_results


def print_summary(results: List[Dict], new_files_processed: int = 0, skipped_files: int = 0) -> None:
  logger.info("\n" + "="*80)
  logger.info("PLAGIARISM DETECTION SUMMARY")
  logger.info("="*80)
  total_files = len(results)
  successful = [r for r in results if r['status'] == 'success']
  failed = [r for r in results if r['status'] == 'error']
  plagiarized = [r for r in successful if r['is_plagiarized']]

  logger.info(f"Total files in results: {total_files}")
  logger.info(f"New files processed: {new_files_processed}")
  logger.info(f"Files skipped: {skipped_files}")
  logger.info(f"Successful analyses: {len(successful)}")
  logger.info(f"Failed analyses: {len(failed)}")
  logger.info(f"Files flagged as plagiarized: {len(plagiarized)}")

  if successful:
    scores = [r['overall_score'] for r in successful]
    logger.info(f"Average similarity score: {sum(scores)/len(scores):.2f}%")
    logger.info(f"Highest similarity score: {max(scores):.2f}%")
    logger.info(f"Lowest similarity score: {min(scores):.2f}%")

  if plagiarized:
    logger.info(f"\nPLAGIARIZED FILES:")
    for result in plagiarized:
      sources_info = f" -> Sources: {', '.join(result.get('plagiarized_sources', []))}" if result.get('plagiarized_sources') else ""
      logger.info(f" > {result['filename']} ({result['overall_score']:.2f}% similarity){sources_info}")

  if failed:
    logger.info(f"\nFAILED ANALYSES:")
    for result in failed:
      logger.info(f" > {result['filename']}: {result.get('error', 'Unknown error')}")


def main() -> None:
  global INPUT_FOLDER, OUTPUT_FOLDER, COLLECTION_NAME, REFERENCE_FOLDER
  try:
    logger.info("Initializing Plagiarism Detector")
    INPUT_FOLDER = input("Enter the input folder path: ").strip()
    OUTPUT_FOLDER = input("Enter the output folder path (default: 'results'): ").strip() or 'results'
    REFERENCE_FOLDER = input("Enter the reference folder path (default: 'resumes'): ").strip() or 'resumes'
    COLLECTION_NAME = input("Enter the Qdrant collection name (default: 'database'): ").strip() or 'database'

    # Validate inputs
    if not INPUT_FOLDER:
      raise ValueError("Input folder path cannot be empty")
    input_path = Path(INPUT_FOLDER)
    if not input_path.exists() or not input_path.is_dir():
      raise ValueError(f"Input folder does not exist or is not a directory: {INPUT_FOLDER}")
    if not REFERENCE_FOLDER:
      raise ValueError("Reference folder path cannot be empty")
    reference_path = Path(REFERENCE_FOLDER)
    if not reference_path.exists() or not reference_path.is_dir():
      raise ValueError(f"Reference folder does not exist or is not a directory: {REFERENCE_FOLDER}")
    if not COLLECTION_NAME:
      raise ValueError("Qdrant collection name cannot be empty")

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    
    if not qdrant_url or not qdrant_api_key:
      raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in environment")
    
    init(qdrant_url, qdrant_api_key, COLLECTION_NAME)
    process_folder(INPUT_FOLDER, OUTPUT_FOLDER)
    logger.info(f"Processing complete! Results saved to '{OUTPUT_FOLDER}' folder")
  except ValueError as ve:
    logger.error(f"Input error: {ve}")
    print("Please provide valid inputs and try again.")
  except Exception as e:
    logger.error(f"Critical error: {e}")
    print("Check the application logs for detailed error information")


if __name__ == "__main__":
  main() 
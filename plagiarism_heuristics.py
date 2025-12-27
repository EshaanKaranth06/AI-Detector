import re
import torch
import numpy as np
from typing import List, Optional
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz.distance import Levenshtein
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from common_utils import get_logger, tfidf_vectorizer

logger = get_logger("logs/plag_heuristics.log")

_bleurt_model: Optional[AutoModelForSequenceClassification] = None
_bleurt_tokenizer: Optional[AutoTokenizer] = None
_device: Optional[str] = None

def initialize_bleurt() -> None:
  global _bleurt_model, _bleurt_tokenizer, _device
  try:
    if _bleurt_model is None:
      _device = "cuda" if torch.cuda.is_available() else "cpu"
      logger.info(f"Loading BLEURT model on {_device}")
      _bleurt_tokenizer = AutoTokenizer.from_pretrained("Elron/bleurt-base-512")
      model = AutoModelForSequenceClassification.from_pretrained("Elron/bleurt-base-512")
      model = model.to(_device) # type: ignore
      model.eval()
      _bleurt_model = model
      logger.info("BLEURT model loaded successfully")
  except Exception as e:
    logger.error(f"Failed to initialize BLEURT: {e}")
    raise RuntimeError("BLEURT initialization failed")

def preprocess_text(text: str) -> str:
  try:
    if not isinstance(text, str) or not text.strip():
      logger.debug("Empty or invalid text input")
      return ""
    text = re.sub(r'\s+', ' ', text.lower().strip())
    text = re.sub(r'[^\w\s]', '', text)
    return text
  except Exception as e:
    logger.error(f"Error preprocessing text: {e}")
    return ""

def generate_ngrams(text: str, n: int) -> List[str]:
  try:
    words = preprocess_text(text).split()
    if len(words) < n:
      return []
    return [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]
  except Exception as e:
    logger.error(f"Error generating n-grams: {e}")
    return []

def ngram_similarity(text1: str, text2: str, n: int = 3) -> float:
  try:
    if not text1.strip() or not text2.strip():
      logger.debug("One or both texts are empty")
      return 0.0
    ngrams1 = generate_ngrams(text1, n)
    ngrams2 = generate_ngrams(text2, n)
    if not ngrams1 or not ngrams2:
      logger.debug("No n-grams generated")
      return 0.0
    counter1 = Counter(ngrams1)
    counter2 = Counter(ngrams2)
    intersection = sum((counter1 & counter2).values())
    union = sum((counter1 | counter2).values())
    return intersection / union if union > 0 else 0.0
  except Exception as e:
    logger.error(f"Error in n-gram similarity: {e}")
    return 0.0

def combined_ngram_similarity(text1: str, text2: str, ngram_sizes: List[int] = [2, 3, 4]) -> float:
  try:
    weights = [0.3, 0.5, 0.2]
    similarities = [ngram_similarity(text1, text2, n) * w for n, w in zip(ngram_sizes, weights)]
    return sum(similarities)
  except Exception as e:
    logger.error(f"Error in combined n-gram similarity: {e}")
    return 0.0

def tfidf_similarity(text1: str, text2: str) -> float:
  try:
    if not text1.strip() or not text2.strip():
      logger.debug("One or both texts are empty")
      return 0.0
    if tfidf_vectorizer is None:
      logger.error("TF-IDF vectorizer is not initialized")
      return 0.0
    tfidf_matrix = tfidf_vectorizer.fit_transform([text1, text2])
    # Convert sparse matrix slices to array for proper indexing
    # Use .toarray() to convert sparse matrix slices before passing to cosine_similarity
    vec1 = tfidf_matrix[0:1].toarray() # type: ignore
    vec2 = tfidf_matrix[1:2].toarray() # type: ignore
    similarity_result = cosine_similarity(vec1, vec2)
    # similarity_result is a numpy array, safe to index
    return float(similarity_result[0, 0])
  except Exception as e:
    logger.error(f"Error in TF-IDF similarity: {e}")
    return 0.0

def fuzzy_similarity(text1: str, text2: str) -> float:
  try:
    if not text1.strip() or not text2.strip():
      logger.debug("One or both texts are empty")
      return 0.0
    max_len = max(len(text1), len(text2))
    if max_len == 0:
      return 0.0
    distance = Levenshtein.distance(text1, text2, weights=(1, 1, 1))
    return 1.0 - (distance / max_len)
  except Exception as e:
    logger.error(f"Error in fuzzy similarity: {e}")
    return 0.0

def normalized_bleurt_score(text1: str, text2: str) -> float:
  global _bleurt_model, _bleurt_tokenizer, _device
  try:
    text1 = text1.strip() if isinstance(text1, str) else ""
    text2 = text2.strip() if isinstance(text2, str) else ""
    if not text1 or not text2:
      logger.debug("One or both texts are empty")
      return 0.0

    initialize_bleurt()
    
    # Check if initialization was successful
    if _bleurt_tokenizer is None or _bleurt_model is None or _device is None:
      logger.error("BLEURT model or tokenizer not initialized")
      return 0.0
    
    inputs = _bleurt_tokenizer( # type: ignore
      text1,
      text2,
      return_tensors="pt",
      max_length=512,
      truncation=True,
      padding=True
    )
    # Type guard: we know inputs is a dict-like object with tensor values
    inputs = {k: v.to(_device) for k, v in inputs.items()} # type: ignore

    with torch.no_grad():
      outputs = _bleurt_model(**inputs) # type: ignore
      # outputs.logits is a tensor, we can safely call .squeeze().cpu().item()
      raw_score = outputs.logits.squeeze().cpu().item() # type: ignore

    normalized_score = (raw_score + 1) / 2
    return float(normalized_score)
  except Exception as e:
    logger.error(f"Error calculating BLEURT score: {e}")
    return 0.0 

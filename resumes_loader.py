from common_utils import (get_logger, validate_environment, load_json_set, save_json_set,
                          safe_upsert_with_retry, ocr_pdf, unstructured_to_markdown,validate_folder,
                          convert_doc_to_pdf, init, SUPPORTED_EXTENSIONS, COLLECTION_NAME)
import os
import time
import json
from qdrant_client.http.models import PointStruct
from pathlib import Path
from typing import Set

logger = get_logger("logs/resumes_loader.log")

def delete_with_retry(file_path: Path, max_retries: int = 3, delay: float = 1.0) -> bool:
    for attempt in range(max_retries):
        try:
            file_path.unlink()
            logger.info(f"Removed intermediate PDF: {file_path}")
            return True
        except PermissionError as e:
            logger.warning(f"PermissionError deleting {file_path} (attempt {attempt + 1}): {e}. Retrying in {delay}s...")
            time.sleep(delay)
        except OSError as e:
            logger.error(f"Failed to delete {file_path}: {e}")
            return False
    logger.error(f"Failed to delete {file_path} after {max_retries} attempts")
    return False

def process_resume(
    file_path: Path,
    qdrant_client, 
    model, 
    splitter, 
    processed_files: Set[str],
    failed_files: Set[str],
    id_counter: int, 
    collection_name: str
) -> int:
    filename = file_path.name
    file_ext = file_path.suffix.lower()
    was_converted = False
    pdf_path = None

    if filename in processed_files:
        logger.info(f"Skipping already processed file: {filename}")
        print(f"Already processed file: {filename}")
        return id_counter

    logger.info(f"Processing {filename}...")
    print(f"Processing {filename}...") 

    processing_path = file_path
    if file_ext in SUPPORTED_EXTENSIONS and file_ext != ".pdf":
        logger.info(f"Converting {filename} to PDF...")
        pdf_path = convert_doc_to_pdf(file_path)
        if pdf_path and pdf_path.exists():
            processing_path = pdf_path
            was_converted = True
        else:
            logger.info(f"LibreOffice conversion failed for {filename}. Using unstructured fallback.")
            md_text = unstructured_to_markdown(file_path)
            if not md_text.strip():
                logger.info(f"Unstructured fallback failed for {filename}. Trying OCR...")
                md_text = ocr_pdf(file_path)
            if not md_text.strip():
                logger.error(f"All extraction methods failed for {filename}")
                failed_files.add(filename)
                return id_counter
    elif file_ext != ".pdf":
        logger.warning(f"Unsupported file type: {filename}")
        failed_files.add(filename)
        return id_counter

    md_text = ""
    try:
        import pymupdf4llm
        md_text = str(pymupdf4llm.to_markdown(str(processing_path))).lower()
        if not md_text.strip():
            logger.info(f"Empty markdown from pymupdf4llm for {filename}. Falling back to unstructured.")
            md_text = unstructured_to_markdown(processing_path)
    except Exception as e:
        logger.error(f"pymupdf4llm extraction failed for {filename}: {e}")
        md_text = unstructured_to_markdown(processing_path)

    if not md_text.strip():
        logger.info(f"Trying OCR fallback for {filename}...")
        md_text = ocr_pdf(processing_path)
        if not md_text.strip():
            logger.error(f"All extraction methods failed for {filename}")
            failed_files.add(filename)
            if was_converted and pdf_path and pdf_path.exists():
                delete_with_retry(pdf_path)
            return id_counter

    if was_converted and pdf_path and pdf_path.exists():
        delete_with_retry(pdf_path)

    try:
        docs = splitter.create_documents([md_text])
        texts = [doc.page_content for doc in docs]
        if not texts:
            logger.error(f"No text chunks created for {filename}")
            failed_files.add(filename)
            return id_counter

        # ✅ ADD E5 INSTRUCTION PREFIX FOR REFERENCE DOCUMENTS
        logger.info(f"Encoding {len(texts)} chunks with 'passage:' prefix for {filename}")
        texts_with_instruction = [f"passage: {text}" for text in texts]
        embeddings = model.encode(texts_with_instruction, show_progress_bar=True)

        # ✅ STORE ORIGINAL TEXT (without prefix) IN PAYLOAD
        points_to_upload = [
            PointStruct(
                id=id_counter + i,
                vector=emb.tolist(),
                payload={"text": texts[i], "source_file": filename}  # Original text without prefix
            )
            for i, emb in enumerate(embeddings)
        ]

        if points_to_upload and safe_upsert_with_retry(qdrant_client, collection_name, points_to_upload):
            processed_files.add(filename)
            logger.info(f"Uploaded {len(points_to_upload)} vectors from {filename}")
            print(f"Uploaded {len(points_to_upload)} vectors from {filename}")
            id_counter += len(points_to_upload)
        else:
            logger.error(f"Skipped saving {filename} due to upload failure")
            failed_files.add(filename)

    except Exception as e:
        logger.error(f"Processing failed for {filename}: {e}")
        failed_files.add(filename)

    return id_counter

def main():
    try:
        validate_environment()
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        # Check for None values
        if not qdrant_url or not qdrant_api_key:
            raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in environment")
        
        init(qdrant_url, qdrant_api_key, COLLECTION_NAME)
        
        from common_utils import qdrant, model, splitter
        
        # Check if components are initialized
        if qdrant is None or model is None or splitter is None:
            raise ValueError("Failed to initialize required components")
        
        resumes_folder = input("Enter the path to the resumes folder: ").strip()
        resumes_path = validate_folder(resumes_folder, folder_type="input")
        
        processed_files_tracker = Path("processing/processed_files.json")
        failed_files_tracker = Path("processing/failed_files.json")
        count_tracker = Path("processing/count.json")

        # Initialize sets for this run
        processed_files: Set[str] = load_json_set(processed_files_tracker)
        failed_files: Set[str] = set()
        logger.info(f"Loaded {len(processed_files)} previously processed files")
        logger.info(f"Starting with empty failed_files set for this run")

        try:
            stats = qdrant.get_collection(COLLECTION_NAME)
            # Use getattr with a default value to handle missing attribute
            id_counter = getattr(stats, 'vectors_count', None) or 0
            logger.info(f"Starting id_counter at {id_counter}")
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}. Starting ID counter at 0.")
            id_counter = 0

        # Create processing directory if it doesn't exist
        Path("processing").mkdir(parents=True, exist_ok=True)
        
        for filename in os.listdir(resumes_path):
            file_path = resumes_path / filename
            if not file_path.is_file():
                logger.warning(f"Skipping non-file entry: {filename}")
                continue
            id_counter = process_resume(
                file_path, qdrant, model, splitter, processed_files, failed_files, id_counter, COLLECTION_NAME
            )

        logger.info(f"Processed files in this run: {processed_files}")
        logger.info(f"Failed files in this run: {failed_files}")

        save_json_set(processed_files, processed_files_tracker)
        save_json_set(failed_files, failed_files_tracker)

        resume_count = len(processed_files)
        with open(count_tracker, "w", encoding='utf-8') as f:
            json.dump({"count": resume_count}, f, indent=2)
        logger.info(f"Updated resume count: {resume_count} in {count_tracker}")

    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        raise

if __name__ == "__main__":
    main()
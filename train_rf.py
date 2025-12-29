"""
TRAINING SCRIPT FOR RANDOM FOREST CLASSIFIER
=============================================

This script trains the RF model using your existing plagiarism results in the damn json file.

"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import logging
from rf_classifier import PlagiarismRFClassifier


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_and_prepare_training_data(
    results_json_path: Path,
    use_existing_labels: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[str], List[Dict]]:
    """
    Load results JSON and convert to training data.
    
    Args:
        results_json_path: Path to plagiarism_results.json
        use_existing_labels: If True, use is_plagiarized from results as labels
                            If False, you'll need to manually label
    
    Returns:
        X: Feature matrix
        y: Labels (0=clean, 1=plagiarized)
        filenames: List of filenames
        raw_results: Original results for inspection
    """
    logger.info(f"Loading data from {results_json_path}")
    
    with open(results_json_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    X_list = []
    y_list = []
    filenames = []
    valid_results = []
    
    classifier = PlagiarismRFClassifier()  # For feature extraction
    
    for result in results:
        # Skip failed analyses
        if result.get('status') != 'success':
            logger.debug(f"Skipping {result.get('filename')}: {result.get('status')}")
            continue
        
        # Need matched chunks with scores
        matched_chunks = result.get('matched_chunks', [])
        if not matched_chunks:
            if not result.get('is_plagiarized', False):
                # Create zero feature vector for clean samples
                features = {name: 0.0 for name in classifier.feature_names}
                feature_array = [features[name] for name in classifier.feature_names]
                
                X_list.append(feature_array)
                y_list.append(0)  # Label as clean
                filenames.append(result['filename'])
                valid_results.append(result)
            continue
        
        # Check if new fields exist (from our modifications)
        if not all(k in matched_chunks[0] for k in ['cosine_score', 'tfidf_score', 'ngram_score', 'fuzzy_score']):
            logger.warning(f"  {result.get('filename')}: Missing new score fields!")
            logger.warning("    You need to re-run detector.py with the modified code first")
            continue
        
        # Build feature list
        chunk_features = []
        for chunk in matched_chunks:
            chunk_features.append({
                "cosine": float(chunk.get("cosine_score", 0.0)),
                "tfidf": float(chunk.get("tfidf_score", 0.0)),
                "ngram": float(chunk.get("ngram_score", 0.0)),
                "fuzzy": float(chunk.get("fuzzy_score", 0.0)),
                "bleurt": float(chunk.get("bleurt_score", 0.0)),
                "composite": float(chunk.get("similarity_score", 0.0)) / 100.0,
            })
        
        # Build source hit map
        source_hit_map: Dict[str, int] = {}
        for chunk in matched_chunks:
            source = chunk.get("source_url", "unknown")
            source_hit_map[source] = source_hit_map.get(source, 0) + 1
        
        # Extract RF features
        features = classifier.extract_features(
            chunk_features,
            result.get('total_chunks', 1),
            source_hit_map
        )
        
        # Convert to array
        feature_array = [features[name] for name in classifier.feature_names]
        X_list.append(feature_array)
        
        # Get label
        if use_existing_labels:
            # Use existing system's decision as label
            label = 1 if result.get('is_plagiarized', False) else 0
        else:
            # You'll need to manually label
            label = result.get('ground_truth_label', None)
            if label is None:
                logger.warning(f"No ground truth label for {result.get('filename')}")
                continue
        
        y_list.append(label)
        filenames.append(result['filename'])
        valid_results.append(result)
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    logger.info(f"Loaded {len(X)} training samples")
    logger.info(f"Class distribution: Clean={np.sum(y==0)}, Plagiarized={np.sum(y==1)}")
    
    return X, y, filenames, valid_results


def analyze_training_data(X: np.ndarray, y: np.ndarray, filenames: List[str]) -> None:
    """Print statistics about the training data."""
    logger.info("\n" + "="*70)
    logger.info("TRAINING DATA ANALYSIS")
    logger.info("="*70)
    
    total = len(y)
    plagiarized = np.sum(y == 1)
    clean = np.sum(y == 0)
    
    logger.info(f"Total samples: {total}")
    logger.info(f"Plagiarized: {plagiarized} ({plagiarized/total*100:.1f}%)")
    logger.info(f"Clean: {clean} ({clean/total*100:.1f}%)")
    
    if plagiarized < 10 or clean < 10:
        logger.warning(" Warning: Very few samples in one class!")
        logger.warning("  Consider getting more training data for better results")
    
    # Feature statistics
    feature_names = PlagiarismRFClassifier().feature_names
    logger.info("\nFeature statistics:")
    for i, name in enumerate(feature_names[:5]):  # Show first 5
        col = X[:, i]
        logger.info(f"  {name:20s}: mean={col.mean():.3f}, std={col.std():.3f}")


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray
) -> PlagiarismRFClassifier:
    """Train and evaluate the RF classifier."""
    logger.info("\n" + "="*70)
    logger.info("TRAINING RANDOM FOREST")
    logger.info("="*70)
    
    # Initialize classifier
    classifier = PlagiarismRFClassifier()
    
    # Train
    logger.info("Training model...")
    metrics = classifier.train(X_train, y_train, X_val, y_val)
    
    # Print results
    logger.info("\n" + "="*70)
    logger.info("TRAINING RESULTS")
    logger.info("="*70)
    logger.info(f"Training Accuracy:   {metrics['train_accuracy']:.3f}")
    logger.info(f"Training Precision:  {metrics['train_precision']:.3f}")
    logger.info(f"Training Recall:     {metrics['train_recall']:.3f}")
    logger.info(f"Training F1:         {metrics['train_f1']:.3f}")
    
    if 'val_accuracy' in metrics:
        logger.info(f"\nValidation Accuracy:  {metrics['val_accuracy']:.3f}")
        logger.info(f"Validation Precision: {metrics['val_precision']:.3f}")
        logger.info(f"Validation Recall:    {metrics['val_recall']:.3f}")
        logger.info(f"Validation F1:        {metrics['val_f1']:.3f}")
    
    # Feature importance
    logger.info("\nTop 10 Most Important Features:")
    feature_imp = sorted(
        metrics['feature_importance'].items(),
        key=lambda x: x[1],
        reverse=True
    )
    for i, (name, importance) in enumerate(feature_imp[:10], 1):
        logger.info(f"  {i:2d}. {name:25s}: {importance:.4f}")
    
    return classifier


def cross_validate_model(X: np.ndarray, y: np.ndarray) -> None:
    """Perform cross-validation to check model stability."""
    logger.info("\n" + "="*70)
    logger.info("CROSS-VALIDATION (5-Fold)")
    logger.info("="*70)
    
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    scores = cross_val_score(clf, X, y, cv=5, scoring='f1')
    logger.info(f"F1 Scores: {[f'{s:.3f}' for s in scores]}")
    logger.info(f"Mean F1: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
    
    if scores.std() > 0.15:
        logger.warning("  High variance in cross-validation!")
        logger.warning("   Model may not generalize well - consider more data")


def manual_validation(
    classifier: PlagiarismRFClassifier,
    X_val: np.ndarray,
    y_val: np.ndarray,
    filenames_val: List[str],
    results_val: List[Dict]
) -> None:
    """Show predictions vs ground truth for manual inspection."""
    logger.info("\n" + "="*70)
    logger.info("SAMPLE PREDICTIONS (First 10)")
    logger.info("="*70)
    
    predictions = classifier.model.predict(X_val)
    probabilities = classifier.model.predict_proba(X_val)
    
    for i in range(min(10, len(X_val))):
        filename = filenames_val[i]
        true_label = "PLAGIARIZED" if y_val[i] == 1 else "CLEAN"
        pred_label = "PLAGIARIZED" if predictions[i] == 1 else "CLEAN"
                # Handle single-class case
        if probabilities.shape[1] == 1:
            confidence = probabilities[i][0]
        else:
            confidence = probabilities[i][1]
        
        match = "✓" if predictions[i] == y_val[i] else "✗"
        
        logger.info(f"\n{match} {filename}")
        logger.info(f"   True: {true_label:12s} | Pred: {pred_label:12s} (conf: {confidence:.3f})")
        
        # Show why it was classified this way
        result = results_val[i]
        logger.info(f"   Chunks: {result.get('flagged_chunks', 0)}/{result.get('total_chunks', 0)}")
        logger.info(f"   Sources: {len(result.get('plagiarized_sources', []))}")


def main():
    """Main training workflow."""
    print("\n" + "="*70)
    print("RANDOM FOREST CLASSIFIER TRAINING")
    print("="*70 + "\n")
    
    # Step 1: Get input path
    results_path = input("Enter path to plagiarism_results.json (or press Enter for 'results/plagiarism_results.json'): ").strip()
    if not results_path:
        results_path = "results/plagiarism_results.json"
    
    results_path = Path(results_path)
    
    if not results_path.exists():
        logger.error(f" File not found: {results_path}")
        logger.error("   Please run your detector first to generate results")
        return
    
    # Step 2: Load data
    try:
        X, y, filenames, raw_results = load_and_prepare_training_data(
            results_path,
            use_existing_labels=True
        )
    except Exception as e:
        logger.error(f" Failed to load training data: {e}")
        logger.error("   Make sure you've re-run detector.py with the modified code")
        return
    
    if len(X) == 0:
        logger.error(" No valid training samples found!")
        logger.error("   Check that your results have the new score fields")
        return
    
    # Check minimum samples
    if len(X) < 20:
        logger.warning("  Warning: Very few training samples (<20)")
        logger.warning("   RF needs more data for good performance")
        proceed = input("   Continue anyway? (y/n): ").strip().lower()
        if proceed != 'y':
            logger.info("Training cancelled")
            return
    
    # Step 3: Analyze data
    analyze_training_data(X, y, filenames)
    
    # Step 4: Split data
    logger.info("\nSplitting data: 80% train, 20% validation")
    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        X, y, np.arange(len(X)),
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    filenames_val = [filenames[i] for i in idx_val]
    results_val = [raw_results[i] for i in idx_val]
    
    # Step 5: Cross-validation (optional but recommended)
    if len(X) >= 30:
        cross_validate_model(X, y)
    
    # Step 6: Train model
    classifier = train_random_forest(X_train, y_train, X_val, y_val)
    
    # Step 7: Manual validation
    manual_validation(classifier, X_val, y_val, filenames_val, results_val)
    
    # Step 8: Save model
    logger.info("\n" + "="*70)
    model_path = Path("models/plagiarism_rf.pkl")
    model_path.parent.mkdir(exist_ok=True)
    
    classifier.save_model(model_path)
    logger.info(f" Model saved to: {model_path}")
    logger.info("="*70)
    
    # Step 9: Next steps
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("1. Review the sample predictions above")
    print("2. If accuracy looks good, you're ready to use RF!")
    print("3. Run detector.py - it will automatically use the trained model")
    print("4. Monitor results and retrain if needed with more data")
    print("\nTo retrain later:")
    print("  - Run detector on more files")
    print("  - Manually review and fix any errors in results JSON")
    print("  - Run this training script again")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
"""
Random Forest classifier for plagiarism detection.

"""

import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import logging
from sklearn.calibration import CalibratedClassifierCV

logger = logging.getLogger(__name__)


class PlagiarismRFClassifier:
    """
    Random Forest classifier that replaces hard-coded thresholds.
    Uses multi-signal features to detect plagiarism patterns.
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize classifier. Load pre-trained model if path provided.
        
        Args:
            model_path: Path to saved RF model (pickle file)
        """
        self.model: Optional[RandomForestClassifier] = None
        self.feature_names: List[str] = [
            # Cosine similarity features
            "max_cosine",
            "mean_cosine",
            "std_cosine",
            
            # BLEURT semantic features
            "max_bleurt",
            "mean_bleurt",
            "std_bleurt",
            
            # N-gram literal overlap
            "max_ngram",
            "mean_ngram",
            "std_ngram",
            
            # TF-IDF features
            "max_tfidf",
            "mean_tfidf",
            
            # Fuzzy matching
            "max_fuzzy",
            "mean_fuzzy",
            
            # Structural patterns
            "chunk_hit_ratio",      # % of input chunks that matched
            "chunk_count",          # absolute number of matches
            "source_concentration", # entropy of matches across sources
            
            # Composite scores
            "max_composite",
            "mean_composite",
        ]
        
        if model_path and model_path.exists():
            self.load_model(model_path)
        else:
            base_rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
    
            # ✅ Calibrate to get realistic confidence scores
            self.model = CalibratedClassifierCV(
                base_rf,
                method="isotonic",
                cv=3  # 3-fold cross-validation
            )
    
    def extract_features(
        self,
        chunk_matches: List[Dict[str, Any]],
        total_chunks: int,
        source_hit_map: Dict[str, int]
    ) -> Dict[str, float]:
        """
        Extract feature vector from chunk-level similarity scores.
        This is the ONLY change to your pipeline - feature extraction.
        
        Args:
            chunk_matches: List of dicts with keys: cosine, tfidf, ngram, fuzzy, bleurt, composite
            total_chunks: Total number of chunks in input document
            source_hit_map: Counter of matches per source file
            
        Returns:
            Feature dictionary ready for RF prediction
        """
        if not chunk_matches:
            # Return zero vector if no matches
            return {name: 0.0 for name in self.feature_names}
        
        # Extract arrays for each metric
        cosines = [float(m["cosine"]) for m in chunk_matches]
        bleurts = [float(m["bleurt"]) for m in chunk_matches]
        ngrams = [float(m["ngram"]) for m in chunk_matches]
        tfidf_scores = [float(m["tfidf"]) for m in chunk_matches]
        fuzzies = [float(m["fuzzy"]) for m in chunk_matches]
        composites = [float(m["composite"]) for m in chunk_matches]
        
        # Calculate source concentration (entropy)
        if source_hit_map:
            total_hits = sum(source_hit_map.values())
            probs = [count / total_hits for count in source_hit_map.values()]
            # Shannon entropy: low = concentrated, high = distributed
            entropy = -sum(p * np.log2(p + 1e-10) for p in probs)
            concentration = 1.0 / (1.0 + entropy)  # Normalize to [0,1], higher = more concentrated
        else:
            concentration = 0.0
        
        features: Dict[str, float] = {
            # Cosine features
            "max_cosine": float(np.max(cosines)),
            "mean_cosine": float(np.mean(cosines)),
            "std_cosine": float(np.std(cosines)),
            
            # BLEURT features
            "max_bleurt": float(np.max(bleurts)),
            "mean_bleurt": float(np.mean(bleurts)),
            "std_bleurt": float(np.std(bleurts)),
            
            # N-gram features
            "max_ngram": float(np.max(ngrams)),
            "mean_ngram": float(np.mean(ngrams)),
            "std_ngram": float(np.std(ngrams)),
            
            # TF-IDF features
            "max_tfidf": float(np.max(tfidf_scores)),
            "mean_tfidf": float(np.mean(tfidf_scores)),
            
            # Fuzzy features
            "max_fuzzy": float(np.max(fuzzies)),
            "mean_fuzzy": float(np.mean(fuzzies)),
            
            # Structural features
            "chunk_hit_ratio": len(chunk_matches) / max(total_chunks, 1),
            "chunk_count": float(len(chunk_matches)),
            "source_concentration": concentration,
            
            # Composite features
            "max_composite": float(np.max(composites)),
            "mean_composite": float(np.mean(composites)),
        }
        
        return features
    
    def predict(self, features: Dict[str, float]) -> Tuple[bool, str, float]:
        """
        Predict plagiarism from feature vector.
        
        Args:
            features: Feature dictionary from extract_features()
            
        Returns:
            Tuple of (is_plagiarized, level, confidence)
            - is_plagiarized: Boolean flag
            - level: "HIGH" | "MODERATE" | "LOW" | "CLEAN"
            - confidence: Probability score [0,1]
        """
        if self.model is None:
            raise ValueError("Model not initialized. Train or load a model first.")
        
        # Convert dict to ordered array
        X = np.array([[features[name] for name in self.feature_names]])
        
        # Get prediction and probability
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        confidence = float(probabilities[1])  # Probability of plagiarism class
        
        # Map to your existing system's output format
        if prediction == 0:
            return False, "CLEAN", confidence
        
        # Classify severity based on confidence
        if confidence > 0.90:
            level = "HIGH"
        elif confidence > 0.75:
            level = "MODERATE"
        elif confidence > 0.55:
            level = "LOW"
        else:
            level = "CLEAN"
        
        return True, level, confidence
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Train the Random Forest classifier.
        
        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training labels (n_samples,) - 0 = clean, 1 = plagiarized
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Training metrics dictionary
        """
        logger.info(f"Training RF with {len(X_train)} samples")
        
        # Train model
        if self.model is None:
            raise ValueError("Model not initialized")
        
        self.model.fit(X_train, y_train)
        
        # Evaluate on training set
        train_pred = self.model.predict(X_train)
        train_report_result = classification_report(y_train, train_pred, output_dict=True)
        
        # Type assertion - classification_report with output_dict=True returns Dict[str, Any]
        # but sklearn's type stubs are incomplete
        train_report: Dict[str, Any] = train_report_result  # type: ignore
        
        metrics: Dict[str, Any] = {
            "train_accuracy": float(train_report["accuracy"]),
            "train_precision": float(train_report["1"]["precision"]),
            "train_recall": float(train_report["1"]["recall"]),
            "train_f1": float(train_report["1"]["f1-score"]),
        }
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_report_result = classification_report(y_val, val_pred, output_dict=True)
            val_report: Dict[str, Any] = val_report_result  # type: ignore
            
            metrics.update({
                "val_accuracy": float(val_report["accuracy"]),
                "val_precision": float(val_report["1"]["precision"]),
                "val_recall": float(val_report["1"]["recall"]),
                "val_f1": float(val_report["1"]["f1-score"]),
            })
            
            logger.info(f"Validation F1: {metrics['val_f1']:.3f}")
        
        try:
            if isinstance(self.model, CalibratedClassifierCV):
                # After fitting, CalibratedClassifierCV has calibrated_classifiers_
                # Each contains a base estimator - use the first one
                if hasattr(self.model, 'calibrated_classifiers_'):
                    base_estimator = self.model.calibrated_classifiers_[0].estimator
                    importances = base_estimator.feature_importances_
                else:
                    # Fallback if not fitted yet
                    logger.warning("Calibrated classifiers not available yet")
                    importances = None
            else:
                # Direct RandomForest
                importances = self.model.feature_importances_
            
            if importances is not None:
                feature_importance: Dict[str, float] = {
                    name: float(imp)
                    for name, imp in zip(self.feature_names, importances)
                }
                metrics["feature_importance"] = feature_importance
                
                # Log top features
                top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                logger.info("Top 5 important features:")
                for name, importance in top_features:
                    logger.info(f"  {name}: {importance:.3f}")
            else:
                metrics["feature_importance"] = {}
                
        except Exception as e:
            logger.warning(f"Could not extract feature importances: {e}")
            metrics["feature_importance"] = {}
        
        return metrics
    
    def save_model(self, path: Path) -> None:
        """Save trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_names': self.feature_names
            }, f)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: Path) -> None:
        """Load trained model from disk."""
        with open(path, 'rb') as f:
            data: Dict[str, Any] = pickle.load(f)
        
        self.model = data['model']
        self.feature_names = data['feature_names']
        
        logger.info(f"Model loaded from {path}")


# ==================== TRAINING UTILITIES ====================

def prepare_training_data(
    results_json_path: Path
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Convert your existing plagiarism_results.json into training data.
    
    This assumes you have manually labeled some results as ground truth.
    
    Args:
        results_json_path: Path to plagiarism_results.json with labeled data
        
    Returns:
        Tuple of (X, y, filenames)
        - X: Feature matrix (n_samples, n_features)
        - y: Labels (n_samples,) - 0 = clean, 1 = plagiarized
        - filenames: List of corresponding filenames
    """
    import json
    
    with open(results_json_path, 'r') as f:
        results: List[Dict[str, Any]] = json.load(f)
    
    X: List[List[float]] = []
    y: List[int] = []
    filenames: List[str] = []
    
    classifier = PlagiarismRFClassifier()  # Just for feature extraction
    
    for result in results:
        # Skip failed analyses
        if result.get('status') != 'success':
            continue
        
        # Extract features from matched_chunks
        matched_chunks: List[Dict[str, Any]] = result.get('matched_chunks', [])
        if not matched_chunks:
            continue
        
        # Build feature list from matched chunks
        chunk_features: List[Dict[str, Any]] = []
        for chunk in matched_chunks:
            chunk_features.append({
                "cosine": chunk.get("cosine_score", 0.0),  # You'll need to add these
                "tfidf": chunk.get("tfidf_score", 0.0),
                "ngram": chunk.get("ngram_score", 0.0),
                "fuzzy": chunk.get("fuzzy_score", 0.0),
                "bleurt": chunk.get("bleurt_score", 0.0),
                "composite": chunk.get("similarity_score", 0.0) / 100.0,
            })
        
        # Build source hit map
        source_hit_map: Dict[str, int] = {}
        for chunk in matched_chunks:
            source = chunk.get("source_url", "unknown")
            source_hit_map[source] = source_hit_map.get(source, 0) + 1
        
        # Extract features
        features = classifier.extract_features(
            chunk_features,
            result.get('total_chunks', 1),
            source_hit_map
        )
        
        # Convert to array
        X.append([features[name] for name in classifier.feature_names])
        
        # Ground truth label
        # IMPORTANT: You need to manually label these or use your current system's output as proxy
        y.append(1 if result.get('is_plagiarized', False) else 0)
        
        filenames.append(result['filename'])
    
    return np.array(X), np.array(y), filenames


def train_from_existing_results(
    results_json_path: Path,
    model_save_path: Path,
    test_size: float = 0.2
) -> Dict[str, Any]:
    """
    Train RF classifier from your existing plagiarism results.
    
    Args:
        results_json_path: Path to plagiarism_results.json
        model_save_path: Where to save trained model
        test_size: Fraction of data for validation
        
    Returns:
        Training metrics
    """
    logger.info("Loading training data...")
    X, y, filenames = prepare_training_data(results_json_path)
    
    logger.info(f"Loaded {len(X)} samples")
    logger.info(f"Class distribution: {np.bincount(y)}")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Train classifier
    classifier = PlagiarismRFClassifier()
    metrics = classifier.train(X_train, y_train, X_val, y_val)
    
    # Save model
    classifier.save_model(model_save_path)
    
    return metrics


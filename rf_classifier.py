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
    
    def __init__(self, model_path: Optional[Path] = None):
        self.model: Optional[RandomForestClassifier] = None
        self.feature_names: List[str] = [
           
            "max_cosine",
            "mean_cosine",
            "std_cosine",
            
            "max_bleurt",
            "mean_bleurt",
            "std_bleurt",
            
            "max_ngram",
            "mean_ngram",
            "std_ngram",
            
            "max_tfidf",
            "mean_tfidf",
            
            "max_fuzzy",
            "mean_fuzzy",
            
            "chunk_hit_ratio",      # % of input chunks that matched
            "chunk_count",          # absolute number of matches
            "source_concentration", # entropy of matches across sources
 
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

        if not chunk_matches:
            
            return {name: 0.0 for name in self.feature_names}
        
        
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
            
            "max_cosine": float(np.max(cosines)),
            "mean_cosine": float(np.mean(cosines)),
            "std_cosine": float(np.std(cosines)),
            
            "max_bleurt": float(np.max(bleurts)),
            "mean_bleurt": float(np.mean(bleurts)),
            "std_bleurt": float(np.std(bleurts)),
            
            "max_ngram": float(np.max(ngrams)),
            "mean_ngram": float(np.mean(ngrams)),
            "std_ngram": float(np.std(ngrams)),
            
            "max_tfidf": float(np.max(tfidf_scores)),
            "mean_tfidf": float(np.mean(tfidf_scores)),
            
            "max_fuzzy": float(np.max(fuzzies)),
            "mean_fuzzy": float(np.mean(fuzzies)),
            
            "chunk_hit_ratio": len(chunk_matches) / max(total_chunks, 1),
            "chunk_count": float(len(chunk_matches)),
            "source_concentration": concentration,
            
            "max_composite": float(np.max(composites)),
            "mean_composite": float(np.mean(composites)),
        }
        
        return features
    
    def predict(self, features: Dict[str, float]) -> Tuple[bool, str, float]:

        if self.model is None:
            raise ValueError("Model not initialized. Train or load a model first.")
        
        # Convert dict to ordered array
        X = np.array([[features[name] for name in self.feature_names]])
        
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        confidence = float(probabilities[1])  # Probability of plagiarism class
        
        if prediction == 0:
            return False, "CLEAN", confidence
        
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
        
        logger.info(f"Training RF with {len(X_train)} samples")
        
        # Train model
        if self.model is None:
            raise ValueError("Model not initialized")
        
        self.model.fit(X_train, y_train)
        
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

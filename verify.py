#!/usr/bin/env python3
"""
VERIFICATION SCRIPT - FIXED FOR WINDOWS ENCODING
=================================================

Run this BEFORE training to check if your detector modifications are working.
This will tell you if you're ready to train the RF model.
"""

import json
from pathlib import Path
import sys


def check_results_file(results_path: Path) -> bool:
    """Check if results file has required fields for RF training."""
    
    print("\n" + "="*70)
    print("CHECKING PLAGIARISM RESULTS")
    print("="*70)
    
    if not results_path.exists():
        print(f"❌ File not found: {results_path}")
        print(f"\nNext step: Run detector.py first to generate results")
        return False
    
    print(f"✅ Found results file: {results_path}")
    
    # Load results
    try:
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False
    
    if not results:
        print("❌ Results file is empty")
        return False
    
    print(f"✅ Loaded {len(results)} results")
    
    # Check for required fields
    required_chunk_fields = ['cosine_score', 'tfidf_score', 'ngram_score', 'fuzzy_score']
    
    samples_with_chunks = 0
    samples_with_new_fields = 0
    
    for result in results:
        if result.get('matched_chunks'):
            samples_with_chunks += 1
            first_chunk = result['matched_chunks'][0]
            
            if all(field in first_chunk for field in required_chunk_fields):
                samples_with_new_fields += 1
    
    print(f"\nResults with matched chunks: {samples_with_chunks}/{len(results)}")
    print(f"Results with new score fields: {samples_with_new_fields}/{samples_with_chunks}")
    
    if samples_with_new_fields == 0:
        print("\n❌ NO RESULTS HAVE NEW SCORE FIELDS!")
        print("\nThis means you haven't re-run detector.py with the modified code.")
        print("\nNext steps:")
        print("1. Make sure you've applied all 8 modifications to detector.py")
        print("2. Re-run detector.py on your input folder")
        print("3. Run this verification script again")
        return False
    
    if samples_with_new_fields < samples_with_chunks:
        print(f"\n⚠️  Warning: Only {samples_with_new_fields}/{samples_with_chunks} have new fields")
        print("   Some results are from before the detector modifications")
        print("   Training will only use the samples with new fields")
    
    # Check class distribution
    plagiarized = sum(1 for r in results if r.get('is_plagiarized', False))
    clean = len(results) - plagiarized
    
    print(f"\nClass distribution:")
    print(f"  Plagiarized: {plagiarized} ({plagiarized/len(results)*100:.1f}%)")
    print(f"  Clean: {clean} ({clean/len(results)*100:.1f}%)")
    
    # Warn if too few samples
    if samples_with_new_fields < 20:
        print("\n⚠️  WARNING: Very few training samples!")
        print(f"   You have: {samples_with_new_fields} samples")
        print(f"   Minimum: 20 samples recommended")
        print(f"   Ideal: 50+ samples")
        print("\n   RF may not train well with so few samples.")
        print("   Consider running detector on more resumes first.")
    
    if plagiarized < 5 or clean < 5:
        print("\n⚠️  WARNING: Class imbalance!")
        print("   You need at least 5 samples of each class (plagiarized & clean)")
        print("   Try to get more balanced training data")
    
    # Success!
    print("\n" + "="*70)
    if samples_with_new_fields >= 20:
        print("✅ READY TO TRAIN!")
        print("="*70)
        print("\nYou have enough data with the correct fields.")
        print("Run: python train_rf.py")
        return True
    else:
        print("⚠️  CAN TRAIN, BUT NOT IDEAL")
        print("="*70)
        print("\nYou can train, but results may not be great with so few samples.")
        print("\nRecommended: Run detector on more resumes first")
        print("Alternative: Proceed anyway for testing (run train_rf.py)")
        return True


def check_detector_modifications() -> None:
    """Check if detector.py has been modified."""
    
    print("\n" + "="*70)
    print("CHECKING DETECTOR MODIFICATIONS")
    print("="*70)
    
    detector_path = Path("plagiarism_detector.py")
    
    if not detector_path.exists():
        print("❌ plagiarism_detector.py not found in current directory")
        return
    
    try:
        # Try UTF-8 first (most common)
        with open(detector_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        try:
            # Fallback to latin-1 which accepts all byte values
            with open(detector_path, 'r', encoding='latin-1') as f:
                content = f.read()
        except Exception as e:
            print(f"❌ Error reading detector.py: {e}")
            print("   File may have encoding issues")
            return
    except Exception as e:
        print(f"❌ Error reading detector.py: {e}")
        return
    
    checks = {
        "RF import": "from rf_classifier import PlagiarismRFClassifier" in content,
        "RF global variable": "rf_classifier" in content and "PlagiarismRFClassifier" in content,
        "Feature collection": "chunk_features_for_rf" in content,
        "RF prediction": "rf_classifier.predict" in content,
    }
    
    all_good = True
    for check_name, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"{status} {check_name}")
        if not passed:
            all_good = False
    
    if all_good:
        print("\n✅ All modifications detected!")
    else:
        print("\n❌ Some modifications are missing!")
        print("\nRefer to integration_checklist.txt for complete instructions")


def check_rf_classifier() -> None:
    """Check if rf_classifier.py exists."""
    
    print("\n" + "="*70)
    print("CHECKING RF CLASSIFIER MODULE")
    print("="*70)
    
    rf_path = Path("rf_classifier.py")
    
    if not rf_path.exists():
        print("❌ rf_classifier.py not found")
        print("\nCopy the rf_classifier.py file to your project directory")
        return
    
    print("✅ rf_classifier.py found")
    
    # Try importing
    try:
        from rf_classifier import PlagiarismRFClassifier
        print("✅ Module imports successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\nCheck dependencies: sklearn, numpy")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


def main():
    print("\n" + "="*70)
    print("RF CLASSIFIER READINESS CHECK")
    print("="*70)
    print("\nThis script checks if you're ready to train the RF model.")
    
    # Check 1: RF classifier module
    check_rf_classifier()
    
    # Check 2: Detector modifications
    check_detector_modifications()
    
    # Check 3: Results file
    results_path = Path("results/plagiarism_results.json")
    ready = check_results_file(results_path)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if ready:
        print("\n✅ You're ready to train the RF model!")
        print("\nNext step: Run 'python train_rf.py'")
    else:
        print("\n❌ Not quite ready yet")
        print("\nFollow the instructions above to prepare for training")
    
    print("\n")


if __name__ == "__main__":
    main()
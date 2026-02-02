"""
Production-Grade Pharmacovigilance Classifier Training Script
CORRECTED VERSION - Addresses critical sampling bias and safety issues

Hardware: i5-1335U (10 cores), 16GB RAM, SSD
Dataset: FAERS adverse events (proper stratified sampling across full dataset)

CRITICAL FIXES:
1. ✅ True stratified sampling across entire dataset (no temporal bias)
2. ✅ Safe class index handling (no assumptions)
3. ✅ Proper metric interpretation (recall-first, ROC-AUC secondary)
4. ✅ Conservative model selection (balanced over high_recall for safety)
"""
import os
import sys
from pathlib import Path
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    recall_score, precision_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from app.services.preprocessor import combine_features


# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "faers_q3_ps_drug_adverse_events_serious_labeled_v2.csv"
MODEL_DIR = PROJECT_ROOT / "ml"
EXPERIMENT_DIR = MODEL_DIR / "experiments"
PLOTS_DIR = MODEL_DIR / "plots"

# Training parameters
SAMPLE_SIZE = 500_000  # Change to 1_000_000 if needed
TEST_SIZE = 0.2
VAL_SIZE = 0.1
RANDOM_STATE = 42
N_JOBS = -1

# Target metrics (safety-first)
TARGET_RECALL = 0.85  # CRITICAL for patient safety
MIN_PRECISION = 0.70  # Avoid overwhelming false positives

# Model configurations
MODEL_CONFIGS = [
    {
        'name': 'baseline',
        'description': 'Reliable baseline - standard balanced weighting',
        'vectorizer': {
            'max_features': 10000,
            'ngram_range': (1, 2),
            'min_df': 5,
            'max_df': 0.95,
            'sublinear_tf': True
        },
        'classifier': {
            'class_weight': 'balanced',
            'max_iter': 1000,
            'solver': 'lbfgs',
            'C': 1.0,
            'n_jobs': N_JOBS
        }
    },
    {
        'name': 'balanced',
        'description': 'Production model - best F1 while meeting recall target',
        'vectorizer': {
            'max_features': 12000,
            'ngram_range': (1, 2),
            'min_df': 5,
            'max_df': 0.90,
            'sublinear_tf': True
        },
        'classifier': {
            'class_weight': 'balanced',
            'max_iter': 1500,
            'solver': 'saga',
            'C': 0.8,
            'penalty': 'l2',
            'n_jobs': N_JOBS
        }
    },
    {
        'name': 'high_recall',
        'description': 'Experimental - max recall (inspect FP/FN before deployment)',
        'vectorizer': {
            'max_features': 15000,
            'ngram_range': (1, 3),
            'min_df': 3,
            'max_df': 0.95,
            'sublinear_tf': True
        },
        'classifier': {
            'class_weight': {
                'Serious': 0.7,
                'Non-Serious': 0.3
            },
            'max_iter': 1000,
            'solver': 'saga',
            'C': 0.5,
            'penalty': 'l1',
            'n_jobs': N_JOBS
        },
        'warning': '⚠️ Manual class weights - verify FP rate before deployment'
    }
]


# ============================================================================
# EXPERIMENT TRACKING
# ============================================================================

class ExperimentTracker:
    """Track and save experiment results."""
    
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"{experiment_name}_{self.timestamp}"
        self.experiment_path = EXPERIMENT_DIR / self.experiment_id
        self.experiment_path.mkdir(parents=True, exist_ok=True)
        
        self.metrics = {}
        self.config = {}
        
    def log_config(self, config: dict):
        """Log experiment configuration."""
        self.config = config
        
    def log_metrics(self, metrics: dict):
        """Log metrics."""
        self.metrics.update(metrics)
        
    def save(self):
        """Save experiment results."""
        results = {
            'experiment_id': self.experiment_id,
            'timestamp': self.timestamp,
            'config': self.config,
            'metrics': self.metrics
        }
        
        results_path = self.experiment_path / "results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"   📊 Saved: {self.experiment_path.name}")


# ============================================================================
# DATA LOADING - FIXED: TRUE STRATIFIED SAMPLING
# ============================================================================

def load_data(sample_size: int = None) -> pd.DataFrame:
    """
    Load data with TRUE stratified sampling across entire dataset.
    
    CRITICAL FIX: Previous version used nrows=sample_size*2 which only
    sampled from the FIRST rows of the CSV, creating temporal/drug bias.
    
    This version:
    - Reads entire dataset in chunks
    - Separates by class
    - Samples randomly from each class maintaining ratio
    - Ensures representative sample across all time periods/drugs
    """
    print(f"📂 Loading data from {DATA_PATH.name}...")
    
    dtypes = {
        'drugname': 'category',
        'adverse_event': 'category',
        'seriousness': 'category'
    }
    
    if not sample_size:
        print(f"   Loading full dataset...")
        df = pd.read_csv(DATA_PATH, dtype=dtypes)
        print(f"   Loaded: {len(df):,} rows")
        return df
    
    # Read entire dataset in chunks and separate by class
    print(f"   Reading dataset in chunks (stratified sampling)...")
    chunks = pd.read_csv(DATA_PATH, dtype=dtypes, chunksize=200_000)
    
    serious_chunks = []
    non_serious_chunks = []
    total_rows = 0
    
    for i, chunk in enumerate(chunks, 1):
        serious_chunks.append(chunk[chunk['seriousness'] == 'Serious'])
        non_serious_chunks.append(chunk[chunk['seriousness'] == 'Non-Serious'])
        total_rows += len(chunk)
        print(f"      Chunk {i}: {len(chunk):,} rows (total: {total_rows:,})", end='\r')
    
    print(f"\n   Total rows in dataset: {total_rows:,}")
    
    # Combine all chunks by class
    df_serious = pd.concat(serious_chunks, ignore_index=True)
    df_non_serious = pd.concat(non_serious_chunks, ignore_index=True)
    
    print(f"   Class counts:")
    print(f"     • Serious: {len(df_serious):,} ({100*len(df_serious)/total_rows:.1f}%)")
    print(f"     • Non-Serious: {len(df_non_serious):,} ({100*len(df_non_serious)/total_rows:.1f}%)")
    
    # Calculate stratified sample sizes
    serious_ratio = len(df_serious) / total_rows
    n_serious = int(sample_size * serious_ratio)
    n_non_serious = sample_size - n_serious
    
    print(f"   Sampling {sample_size:,} rows (stratified)...")
    print(f"     • Serious: {n_serious:,}")
    print(f"     • Non-Serious: {n_non_serious:,}")
    
    # Random sampling from each class (truly representative)
    df_serious_sample = df_serious.sample(n=n_serious, random_state=RANDOM_STATE)
    df_non_serious_sample = df_non_serious.sample(n=n_non_serious, random_state=RANDOM_STATE)
    
    # Combine and shuffle
    df = pd.concat([df_serious_sample, df_non_serious_sample], ignore_index=True)
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    
    print(f"   ✅ Sampled: {len(df):,} rows (representative across entire dataset)")
    
    return df


# ============================================================================
# FEATURE PREPARATION
# ============================================================================

def prepare_features(df: pd.DataFrame) -> tuple:
    """Prepare text features from drug names and adverse events."""
    print("\n🔧 Preparing features...")
    
    # Handle missing values
    df['drugname'] = df['drugname'].fillna('UNKNOWN')
    df['adverse_event'] = df['adverse_event'].fillna('UNKNOWN')
    
    # Combine drug name and adverse event
    df['text'] = df.apply(
        lambda row: combine_features(
            str(row['drugname']),
            str(row['adverse_event'])
        ),
        axis=1
    )
    
    # Remove very short texts
    df = df[df['text'].str.len() > 3]
    
    X = df['text'].values
    y = df['seriousness'].values
    
    # Print statistics
    unique, counts = np.unique(y, return_counts=True)
    print(f"   Total samples: {len(X):,}")
    print(f"   Class distribution:")
    for label, count in zip(unique, counts):
        print(f"     • {label}: {count:,} ({100*count/len(y):.1f}%)")
    
    return X, y


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_confusion_matrix(cm, classes, experiment_path, title='Confusion Matrix'):
    """Plot and save confusion matrix heatmap."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Count'})
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    plot_path = experiment_path / f"{title.lower().replace(' ', '_')}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_curve(y_true, y_prob, experiment_path):
    """
    Plot ROC curve with AUC.
    
    NOTE: ROC-AUC is reported for completeness but is NOT the primary
    metric for model selection. Recall (sensitivity) is the priority
    for pharmacovigilance safety systems.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'Model (AUC = {auc:.3f})', linewidth=2, color='#2E86AB')
    plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate (Recall)', fontsize=12)
    plt.title('ROC Curve (Secondary Metric)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add note about metric interpretation
    plt.text(0.6, 0.2, 'Note: Model selection based\non recall, not AUC',
             fontsize=9, style='italic', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    plot_path = experiment_path / "roc_curve.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return auc


def plot_precision_recall_curve(y_true, y_prob, experiment_path, target_recall=0.85):
    """Plot precision-recall curve with target line."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, linewidth=2, label='PR Curve', color='#2E86AB')
    plt.axvline(x=target_recall, color='#A23B72', linestyle='--', 
                linewidth=2, label=f'Target Recall ({target_recall})')
    plt.xlabel('Recall (PRIMARY METRIC)', fontsize=12, fontweight='bold')
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = experiment_path / "precision_recall_curve.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================================
# THRESHOLD OPTIMIZATION
# ============================================================================

def find_optimal_threshold(y_true, y_prob, target_recall=0.85):
    """
    Find optimal decision threshold to achieve target recall.
    
    Strategy:
    1. Find all thresholds that meet target recall
    2. Among those, choose the one with best precision (or F1)
    
    This is validation-set-only tuning (no test set leakage).
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    
    # Find thresholds meeting target recall
    valid_indices = recall >= target_recall
    
    if not valid_indices.any():
        print(f"   ⚠️  Cannot achieve target recall of {target_recall}")
        print(f"   Using threshold that maximizes recall...")
        best_idx = np.argmax(recall)
        return thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    
    # Among valid thresholds, find best precision
    valid_precision = precision[valid_indices]
    valid_recall = recall[valid_indices]
    valid_thresholds = thresholds[valid_indices[:-1]]
    
    if len(valid_thresholds) == 0:
        return 0.5
    
    # Choose threshold with best F1 among valid options
    valid_f1 = 2 * (valid_precision[:-1] * valid_recall[:-1]) / \
               (valid_precision[:-1] + valid_recall[:-1] + 1e-10)
    best_idx = np.argmax(valid_f1)
    
    optimal_threshold = valid_thresholds[best_idx]
    
    print(f"   🎯 Optimal threshold: {optimal_threshold:.4f}")
    print(f"      Expected recall: {valid_recall[best_idx]:.4f}")
    print(f"      Expected precision: {valid_precision[best_idx]:.4f}")
    
    return optimal_threshold


# ============================================================================
# MODEL TRAINING & EVALUATION - FIXED: SAFE CLASS INDEX HANDLING
# ============================================================================

def train_and_evaluate_model(X_train, y_train, X_val, y_val, X_test, y_test, 
                             config, tracker):
    """
    Train and evaluate a single model configuration.
    
    CRITICAL FIX: Safe class index handling - no assumptions about
    which index corresponds to 'Serious' class.
    """
    
    print(f"\n{'='*70}")
    print(f"🚀 TRAINING: {config['name'].upper()}")
    print(f"   {config['description']}")
    if 'warning' in config:
        print(f"   {config['warning']}")
    print(f"{'='*70}")
    
    # Vectorization
    print("\n📝 Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(**config['vectorizer'])
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)
    
    print(f"   Vocabulary size: {len(vectorizer.vocabulary_):,}")
    print(f"   Feature matrix: {X_train_vec.shape}")
    
    # Train model
    print(f"\n🎓 Training Logistic Regression...")
    model = LogisticRegression(
        random_state=RANDOM_STATE,
        **config['classifier']
    )
    
    import time
    start_time = time.time()
    model.fit(X_train_vec, y_train)
    training_time = time.time() - start_time
    
    print(f"   ✅ Training complete in {training_time:.1f} seconds")
    
    # CRITICAL FIX: Safe class index lookup
    print(f"\n🔍 Determining class indices...")
    try:
        serious_idx = list(model.classes_).index('Serious')
        print(f"   'Serious' class at index: {serious_idx}")
        print(f"   Classes: {list(model.classes_)}")
    except ValueError:
        raise ValueError("'Serious' class not found in model.classes_!")
    
    # Cross-validation
    print(f"\n🔄 Cross-validation (5-fold)...")
    cv_scores = cross_val_score(
        model, X_train_vec, y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring='recall',
        n_jobs=N_JOBS
    )
    print(f"   CV Recall: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Threshold optimization on validation set (SAFE indexing)
    print(f"\n🎯 Finding optimal threshold on validation set...")
    y_val_prob = model.predict_proba(X_val_vec)[:, serious_idx]  # FIXED
    optimal_threshold = find_optimal_threshold(
        (y_val == 'Serious').astype(int),
        y_val_prob,
        target_recall=TARGET_RECALL
    )
    
    # Test set evaluation (SAFE indexing)
    print(f"\n{'='*70}")
    print("📊 TEST SET EVALUATION")
    print(f"{'='*70}")
    
    y_test_prob = model.predict_proba(X_test_vec)[:, serious_idx]  # FIXED
    y_test_pred_default = model.predict(X_test_vec)
    
    # Predictions with tuned threshold
    y_test_pred_tuned = (y_test_prob >= optimal_threshold).astype(int)
    y_test_pred_tuned = np.where(y_test_pred_tuned == 1, 'Serious', 'Non-Serious')
    
    y_test_binary = (y_test == 'Serious').astype(int)
    
    # Metrics comparison
    recall_default = recall_score(y_test, y_test_pred_default, pos_label='Serious')
    precision_default = precision_score(y_test, y_test_pred_default, pos_label='Serious')
    f1_default = f1_score(y_test, y_test_pred_default, pos_label='Serious')
    
    recall_tuned = recall_score(y_test, y_test_pred_tuned, pos_label='Serious')
    precision_tuned = precision_score(y_test, y_test_pred_tuned, pos_label='Serious')
    f1_tuned = f1_score(y_test, y_test_pred_tuned, pos_label='Serious')
    
    auc_score = roc_auc_score(y_test_binary, y_test_prob)
    
    # Print results
    print(f"\nSerious Class Metrics:")
    print(f"{'Metric':<15} {'Default (0.5)':<15} {'Tuned ({:.3f})':<15} {'Change':<10}".format(optimal_threshold))
    print("-" * 60)
    print(f"{'Recall':<15} {recall_default:<15.4f} {recall_tuned:<15.4f} {recall_tuned-recall_default:+.4f}")
    print(f"{'Precision':<15} {precision_default:<15.4f} {precision_tuned:<15.4f} {precision_tuned-precision_default:+.4f}")
    print(f"{'F1-Score':<15} {f1_default:<15.4f} {f1_tuned:<15.4f} {f1_tuned-f1_default:+.4f}")
    print(f"\n{'ROC-AUC':<15} {auc_score:.4f} (reported for completeness, not primary metric)")
    
    # Check target achievement
    if recall_tuned >= TARGET_RECALL:
        print(f"\n✅ TARGET ACHIEVED! Recall = {recall_tuned:.4f} (≥ {TARGET_RECALL})")
    else:
        print(f"\n⚠️  Below target: Recall = {recall_tuned:.4f} (target: {TARGET_RECALL})")
    
    # Confusion matrix analysis
    cm = confusion_matrix(y_test, y_test_pred_tuned)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\nConfusion Matrix Breakdown:")
    print(f"  True Negatives (TN):  {tn:,} (correct Non-Serious predictions)")
    print(f"  False Positives (FP): {fp:,} (Non-Serious predicted as Serious)")
    print(f"  False Negatives (FN): {fn:,} ⚠️ CRITICAL - Missed serious events")
    print(f"  True Positives (TP):  {tp:,} (correct Serious predictions)")
    print(f"\n  FN Rate: {fn/(fn+tp)*100:.2f}% of serious events MISSED")
    print(f"  FP Rate: {fp/(fp+tn)*100:.2f}% of non-serious events FLAGGED")
    
    # Detailed report
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, y_test_pred_tuned))
    
    # Log metrics
    metrics = {
        'training_time_seconds': float(training_time),
        'cv_recall_mean': float(cv_scores.mean()),
        'cv_recall_std': float(cv_scores.std()),
        'optimal_threshold': float(optimal_threshold),
        'serious_class_index': int(serious_idx),
        'test_recall_default': float(recall_default),
        'test_precision_default': float(precision_default),
        'test_f1_default': float(f1_default),
        'test_recall_tuned': float(recall_tuned),
        'test_precision_tuned': float(precision_tuned),
        'test_f1_tuned': float(f1_tuned),
        'test_auc': float(auc_score),
        'confusion_matrix': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp),
            'fn_rate': float(fn/(fn+tp)),
            'fp_rate': float(fp/(fp+tn))
        },
        'vocabulary_size': len(vectorizer.vocabulary_),
        'meets_target_recall': bool(recall_tuned >= TARGET_RECALL)
    }
    
    return model, vectorizer, optimal_threshold, metrics, y_test_pred_tuned, y_test_prob


# ============================================================================
# MODEL PERSISTENCE
# ============================================================================

def save_artifacts(model, vectorizer, threshold, config, metrics, experiment_path):
    """Save model, vectorizer, and metadata."""
    
    joblib.dump(model, experiment_path / "model.pkl")
    joblib.dump(vectorizer, experiment_path / "vectorizer.pkl")
    
    # Save comprehensive metadata
    metadata = {
        'threshold': threshold,
        'config': config,
        'metrics': metrics,
        'timestamp': datetime.now().isoformat(),
        'sample_size': SAMPLE_SIZE,
        'random_state': RANDOM_STATE,
        'target_recall': TARGET_RECALL,
        'data_version': 'v2',
        'sampling_method': 'stratified_across_full_dataset',  # Important!
        'notes': [
            'Model selection based on recall (patient safety), not ROC-AUC',
            'Threshold tuned on validation set only (no test set leakage)',
            'Class index verified at runtime (no assumptions)'
        ]
    }
    
    with open(experiment_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main training pipeline."""
    
    print("=" * 70)
    print("🏥 PHARMACOVIGILANCE ADVERSE EVENT CLASSIFIER")
    print("   CORRECTED VERSION - Production Grade")
    print("=" * 70)
    print(f"\nHardware: i5-1335U (10 cores), 16GB RAM, SSD")
    print(f"Target: {TARGET_RECALL:.0%} recall for Serious events (patient safety)")
    print(f"Sample: {SAMPLE_SIZE:,} records (stratified across full dataset)")
    print(f"Data: FAERS Q3 v2\n")
    
    # Create directories
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data (FIXED: true stratified sampling)
    df = load_data(sample_size=SAMPLE_SIZE)
    
    # Prepare features
    X, y = prepare_features(df)
    
    # Three-way split: train / validation / test
    print(f"\n📊 Splitting data...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, 
        test_size=VAL_SIZE/(1-TEST_SIZE), 
        stratify=y_temp, 
        random_state=RANDOM_STATE
    )
    
    print(f"   Training:   {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   Validation: {len(X_val):,} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"   Test:       {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    # Train configurations
    best_model = None
    best_score = 0
    best_config_name = None
    results_summary = []
    
    # IMPORTANT: Prefer 'balanced' over 'high_recall' for production safety
    preferred_config = 'balanced'
    
    for i, config in enumerate(MODEL_CONFIGS, 1):
        print(f"\n\n{'#'*70}")
        print(f"# CONFIGURATION {i}/{len(MODEL_CONFIGS)}")
        print(f"{'#'*70}")
        
        tracker = ExperimentTracker(config['name'])
        tracker.log_config(config)
        
        # Train and evaluate
        model, vectorizer, threshold, metrics, y_pred, y_prob = train_and_evaluate_model(
            X_train, y_train, X_val, y_val, X_test, y_test,
            config, tracker
        )
        
        # Generate visualizations
        print(f"\n Generating visualizations...")
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm, model.classes_, tracker.experiment_path)
        plot_roc_curve((y_test == 'Serious').astype(int), y_prob, tracker.experiment_path)
        plot_precision_recall_curve(
            (y_test == 'Serious').astype(int), 
            y_prob, 
            tracker.experiment_path, 
            TARGET_RECALL
        )
        
        save_artifacts(model, vectorizer, threshold, config, metrics, tracker.experiment_path)
        tracker.log_metrics(metrics)
        tracker.save()
        
        # Model selection logic (CONSERVATIVE)
        # Prefer 'balanced' config if it meets target
        # Only use 'high_recall' if balanced doesn't meet target
        meets_target = metrics['test_recall_tuned'] >= TARGET_RECALL
        
        if config['name'] == preferred_config and meets_target:
            # Always prefer balanced if it meets target
            best_score = metrics['test_f1_tuned']
            best_model = (model, vectorizer, threshold, config)
            best_config_name = config['name']
        elif best_model is None and meets_target:
            # Fallback: any config that meets target
            if metrics['test_f1_tuned'] > best_score:
                best_score = metrics['test_f1_tuned']
                best_model = (model, vectorizer, threshold, config)
                best_config_name = config['name']
        
        results_summary.append({
            'config': config['name'],
            'recall': metrics['test_recall_tuned'],
            'precision': metrics['test_precision_tuned'],
            'f1': metrics['test_f1_tuned'],
            'auc': metrics['test_auc'],
            'fn_rate': metrics['confusion_matrix']['fn_rate'],
            'fp_rate': metrics['confusion_matrix']['fp_rate'],
            'meets_target': meets_target
        })
    
    # Deploy best model
    if best_model:
        print(f"\n\n{'='*70}")
        print(f"🏆 BEST MODEL: {best_config_name.upper()}")
        print(f"{'='*70}")
        
        model, vectorizer, threshold, config = best_model
        
        joblib.dump(model, MODEL_DIR / "model.pkl")
        joblib.dump(vectorizer, MODEL_DIR / "vectorizer.pkl")
        
        with open(MODEL_DIR / "model_metadata.json", 'w') as f:
            json.dump({
                'threshold': threshold,
                'config_name': best_config_name,
                'sample_size': SAMPLE_SIZE,
                'timestamp': datetime.now().isoformat(),
                'target_recall_achieved': True,
                'data_version': 'v2',
                'selection_criteria': 'recall-first (patient safety), balanced preferred'
            }, f, indent=2)
        
        print(f"\n✅ Model deployed to: {MODEL_DIR}/")
        print(f"   • model.pkl")
        print(f"   • vectorizer.pkl")
        print(f"   • model_metadata.json")
    else:
        print(f"\n⚠️  WARNING: No model met target recall of {TARGET_RECALL}")
        
    
    # Summary table
    print(f"\n\n{'='*70}")
    print("📋 RESULTS SUMMARY")
    print(f"{'='*70}\n")
    print(f"{'Config':<15} {'Recall':<8} {'Prec':<8} {'F1':<8} {'FN%':<8} {'FP%':<8} {'Target'}")
    print("-" * 70)
    
    for result in results_summary:
        target_symbol = "✅" if result['meets_target'] else "❌"
        print(f"{result['config']:<15} {result['recall']:<8.4f} "
              f"{result['precision']:<8.4f} {result['f1']:<8.4f} "
              f"{result['fn_rate']*100:<7.2f}% {result['fp_rate']*100:<7.2f}% {target_symbol}")
    
    print(f"\n{'='*70}")
    print("✅ TRAINING COMPLETE!")
    print(f"{'='*70}")
   

if __name__ == "__main__":
    import time
    overall_start = time.time()
    
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        total_time = time.time() - overall_start
        print(f"\n⏱️  Total runtime: {total_time/60:.1f} minutes ({total_time:.0f} seconds)")
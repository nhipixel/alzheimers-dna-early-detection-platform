from model.models.xgboost.model import XGBoostModel
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (precision_score, recall_score, accuracy_score, f1_score,
                           classification_report, confusion_matrix)
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

import pandas as pd
import numpy as np
import time, os, argparse
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append('./model')

from data.loaders.loader_xgboost import load_data, load_data_h5

def enhanced_kfold_cv(model, X, y, k=5, random_state=42):
    """
    Enhanced K-Fold Cross Validation with stratification and detailed metrics
    """
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    
    metrics = {
        'precision_weighted': [], 'precision_macro': [],
        'recall_weighted': [], 'recall_macro': [],
        'accuracy': [], 'f1_weighted': [], 'f1_macro': []
    }
    
    fold_predictions = []
    fold_true_labels = []
    
    print(f"Starting {k}-fold cross validation...")
    
    for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
        print(f"Training fold {fold + 1}/{k}...")
        
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        # Create new model instance for each fold
        fold_model = XGBoostModel(params=model.params, 
                                 use_feature_selection=model.use_feature_selection,
                                 n_features=model.n_features)
        
        # Train model
        fold_model.train(X_train, y_train)
        y_pred = fold_model.predict(X_val)
        
        # Store predictions for ensemble analysis
        fold_predictions.extend(y_pred)
        fold_true_labels.extend(y_val)
        
        # Calculate metrics
        metrics['precision_weighted'].append(precision_score(y_val, y_pred, average='weighted', zero_division=0))
        metrics['precision_macro'].append(precision_score(y_val, y_pred, average='macro', zero_division=0))
        metrics['recall_weighted'].append(recall_score(y_val, y_pred, average='weighted', zero_division=0))
        metrics['recall_macro'].append(recall_score(y_val, y_pred, average='macro', zero_division=0))
        metrics['accuracy'].append(accuracy_score(y_val, y_pred))
        metrics['f1_weighted'].append(f1_score(y_val, y_pred, average='weighted', zero_division=0))
        metrics['f1_macro'].append(f1_score(y_val, y_pred, average='macro', zero_division=0))
        
        print(f"Fold {fold + 1} - Accuracy: {metrics['accuracy'][-1]:.4f}, "
              f"F1 (weighted): {metrics['f1_weighted'][-1]:.4f}")
    
    return metrics, fold_predictions, fold_true_labels

def perform_feature_analysis(X, y, feature_names=None, top_k=1000):
    """
    Analyze feature importance using statistical tests
    """
    print("Performing feature importance analysis...")
    
    # Statistical feature selection
    selector_f = SelectKBest(score_func=f_classif, k=min(top_k, X.shape[1]))
    selector_mi = SelectKBest(score_func=mutual_info_classif, k=min(top_k, X.shape[1]))
    
    # Fit selectors
    selector_f.fit(X, y)
    selector_mi.fit(X, y)
    
    # Get scores
    f_scores = selector_f.scores_
    mi_scores = selector_mi.scores_
    
    if feature_names is not None:
        feature_importance = pd.DataFrame({
            'feature': feature_names[:len(f_scores)],
            'f_score': f_scores,
            'mi_score': mi_scores
        })
        feature_importance = feature_importance.sort_values('f_score', ascending=False)
        
        # Save top features
        top_features_path = './model/models/xgboost/top_features.csv'
        feature_importance.head(top_k).to_csv(top_features_path, index=False)
        print(f"Top {top_k} features saved to {top_features_path}")
        
        return feature_importance
    
    return f_scores, mi_scores

def plot_training_results(metrics, save_path='./model/models/xgboost/'):
    """
    Create comprehensive training result visualizations
    """
    os.makedirs(save_path, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy distribution
    axes[0, 0].hist(metrics['accuracy'], bins=10, alpha=0.7, color='blue')
    axes[0, 0].axvline(np.mean(metrics['accuracy']), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(metrics["accuracy"]):.4f}')
    axes[0, 0].set_title('Accuracy Distribution Across Folds')
    axes[0, 0].set_xlabel('Accuracy')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    
    # F1 Score comparison
    axes[0, 1].boxplot([metrics['f1_weighted'], metrics['f1_macro']], 
                       labels=['Weighted F1', 'Macro F1'])
    axes[0, 1].set_title('F1 Score Distribution')
    axes[0, 1].set_ylabel('F1 Score')
    
    # Precision-Recall comparison
    axes[1, 0].scatter(metrics['precision_weighted'], metrics['recall_weighted'], 
                       alpha=0.7, c='blue', label='Weighted')
    axes[1, 0].scatter(metrics['precision_macro'], metrics['recall_macro'], 
                       alpha=0.7, c='red', label='Macro')
    axes[1, 0].set_xlabel('Precision')
    axes[1, 0].set_ylabel('Recall')
    axes[1, 0].set_title('Precision vs Recall')
    axes[1, 0].legend()
    
    # Metric comparison across folds
    fold_nums = range(1, len(metrics['accuracy']) + 1)
    axes[1, 1].plot(fold_nums, metrics['accuracy'], 'o-', label='Accuracy', alpha=0.7)
    axes[1, 1].plot(fold_nums, metrics['f1_weighted'], 's-', label='F1 Weighted', alpha=0.7)
    axes[1, 1].plot(fold_nums, metrics['f1_macro'], '^-', label='F1 Macro', alpha=0.7)
    axes[1, 1].set_xlabel('Fold')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Metrics Across Folds')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_results.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training visualization saved to {save_path}training_results.png")

def save_comprehensive_results(model, metrics, fold_predictions, fold_true_labels, 
                              feature_importance=None, save_path='./model/models/xgboost/'):
    """
    Save comprehensive training results and model artifacts
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Calculate overall metrics
    overall_metrics = {}
    for key, values in metrics.items():
        overall_metrics[key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    
    # Classification report on all fold predictions
    class_names = ['Control', 'MCI', "Alzheimer's"]
    classification_rep = classification_report(
        fold_true_labels, fold_predictions, 
        target_names=class_names, output_dict=True
    )
    
    # Confusion matrix
    conf_matrix = confusion_matrix(fold_true_labels, fold_predictions)
    
    # Save results
    results = {
        'cv_metrics': overall_metrics,
        'classification_report': classification_rep,
        'confusion_matrix': conf_matrix.tolist(),
        'model_params': model.params,
        'feature_selection': model.use_feature_selection,
        'n_features_selected': model.n_features,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    if feature_importance is not None:
        results['top_features'] = feature_importance.head(100).to_dict('records')
    
    # Save as pickle and JSON for different use cases
    with open(os.path.join(save_path, 'training_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    # Save readable summary
    with open(os.path.join(save_path, 'training_summary.txt'), 'w') as f:
        f.write("XGBoost Model Training Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Training completed: {results['timestamp']}\n\n")
        
        f.write("Cross-Validation Results (5-fold):\n")
        f.write("-" * 30 + "\n")
        for metric, values in overall_metrics.items():
            f.write(f"{metric.replace('_', ' ').title()}: "
                   f"{values['mean']:.4f} ± {values['std']:.4f} "
                   f"(range: {values['min']:.4f}-{values['max']:.4f})\n")
        
        f.write(f"\nOverall Classification Report:\n")
        f.write("-" * 30 + "\n")
        for class_name in class_names:
            if class_name in classification_rep:
                metrics_str = (f"Precision: {classification_rep[class_name]['precision']:.4f}, "
                              f"Recall: {classification_rep[class_name]['recall']:.4f}, "
                              f"F1: {classification_rep[class_name]['f1-score']:.4f}")
                f.write(f"{class_name}: {metrics_str}\n")
    
    print(f"Comprehensive results saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Enhanced XGBoost training with comprehensive evaluation")
    parser.add_argument('--grid-search', action='store_true', 
                       help='Run grid search for hyperparameters')
    parser.add_argument('--feature-selection', action='store_true', default=True,
                       help='Use feature selection')
    parser.add_argument('--n-features', type=int, default=10000,
                       help='Number of features to select')
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='Number of cross-validation folds')
    args = parser.parse_args()

    print("Enhanced XGBoost Training Pipeline")
    print("=" * 50)
    
    # Enhanced parameters optimized for Alzheimer's classification
    params = {
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "use_label_encoder": False,
        "learning_rate": 0.05,
        "max_depth": 8,
        "n_estimators": 500,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "gamma": 0.1,
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist"
    }

    # Data paths
    data_train_h5 = './model/data/train/methylation.h5'
    idmap_train_path = './model/data/train/idmap.csv'

    # Load data
    print("Loading training data...")
    try:
        X_train, y_train = load_data_h5(data_train_h5, idmap_train_path, indices=None)
        print(f"Data loaded successfully: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    except Exception as e:
        print(f"Error loading data: {e}")
        # Fallback to CSV
        data_train_path = './model/data/train/methylation.csv'
        X_train, y_train = load_data(data_train_path, idmap_train_path, indices=None)
        print(f"Data loaded from CSV: {X_train.shape[0]} samples, {X_train.shape[1]} features")

    # Initialize model
    model = XGBoostModel(params=params, 
                        use_feature_selection=args.feature_selection,
                        n_features=args.n_features)
    
    print(f"Model initialized with feature selection: {args.feature_selection}")
    if args.feature_selection:
        print(f"Will select top {args.n_features} features")

    if args.grid_search:
        print("Running grid search for hyperparameter optimization...")
        
        # Enhanced parameter grid
        search_params = {
            "classifier__max_depth": [6, 8, 10],
            "classifier__learning_rate": [0.03, 0.05, 0.1],
            "classifier__n_estimators": [300, 500, 700],
            "classifier__subsample": [0.7, 0.8, 0.9],
            "classifier__reg_alpha": [0.05, 0.1, 0.2],
            "classifier__reg_lambda": [0.5, 1.0, 1.5]
        }
        
        best_model, best_params, best_score = model.search_cv(
            search_params, X_train, y_train, cv=args.cv_folds
        )
        
        print(f"Best parameters found: {best_params}")
        print(f"Best cross-validation score: {best_score:.4f}")
        
        # Update model with best parameters
        model.model = best_model
        
        # Save model
        save_path = './model/models/xgboost/'
        model_path = model.save_model(save_path)
        print(f"Best model saved to {model_path}")
        
    else:
        print(f"Starting {args.cv_folds}-fold cross validation...")
        
        # Perform enhanced cross-validation
        metrics, fold_predictions, fold_true_labels = enhanced_kfold_cv(
            model, X_train, y_train, k=args.cv_folds
        )
        
        # Print detailed results
        print("\nDetailed Cross-Validation Results:")
        print("=" * 50)
        for metric, values in metrics.items():
            print(f"{metric.replace('_', ' ').title()}: "
                  f"{np.mean(values):.4f} ± {np.std(values):.4f}")
        
        # Feature analysis
        feature_importance = None
        if args.feature_selection:
            try:
                feature_names = [f"CpG_{i}" for i in range(X_train.shape[1])]
                feature_importance = perform_feature_analysis(
                    X_train, y_train, feature_names, args.n_features
                )
            except Exception as e:
                print(f"Feature analysis failed: {e}")
        
        # Train final model on full dataset
        print("\nTraining final model on full dataset...")
        model.train(X_train, y_train)
        
        # Save model and results
        save_path = './model/models/xgboost/'
        model_path = model.save_model(save_path)
        
        # Generate visualizations
        plot_training_results(metrics, save_path)
        
        # Save comprehensive results
        save_comprehensive_results(
            model, metrics, fold_predictions, fold_true_labels, 
            feature_importance, save_path
        )
        
        print(f"Final model saved to {model_path}")
        print(f"Training completed successfully!")

if __name__ == "__main__":
    main()
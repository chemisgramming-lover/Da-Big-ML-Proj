"""
Evaluation Module
Comprehensive evaluation metrics and visualization
"""

import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from pathlib import Path

from data_loader import load_cifar10, get_class_name, get_all_class_names
from utils import plot_confusion_matrix, plot_sample_predictions


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = 'model'
):
    """
    Evaluate model on test set.
    
    Args:
        model: Trained Keras model
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model
        
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\n{'='*60}")
    print(f"📊 Evaluating {model_name.upper()}")
    print(f"{'='*60}")
    
    # Make predictions
    print("🔮 Making predictions...")
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Print metrics
    print(f"\n📈 Overall Metrics:")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    
    # Per-class metrics
    print(f"\n📋 Per-Class Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=get_all_class_names(),
        digits=4
    ))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Visualizations
    plot_confusion_matrix(cm, model_name)
    plot_sample_predictions(model, X_test, y_test, model_name)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    return metrics


def compare_models(model_paths: dict, X_test: np.ndarray, y_test: np.ndarray):
    """
    Compare multiple models on test set.
    
    Args:
        model_paths: Dictionary of {model_name: model_path}
        X_test: Test features
        y_test: Test labels
    """
    print(f"\n{'='*60}")
    print("🏆 MODEL COMPARISON")
    print(f"{'='*60}")
    
    results = {}
    
    for model_name, model_path in model_paths.items():
        print(f"\nLoading {model_name}...")
        model = keras.models.load_model(model_path)
        metrics = evaluate_model(model, X_test, y_test, model_name)
        results[model_name] = metrics
    
    # Create comparison table
    print(f"\n{'='*60}")
    print("📊 RESULTS SUMMARY")
    print(f"{'='*60}\n")
    
    print(f"{'Model':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 63)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<15} {metrics['accuracy']:<12.4f} "
              f"{metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
              f"{metrics['f1']:<12.4f}")
    
    return results


def load_and_evaluate_model(model_path: str, X_test: np.ndarray, y_test: np.ndarray):
    """
    Load a trained model and evaluate it.
    
    Args:
        model_path: Path to saved model
        X_test: Test features
        y_test: Test labels
    """
    print(f"Loading model from {model_path}...")
    model = keras.models.load_model(model_path)
    
    model_name = Path(model_path).stem
    metrics = evaluate_model(model, X_test, y_test, model_name)
    
    return model, metrics


def predict_image(image_path: str, model_path: str):
    """
    Predict class of a single image.
    
    Args:
        image_path: Path to image file
        model_path: Path to saved model
        
    Returns:
        Prediction and confidence
    """
    import cv2
    
    # Load model
    model = keras.models.load_model(model_path)
    
    # Load and preprocess image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (32, 32))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Make prediction
    pred = model.predict(img, verbose=0)
    class_idx = np.argmax(pred[0])
    confidence = pred[0][class_idx]
    
    return {
        'class': get_class_name(class_idx),
        'class_idx': class_idx,
        'confidence': float(confidence),
        'all_probs': pred[0].tolist()
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate CV model on CIFAR-10')
    parser.add_argument('--model_path', type=str,
                        help='Path to saved model')
    parser.add_argument('--image_path', type=str,
                        help='Path to image for prediction')
    parser.add_argument('--compare', action='store_true',
                        help='Compare all models in models/ directory')
    
    args = parser.parse_args()
    
    # Load test data
    print("📥 Loading CIFAR-10 test set...")
    _, _, X_test, y_test = load_cifar10()
    
    if args.image_path:
        result = predict_image(args.image_path, args.model_path)
        print(f"\n🎯 Prediction: {result['class']}")
        print(f"📊 Confidence: {result['confidence']:.4f}")
    
    elif args.compare:
        # Find all models in models/ directory
        model_dir = Path('models')
        model_paths = {
            p.stem: str(p) 
            for p in model_dir.glob('*.h5') 
            if 'best' in p.name
        }
        
        if model_paths:
            compare_models(model_paths, X_test, y_test)
        else:
            print("❌ No trained models found in models/ directory")
    
    elif args.model_path:
        load_and_evaluate_model(args.model_path, X_test, y_test)
    
    else:
        # Evaluate all best models
        model_dir = Path('models')
        model_paths = {
            p.stem: str(p) 
            for p in model_dir.glob('*_best.h5')
        }
        
        if model_paths:
            compare_models(model_paths, X_test, y_test)
        else:
            print("❌ No trained models found. Train models first using train.py")


if __name__ == "__main__":
    main()

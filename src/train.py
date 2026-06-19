"""
Training Pipeline
Handles model training with data augmentation and callbacks
"""

import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from pathlib import Path

from data_loader import (
    load_cifar10, 
    create_train_val_split, 
    get_data_augmentation,
    get_class_name
)
from models import (
    build_custom_cnn,
    build_resnet50,
    build_vgg16,
    compile_model,
    get_model_callbacks
)
from utils import plot_training_history, save_model


def train_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 128,
    learning_rate: float = 0.001
):
    """
    Train a model on CIFAR-10.
    
    Args:
        model_name: Name of the model ('custom', 'resnet50', 'vgg16')
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
    """
    print(f"\n{'='*60}")
    print(f"🚀 Training {model_name.upper()}")
    print(f"{'='*60}")
    
    # Build model
    if model_name == 'custom':
        model = build_custom_cnn()
    elif model_name == 'resnet50':
        model = build_resnet50()
    elif model_name == 'vgg16':
        model = build_vgg16()
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Compile model
    model = compile_model(model, learning_rate=learning_rate)
    
    # Get callbacks
    callbacks = get_model_callbacks(model_name)
    
    # Create data augmentation
    datagen = get_data_augmentation()
    
    # Train model
    print(f"\n📊 Training configuration:")
    print(f"   - Epochs: {epochs}")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Learning rate: {learning_rate}")
    print(f"   - Training samples: {len(X_train)}")
    print(f"   - Validation samples: {len(X_val)}")
    
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        steps_per_epoch=len(X_train) // batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Save model
    save_model(model, model_name)
    
    # Plot training history
    plot_training_history(history, model_name)
    
    print(f"\n✅ {model_name.upper()} training completed!")
    
    return model, history


def main():
    parser = argparse.ArgumentParser(description='Train CV model on CIFAR-10')
    parser.add_argument('--model', type=str, default='custom',
                        choices=['custom', 'resnet50', 'vgg16', 'all'],
                        help='Model to train')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio')
    
    args = parser.parse_args()
    
    # Create necessary directories
    Path('models').mkdir(exist_ok=True)
    Path('results').mkdir(exist_ok=True)
    Path('logs').mkdir(exist_ok=True)
    
    # Load data
    print("📥 Loading CIFAR-10 dataset...")
    X_train, y_train, X_test, y_test = load_cifar10()
    
    # Create validation split
    X_train, y_train, X_val, y_val = create_train_val_split(
        X_train, y_train, val_split=args.val_split
    )
    
    # Train models
    if args.model == 'all':
        models_dict = {}
        for model_name in ['custom', 'resnet50', 'vgg16']:
            model, history = train_model(
                model_name,
                X_train, y_train,
                X_val, y_val,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr
            )
            models_dict[model_name] = (model, history)
    else:
        model, history = train_model(
            args.model,
            X_train, y_train,
            X_val, y_val,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr
        )
    
    print("\n" + "="*60)
    print("🎉 Training complete! Check 'models/' for saved weights.")
    print("="*60)


if __name__ == "__main__":
    main()

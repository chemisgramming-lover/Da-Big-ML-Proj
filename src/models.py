"""
Model Architecture Definitions
Includes ResNet50, VGG16, Custom CNN, and Ensemble
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np


def build_custom_cnn(input_shape: tuple = (32, 32, 3), num_classes: int = 10) -> models.Model:
    """
    Build a custom CNN optimized for CIFAR-10.
    
    Args:
        input_shape: Input image shape
        num_classes: Number of output classes
        
    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                      input_shape=input_shape, name='conv1_1'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_2'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2), name='pool1'),
        layers.Dropout(0.25),
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_1'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_2'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2), name='pool2'),
        layers.Dropout(0.25),
        
        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_1'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_2'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2), name='pool3'),
        layers.Dropout(0.25),
        
        # Flatten and Dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu', name='fc1'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu', name='fc2'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax', name='output')
    ])
    
    return model


def build_resnet50(input_shape: tuple = (32, 32, 3), num_classes: int = 10) -> models.Model:
    """
    Build ResNet50 with transfer learning for CIFAR-10.
    
    Args:
        input_shape: Input image shape
        num_classes: Number of output classes
        
    Returns:
        Compiled Keras model
    """
    # Load pre-trained ResNet50
    base_model = ResNet50(
        weights='imagenet',
        input_shape=input_shape,
        include_top=False,
        pooling='avg'
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Add custom top layers
    model = models.Sequential([
        base_model,
        layers.Dense(512, activation='relu', name='fc1'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu', name='fc2'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax', name='output')
    ])
    
    return model


def build_vgg16(input_shape: tuple = (32, 32, 3), num_classes: int = 10) -> models.Model:
    """
    Build VGG16 with transfer learning for CIFAR-10.
    
    Args:
        input_shape: Input image shape
        num_classes: Number of output classes
        
    Returns:
        Compiled Keras model
    """
    # Load pre-trained VGG16
    base_model = VGG16(
        weights='imagenet',
        input_shape=input_shape,
        include_top=False,
        pooling='avg'
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Add custom top layers
    model = models.Sequential([
        base_model,
        layers.Dense(512, activation='relu', name='fc1'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu', name='fc2'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax', name='output')
    ])
    
    return model


def compile_model(model: models.Model, learning_rate: float = 0.001) -> models.Model:
    """
    Compile model with optimizer, loss, and metrics.
    
    Args:
        model: Keras model
        learning_rate: Learning rate for optimizer
        
    Returns:
        Compiled model
    """
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def get_model_callbacks(model_name: str):
    """
    Get training callbacks for the model.
    
    Args:
        model_name: Name of the model (for checkpoint naming)
        
    Returns:
        List of Keras callbacks
    """
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            f'models/{model_name}_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.TensorBoard(
            log_dir=f'./logs/{model_name}',
            histogram_freq=1
        )
    ]
    
    return callbacks


class EnsembleModel:
    """Ensemble of multiple models for improved predictions."""
    
    def __init__(self, models_list: list):
        """
        Initialize ensemble with list of models.
        
        Args:
            models_list: List of trained Keras models
        """
        self.models = models_list
        self.num_models = len(models_list)
    
    def predict(self, X, voting: str = 'soft'):
        """
        Make ensemble predictions.
        
        Args:
            X: Input data
            voting: 'soft' for averaging probabilities, 'hard' for majority voting
            
        Returns:
            Ensemble predictions
        """
        if voting == 'soft':
            # Average probability predictions
            predictions = np.zeros((X.shape[0], self.models[0].output_shape[-1]))
            for model in self.models:
                predictions += model.predict(X)
            predictions /= self.num_models
            return np.argmax(predictions, axis=1)
        
        elif voting == 'hard':
            # Majority voting
            all_preds = np.array([
                np.argmax(model.predict(X), axis=1) 
                for model in self.models
            ])
            # Mode along axis 0
            ensemble_pred = np.apply_along_axis(
                lambda x: np.bincount(x).argmax(), 
                axis=0, 
                arr=all_preds
            )
            return ensemble_pred
        
        else:
            raise ValueError("voting must be 'soft' or 'hard'")
    
    def predict_proba(self, X):
        """
        Get ensemble probability predictions.
        
        Args:
            X: Input data
            
        Returns:
            Average probability predictions across all models
        """
        predictions = np.zeros((X.shape[0], self.models[0].output_shape[-1]))
        for model in self.models:
            predictions += model.predict(X)
        return predictions / self.num_models


if __name__ == "__main__":
    # Test model building
    print("🏗️ Building Custom CNN...")
    custom_cnn = build_custom_cnn()
    custom_cnn = compile_model(custom_cnn)
    print(f"✅ Custom CNN created with {len(custom_cnn.layers)} layers")
    
    print("\n🏗️ Building ResNet50...")
    resnet = build_resnet50()
    resnet = compile_model(resnet)
    print(f"✅ ResNet50 created with {len(resnet.layers)} layers")
    
    print("\n🏗️ Building VGG16...")
    vgg = build_vgg16()
    vgg = compile_model(vgg)
    print(f"✅ VGG16 created with {len(vgg.layers)} layers")

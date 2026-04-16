"""
F1 Pit Stop Prediction — Inference Pipeline

Standalone module for deployment (Streamlit integration).
Reads all artifacts from Dataset/ — zero dependency on notebooks.

Usage:
    from inference import F1PitPredictor
    predictor = F1PitPredictor()

    # From raw input dict
    result = predictor.predict_from_raw({
        'Driver': 'VER', 'Compound': 'SOFT', 'LapNumber': 30, 'Stint': 2,
        'TyreLife': 15, 'Position': 1, 'LapTime (s)': 91.5, 'LapTime_Delta': 0.3,
        'Cumulative_Degradation': -10.5, 'RaceProgress': 0.5,
        'Position_Change': 0, 'Degradation_Rate': 0.02
    })
    print(result)
    # {'probability': 0.73, 'prediction': 1, 'label': 'PIT'}
"""

import numpy as np
import pandas as pd
import json
import os


class F1PitPredictor:
    """End-to-end predictor: raw input → preprocess → forward pass → prediction."""

    def __init__(self, artifact_dir='Dataset/'):
        self.artifact_dir = artifact_dir
        self._load_artifacts()

    # ── Load all artifacts ──
    def _load_artifacts(self):
        # 1. Feature order
        with open(os.path.join(self.artifact_dir, 'feature_order.json')) as f:
            self.feature_order = json.load(f)

        # 2. Scaler params (mean/std for numeric features)
        scaler = pd.read_csv(
            os.path.join(self.artifact_dir, 'scaler_params.csv'), index_col=0
        )
        self.scaler_mean = scaler['mean'].to_dict()
        self.scaler_std = scaler['std'].to_dict()
        # Features that get standardized = those present in scaler_params
        self.scaled_features = set(scaler.index)

        # 3. Driver mapping
        driver_df = pd.read_csv(
            os.path.join(self.artifact_dir, 'driver_mapping.csv')
        )
        self.driver_map = dict(zip(driver_df['Driver'], driver_df['Encoded']))

        # 4. Model config
        with open(os.path.join(self.artifact_dir, 'model_config.json')) as f:
            self.config = json.load(f)
        self.threshold = self.config['threshold']
        self.model_type = self.config['selected_model']
        self.layer_sizes = self.config['layer_sizes']

        # 5. Model weights
        data = np.load(os.path.join(self.artifact_dir, 'model_weights.npz'))
        if self.model_type == 'lr':
            self.lr_W = data['lr_W']
            self.lr_b = data['lr_b'][0]
        else:
            prefix = 'ann1_' if self.model_type == 'ann_1layer' else 'ann2_'
            n_layers = len(self.layer_sizes) - 1
            self.weights = [data[f'{prefix}W{i}'] for i in range(n_layers)]
            self.biases = [data[f'{prefix}b{i}'] for i in range(n_layers)]

    # ── Activation functions ──
    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def _relu(z):
        return np.maximum(0, z)

    # ── Preprocessing ──
    def preprocess(self, raw_input: dict) -> np.ndarray:
        """
        Convert raw input dict → standardized feature vector in correct order.

        Args:
            raw_input: dict with keys like 'Driver', 'Compound', 'LapNumber', etc.
        Returns:
            np.ndarray of shape (1, n_features)
        """
        features = {}

        # Encode driver
        driver = raw_input.get('Driver', 'UNK')
        features['Driver_encoded'] = float(self.driver_map.get(driver, 0))

        # One-hot encode compound
        compound = raw_input.get('Compound', 'MEDIUM')
        for c in ['HARD', 'INTERMEDIATE', 'MEDIUM', 'SOFT', 'WET']:
            features[f'Compound_{c}'] = 1.0 if compound == c else 0.0

        # Numeric features
        numeric_keys = [
            'LapNumber', 'Stint', 'TyreLife', 'Position', 'LapTime (s)',
            'LapTime_Delta', 'Cumulative_Degradation', 'RaceProgress',
            'Position_Change', 'Degradation_Rate'
        ]
        for key in numeric_keys:
            features[key] = float(raw_input.get(key, 0.0))

        # Standardize only features that were scaled during training
        for key in numeric_keys:
            if key in self.scaler_mean:
                mean = self.scaler_mean[key]
                std = self.scaler_std[key]
                features[key] = (features[key] - mean) / std

        # Build feature vector in locked order
        X = np.array([[features[f] for f in self.feature_order]], dtype=np.float64)
        return X

    # ── Forward pass ──
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass through the selected model.

        Args:
            X: np.ndarray of shape (m, n_features), already preprocessed
        Returns:
            np.ndarray of shape (m,) with probabilities
        """
        if self.model_type == 'lr':
            z = X @ self.lr_W + self.lr_b
            return self._sigmoid(z).flatten()
        else:
            A = X
            n_layers = len(self.weights)
            for i in range(n_layers):
                Z = A @ self.weights[i] + self.biases[i]
                if i < n_layers - 1:
                    A = self._relu(Z)
                else:
                    A = self._sigmoid(Z)
            return A.flatten()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class (0 or 1) using the tuned threshold."""
        proba = self.predict_proba(X)
        return (proba >= self.threshold).astype(int)

    # ── High-level API (for Streamlit) ──
    def predict_from_raw(self, raw_input: dict) -> dict:
        """
        End-to-end: raw input → preprocessed → prediction.

        Args:
            raw_input: dict with raw feature values
        Returns:
            dict with 'probability', 'prediction', 'label'
        """
        X = self.preprocess(raw_input)
        proba = self.predict_proba(X)[0]
        pred = int(proba >= self.threshold)
        return {
            'probability': round(float(proba), 4),
            'prediction': pred,
            'label': 'PIT' if pred == 1 else 'NO PIT',
            'threshold': self.threshold,
            'model': self.model_type
        }


# ── Quick self-test ──
if __name__ == '__main__':
    predictor = F1PitPredictor(artifact_dir='Dataset/')
    print(f'Model: {predictor.model_type}')
    print(f'Threshold: {predictor.threshold:.4f}')
    print(f'Features: {predictor.feature_order}')
    print(f'Layer sizes: {predictor.layer_sizes}')

    # Test with a sample input
    sample = {
        'Driver': 'VER', 'Compound': 'SOFT',
        'LapNumber': 30, 'Stint': 2, 'TyreLife': 15,
        'Position': 1, 'LapTime (s)': 91.5,
        'LapTime_Delta': 0.3, 'Cumulative_Degradation': -10.5,
        'RaceProgress': 0.5, 'Position_Change': 0,
        'Degradation_Rate': 0.02
    }
    result = predictor.predict_from_raw(sample)
    print(f'\nSample prediction: {result}')

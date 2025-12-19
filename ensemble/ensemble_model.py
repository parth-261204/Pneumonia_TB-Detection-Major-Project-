import numpy as np
import json
import yaml
from tensorflow.keras.models import load_model

class EnsembleModel:
    """Weighted ensemble of multiple models"""
    
    def __init__(self, model_paths, weights=None, config_path='config/ensemble_config.yaml'):
        """Initialize ensemble with model paths and weights"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.models = [load_model(path) for path in model_paths]
        self.model_names = [path.split('/')[-1].replace('_best.h5', '') for path in model_paths]
        
        if weights is None:
            model_weights = {m['name']: m['weight'] for m in config['ensemble']['models']}
            self.weights = [model_weights.get(name, 0.25) for name in self.model_names]
        else:
            self.weights = weights
        
        self.weights = np.array(self.weights)
        self.weights = self.weights / np.sum(self.weights)
        
        self.class_names = ['Normal', 'Tuberculosis', 'Pneumonia']
    
    def predict(self, image):
        """Make weighted ensemble prediction"""
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        predictions = []
        for model in self.models:
            pred = model.predict(image, verbose=0)
            predictions.append(pred[0])
        
        predictions = np.array(predictions)
        
        weighted_pred = np.average(predictions, axis=0, weights=self.weights)
        
        pred_class = np.argmax(weighted_pred)
        confidence = weighted_pred[pred_class]
        
        return {
            'predicted_class': self.class_names[pred_class],
            'predicted_index': int(pred_class),
            'confidence': float(confidence),
            'probabilities': {
                self.class_names[i]: float(weighted_pred[i]) 
                for i in range(len(self.class_names))
            },
            'individual_predictions': {
                self.model_names[i]: {
                    'class': self.class_names[np.argmax(predictions[i])],
                    'confidence': float(np.max(predictions[i])),
                    'probabilities': predictions[i].tolist()
                }
                for i in range(len(self.models))
            }
        }
    
    def save_weights(self, filepath='ensemble/ensemble_weights.json'):
        """Save ensemble weights to file"""
        weights_dict = {
            name: float(weight) 
            for name, weight in zip(self.model_names, self.weights)
        }
        
        with open(filepath, 'w') as f:
            json.dump(weights_dict, f, indent=4)
    
    @classmethod
    def load_from_config(cls, config_path='config/ensemble_config.yaml'):
        """Load ensemble from configuration file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        with open('config/paths.yaml', 'r') as f:
            paths = yaml.safe_load(f)
        
        model_paths = [
            paths['models'][m['name']] 
            for m in config['ensemble']['models']
        ]
        
        weights = [m['weight'] for m in config['ensemble']['models']]
        
        return cls(model_paths, weights, config_path)

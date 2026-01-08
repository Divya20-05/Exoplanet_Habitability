"""
Flask Backend API for Exoplanet Habitability Prediction
========================================================

REST API endpoints:
- POST /api/predict - Single exoplanet prediction
- POST /api/predict_batch - Batch predictions
- GET /api/model_info - Model metadata
- GET /api/features - Required features list
- GET /api/health - Health check
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime
import traceback

# Initialize Flask app
app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)  # Enable CORS for frontend access

# Load configuration
import sys
sys.path.append(str(Path(__file__).parent))
import config

# Load production model
# Load production model (Switching to simplified model for better performance on 8-feature inputs)
MODEL_PATH = config.MODELS_DIR / 'production' / 'simplified_8feature_model.pkl'
# We still use the full model card for general metadata, or we could assume it applies roughly
MODEL_CARD_PATH = config.MODELS_DIR / 'production' / 'model_card_v1.0.0.json'
FEATURE_NAMES_PATH = config.MODELS_DIR / 'production' / 'simplified_features.json'

print("Loading simplified production model...")
model = joblib.load(MODEL_PATH)
print(f"✓ Model loaded from: {MODEL_PATH}")

# Load model metadata
with open(MODEL_CARD_PATH, 'r') as f:
    model_card = json.load(f)

with open(FEATURE_NAMES_PATH, 'r') as f:
    feature_info = json.load(f)
    required_features = feature_info['features']

print(f"✓ Model ready with {len(required_features)} features")

# Class names - Binary classification (combining Habitable and Optimistic Habitable)
CLASS_NAMES = {
    0: "Non-Habitable",
    1: "Habitable",
    2: "Habitable"  # Optimistic Habitable now mapped to Habitable
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def validate_input(data):
    """Validate input data structure"""
    if not isinstance(data, dict):
        return False, "Input must be a JSON object"
    
    # Validation needs to handle the S_TEMP alias for S_TEMPERATURE
    # Create a copy of keys to check against required features
    input_keys = set(data.keys())
    if 'S_TEMP' in input_keys:
        input_keys.add('S_TEMPERATURE')
        
    # Check if all required features are present
    missing_fields = [f for f in required_features if f not in input_keys]
    if missing_fields:
        return False, f"Missing required fields: {missing_fields}"
    
    # Check for numeric values
    for field in required_features:
        # Check actual field name or alias
        check_field = field
        if field == 'S_TEMPERATURE' and 'S_TEMP' in data:
            check_field = 'S_TEMP'
            
        if check_field in data:
            try:
                float(data[check_field])
            except (ValueError, TypeError):
                return False, f"Field '{check_field}' must be numeric"
    
    return True, "Valid"

def prepare_input(data):
    """Convert simple input dict to feature vector for simplified model"""
    # The simplified model pipeline includes scaling, so we just need
    # to provide the raw values in the correct order
    
    # Create dictionary with correct field names (mapping UI names if needed, but they seem to match now)
    # The simplified_features.json uses 'S_TEMPERATURE' but UI sends 'S_TEMP'
    # We need to map 'S_TEMP' -> 'S_TEMPERATURE'
    
    processed_data = {}
    
    # Direct mapping for most fields
    for feature in required_features:
        if feature in data:
            # Round to 2 decimal places
            processed_data[feature] = round(float(data[feature]), 2)
        elif feature == 'S_TEMPERATURE' and 'S_TEMP' in data:
            # Round to 2 decimal places
            processed_data[feature] = round(float(data['S_TEMP']), 2)
            
    # Create DataFrame in the exact order the model expects
    df = pd.DataFrame([processed_data])[required_features]
    
    return df

def format_prediction_response(prediction, probabilities, input_data=None):
    """Format prediction response with binary classification"""
    pred_class = int(prediction[0])
    proba = probabilities[0].tolist()
    
    # Combine probabilities for binary classification
    # Classes 1 and 2 are both "Habitable"
    prob_non_habitable = float(proba[0])
    prob_habitable = float(proba[1] + proba[2])  # Combine Habitable + Optimistic Habitable
    
    response = {
        'prediction': {
            'class': pred_class,
            'class_name': CLASS_NAMES[pred_class],
            'confidence': float(max(prob_non_habitable, prob_habitable)),
            'probabilities': {
                'non_habitable': prob_non_habitable,
                'habitable': prob_habitable
            }
        },
        'timestamp': datetime.now().isoformat()
    }
    
    if input_data:
        response['input_summary'] = {
            'n_features': len(input_data),
            'sample_features': dict(list(input_data.items())[:5])  # First 5 features
        }
    
    return response

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/')
def index():
    """Serve the main frontend page"""
    return app.send_static_file('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/model_info', methods=['GET'])
def get_model_info():
    """Get model metadata and performance metrics"""
    return jsonify({
        'model_info': model_card['model_info'],
        'performance_metrics': model_card['performance_metrics'],
        'training_info': model_card['training_info'],
        'usage': model_card['usage']
    })

@app.route('/api/features', methods=['GET'])
def get_features():
    """Get list of required features"""
    # Get top important features for display
    try:
        importance_df = pd.read_csv(config.REPORTS_DIR / 'feature_importance.csv')
        top_features = importance_df.head(20).to_dict('records')
    except:
        top_features = []
    
    return jsonify({
        'total_features': len(required_features),
        'required_features': required_features[:50],  # First 50 for display
        'top_important_features': top_features,
        'note': 'All features must be provided for prediction'
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Single exoplanet habitability prediction"""
    try:
        # Get input data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        # Validate input
        is_valid, message = validate_input(data)
        if not is_valid:
            return jsonify({'error': message}), 400
        
        # Prepare input
        X = prepare_input(data)
        
        # Make prediction
        prediction = model.predict(X)
        probabilities = model.predict_proba(X)
        
        # Format response
        response = format_prediction_response(prediction, probabilities, data)
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'error': 'Prediction failed',
            'details': str(e)
        }), 500

@app.route('/api/predict_batch', methods=['POST'])
def predict_batch():
    """Batch exoplanet habitability predictions"""
    try:
        # Get input data
        data = request.get_json()
        
        if not data or 'samples' not in data:
            return jsonify({'error': 'No samples provided. Use {"samples": [...]}'})  , 400
        
        samples = data['samples']
        
        if not isinstance(samples, list):
            return jsonify({'error': 'Samples must be a list'}), 400
        
        if len(samples) == 0:
            return jsonify({'error': 'Empty samples list'}), 400
        
        if len(samples) > 100:
            return jsonify({'error': 'Maximum 100 samples per batch'}), 400
        
        # Validate and prepare all samples
        results = []
        for idx, sample in enumerate(samples):
            is_valid, message = validate_input(sample)
            if not is_valid:
                results.append({
                    'index': idx,
                    'error': message,
                    'prediction': None
                })
            else:
                try:
                    X = prepare_input(sample)
                    prediction = model.predict(X)
                    probabilities = model.predict_proba(X)
                    
                    pred_response = format_prediction_response(prediction, probabilities)
                    results.append({
                        'index': idx,
                        'prediction': pred_response['prediction'],
                        'error': None
                    })
                except Exception as e:
                    results.append({
                        'index': idx,
                        'error': str(e),
                        'prediction': None
                    })
        
        # Summary
        successful = sum(1 for r in results if r['error'] is None)
        failed = len(results) - successful
        
        return jsonify({
            'results': results,
            'summary': {
                'total': len(results),
                'successful': successful,
                'failed': failed
            },
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        print(f"Error in batch prediction: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'error': 'Batch prediction failed',
            'details': str(e)
        }), 500

@app.route('/api/example', methods=['GET'])
def get_example():
    """Get example input data for testing"""
    # Return Earth as the default example
    return jsonify({
        'example_input': {
            'P_MASS_EST': 1.0,
            'P_RADIUS_EST': 1.0,
            'P_TEMP_EQUIL': 288.0,
            'P_PERIOD': 365.25,
            'P_FLUX': 1.0,
            'S_MASS': 1.0,
            'S_RADIUS': 1.0,
            'S_TEMP': 5778.0
        },
        'note': 'Earth - our reference for habitability'
    }), 200

@app.route('/api/planets', methods=['GET'])
def get_planets():
    """Get curated list of planets for dropdown selection"""
    try:
        planet_presets_path = config.OUTPUT_DIR / 'planet_presets.json'
        
        if not planet_presets_path.exists():
            return jsonify({
                'error': 'Planet presets not found',
                'details': 'Run prepare_planet_data.py to generate planet presets'
            }), 404
        
        with open(planet_presets_path, 'r') as f:
            planet_data = json.load(f)
        
        return jsonify(planet_data), 200
        
    except Exception as e:
        print(f"Error loading planet presets: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'error': 'Failed to load planet presets',
            'details': str(e)
        }), 500

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("EXOPLANET HABITABILITY PREDICTION API")
    print("=" * 80)
    print(f"\n🚀 Starting Flask server...")
    print(f"📊 Model: {model_card['model_info']['name']} v{model_card['model_info']['version']}")
    print(f"🎯 F1 Score: {model_card['performance_metrics']['f1_score']:.4f}")
    print(f"\n📡 API Endpoints:")
    print(f"  GET  /                    - Frontend UI")
    print(f"  GET  /api/health          - Health check")
    print(f"  GET  /api/model_info      - Model metadata")
    print(f"  GET  /api/features        - Required features")
    print(f"  GET  /api/planets         - Planet presets")
    print(f"  GET  /api/example         - Example input")
    print(f"  POST /api/predict         - Single prediction")
    print(f"  POST /api/predict_batch   - Batch predictions")
    
    # Use PORT environment variable for Render
    port = int(os.environ.get("PORT", 5050))
    print(f"\n🌐 Server running on: http://localhost:{port}")
    print("=" * 80)
    
    app.run(debug=False, host='0.0.0.0', port=port)

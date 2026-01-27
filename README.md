## 🚀 Deployment to Render

This project is optimized for deployment as a Web Service on **[Render](https://render.com/)**.

### How to Deploy:
1. **GitHub Link**: `https://github.com/Divya20-05/Exoplanet_Habitability.git`
2. **Setup on Render**:
   - Create a new **Web Service**.
   - Connect the repository.
   - **Python Version**: Select **Python 3.11** (auto-detected via `runtime.txt`).
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   
### Feature List
- **High-Performance ML Model**: 99.65% F1 Score.
- **REST API**: Flask backend with comprehensive endpoints.
- **Production-Ready**: Configured with Procfile, Gunicorn, and relative paths.

## 📡 API Endpoints

### Health Check
```bash
GET /api/health
```

### Model Information
```bash
GET /api/model_info
```

### Single Prediction
```bash
POST /api/predict
Content-Type: application/json

{
  "P_MASS_EST": 1.0,
  "P_RADIUS_EST": 1.0,
  "P_TEMP_EQUIL": 288,
  "P_PERIOD": 365,
  "P_FLUX": 1.0,
  "S_MASS": 1.0,
  "S_RADIUS": 1.0,
  "S_TEMP": 5778
}
```

### Batch Predictions
```bash
POST /api/predict_batch
Content-Type: application/json

{
  "samples": [
    { /* exoplanet data 1 */ },
    { /* exoplanet data 2 */ }
  ]
}
```

### Get Example Data
```bash
GET /api/example
```

### Get Required Features
```bash
GET /api/features
```

## 🏗️ Project Structure

```
Infosys_Exoplanet/
├── app.py                          # Flask backend API
├── config.py                       # Configuration settings
├── 01_data_quality_assessment.py  # Phase 1: Data quality
├── 02_data_cleaning.py             # Phase 2: Data cleaning
├── 03_encoding_scaling.py          # Phase 3: Preprocessing
├── 04_eda.py                       # Phase 4: EDA
├── 05_dimensionality_reduction.py  # Phase 5: Dimensionality
├── 06_baseline_models.py           # Phase 6: Baseline models
├── 07_smote_sampling.py            # Phase 7: SMOTE techniques
├── 08_full_pipeline.py             # Phase 9: Full pipeline
├── 09_model_evaluation.py          # Phase 10: Evaluation
├── 10_hyperparameter_tuning.py     # Phase 11: Tuning
├── 11_model_interpretability.py    # Phase 12: Interpretability
├── 12_final_model_deployment.py    # Phases 13-15: Deployment
├── static/
│   ├── index.html                  # Frontend UI
│   ├── css/style.css               # Premium styling
│   └── js/app.js                   # Frontend logic
├── outputs/
│   ├── models/
│   │   └── production/             # Production models
│   ├── reports/                    # Analysis reports
│   └── plots/                      # Visualizations
└── requirements.txt                # Python dependencies

```

## 🎯 ML Pipeline Phases

1. **Data Quality Assessment** - Analyze dataset quality
2. **Data Cleaning** - Handle missing values and outliers
3. **Encoding & Scaling** - Preprocess features
4. **EDA** - Exploratory data analysis
5. **Dimensionality Reduction** - PCA visualization
6. **Baseline Models** - Initial model training
7. **SMOTE Sampling** - Handle class imbalance
8. **Full Pipeline** - End-to-end ML pipeline
9. **Model Evaluation** - Comprehensive metrics
10. **Hyperparameter Tuning** - Optimize performance
11. **Model Interpretability** - Feature importance
12. **Final Deployment** - Production-ready model

## 🌐 Web Interface Features

- **Input Form**: Enter exoplanet parameters
- **Real-time Predictions**: Instant habitability classification
- **Probability Distribution**: Visual breakdown of prediction confidence
- **Model Information**: View model metadata and performance
- **Example Data**: Load sample exoplanet for testing
- **Responsive Design**: Works on desktop, tablet, and mobile

## 🔬 Technology Stack

- **Backend**: Flask, Python 3.13
- **ML Framework**: scikit-learn, imbalanced-learn
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Visualization**: Chart.js, Matplotlib, Seaborn
- **Styling**: Bootstrap 5, Custom CSS with Glassmorphism

## 📈 Model Details

- **Algorithm**: Linear SVM with SMOTE sampling
- **Training Samples**: 3,238
- **Test Samples**: 810
- **Features**: 6,509 (after encoding)
- **Classes**: 3 (Non-Habitable, Habitable, Optimistic Habitable)

## 🎨 UI Design

- **Theme**: Modern space-themed with vibrant gradients
- **Effects**: Glassmorphism, smooth animations, hover effects
- **Colors**: Purple/blue gradients with accent colors
- **Typography**: Inter font family
- **Responsive**: Mobile-first design approach

## 📝 License

This project is part of the Infosys Springboard program.

## 👥 Contributors

Divya Vishwanath

## 🙏 Acknowledgments

- Infosys Springboard for the learning opportunity
- PHL Exoplanet Catalog for the dataset
- scikit-learn and imbalanced-learn communities

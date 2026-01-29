# Alzheimer's DNA Methylation Early Detection Platform

> **🔄 Enhanced Fork** from [hackbio-ca/epigenetic-memory-loss-methylation](https://github.com/hackbio-ca/epigenetic-memory-loss-methylation)  
> This fork improves the original hackathon project that my team did during HackBio'25 with enhanced PyTorch CNN + XGBoost models, advanced feature selection, more accuracy, and XAI capabilities.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org)
[![XGBoost](https://img.shields.io/badge/xgboost-latest-orange.svg)](https://xgboost.readthedocs.io)

## 🧬 Abstract

This platform leverages epigenetic markers for early detection of cognitive impairment and Alzheimer's disease. By analyzing DNA methylation profiles from blood samples, our enhanced machine learning models predict disease risk, monitor cognitive decline, and identify key biomarkers.

Epigenetics involves the changes in gene expression without altering the DNA sequence and can be passed down through generations. It is regulated by chemical modifications which alter the DNA molecule to inhibit or express a gene. One such mechanism is called DNA methylation which adds a methyl group to the CpG dinucleotide of the DNA molecule. Both environmental factors and genetics can shape DNA methylation patterns and whether or not a gene is expressed. Alzheimer’s disease, a neurodegenerative condition affected by both genetic factors and environment, may therefore be studied through these epigenetic signatures. By analyzing DNA methylation profiles, we aim to monitor cognitive decline, identify disease-associated genes and CpG sites, and predict Alzheimer’s risk using machine learning. In this project, DNA methylation profiles extracted from blood samples of individuals with Alzheimer’s disease, those experiencing cognitive impairment, and healthy controls will be used to create a dataset for training the machine learning model. This model will predict the likelihood of Alzheimer’s as well as monitor cognitive decline based on methylation patterns. Furthermore, by analyzing feature importance scores from the trained model, we can identify the specific methylation sites and genes most strongly associated with Alzheimer’s, offering valuable insights into disease mechanisms, potential biomarkers, and cognitive impairment.

**Key Features:**
- **Multi-Model Architecture**: Enhanced PyTorch CNN + Optimized XGBoost for ~850k CpG sites
- **Feature Selection**: EWAS-based statistical feature selection for improved accuracy  
- **Explainable AI**: SHAP analysis reveals critical CpG sites and gene associations
- **Interactive Platform**: Next.js frontend with FastAPI backend for seamless predictions

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- Git

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/alzheimers-dna-early-detection-platform.git
cd alzheimers-dna-early-detection-platform
```

2. **Backend Setup**
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
pip install -r backend/requirements.txt
```

3. **Frontend Setup**
```bash
cd frontend
npm install
# or
pnpm install
```


### Running the Application

1. **Start Backend Server**
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

2. **Start Frontend Development Server**
```bash
cd frontend
npm run dev
# or
pnpm dev
```

3. **Access the Platform**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## Project Architecture

```
├── backend/           # FastAPI backend with ML inference
│   ├── app/          # Application logic
│   └── main.py       # FastAPI server entry point
├── frontend/         # Next.js React frontend
│   ├── components/   # Reusable UI components  
│   ├── app/         # Next.js app router pages
│   └── lib/         # Utility functions
├── model/           # ML pipeline and models
│   ├── models/      # Enhanced PyTorch CNN + XGBoost
│   ├── train/       # Training scripts with cross-validation
│   ├── data/        # Data loaders and preprocessing
│   └── utils/       # ML utilities and helpers
└── docs/           # Documentation and examples
```

## Usage

### Web Interface
1. Upload CSV file containing DNA methylation data (CpG beta values)
2. Select analysis parameters and model preferences  
3. View predictions with risk scores and confidence intervals
4. Explore SHAP visualizations showing important CpG sites
5. Export results for clinical or research use

### API Integration
```python
import requests

# Upload methylation data
files = {'file': open('methylation_data.csv', 'rb')}
response = requests.post('http://localhost:8000/api/predict', files=files)

# Get predictions and SHAP analysis
results = response.json()
predictions = results['results']
shap_values = results['shap_analysis']
```

### Command Line Training
```bash
# Enhanced XGBoost with grid search
python -m model.train.xgboost.train --grid-search --cv-folds 5

# PyTorch CNN with different architectures  
python -m model.train.pytorch.train --model convnet --epochs 50
python -m model.train.pytorch.train --model hybrid --epochs 100
```

## 📊 Data Format

Input CSV should contain:
- Rows: Samples/Participants  
- Columns: CpG site beta values (0-1 range)
- First column: Sample IDs
- Mapping file: Sample metadata with disease states

Example structure:
```csv
sample_id,cg00000029,cg00000108,cg00000109,...
SAMPLE_001,0.8234,0.1456,0.9876,...
SAMPLE_002,0.7123,0.2345,0.8765,...
```

## 🧪 Development

### Testing
```bash
# Backend tests
cd backend
python -m pytest tests/

# Frontend tests  
cd frontend
npm test
```

### Contributing
Refer to the [original epigenetic-memory-loss-methylation project](https://github.com/hackbio-ca/epigenetic-memory-loss-methylation) and contribution guidelines for more information.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Fork Notice
This is an enhanced fork of the [original epigenetic-memory-loss-methylation project](https://github.com/hackbio-ca/epigenetic-memory-loss-methylation) developed by the HackBio team during a hackathon. The original work was a collaborative effort, and this fork includes significant improvements to the machine learning models and platform architecture.

**Original Repository**: [hackbio-ca/epigenetic-memory-loss-methylation](https://github.com/hackbio-ca/epigenetic-memory-loss-methylation)  
**Fork Enhancements**: 
- Enhanced PyTorch CNN architecture
- Optimized XGBoost with advanced feature selection (F-test + mutual information)
- Improved model accuracy through stratified 5-fold cross-validation
- Comprehensive SHAP-based explainable AI implementation
- Backend and frontend enhancements

## Acknowledgments

- **HackBio Canada** and my original hackathon team for the foundational work and collaborative project

## Support

For questions about this fork or technical issues:
- Open an [issue](https://github.com/your-username/alzheimers-dna-early-detection-platform/issues)
- Check existing documentation in the `/docs` folder
- Review the API documentation at `/docs` endpoint

---

**Disclaimer**: This platform is for research purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment.
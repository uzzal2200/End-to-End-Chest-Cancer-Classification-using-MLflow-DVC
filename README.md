# 🫁 End-to-End Chest Cancer Classification using MLflow & DVC

[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Docker](https://img.shields.io/badge/docker-enabled-brightgreen.svg)](https://www.docker.com/)
[![MLflow](https://img.shields.io/badge/MLflow-tracking-blue.svg)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/DVC-pipeline-orange.svg)](https://dvc.org/)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-success.svg)](https://github.com/features/actions)

An end-to-end deep learning project for classifying chest CT scan images to detect **Adenocarcinoma Cancer** using **VGG16** transfer learning. The project implements MLOps best practices with **MLflow** for experiment tracking and **DVC** for data versioning and pipeline orchestration, deployed using **AWS ECR**, **EC2**, and **GitHub Actions CI/CD**.

---

## 📋 Table of Contents

- [Features](#-features)
- [Project Architecture](#-project-architecture)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Usage](#-usage)
- [Development Workflow](#-development-workflow)
- [MLflow Integration](#-mlflow-integration)
- [DVC Pipeline](#-dvc-pipeline)
- [AWS Deployment](#-aws-deployment-cicd)
- [Project Structure](#-project-structure)
- [Technologies Used](#-technologies-used)

---

## ✨ Features

- 🔬 **Deep Learning Model**: VGG16-based transfer learning for binary classification (Normal vs Adenocarcinoma)
- 📊 **Experiment Tracking**: MLflow integration for tracking experiments, metrics, and model versioning
- 🔄 **Data Versioning**: DVC for managing large datasets and creating reproducible ML pipelines
- 🐳 **Containerization**: Docker support for consistent environments
- ☁️ **Cloud Deployment**: Automated deployment on AWS (ECR + EC2)
- 🚀 **CI/CD Pipeline**: GitHub Actions for continuous integration and deployment
- 🎯 **Modular Design**: Clean, maintainable code structure with configuration management
- 📈 **Model Evaluation**: Comprehensive evaluation with MLflow logging

---

## 🏗️ Project Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│   Data      │───▶│ Prepare Base │───▶│   Model     │───▶│   Model      │
│  Ingestion  │    │    Model     │    │  Training   │    │  Evaluation  │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
      │                    │                   │                   │
      └────────────────────┴───────────────────┴───────────────────┘
                                   │
                           ┌───────▼────────┐
                           │  MLflow + DVC  │
                           │   Tracking     │
                           └───────┬────────┘
                                   │
                           ┌───────▼────────┐
                           │  Docker Image  │
                           └───────┬────────┘
                                   │
                           ┌───────▼────────┐
                           │  AWS ECR + EC2 │
                           └────────────────┘
```

---

## 📊 Dataset

The project uses the **Chest CT Scan Dataset** containing:
- **Normal** chest CT scans
- **Adenocarcinoma** cancer CT scans

Dataset is automatically downloaded and extracted during the data ingestion pipeline stage.

---

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Git
- Docker (optional, for containerization)
- AWS Account (for deployment)

### Local Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/End-to-End-Chest-Cancer-Classification-using-MLflow-DVC.git
cd End-to-End-Chest-Cancer-Classification-using-MLflow-DVC
```

2. **Create a virtual environment**
```bash
python -m venv cancer
source cancer/bin/activate  # On Windows: cancer\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure the project**
   - Update `config/config.yaml` with your settings
   - Update `params.yaml` with model hyperparameters

---

## 🚀 Usage

### Training the Model

Run the complete pipeline:
```bash
python main.py
```

This executes all stages:
1. Data Ingestion
2. Prepare Base Model
3. Model Training
4. Model Evaluation with MLflow

### Running the Web Application

```bash
python app.py
```

Access at `http://localhost:8080`

### Using Docker

```bash
# Build Docker image
docker build -t chest-cancer-classifier .

# Run container
docker run -p 8080:8080 chest-cancer-classifier
```

---

## 🔄 Development Workflow

When adding new features, follow this workflow:

1. **Update Configuration Files**
   - `config/config.yaml`: Data paths, artifacts, parameters
   - `params.yaml`: Model hyperparameters

2. **Update Entity**
   - Define data classes in `src/cnnClassifier/entity/config_entity.py`

3. **Update Configuration Manager**
   - Add configuration methods in `src/cnnClassifier/config/configuration.py`

4. **Create Components**
   - Implement logic in `src/cnnClassifier/components/`

5. **Create Pipeline Stage**
   - Add pipeline in `src/cnnClassifier/pipeline/`

6. **Update Main Pipeline**
   - Integrate in `main.py`

7. **Update DVC Pipeline**
   - Add stage to `dvc.yaml`

---

## 📊 MLflow Integration

### Tracking Experiments

```bash
# Start MLflow UI
mlflow ui
```

Access at `http://localhost:5000`

### Using DagsHub for Remote Tracking

```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/yourusername/chest-Disease-Classification-MLflow-DVC.mlflow
export MLFLOW_TRACKING_USERNAME=yourusername
export MLFLOW_TRACKING_PASSWORD=your_token

python main.py
```

### What MLflow Tracks
- ✅ Model parameters (learning rate, batch size, epochs)
- ✅ Training metrics (loss, accuracy)
- ✅ Model artifacts
- ✅ Experiment metadata

### MLflow Benefits
- 🎯 **Production-grade**: Enterprise-ready experiment tracking
- 📈 **Comprehensive Logging**: Track all experiments and model versions
- 🏷️ **Model Registry**: Tag and version your models
- 🔄 **Reproducibility**: Reproduce any experiment with logged parameters

---

## 🔄 DVC Pipeline

### Initialize DVC
```bash
dvc init
```

### Reproduce Pipeline
```bash
dvc repro
```

### Visualize Pipeline DAG
```bash
dvc dag
```

### DVC Benefits
- 💡 **Lightweight**: Perfect for POCs and small projects
- 📦 **Data Versioning**: Version control for large datasets
- 🔄 **Pipeline Orchestration**: Create and manage ML pipelines
- 🎯 **Reproducibility**: Ensure consistent results across runs

---

## ☁️ AWS Deployment (CI/CD)

### Prerequisites

#### 1. AWS IAM User
Create IAM user with policies:
- `AmazonEC2ContainerRegistryFullAccess`
- `AmazonEC2FullAccess`

#### 2. Create ECR Repository
```bash
aws ecr create-repository --repository-name chest-cancer-classifier --region us-east-1
```
Save the repository URI (e.g., `566373416292.dkr.ecr.us-east-1.amazonaws.com/chest-cancer-classifier`)

#### 3. Launch EC2 Instance
- Choose Ubuntu Server
- Instance type: t2.medium or higher
- Configure security group to allow port 8080

#### 4. Install Docker on EC2
```bash
sudo apt-get update -y
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker
```

#### 5. Configure Self-Hosted Runner
1. Go to GitHub repo → Settings → Actions → Runners
2. Click "New self-hosted runner"
3. Select Linux
4. Run the provided commands on your EC2 instance

#### 6. Configure GitHub Secrets
Add these secrets in GitHub repo → Settings → Secrets and variables → Actions:

```
AWS_ACCESS_KEY_ID=<your_access_key>
AWS_SECRET_ACCESS_KEY=<your_secret_key>
AWS_REGION=us-east-1
AWS_ECR_LOGIN_URI=<your_ecr_uri>
ECR_REPOSITORY_NAME=chest-cancer-classifier
```

---

### CI/CD Workflow

The GitHub Actions workflow (`.github/workflows/main.yaml`) automatically:

1. **Continuous Integration**
   - Lints code
   - Runs unit tests

2. **Continuous Delivery**
   - Builds Docker image
   - Pushes to AWS ECR

3. **Continuous Deployment**
   - Pulls latest image from ECR
   - Deploys on EC2
   - Cleans up old containers

**Trigger**: Push to `main` branch (excluding README.md changes)

---

## 📁 Project Structure

```
End-to-End-Chest-Cancer-Classification/
│
├── .github/
│   └── workflows/
│       └── main.yaml              # CI/CD pipeline configuration
│
├── artifacts/                     # Generated artifacts
│   ├── data_ingestion/           # Downloaded dataset
│   ├── prepare_base_model/       # Base VGG16 models
│   └── training/                 # Trained model
│
├── config/
│   └── config.yaml               # Main configuration file
│
├── logs/                         # Application logs
│
├── mlruns/                       # MLflow experiment tracking
│
├── research/                     # Jupyter notebooks for experimentation
│   ├── 01_data_ingestion.ipynb
│   ├── 02_prepare_base_model.ipynb
│   ├── 03_model_trainer.ipynb
│   └── 04_model_evaluation_with_mlflow.ipynb
│
├── src/cnnClassifier/
│   ├── components/               # Core ML components
│   │   ├── data_ingestion.py
│   │   ├── prepare_base_model.py
│   │   ├── model_trainer.py
│   │   └── model_evaluation_mlflow.py
│   │
│   ├── config/                   # Configuration management
│   │   └── configuration.py
│   │
│   ├── constants/                # Constants and paths
│   │   └── __init__.py
│   │
│   ├── entity/                   # Data classes
│   │   └── config_entity.py
│   │
│   ├── pipeline/                 # Training and prediction pipelines
│   │   ├── stage_01_data_ingestion.py
│   │   ├── stage_02_prepare_base_model.py
│   │   ├── stage_03_model_trainer.py
│   │   ├── stage_04_model_evaluation.py
│   │   └── prediction.py
│   │
│   └── utils/                    # Utility functions
│       └── common.py
│
├── templates/
│   └── index.html               # Web UI template
│
├── app.py                       # Flask web application
├── main.py                      # Main training pipeline
├── Dockerfile                   # Docker configuration
├── dvc.yaml                     # DVC pipeline definition
├── params.yaml                  # Model hyperparameters
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup
└── README.md                    # Project documentation
```

---

## 🛠️ Technologies Used

### Core Technologies
- **Python 3.8**: Programming language
- **TensorFlow/Keras**: Deep learning framework
- **VGG16**: Pre-trained CNN model for transfer learning

### MLOps Tools
- **MLflow**: Experiment tracking and model registry
- **DVC**: Data versioning and pipeline orchestration
- **DagsHub**: Remote MLflow tracking server

### Deployment
- **Docker**: Containerization
- **AWS ECR**: Container registry
- **AWS EC2**: Cloud computing
- **GitHub Actions**: CI/CD automation

### Web Framework
- **Flask**: Web application framework

### Development Tools
- **Jupyter Notebooks**: Experimentation
- **Git**: Version control

---

## 📝 Configuration Files

### `config/config.yaml`
Contains paths for data, models, and artifacts:
- Data ingestion settings
- Model preparation parameters
- Training configurations
- Evaluation settings

### `params.yaml`
Model hyperparameters:
- `IMAGE_SIZE`: Input image dimensions
- `BATCH_SIZE`: Training batch size
- `EPOCHS`: Number of training epochs
- `LEARNING_RATE`: Optimizer learning rate
- `CLASSES`: Number of output classes
- `WEIGHTS`: Pre-trained weights (imagenet)
- `INCLUDE_TOP`: Whether to include top layers

### `dvc.yaml`
Pipeline stages and dependencies:
- Data ingestion stage
- Base model preparation stage
- Training stage
- Evaluation stage

---

## 🧪 Model Details

### Architecture
- **Base Model**: VGG16 (pre-trained on ImageNet)
- **Custom Layers**: 
  - Flatten layer
  - Dense layer (output: 2 classes)
  - Softmax activation

### Training Configuration
- **Optimizer**: Adam/SGD
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy
- **Data Augmentation**: Applied during training

### Performance Metrics
Tracked in MLflow:
- Training accuracy
- Training loss
- Validation accuracy (if applicable)
- Model parameters

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

- **Your Name** - [GitHub](https://github.com/uzzal2200)

---

## 🙏 Acknowledgments

- VGG16 model from Keras Applications
- MLflow documentation and community
- DVC team for pipeline orchestration tools
- AWS for cloud infrastructure

---

## 📧 Contact

For questions or feedback, please reach out:
- GitHub Issues: [Create an issue](https://github.com/yourusername/End-to-End-Chest-Cancer-Classification-using-MLflow-DVC/issues)
- Email: uzzal.220605@s.pust.ac.bd

---

## 🔮 Future Enhancements

- [ ] Add more cancer types for multi-class classification
- [ ] Implement model serving with TensorFlow Serving
- [ ] Add Kubernetes deployment support
- [ ] Integrate Prometheus and Grafana for monitoring
- [ ] Add comprehensive unit and integration tests
- [ ] Implement A/B testing for model versions
- [ ] Add explainability with Grad-CAM visualizations

---

**⭐ If you find this project useful, please give it a star!**


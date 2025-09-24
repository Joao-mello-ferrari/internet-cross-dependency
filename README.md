# Internet Cross-Dependency Analysis

A comprehensive tool for analyzing internet infrastructure dependencies across countries, including latency measurements, CDN locality analysis, and website classification.

---

## 📋 Prerequisites

This project requires both **Python 3.12.x+** and **Node.js 22.17.x+**.

### System Requirements

- Python 3.12.x
- Node.js 22.17.x
- Google Cloud SDK (for BigQuery access)
- PostgreSQL (for geolocation data)

---

## 🚀 Installation

### 1. Python Dependencies

First, create and activate a virtual environment (recommended):

```sh
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install Python dependencies:

```sh
pip install -r requirements.txt
```

### 2. Node.js Dependencies

Navigate to the JavaScript project directory and install dependencies:

```sh
cd src/steps/locality/locedge/classify_headers
npm install
```

### 3. Environment Configuration

Create a `.env` file in the project root with your API keys:

```sh
# OpenAI API for website classification
OPENAI_API_KEY=your_openai_api_key_here
RIPE_ATLAS_API_KEY=your_key
RIPE_ATLAS_BILL_TO_EMAIL=your_email
# Add other environment variables as needed
```

### 4. Verify Installation

Run the dependency verification script to ensure everything is installed correctly:

```sh
python verify_dependencies.py
```

This script will check:

- Python version compatibility (3.12.4+)
- All required Python packages
- Node.js version compatibility (22.17.0+)
- JavaScript dependencies installation

---

## 🛠️ Setup

```sh
make setup COUNTRY=Brazil CODE=br SEMESTER=202501
```

## ⏱️ Latency Measurement

```sh
make latency COUNTRY=Brazil CODE=br
```

## 🌐 Locality Check (with VPN)

```sh
make locality COUNTRY=Brazil CODE=br VPN=<vpn_name_label>
```

## 📊 Run Analysis

```sh
make analysis COUNTRY=Brazil CODE=br
```

---

## ☁️ Google Cloud Configuration

1. Set your GCP project:

```sh
gcloud config set project <your-project-id>
```

2. Authenticate with Application Default Credentials:

```sh
gcloud auth application-default login
```

---

## 🏗️ Project Structure

```
├── src/
│   ├── steps/
│   │   ├── website_fetching/       # BigQuery website data collection
│   │   ├── website_classification/ # AI-powered website categorization
│   │   ├── latency/               # Network latency measurements
│   │   ├── locality/              # CDN and geolocation analysis
│   │   └── analysis/              # Data visualization and statistics
│   ├── setup.py                   # Main pipeline setup script
│   ├── latency.py                # Latency measurement pipeline
│   ├── locality.py               # Locality analysis pipeline
│   └── analysis.py               # Analysis and visualization pipeline
├── results/                      # Generated data and analysis results
├── requirements.txt              # Python dependencies
└── README.md                    # This file
```

---

> **Note:** Replace `<vpn_name_label>` and `<your-project-id>` with your actual VPN label and Google Cloud project ID.

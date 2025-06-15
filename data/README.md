# Job Fraud Detection System

A machine learning-based system to detect fraudulent job postings with an interactive dashboard.

## Features

- Binary classification of job postings (genuine vs fraudulent)
- CSV file upload and processing
- Interactive dashboard with visualizations
- Fraud probability scoring
- Top suspicious listings identification

## Project Structure

```
job-fraud-detection/
├── data/                   # Data directory
├── src/                    # Source code
│   ├── model/             # ML model code
│   ├── dashboard/         # Streamlit dashboard
│   └── utils/             # Utility functions
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Setup Instructions

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the dashboard:
```bash
streamlit run src/dashboard/app.py
```

## Model Details

The system uses a binary classifier trained on job posting features including:
- Job title
- Description
- Location
- Company information
- Salary details

The model outputs both a classification (genuine/fraudulent) and a probability score.

## Dashboard Features

- Interactive table of job postings with fraud probabilities
- Histogram of fraud probability distribution
- Pie chart showing genuine vs fraudulent job distribution
- Top 10 most suspicious listings
- CSV upload and processing interface 
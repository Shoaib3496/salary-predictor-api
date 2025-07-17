# 💼 Salary Predictor API

This is a REST API that predicts salaries based on years of experience using ML models.
Built with Flask, Swagger (Flask-RESTX), Docker, and unit tested with pytest.

## Features
- Multiple model versions (Linear Regression, Random Forest)
- Swagger documentation at http://localhost:5000/
- Supports GET and POST
- Docker compatible
- Unit tested

## Setup
```bash
pip install -r requirements.txt
python train_model.py
python app.py
```

## Docker
```bash
docker build -t salary-api .
docker run -p 5000:5000 salary-api
```
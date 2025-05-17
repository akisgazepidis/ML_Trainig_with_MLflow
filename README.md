# ML Training with MLflow

This repository is designed for deep diving into MLflow, an open-source platform for managing the end-to-end machine learning lifecycle. The project demonstrates how to train a machine learning model using scikit-learn and track experiments, models, and metrics with MLflow.

## Project Structure

```
ML_Training_with_MLflow/
│
├── main.py                # Entry point for training and logging a model with MLflow
├── requirements.txt       # Python dependencies
├── .gitignore             # Files and folders to be ignored by git
├── utils/
│   ├── __init__.py        # Marks utils as a Python package
│   └── utils.py           # Utility functions (e.g., train_with_mlflow)
├── mlruns/                # MLflow experiment tracking data (ignored by git)
│   └── ...                # Experiment runs, models, metrics, and artifacts
```

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the training script:
   ```bash
   python main.py
   ```
3. Launch the MLflow UI:
   ```bash
   mlflow ui --host 127.0.0.1 --port 5000
   ```
   Visit [http://127.0.0.1:5000](http://127.0.0.1:5000) to explore your experiment runs.

## Notes
- The `mlruns/` directory contains experiment logs and artifacts and is excluded from version control.
- Utility functions for training and logging models are located in `utils/utils.py`.

## License
This project is for educational purposes.

from setuptools import setup, find_packages

setup(
    name="ml_training_with_mlflow",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "mlflow",
        "scikit-learn",
        "pytest",
        "pytest-cov"
    ],
)
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def train_with_mlflow():
    # Load dataset
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    input_example = X_test[:1]  # Use a single sample as input example

    # Start MLflow run
    with mlflow.start_run():
        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)

        # Log model and metrics with input_example for signature inference
        mlflow.sklearn.log_model(model, "model", input_example=input_example)
        mlflow.log_metric("mse", mse)
        print(f"Logged model with MSE: {mse}")
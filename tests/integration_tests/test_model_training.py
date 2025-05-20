import pytest
import mlflow
from utils.utils import train_with_mlflow

@pytest.mark.integration
def test_model_training_and_logging():
    with mlflow.start_run() as run:
        train_with_mlflow(active_run=run)
        
        # Verify that metrics were logged
        run_metrics = mlflow.get_run(run.info.run_id).data.metrics
        assert 'mse' in run_metrics
        assert run_metrics['mse'] > 0
        
        # Verify that model was logged
        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(run.info.run_id)
        assert any(artifact.path == "model" for artifact in artifacts)
        
        # Load and verify the model
        model_uri = f"runs:/{run.info.run_id}/model"
        loaded_model = mlflow.sklearn.load_model(model_uri)
        assert hasattr(loaded_model, 'predict')
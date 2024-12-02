import logging
import os
from pathlib import Path

from common import (
    PYTHON,
    TRAINING_BATCH_SIZE,
    TRAINING_EPOCHS,
    FlowMixin,
    build_features_transformer,
    build_model,
    build_target_transformer,
    configure_logging,
    packages,
)
from inference import Model
from metaflow import (
    FlowSpec,
    Parameter,
    card,
    current,
    environment,
    project,
    pypi_base,
    resources,
    step,
)

configure_logging()


@project(name="penguins")
@pypi_base(
    python=PYTHON,
    packages=packages(
        "scikit-learn",
        "pandas",
        "numpy",
        "keras",
        "jax[cpu]",
        "boto3",
        "packaging",
        "mlflow",
        "setuptools",
        "python-dotenv",
    ),
)
class Training(FlowSpec, FlowMixin):
    """Training pipeline.

    This pipeline trains, evaluates, and registers a model to predict the species of
    penguins.
    """

    accuracy_threshold = Parameter(
        "accuracy-threshold",
        help=(
            "Minimum accuracy threshold required to register the model at the end of "
            "the pipeline. The model will not be registered if its accuracy is below "
            "this threshold."
        ),
        default=0.7,
    )

    @card
    @environment(
        vars={
            "MLFLOW_TRACKING_URI": os.getenv(
                "MLFLOW_TRACKING_URI",
                "http://127.0.0.1:5000",
            ),
        },
    )
    @step
    def start(self):
        """Start and prepare the Training pipeline."""
        import mlflow

        self.mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

        logging.info("MLFLOW_TRACKING_URI: %s", self.mlflow_tracking_uri)
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        self.mode = "production" if current.is_production else "development"
        logging.info("Running flow in %s mode.", self.mode)

        self.data = self.load_dataset()

        try:
            # Let's start a new MLFlow run to track everything that happens during the
            # execution of this flow. We want to set the name of the MLFlow
            # experiment to the Metaflow run identifier so we can easily
            # recognize which experiment corresponds with each run.
            run = mlflow.start_run(run_name=current.run_id)
            self.mlflow_run_id = run.info.run_id
        except Exception as e:
            message = f"Failed to connect to MLflow server {self.mlflow_tracking_uri}."
            raise RuntimeError(message) from e

        # This is the configuration we'll use to train the model. We want to set it up
        # at this point so we can reuse it later throughout the flow.
        self.training_parameters = {
            "epochs": TRAINING_EPOCHS,
            "batch_size": TRAINING_BATCH_SIZE,
        }

        # Now that everything is set up, we want to run a cross-validation process
        # to evaluate the model and train a final model on the entire dataset. Since
        # these two steps are independent, we can run them in parallel.
        self.next(self.cross_validation, self.transform)

    @card
    @step
    def cross_validation(self):
        """Generate the indices to split the data for the cross-validation process."""
        from sklearn.model_selection import KFold

        # We are going to use a 5-fold cross-validation process to evaluate the model,
        # so let's set it up. We'll shuffle the data before splitting it into batches.
        kfold = KFold(n_splits=5, shuffle=True)

        # We can now generate the indices to split the dataset into training and test
        # sets. This will return a tuple with the fold number and the training and test
        # indices for each of 5 folds.
        self.folds = list(enumerate(kfold.split(self.data)))

        # We want to transform the data and train a model using each fold, so we'll use
        # `foreach` to run every cross-validation iteration in parallel. Notice how we
        # pass the tuple with the fold number and the indices to next step.
        self.next(self.transform_fold, foreach="folds")

    @step
    def transform_fold(self):
        """Transform the data to build a model during the cross-validation process.

        This step will run for each fold in the cross-validation process. It uses
        a SciKit-Learn pipeline to preprocess the dataset before training a model.
        """
        # Let's start by unpacking the indices representing the training and test data
        # for the current fold. We computed these values in the previous step and passed
        # them as the input to this step.
        self.fold, (self.train_indices, self.test_indices) = self.input

        logging.info("Transforming fold %d...", self.fold)

        # We need to turn the target column into a shape that the Scikit-Learn
        # pipeline understands.
        species = self.data.species.to_numpy().reshape(-1, 1)

        # We can now build the SciKit-Learn pipeline to process the target column,
        # fit it to the training data and transform both the training and test data.
        target_transformer = build_target_transformer()
        self.y_train = target_transformer.fit_transform(
            species[self.train_indices],
        )
        self.y_test = target_transformer.transform(
            species[self.test_indices],
        )

        # Finally, let's build the SciKit-Learn pipeline to process the feature columns,
        # fit it to the training data and transform both the training and test data.
        features_transformer = build_features_transformer()
        self.x_train = features_transformer.fit_transform(
            self.data.iloc[self.train_indices],
        )
        self.x_test = features_transformer.transform(
            self.data.iloc[self.test_indices],
        )

        # After processing the data and storing it as artifacts in the flow, we want
        # to train a model.
        self.next(self.train_fold)

    @card
    @environment(
        vars={
            "KERAS_BACKEND": os.getenv("KERAS_BACKEND", "jax"),
        },
    )
    @resources(memory=4096)
    @step
    def train_fold(self):
        """Train a model as part of the cross-validation process.

        This step will run for each fold in the cross-validation process. It trains the
        model using the data we processed in the previous step.
        """
        import mlflow

        logging.info("Training fold %d...", self.fold)

        # Let's track the training process under the same experiment we started at the
        # beginning of the flow. Since we are running cross-validation, we can create
        # a nested run for each fold to keep track of each separate model individually.
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        with (
            mlflow.start_run(run_id=self.mlflow_run_id),
            mlflow.start_run(
                run_name=f"cross-validation-fold-{self.fold}",
                nested=True,
            ) as run,
        ):
            # Let's store the identifier of the nested run in an artifact so we can
            # reuse it later when we evaluate the model from this fold.
            self.mlflow_fold_run_id = run.info.run_id

            # Let's configure the autologging for the training process. Since we are
            # training the model corresponding to one of the folds, we won't log the
            # model itself.
            mlflow.autolog(log_models=False)

            # Let's now build and fit the model on the training data. Notice how we are
            # using the training data we processed and stored as artifacts in the
            # `transform` step.
            self.model = build_model(self.x_train.shape[1])
            self.model.fit(
                self.x_train,
                self.y_train,
                verbose=0,
                **self.training_parameters,
            )

        # After training a model for this fold, we want to evaluate it.
        self.next(self.evaluate_fold)

    @card
    @environment(
        vars={
            "KERAS_BACKEND": os.getenv("KERAS_BACKEND", "jax"),
        },
    )
    @step
    def evaluate_fold(self):
        """Evaluate the model we created as part of the cross-validation process.

        This step will run for each fold in the cross-validation process. It evaluates
        the model using the test data for this fold.
        """
        import mlflow
        from sklearn.metrics import precision_score, recall_score
        import numpy as np

        logging.info("Evaluating fold %d...", self.fold)

        # Predict the test data
        y_pred = np.argmax(self.model.predict(self.x_test), axis=1)
        y_true = np.argmax(self.y_test, axis=1)

        # Compute metrics for this fold
        self.loss, self.accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=2)
        self.precision = precision_score(y_true, y_pred, average="weighted")
        self.recall = recall_score(y_true, y_pred, average="weighted")

        logging.info(
            "Fold %d - loss: %f - accuracy: %f - precision: %f - recall: %f",
            self.fold,
            self.loss,
            self.accuracy,
            self.precision,
            self.recall,
        )

        # Log metrics to the nested MLflow run for this fold
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        with mlflow.start_run(run_id=self.mlflow_fold_run_id):
            mlflow.log_metrics(
                {
                    "test_loss": self.loss,
                    "test_accuracy": self.accuracy,
                    "test_precision": self.precision,
                    "test_recall": self.recall,
                }
            )

        # When we finish evaluating every fold in the cross-validation process, we want
        # to evaluate the overall performance of the model by averaging the scores from
        # each fold.
        self.next(self.evaluate_model)


    @card
    @step
    def evaluate_model(self, inputs):
        """Evaluate the overall cross-validation process.

        This function averages the score computed for each individual model to
        determine the final model performance.
        """
        import mlflow
        import numpy as np

        # Merge artifacts from the incoming branches (fold evaluations)
        self.merge_artifacts(inputs, include=["mlflow_run_id", "mlflow_tracking_uri"])

        # Aggregate metrics across folds
        accuracies = [i.accuracy for i in inputs]
        losses = [i.loss for i in inputs]
        precisions = [i.precision for i in inputs]
        recalls = [i.recall for i in inputs]

        # Calculate averages and standard deviations
        self.accuracy = np.mean(accuracies)
        self.accuracy_std = np.std(accuracies)
        self.loss = np.mean(losses)
        self.loss_std = np.std(losses)
        self.precision = np.mean(precisions)
        self.precision_std = np.std(precisions)
        self.recall = np.mean(recalls)
        self.recall_std = np.std(recalls)

        logging.info("Accuracy: %f ±%f", self.accuracy, self.accuracy_std)
        logging.info("Loss: %f ±%f", self.loss, self.loss_std)
        logging.info("Precision: %f ±%f", self.precision, self.precision_std)
        logging.info("Recall: %f ±%f", self.recall, self.recall_std)

        # Log aggregated metrics to MLflow
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        with mlflow.start_run(run_id=self.mlflow_run_id):
            mlflow.log_metrics(
                {
                    "cross_validation_accuracy": self.accuracy,
                    "cross_validation_accuracy_std": self.accuracy_std,
                    "cross_validation_loss": self.loss,
                    "cross_validation_loss_std": self.loss_std,
                    "cross_validation_precision": self.precision,
                    "cross_validation_precision_std": self.precision_std,
                    "cross_validation_recall": self.recall,
                    "cross_validation_recall_std": self.recall_std,
                }
            )

        # Proceed to the model registration step
        self.next(self.register_model)


    @card
    @step
    def transform(self):
        """Apply the transformation pipeline to the entire dataset.

        This function transforms the columns of the entire dataset because we'll
        use all of the data to train the final model.

        We want to store the transformers as artifacts so we can later use them
        to transform the input data during inference.
        """
        # Let's build the SciKit-Learn pipeline to process the target column and use it
        # to transform the data.
        self.target_transformer = build_target_transformer()
        self.y = self.target_transformer.fit_transform(
            self.data.species.to_numpy().reshape(-1, 1),
        )

        # Let's build the SciKit-Learn pipeline to process the feature columns and use
        # it to transform the training.
        self.features_transformer = build_features_transformer()
        self.x = self.features_transformer.fit_transform(self.data)

        # Now that we have transformed the data, we can train the final model.
        self.next(self.train_model)

    @card
    @environment(
        vars={
            "KERAS_BACKEND": os.getenv("KERAS_BACKEND", "jax"),
        },
    )
    @resources(memory=4096)
    @step
    def train_model(self):
        """Train the model that will be deployed to production.

        This function will train the model using the entire dataset.
        """
        import mlflow

        # Let's log the training process under the experiment we started at the
        # beginning of the flow.
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        with mlflow.start_run(run_id=self.mlflow_run_id):
            # Let's disable the automatic logging of models during training so we
            # can log the model manually during the registration step.
            mlflow.autolog(log_models=False)

            # Let's now build and fit the model on the entire dataset.
            self.model = build_model(self.x.shape[1])
            self.model.fit(
                self.x,
                self.y,
                verbose=2,
                **self.training_parameters,
            )

            # Let's log the training parameters we used to train the model.
            mlflow.log_params(self.training_parameters)

        # After we finish training the model, we want to register it.
        self.next(self.register_model)

    @environment(
        vars={
            "KERAS_BACKEND": os.getenv("KERAS_BACKEND", "jax"),
        },
    )
    @step
    def register_model(self, inputs):
        """Register the model in the Model Registry.

        This function will prepare and register the final model in the Model Registry.
        The model will only be registered if its accuracy is higher than the accuracy
        of the currently registered model in MLflow.
        """
        import tempfile
        import mlflow
        from mlflow.tracking import MlflowClient

        # Merge artifacts from the incoming branches
        self.merge_artifacts(inputs)

        # Initialize MLflow Client
        client = MlflowClient()

        # Get the latest registered model's accuracy
        try:
            model_versions = client.get_latest_versions("penguins")
            if model_versions:
                # Assume the latest model has the highest accuracy (optional: sort to confirm)
                latest_version = model_versions[-1]
                latest_run_id = latest_version.run_id
                latest_run = client.get_run(latest_run_id)
                latest_accuracy = float(latest_run.data.metrics["cross_validation_accuracy"])
                logging.info("Latest registered model accuracy: %f", latest_accuracy)
            else:
                latest_accuracy = 0.0  # No models in the registry yet
                logging.info("No previously registered models found.")
        except Exception as e:
            latest_accuracy = 0.0
            logging.warning("Failed to retrieve the latest model's accuracy: %s", e)

        # Check if the current model's accuracy exceeds the latest registered model's accuracy
        if self.accuracy > latest_accuracy:
            logging.info("Current model accuracy (%f) exceeds the latest model accuracy (%f).", self.accuracy, latest_accuracy)
            logging.info("Registering the model...")

            # Register the model in MLflow
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            with (
                mlflow.start_run(run_id=self.mlflow_run_id),
                tempfile.TemporaryDirectory() as directory,
            ):
                # Prepare artifacts for registration
                mlflow.pyfunc.log_model(
                    python_model=Model(data_capture=False),
                    registered_model_name="penguins",
                    artifact_path="model",
                    code_paths=[(Path(__file__).parent / "inference.py").as_posix()],
                    artifacts=self._get_model_artifacts(directory),
                    pip_requirements=self._get_model_pip_requirements(),
                    signature=self._get_model_signature(),
                    # Set example_no_conversion to True for saving input examples directly
                    example_no_conversion=True,
                )
        else:
            logging.info(
                "Current model accuracy (%f) is not higher than the latest model accuracy (%f). Model registration skipped.",
                self.accuracy,
                latest_accuracy,
            )

        # Proceed to the end step
        self.next(self.end)


    @step
    def end(self):
        """End the Training pipeline."""
        logging.info("The pipeline finished successfully.")

    def _get_model_artifacts(self, directory: str):
        """Return the list of artifacts that will be included with model.

        The model must preprocess the raw input data before making a prediction, so we
        need to include the Scikit-Learn transformers as part of the model package.
        """
        import joblib

        # Let's start by saving the model inside the supplied directory.
        model_path = (Path(directory) / "model.keras").as_posix()
        self.model.save(model_path)

        # We also want to save the Scikit-Learn transformers so we can package them
        # with the model and use them during inference.
        features_transformer_path = (Path(directory) / "features.joblib").as_posix()
        target_transformer_path = (Path(directory) / "target.joblib").as_posix()
        joblib.dump(self.features_transformer, features_transformer_path)
        joblib.dump(self.target_transformer, target_transformer_path)

        return {
            "model": model_path,
            "features_transformer": features_transformer_path,
            "target_transformer": target_transformer_path,
        }

    def _get_model_signature(self):
        """Return the model's signature.

        The signature defines the expected format for model inputs and outputs. This
        definition serves as a uniform interface for appropriate and accurate use of
        a model.
        """
        from mlflow.models import infer_signature

        return infer_signature(
            model_input={
                "island": "Biscoe",
                "culmen_length_mm": 48.6,
                "culmen_depth_mm": 16.0,
                "flipper_length_mm": 230.0,
                "body_mass_g": 5800.0,
                "sex": "MALE",
            },
            model_output={"prediction": "Adelie", "confidence": 0.90},
            params={"data_capture": False},
        )

    def _get_model_pip_requirements(self):
        """Return the list of required packages to run the model in production."""
        return [
            f"{package}=={version}"
            for package, version in packages(
                "scikit-learn",
                "pandas",
                "numpy",
                "keras",
                "jax[cpu]",
            ).items()
        ]


if __name__ == "__main__":
    Training()

import pickle
import os
import argparse
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import numpy as np
import json
import subprocess
from typing import Tuple, List

from azureml.core.run import Run
from azureml.core.model import Model

run = Run.get_context()
exp = run.experiment
ws = run.experiment.workspace

outputs_folder = './model'
os.makedirs(outputs_folder, exist_ok=True)

print("Loading training data...")
# https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html
X, y = load_diabetes(return_X_y=True)
columns = ["age", "gender", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
data = {"train": {"X": X_train, "y": y_train}, "test": {"X": X_test, "y": y_test}}

print("Training the model...")
# Randomly pic alpha
alphas = np.arange(0.0, 1.0, 0.05)
alpha = alphas[np.random.choice(alphas.shape[0], 1, replace=False)][0]
print(alpha)
run.log("alpha", alpha)
reg = Ridge(alpha=alpha)
reg.fit(data["train"]["X"], data["train"]["y"])
preds = reg.predict(data["test"]["X"])
run.log("mse", mean_squared_error(preds, data["test"]["y"]))

# Save model as part of the run history
print("Exporting the model as pickle file...")
model_name = "sklearn_regression_model"
model_filename = "sklearn_regression_model.pkl"
model_path = os.path.join(outputs_folder, model_filename)

with open(model_path, "wb") as file:
    joblib.dump(value=reg, filename=model_filename)

# upload the model file explicitly into artifacts
print("Uploading the model into run artifacts...")
run.upload_file(name="./outputs/" + model_filename, path_or_stream=model_path)
print("Uploaded the model {} to experiment {}".format(model_filename, run.experiment.name))
dirpath = os.getcwd()
print(dirpath)
print("Following files are uploaded ")
print(run.get_file_names())

try:
    # Get most recently registered model, we assume that is the model in production. Download this model and compare it with the recently trained model by running test with same data set.
    model_list = Model.list(ws)
    production_model = next(
        filter(
            lambda x: x.created_time == max(model.created_time for model in model_list),
            model_list,
        )
    )
    production_model_run_id = production_model.tags.get("run_id")
    run_list = exp.get_runs()

    # Get the run history for both production model and newly trained model and compare mse
    production_model_run = Run(exp, run_id=production_model_run_id)

    production_model_mse = production_model_run.get_metrics().get("mse")
    new_model_mse = run.get_metrics().get("mse")
    print(
        "Current Production model mse: {}, New trained model mse: {}".format(
            production_model_mse, new_model_mse
        )
    )

    promote_new_model = False
    if new_model_mse < production_model_mse:
        promote_new_model = True
        print("New trained model performs better, thus it will be registered")
except:
    promote_new_model = True
    print("This is the first model to be trained, thus nothing to evaluate for now")

# register the model
if promote_new_model:
    print('Registering the model...')
    model = Model.register(
        model_path=model_path,
        model_name=model_name,
        id=run.id,
        tags={"data": "diabetes", "model": "regression", "run_id": run.id},
        description="Linear model using diabetes dataset",
        workspace=ws
    )

run.complete()
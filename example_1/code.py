"""
pycharm
-------------------------------
 - Eugenio Marinetto
 - github@nenetto.page
-------------------------------
Created 10-12-2018
"""

import os
from random import random, randint
import mlflow


if __name__ == "__main__":

    print("Example #2: Create an experiment")

    mlflow.set_experiment('My First Experiment')

    # Log a parameter
    mlflow.log_param("Parameter 1", randint(0, 100))

    # Log a metric
    mlflow.log_metric("Metric 1", random())
    mlflow.log_metric("Metric 2", random() + 1)
    mlflow.log_metric("Metric 3", random() + 2)

    # Log local files/folders
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    with open("outputs/test.txt", "w") as f:
        f.write("hello world!")

        mlflow.log_artifacts("outputs")
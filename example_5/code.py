"""
pycharm
-------------------------------
 - Eugenio Marinetto
 - github@nenetto.page
-------------------------------
Created 10-12-2018
"""

import os
import mlflow
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import sklearn.metrics

if __name__ == "__main__":
    client = mlflow.tracking.MlflowClient()

    exp_name = 'Iris HyperOpt'
    ver_name = '0.0'

    print("Experiment [{0}]".format(exp_name))
    experiments = client.list_experiments()

    experiment = next((e for e in experiments if e.name == exp_name), None)
    if experiment is None:
        experiment = client.create_experiment(exp_name)


    # Perform HyperParam Search

    print('Reading dataset')
    mlflow.log_param("Dataset", 'Iris')
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Split the dataset in two equal parts
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    clf = GridSearchCV(SVC(), tuned_parameters, cv=15, iid=False, return_train_score=True,
                       scoring=sklearn.metrics.make_scorer(sklearn.metrics.normalized_mutual_info_score,
                                                           average_method='arithmetic'))
    clf.fit(X_train, y_train)


    i = 0
    for p in clf.cv_results_['params']:
        # Log best params
        run = client.create_run(experiment_id=experiment.experiment_id,
                                source_version=ver_name,
                                run_name='run_{0}'.format(str(i)))

        with mlflow.start_run(run_uuid=run.info.run_uuid, nested=True):

            for k, v in p.items():
                mlflow.log_param(k, v)

            for k in ['rank_test_score',
                      'mean_test_score',
                      'std_test_score',
                      'mean_fit_time',
                      'std_fit_time']:
                mlflow.log_metric(k, clf.cv_results_[k][i])

            for cvix in range(clf.n_splits_):
                keyname = 'split{0}_train_score'.format(cvix)
                mlflow.log_metric('train_score_cv', clf.cv_results_[keyname][i])

                keyname = 'split{0}_test_score'.format(cvix)
                mlflow.log_metric('test_score_cv', clf.cv_results_[keyname][i])

        i = i+1



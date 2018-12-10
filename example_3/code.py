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
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import sklearn.metrics


if __name__ == "__main__":

    exp_name = 'Iris'
    ver_name = '0.0'
    run_name = 'Iris Run'

    print("Experiment [{0}]".format(exp_name))
    mlflow.set_experiment(exp_name)

    with mlflow.start_run(source_version=ver_name,
                          run_name=run_name):

        print('Reading dataset')
        mlflow.log_param("Dataset", 'Iris')
        iris = sklearn.datasets.load_iris()
        X = iris.data
        y = iris.target

        print('Splitting')
        test_proportion = 0.4
        mlflow.log_param("Test proportion", test_proportion)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_proportion, shuffle=True)

        print('Creating estimator using Kmeans')
        kmeans_n_clusters = 3
        mlflow.log_param("Kmeans N Clusters", kmeans_n_clusters)
        est = KMeans(n_clusters=kmeans_n_clusters)

        print('Fitting estimator')
        est.fit(X_train)
        y_predict_train = est.predict(X_train)

        print('Computing training metrics')
        nmi_train = sklearn.metrics.normalized_mutual_info_score(y_train, y_predict_train, average_method='arithmetic')

        print('  - normalized_mutual_info_score:', nmi_train)

        mlflow.log_metric("nmi_train", nmi_train)

        print('Testing')
        y_predict = est.predict(X_test)

        print('Computing testing metrics')
        nmi = sklearn.metrics.normalized_mutual_info_score(y_test, y_predict, average_method='arithmetic')

        print('  - normalized_mutual_info_score:', nmi)

        mlflow.log_metric("nmi", nmi_train)

        print('Generate plot')
        fig = plt.figure(figsize=(4, 4))
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
        labels = y_predict

        ax.scatter(X_test[:, 3], X_test[:, 0], X_test[:, 2],
                   c=labels.astype(np.float), edgecolor='k')

        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])
        ax.set_xlabel('Petal width')
        ax.set_ylabel('Sepal length')
        ax.set_zlabel('Petal length')
        ax.set_title('Kmeans results')
        ax.dist = 12

        if not os.path.exists("figures"):
            os.makedirs("figures")

        mlflow.log_artifacts("figures")

        fig.savefig("figures/prediction.png")

        # Plot the ground truth
        fig = plt.figure(figsize=(4, 4))
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

        for name, label in [('Setosa', 0),
                            ('Versicolour', 1),
                            ('Virginica', 2)]:
            ax.text3D(X_test[y_test == label, 3].mean(),
                      X_test[y_test == label, 0].mean(),
                      X_test[y_test == label, 2].mean() + 2, name,
                      horizontalalignment='center',
                      bbox=dict(alpha=.2, edgecolor='w', facecolor='w'))
        # Reorder the labels to have colors matching the cluster results
        y = np.choose(y_test, [1, 2, 0]).astype(np.float)
        ax.scatter(X_test[:, 3], X_test[:, 0], X_test[:, 2], c=y, edgecolor='k')

        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])
        ax.set_xlabel('Petal width')
        ax.set_ylabel('Sepal length')
        ax.set_zlabel('Petal length')
        ax.set_title('Ground Truth')
        ax.dist = 12

        fig.savefig("figures/ground_truth.png")
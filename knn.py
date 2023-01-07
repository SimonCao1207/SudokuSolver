# https://github.com/neeru1207/AI_Sudoku/blob/master/KNN.py

import numpy as np
from sklearn import datasets
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
import pickle
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--k', default='3')
args = parser.parse_args()

SEED = 1208
np.random.seed(seed=SEED)

KNN_PATH = "./checkpoints/knn.sav"


class KNN:
    def __init__(self, k=int(args.k)):
        self.mnist = datasets.fetch_openml('mnist_784', data_home='mnist_dataset/')
        self.data, self.target = self.mnist.data, self.mnist.target
        self.indx = np.random.choice(len(self.target), 70000, replace=False)
        self.classifier = KNeighborsClassifier(n_neighbors=k)
        self.k = k

    def mk_dataset(self, size):
        """makes a dataset of size "size", and returns that datasets images and targets
        This is used to make the dataset that will be stored by a model and used in
        experimenting with different stored dataset sizes
        """

        print("Making dataset: ...", self.data)
        
        train_img = [self.data.iloc[i] for i in self.indx[:size]]
        train_img = np.array(train_img)
        train_target = [self.target[i] for i in self.indx[:size]]
        train_target = np.array(train_target)

        return train_img, train_target

    def skl_knn(self):
        """k: number of neighbors to use in classification
        test_data: the data/targets used to test the classifier
        stored_data: the data/targets used to classify the test_data
        """
        fifty_x, fifty_y = self.mk_dataset(50000)
        test_img = [self.data.iloc[i] for i in self.indx[60000:70000]]
        test_img1 = np.array(test_img)
        test_target = [self.target[i] for i in self.indx[60000:70000]]
        test_target1 = np.array(test_target)

        print(f"Classifier fitting: ... k = {self.k}")
        self.classifier.fit(fifty_x, fifty_y)

        y_pred = self.classifier.predict(test_img1)
        print("unique: ", np.unique(test_img))
        pickle.dump(self.classifier, open('./checkpoints/knn.sav', 'wb'))
        print(classification_report(test_target1, y_pred))
        print("KNN Classifier model saved as knn.sav!")

if __name__ == "__main__":
    knn = KNN()
    knn.skl_knn() 
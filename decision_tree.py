import numpy as np
class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.n_classes_ = len(np.unique(y))
        self.n_features_ = X.shape[1]
        self.tree_ = self._grow_tree(X, y)

    def _best_split(self, X, y):
        # size of current dataset
        m = y.size
        if m <= 1:
            return None, None
        # num of pt in each class
        num_parent = [np.sum(y == c) for c in range(self.n_classes_)]
        # gini impurity
        best_gini = 1.0 - sum((n / m) ** 2 for n in num_parent)
        best_idx, best_thr = None, None
        
        # iterate each possible split point(threshold) in each feature in X(m, n_featurea)
        for idx in range(self.n_features_):
            # sort the pts to better iterate over
            # [(1,1), (2,1), (3,1), ...] (x, y)
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
            # initialize num of pt at left of thred and right
            num_left = [0] * self.n_classes_
            num_right = num_parent.copy()
            # each x data point is a possible split point(threshold)
            for i in range(1, m):
                # try to split at cur pt x value 3
                # e.g. [(1,1), (2,1), (3,1), ...], split btw x = 2 and x = 3
                # right (1,1)(2,1) | (3,1) left
                # so move (2,1) from right-- to left++, and c = 2 -> classes[i-1]
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                gini_left = 1.0 - sum(
                    (num_left[x] / i) ** 2 for x in range(self.n_classes_)
                )
                gini_right = 1.0 - sum(
                    (num_right[x] / (m - i)) ** 2 for x in range(self.n_classes_)
                )
                gini = (i * gini_left + (m - i) * gini_right) / m
                # skip duplicate, if we have 2 splits with same threshold, e.g. [(2,1), (2,1), (2,1)]
                # since we have split on threshold=(2+2)/2=2 before, don't need to recompute the split
                # because the splited result will be the same.
                if thresholds[i] == thresholds[i - 1]:
                    continue
                # if gini_impurity is smaller: purier
                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    # we are spliting btw the cur pt i and previous pt i - 1
                    # so we start from pt 1 to m-1
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2
        
        return best_idx, best_thr

    def _grow_tree(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        # assign a class for currect node(dataset), prunning, early_stop
        predicted_class = np.argmax(num_samples_per_class)
        node = Node(predicted_class=predicted_class)
        if depth < self.max_depth:
            # best_split feature, best_split value base on gini
            idx, thr = self._best_split(X, y)
            if idx is not None:
                # split cur dataset into right and left set
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature_index = idx
                node.threshold = thr
                node.left = self._grow_tree(X_left, y_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, depth + 1)
        return node
    
    def predict(self, X):
        return [self._predict(x) for x in X]

    def _predict(self, x):
        # predicted class for a single point x
        node = self.tree_
        while node.left:
            if x[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class
    
class Node:
    def __init__(self, *, predicted_class):
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0.0 
        self.left = None
        self.right = None

    def is_leaf_node(self):
        return self.left is None and self.right is None
    
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
def test():
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Train the decision tree
    tree = DecisionTree(max_depth=3)
    tree.fit(X_train, y_train)
    # Predict
    y_pred = tree.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    # 1.0
    print(f"Accuracy: {accuracy}")

if __name__ == '__main__':
    test()
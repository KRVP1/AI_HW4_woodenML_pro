import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import List



def find_best_split(feature_vector, target_vector):
    matrix = np.stack((feature_vector, target_vector), axis=1)
    sorted_indices = np.argsort(matrix[:, 0]) 
    sorted_array = matrix[sorted_indices]
    if len(np.unique(sorted_array[:, 0])) == 1:
        return [], [], sorted_array[:, 0][0], -10

    # print(sorted_array[:, 0].reshape(-1, 1).T >= sorted_array[:, 0].reshape(-1, 1))
    total_count = len(sorted_array[:, 1])
    total_ones = np.sum(sorted_array[:, 1])
    total_zeros = total_count - total_ones

    prefix_ones = np.cumsum(sorted_array[:, 1])
    prefix_zeros = np.arange(1, total_count + 1) - prefix_ones

    positive_class_l = prefix_ones[:-1]
    negative_class_l = prefix_zeros[:-1]
    positive_class_r = total_ones - positive_class_l
    negative_class_r = total_zeros - negative_class_l

    # drop_index = np.where(np.logical_or(sorted_array[:, 0] == sorted_array[:, 0].min(), positive_class_r + negative_class_r == 0))
    drop_index =  np.where(sorted_array[:, 0] == sorted_array[:, 0].min())

    positive_class_l = np.delete(positive_class_l, drop_index)
    positive_class_r = np.delete(positive_class_r, drop_index)
    negative_class_l = np.delete(negative_class_l, drop_index)
    negative_class_r = np.delete(negative_class_r, drop_index)
    thresholds = np.delete(sorted_array[:, 0], drop_index)

    Hl = 1 - (positive_class_l / (positive_class_l + negative_class_l)) ** 2 - (negative_class_l / (positive_class_l + negative_class_l)) ** 2
    Hr = 1 - (positive_class_r / (positive_class_r + negative_class_r)) ** 2 - (negative_class_r / (positive_class_r + negative_class_r)) ** 2
    Gini = - (positive_class_l + negative_class_l) / target_vector.shape[0] * Hl - (positive_class_r + negative_class_r) / target_vector.shape[0] * Hr
    if len(thresholds) == 1:
        threshold_best = thresholds[0]
        Gini_best = 0
    else:
        threshold_best = thresholds[np.where(np.abs(Gini) == np.min(np.abs(Gini)))[0][0]]
        Gini_best = -np.min(np.abs(Gini))

    return thresholds, Gini, threshold_best, Gini_best


class DecisionTree(BaseEstimator, ClassifierMixin):
    def __init__(self, feature_types: List[str], max_depth=None, min_samples_split=0, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self.feature_types = feature_types
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.requires_fit = False
        return tags

    def _fit_node(self, sub_X, sub_y, node):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        categories_map_best, feature_best, threshold_best, gini_best, split = None, None, None, None, None
        for feature in range(0, sub_X.shape[1]):
            feature_type = self.feature_types[feature]
            categories_map = {}
            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0.0
                    ratio[key] = current_click / current_count
                sorted_categories = dict(sorted(ratio.items(), key=lambda x: x[1]))
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))
                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))
            else:
                raise ValueError

            if len(feature_vector) <= self.min_samples_split:
                continue
            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                categories_map_best = categories_map
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = threshold
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self.feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self.feature_types[feature_best] == "categorical":
            node["categories_split"] = list(map(lambda x: x[0],
                                            filter(lambda x: x[1] < threshold_best, categories_map_best.items())))
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"])
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"])

    def _predict_node(self, x, node) -> int:
        if node == {}:
            return
        if node["type"] == "terminal":
            return node["class"]
        try:
            if x[node["feature_split"]] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        except:
            if x[node["feature_split"]] in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)
        return self
    
    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
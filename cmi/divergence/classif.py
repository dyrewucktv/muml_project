import xgboost
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def make_dataset(v2, vp):
    # make dataset for classification
    dataset = np.row_stack([v2, vp])
    target = np.row_stack([np.ones((v2.shape[0], 1)), np.zeros((vp.shape[0], 1))])

    dataset_shuffle = np.random.permutation(dataset.shape[0])

    dataset = dataset[dataset_shuffle]
    target = target[dataset_shuffle]

    x_train, x_test, y_train, y_test = train_test_split(dataset, target, test_size=.33, shuffle=True, stratify=target)
    return x_train, x_test, y_train, y_test


def classif_based_cmi(v2, vp):
    """
    calculate cmi using classifier based estimation
    based on DV representation of KL
    """
    x_train, x_test, y_train, y_test = make_dataset(v2, vp)

    # fit model
    model = xgboost.XGBClassifier(min_child_weight=.1)
    model.fit(x_train, y_train)

    # calculate cmi
    probs = model.predict_proba(x_test)
    print(f"model acc: {accuracy_score(y_test, model.predict(x_test))}")
    f_index = y_test.squeeze().astype(bool)
    l_w_f = (probs[f_index]/(1-probs[f_index]))[:, 1]
    l_w_g = (probs[~f_index]/(1-probs[~f_index]))[:, 1]
    return np.mean(np.log(l_w_f)) - np.log(np.mean(l_w_g))

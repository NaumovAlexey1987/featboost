import numpy as np

from tqdm import tqdm

from sklearn.metrics import auc, accuracy_score, mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split

class FeatBoost():
    """FeatBoost algorithm for feature selection.
    Inputs: two classificators for feature ranking and estimating. \n
    classificator_1 must support .features_importances_ and .predict_log_proba (check skleran models for example).
    """
    def __init__(self, classificator_1 = None, classificator_2 = None) -> None:
        self.x = None
        self.y = None
        self.trained = False

        self.classificator_1 = classificator_1
        self.classificator_2 = classificator_2

        self.k = 5
        self.m = 3
        self.err = 0.01
        self.test_size = 0.2

        self.selected_features = None
        self.n_features = None

        self.sample_weights = None
        self.alpha_coef = None
        self.accuracy_list = None
        pass

    def fit(self, X, y) -> None:
        """Fit the framework. \n
        Inputs: \n
        X: array-like,
        y: array-like, \n
        This array will fit two chosen classificators.
        """
        self.x = X
        self.y = y
        self.trained = False
        pass

    def set_classificators(self, new_classificator_1 = None, new_classificator_2 = None):
        """ Set new classificator to the framework.
        Inputs:
        new_classififcator_1: Model to calculate features ranking. 
        new_classificator_2: Model to evaluate model with selected top features. 
        """
        if new_classificator_1 != None:
            self.classificator_1 = new_classificator_1
        if new_classificator_2 != None:
            self.classificator_2 = new_classificator_2
        pass

    def set_params(self, k = None, m = None, err = None, test_size = None):
        """ Set additional parameters for feature selecting algorithm. \n
        Inputs:
        k: int, default = 5 
        m: int, default = 3 
        err: float, default = 1e-3
        test_size: float in (0, 1), default = 0.2

        k - parameters for k-fold cross-validation \n
        m - number of chosen features after features ranking with classificator_1 \n
        err - error size to evaluate new feature to continue \n
        test_size: fit sklearn.model_selection.train_test_split for test set size \n
        """
        if k != None:
            self.k = k
        if m != None:
            self.m = m
        if err != None:
            self.err = err
        if test_size != None:
            self.test_size = test_size
        pass

    def get_logs(self):
        """ Get some information in feature selecting algorithm \n
        Return: [sample_weights, alpha_coefficients, accuracy_list]
        """
        return [self.sample_weights, self.alpha_coef, self.accuracy_list]

    def feature_selecting(self, n_features) -> None:
        """ Start the feature selected algorithm. \n
        Before start you need fit framework with .fit(X, y) \n
        Also you can set parameters with .set_params() and classificators with .set_classificators()
        """
        
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=self.test_size)

        n = x_train.shape[0]
        self.selected_features = np.array([], dtype=int)
        self.accuracy_list = []
        self.alpha_coef = [np.ones(n)]
        self.n_features = n_features

        model_1 = self.classificator_1
        model_2 = self.classificator_2

        acc_delta = 0

        for i in tqdm(range(self.n_features)):
            features_ranking = []
            cross_val = []
            pass_fisrt = False

            self.sample_weights = np.ones(n) / n

            model_1.fit(x_train, y_train, self.sample_weights)

            for feature_idx in np.argsort(model_1.feature_importances_)[::-1]:
                if feature_idx not in self.selected_features:
                    features_ranking.append(feature_idx)
                if len(features_ranking) == self.m:
                    break
            
            for feature_idx in features_ranking:
                cross_val.append((cross_val_score(model_2, x_train[:, np.append(self.selected_features, feature_idx)], y_train, cv=self.k).mean(), feature_idx))
            
            top_feature = sorted(cross_val, key=lambda x: x[0])[-1][1]

            if not pass_fisrt:
                model_2.fit(x_train[:, top_feature].reshape(-1, 1), y_train)
                self.accuracy_list.append(accuracy_score(y_test, model_2.predict(x_test[:, top_feature].reshape(-1, 1))))
                pass_fisrt = True
                acc_delta = self.err + 1
                pass
            else:
                model_2.fit(x_train[:, np.append(self.selected_features, top_feature)], y_train)
                self.accuracy_list.append(accuracy_score(y_test, model_2.predict(x_test[:, np.append(self.selected_features, top_feature)])))
                acc_delta = np.abs(self.accuracy_list[-1] - self.accuracy_list[-2])
            
            if acc_delta > self.err:
                self.selected_features = np.append(self.selected_features, top_feature)

                model_1.fit(x_train[:, self.selected_features], y_train)

                self.alpha_coef.append(model_1.predict_log_proba(x_train[:, self.selected_features]))
                self.alpha_coef[-1][self.alpha_coef[-1] == -np.inf] = -1
                self.alpha_coef[-1] = -self.alpha_coef[-1].sum(axis=1) / self.alpha_coef[-2]

                self.sample_weights *= self.alpha_coef[-1]
                self.sample_weights /= self.sample_weights.sum()
            else:
                self.sample_weights = np.ones(n) / n
                break
        self.trained = True
        pass

    def get_subset(self, n_features = None):
        """ Get subset of selected features. \n
        If .feature_selecting didn't run before, input n_features.
        Return: np.array
        """
        if self.trained == True:
            return self.selected_features
        else:
            self.feature_selecting(n_features=n_features)
            return self.selected_features
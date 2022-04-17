from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import pandas
from tqdm.notebook import tqdm
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, roc_curve, RocCurveDisplay, precision_recall_curve, PrecisionRecallDisplay
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split


class Samplers:
    @staticmethod
    def random_oversampler(sampling_strategy) -> RandomOverSampler:
        return RandomOverSampler(sampling_strategy=sampling_strategy, random_state=0)

    @staticmethod
    def random_undersampler(sampling_strategy) -> RandomUnderSampler:
        return RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=0)

    @staticmethod
    def smote_oversampler(sampling_strategy, k_neighbors: int = 55) -> SMOTE:
        return SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, random_state=0)


class RandomForest:
    def __init__(self, dataset: pandas.DataFrame, labels: pandas.DataFrame, n_forests: int, random_state: int,
                 n_folds: int, show_bar: bool, balance: bool):
        self.dataset = dataset.to_numpy()
        self.n_forests = n_forests
        self.random_state = random_state
        self.labels = labels.to_numpy()
        self.pieces = n_folds
        self.show_bar = show_bar
        self.splitter = StratifiedKFold(n_splits=n_folds)

        if balance:
            self.classifier = RandomForestClassifier(n_estimators=n_forests, random_state=random_state,
                                                     class_weight='balanced_subsample')
        else:
            self.classifier = RandomForestClassifier(n_estimators=n_forests, random_state=random_state)

        self._trained_classifier = None
        self.sensitivity, self.accuracy, self.specificity, self.roc_auc, self.prc_auc, self.fpr, self.tpr = [None] * 7
        self.f_score = {0: None, 1: None}
        self.roc_curve = {'fpr': None, 'tpr': None, 'thresholds': None}
        self.prc_curve = {'precision': None, 'recall': None, 'thresholds': None}
        self.roc_plot = None

    @property
    def trained_classifier(self):
        if self._trained_classifier is None:
            raise ValueError('Classifier is not yet classified, use run().')

        return self._trained_classifier

    @trained_classifier.setter
    def trained_classifier(self, classifier):
        self._trained_classifier = classifier

    def predict(self, dataset: pandas.DataFrame):
        prediction = self.trained_classifier.predict(dataset)
        return prediction

    def positive_proba(self, dataset):
        probs = self.trained_classifier.predict_proba(dataset)[:, 1]
        return probs

    def run(self, sampling_pipe: list = ()):
        folds_indexes = self.splitter.split(self.dataset, self.labels)
        iterator = tqdm(list(folds_indexes), desc='Training', leave=False) if self.show_bar else folds_indexes

        test_predictions = np.array([0] * len(self.dataset))
        test_probs = np.array([0.0] * len(self.dataset))

        for train_idx, test_idx in iterator:
            X_train, y_train = self.resample(self.dataset[train_idx], self.labels[train_idx], sampling_pipe)
            X_test, y_test = self.dataset[test_idx], self.labels[test_idx]

            self.trained_classifier = self.classifier.fit(X_train, y_train)

            test_predictions[test_idx] = self.predict(X_test)
            test_probs[test_idx] = self.positive_proba(X_test)

        report = classification_report(self.labels, test_predictions, target_names=[0, 1], output_dict=True)

        self.sensitivity = report[1]['recall']
        self.accuracy = report['accuracy']
        self.specificity = report[0]['recall']

        self.f_score[0] = report[0]['f1-score']
        self.f_score[1] = report[1]['f1-score']

        self.roc_auc = roc_auc_score(self.labels, test_probs)
        self.prc_auc = average_precision_score(self.labels, test_probs)
        self.roc_curve['fpr'], self.roc_curve['tpr'], self.roc_curve['thresholds'] = roc_curve(self.labels, test_probs)
        self.prc_curve['precision'], self.prc_curve['recall'], self.prc_curve['thresholds'] = precision_recall_curve(self.labels, test_probs)
        self.roc_plot = RocCurveDisplay(fpr=self.roc_curve['fpr'], tpr=self.roc_curve['tpr'], roc_auc=self.roc_auc)
        self.prc_plot = PrecisionRecallDisplay(self.prc_curve['precision'], self.prc_curve['recall'])

    def report(self) -> dict:
        return {'sensitivity': self.sensitivity,
                'specificity': self.specificity,
                'accuracy': self.accuracy,
                'f_score': self.f_score,
                'roc_auc': self.roc_auc,
                'prc_auc': self.prc_auc}

    @staticmethod
    def resample(X, y, sampling_pipeline):
        X_, y_ = X, y

        for sampling_phase in sampling_pipeline:
            X_, y_ = sampling_phase.fit_resample(X_, y_)

        return X_, y_

    # print(rf.report())
    # trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2)
    # rf.classifier.fit(trainX, trainy)
    # probs = rf.classifier.predict_proba(testX)[:, 1]
    # print(probs)

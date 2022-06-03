import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml import sklearn2pmml
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn2pmml import make_pmml_pipeline


class Resampling:
    def __init__(self, dataset, target_col, bads_label, goods_label):
        self.df = dataset
        self.target_col = target_col
        self.goods_label = goods_label
        self.bads_label = bads_label
        self.goods = self.df[self.df[target_col] == (type(self.df[target_col][0])(goods_label))]
        self.bads = self.df[self.df[target_col] == (type(self.df[target_col][0])(bads_label))]
        self.hash_balanced_resamples = {}
        self.X_train = pd.DataFrame()
        self.y_train = pd.DataFrame()
        self.X_test = pd.DataFrame()
        self.y_test = pd.DataFrame()
        self.model = []
        self.dummies = []
        self.features = []
        self.flag_missings = False
        self.binary_result_train = []
        self.binary_result_test = []
        self.proba_result_train = []
        self.proba_result_test = []
        self.model_name = []

    def _label_encoder(self, labels):
        arr = []
        for i in range(len(labels)):
            if labels[i] == self.bads_label:
                arr.append(1)
            else:
                arr.append(0)

        return pd.Series(arr)

    def feature_select_balanced_samples(self, features, fill_missings=True):
        if len(self.hash_balanced_resamples) == 0:
            raise Warning('First you need to resample dataset by "balanced_resamples" function')
        else:
            self.features = features
            self.flag_missings = fill_missings
            for i, item in enumerate(self.hash_balanced_resamples):
                df = self.hash_balanced_resamples[i]
                df = df[features]
                if fill_missings == True:
                    # df = df.astype(np.float32)
                    df = df.replace([np.inf, -np.inf], np.nan)
                    df = df.replace(r' ', np.nan, regex=True)
                    df = df.replace(r'', np.nan, regex=True)
                    df = df.replace(r'^\s*$', np.nan, regex=True)
                    self.hash_balanced_resamples[i] = df.fillna(int(-1))
                else:
                    self.hash_balanced_resamples[i] = df

    def _preprocess_data(self, df):
        df = df[self.features]
        if self.flag_missings == True:
            # df = df.astype(np.float32)
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.replace(r' ', np.nan, regex=True)
            df = df.replace(r'', np.nan, regex=True)
            df = df.replace(r'^\s*$', np.nan, regex=True)
            df = df.fillna(int(-1))
        return df

    def balanced_resamples_ratio(self, ratio_start=0.5, ratio_end=0.5):
        if ratio_start > ratio_end:
            raise Warning("ratio_start should be <= than ratio_end")
        if ratio_start < len(self.bads) / len(self.goods) or ratio_end < len(self.bads) / len(self.goods):
            raise Warning("ratio should be higher than initiale ratio. Choose higher ratio value than {}".format(
                len(self.bads) / len(self.goods)))
        else:
            i, cnt, prev = 0, 0, 0

            while i < len(self.goods):
                bad_dist = random.uniform(ratio_start, ratio_end)
                good_dist = 1 - bad_dist

                i += int((len(self.bads) * good_dist) // bad_dist)
                # st.write('Iterative i-th value:', i)
                if i + len(self.bads) <= len(self.goods):
                    self.hash_balanced_resamples[cnt] = pd.concat([self.goods.iloc[prev:i], self.bads],
                                                                  ignore_index=True)
                else:
                    self.hash_balanced_resamples[cnt] = pd.concat(
                        [self.goods.iloc[prev:], self.bads[:len(self.goods) - prev]], ignore_index=True)

                prev = i
                cnt += 1

    def get_tables(self):
        return self.hash_balanced_resamples

    def ensemble_train(self, algorithm='tree', voting='soft'):
        models = []
        self.dummies = pd.get_dummies(
            self.hash_balanced_resamples[0][self.hash_balanced_resamples[0].columns.difference([self.target_col])])

        for i in self.hash_balanced_resamples:
            y = self._label_encoder(self.hash_balanced_resamples[i].pop(self.target_col))
            X = pd.get_dummies(self.hash_balanced_resamples[i]).reindex(columns=self.dummies.columns)

            self.X_train = pd.concat([self.X_train, X])
            self.y_train = pd.concat([self.y_train, y])

            if algorithm == 'tree':
                dtc = DecisionTreeClassifier()
                self.model_name = DecisionTreeClassifier()
                models.append(('clf_{}'.format(i), dtc))

            elif algorithm == 'adaboost':
                adab = AdaBoostClassifier()
                self.model_name = AdaBoostClassifier()
                models.append(('clf_{}'.format(i), adab))

            elif algorithm == 'gradb':
                gradb = GradientBoostingClassifier(max_depth=3, n_estimators=50, learning_rate=0.1)
                self.model_name = GradientBoostingClassifier(max_depth=3, n_estimators=50, learning_rate=0.1)
                models.append(('clf_{}'.format(i), gradb))

            elif algorithm == 'randomf':
                rf = RandomForestClassifier()
                self.model_name = RandomForestClassifier()
                models.append(('clf_{}'.format(i), rf))

            elif algorithm == 'knn':
                bg = KNeighborsClassifier(n_neighbors=3)
                self.model_name = KNeighborsClassifier(n_neighbors=3)
                models.append(('clf_{}'.format(i), bg))

        if voting == 'soft':
            clf = VotingClassifier(estimators=models, voting='soft')
        elif voting == 'hard':
            clf = VotingClassifier(estimators=models, voting='hard')

        st.write(self.X_train.head())

        pipeline = PMMLPipeline([('classifier', self.model_name)])
        pipeline.fit(self.X_train, self.y_train)
        # sklearn2pmml(pipeline, "card_card.pmml", with_repr = True)   # Сохранение в pmml
        self.model = clf.fit(self.X_train, self.y_train)
        self.proba_result_train = self.model.predict_proba(self.X_train)[:, 1]

        st.code('Train size: ' + str(len(self.X_train)) + '\nTrain results: \n' + classification_report(self.y_train, self.model.predict(self.X_train)))

        fpr, tpr, thresholds = metrics.roc_curve(self.y_train, self.model.predict_proba(self.X_train)[:, 1])
        auc = metrics.auc(fpr, tpr)
        st.write('AUC: ', auc)
        st.write('GINI: ', 2 * auc - 1)

    def ensemble_predict(self, X):
        # out of sample
        X = self._preprocess_data(X)
        y = self._label_encoder(X.pop(self.target_col))
        X = pd.get_dummies(X).reindex(columns=self.dummies.columns)
        st.code('Test size: ' + str(len(X)) + '\nTest results: \n' + classification_report(y, self.model.predict(X)))
        fpr, tpr, thresholds = metrics.roc_curve(y, self.model.predict_proba(X)[:, 1])
        auc = metrics.auc(fpr, tpr)
        st.write('AUC: ', auc)
        st.write('GINI: ', 2 * auc - 1)
        self.binary_result = self.model.predict(X)
        self.proba_result_test = self.model.predict_proba(X)[:, 1]
        self.y_test = y
        self.X_test = X
        return self.binary_result

    def ensemble_predict_proba(self, X):
        X = self._preprocess_data(X)
        y = self._label_encoder(X.pop(self.target_col))
        X = pd.get_dummies(X).reindex(columns=self.dummies.columns)

        fpr, tpr, thresholds = metrics.roc_curve(y, self.model.predict_proba(X)[:, 1])
        auc = metrics.auc(fpr, tpr)

        self.proba_result = self.model.predict_proba(X)
        self.proba_result_test = self.model.predict_proba(X)[:, 1]
        self.y_test = y
        self.X_test = X
        st.write(int(1000 * (1 - self.model.predict_proba(X)[:, 1])))

        return self.proba_result

    def upload_data(self, sample_type='train'):
        data = pd.DataFrame()

        data['proba'] = self.proba_result_test
        data['score'] = data['proba'].apply(lambda x: int(1000 * (1 - x)))
        data['target'] = self.y_test.values

        if sample_type == 'train':
            data.to_excel('train_output.xlsx')
        elif sample_type == 'test':
            data.to_excel('test_output.xlsx')
        elif sample_type == 'out_sample':
            data.to_excel('out_time_output.xlsx')

    def ensemble_metrics(self):
        probs = self.model.predict_proba(self.X_train)
        preds = probs[:, 1]

        # prc_train_p, prc_train_r, prc_train_th = metrics.precision_recall_curve(self.y_train, preds)
        # fig, ax = plt.subplots()
        # ax.plot(prc_train_r, prc_train_p, color='purple')
        # ax.set_title('Train Precision-Recall Curve')
        # ax.set_ylabel('Precision')
        # ax.set_xlabel('Recall')
        # st.pyplot(plt)

        fpr, tpr, threshold = metrics.roc_curve(self.y_train, preds)
        roc_auc = metrics.auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.title('Train Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig('train_auc.png')
        st.pyplot(plt)

        probs = self.model.predict_proba(self.X_test)
        preds = probs[:, 1]

        # prc_test_p, prc_test_r, prc_test_th = metrics.precision_recall_curve(self.y_test, preds)
        # fig, ax = plt.subplots()
        # ax.plot(prc_train_r, prc_train_p, color='purple')
        # ax.set_title('Test Precision-Recall Curve')
        # ax.set_ylabel('Precision')
        # ax.set_xlabel('Recall')
        # st.pyplot(plt)


        fpr, tpr, threshold = metrics.roc_curve(self.y_test, preds)
        roc_auc = metrics.auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.title('Test Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig('test_auc.png')
        st.pyplot(plt)
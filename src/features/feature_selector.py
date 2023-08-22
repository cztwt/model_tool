import numpy as np
import pandas as pd
import warnings
import logging
import os

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import roc_auc_score

import optuna
import shap

from itertools import chain

import gc

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

curr_dic = os.path.dirname(os.path.abspath(__file__))

class FeatureSelector:
    def __init__(self, data, labels=None):
        self.data = data
        self.labels = labels

        self.remove_features = {}  # 存储每个特征筛选方法过滤的特征
        self.base_features = list(data.columns)
        self.ops = {}

        # 记录过滤特征的相关信息
        self.record_coll = None # 相关系数过滤特征信息
        self.record_low_importance = None # 树模型过滤特征信息
        self.record_missing = None # 缺失过滤特征信息
        
        self.feature_importances = None # 树模型特征重要性排序
        self.missing_stats = None # 特征缺失值信息
    
    def missing_filter(self, missing_threshold):
        """过滤掉缺失值占比missing_threshold以上的特征"""
        
        # 计算每个特征的空值占比
        missing_series = self.data.isnull().sum() / self.data.shape[0]
        self.missing_stats = pd.DataFrame(missing_series).rename(
            columns = {'index': 'feature', 0: 'missing_rate'}
        )
        # 根据缺失率占比进行排序
        self.missing_stats = self.missing_stats.sort_values('missing_rate', ascending = False)
        # 找到缺失率占比在missing_threshold以上的特征
        record_missing = pd.DataFrame(
            missing_series[missing_series > missing_threshold]
        ).reset_index().rename(columns = {'index': 'feature', 0: 'missing_rate'})
        missing_to_drop = list(record_missing['feature'])

        self.record_missing = record_missing
        self.remove_features['missing'] = missing_to_drop
        logger.info('过滤缺失率>{}的特征，共{}个特征'.format(missing_threshold, len(missing_to_drop)))

    def variance_filter(self, threshold=0):
        """过滤特征中方差<=threshold的特征
        参数：
            - threshold: 方差阈值, 默认为0
        """
        df = self.data

        selector = VarianceThreshold(threshold=threshold)
        X_filtered = selector.fit_transform(df)
        filtered_features = []

        if X_filtered.shape[1] > 0:
            mask = np.logical_not(selector.get_support())
            filtered_features = df.columns[mask].tolist()
        self.remove_features["var_filter"] = filtered_features
        logger.info("过滤方差<={}的特征，共{}个特征".format(threshold, len(filtered_features)))

    def corr_filter(self, threshold):
        """根据特征之间的相关系数, 找到相关系数大于threshold的每对特征, 并删掉其中一个
        参数：
            - threshold: 相关系数阈值
        """
        df = self.data
        # 计算特征之间的相关性矩阵
        corr_matrix = df.corr()
        # 获取相关性矩阵的上三角部分
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool)
        )
        # 找到相关性大于阈值的特征
        features_to_drop = [
            column for column in upper.columns if any(upper[column].abs() > threshold)
        ]
        # 记录相关性大于阈值的特征
        record_coll = pd.DataFrame(
            columns=["drop_feature", "corr_feature", "corr_value"]
        )
        for column in features_to_drop:
            corr_features = list(upper.index[upper[column].abs() > threshold])
            corr_values = list(upper[column][upper[column].abs() > threshold])
            drop_features = [column for _ in range(len(corr_features))]
            temp_df = pd.DataFrame.from_dict(
                {
                    "drop_feature": drop_features,
                    "corr_feature": corr_features,
                    "corr_value": corr_values,
                }
            )
            record_coll = record_coll.append(temp_df, ignore_index=True)
        self.record_coll = record_coll
        self.remove_features["corr_filter"] = features_to_drop
        logger.info("过滤相关系数>{}的特征，共{}个特征".format(threshold, len(features_to_drop)))

    def feature_importance_filter(
        self, task, n_iterations=10, early_stop=True, early_stopping_rounds=100, importance_type='split',
        eval_metric=None, n_estimators=2000, learning_rate=0.05, cumulative_importance=0.95,
        max_depth=5
    ):
        """根据树模型的特征重要性, 过滤累积系数大于cumulative_importance的特征
        
        参数：
            - task: 树模型任务'classification' or 'regression'
            - n_iterations: 迭代次数, 训练几次树模型
            - early_stop: 是否早停
            - early_stopping_rounds: 早停参数
            - importance_type: 特征重要性参数'split'、'gain' or 'shap'(采用shap value方法)
            - eval_metric: 验证集评估指标
            - n_estimators: 树的个数
            - learning_rate: 步长学习率
            - max_depth: 每棵树的深度
            - cumulative_importance: 累积系数(将特征重要性归一化按照降序排列求cumsum, 过滤掉大于cumulative_importance的)
        """
        if early_stop and eval_metric is None:
            raise ValueError('eval_metric必须和early_stop一起提供, 用来作为验证集, 分类可选auc, 回归可选l2')
        
        if self.labels is None:
            raise ValueError("请先提供数据的label标签.")
        
        features, labels = np.array(self.data), np.array(self.labels).reshape((-1,))
        feature_names = list(self.data.columns)
        
        logger.info("训练lgbm模型....")

        lgb_params = {
            "n_jobs": -1,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth
        }
        
        feature_importance_values = np.zeros(len(feature_names))
        for i in range(n_iterations):
            if task == 'classification':
                model = lgb.LGBMClassifier(**lgb_params)
            elif task == "regression":
                model = lgb.LGBMRegressor(**lgb_params)
            else:
                raise ValueError('task必须是"classification"或者"regression"')
            
            train_x, valid_x, train_y, valid_y = train_test_split(features, labels, test_size=0.2)
            if early_stop:
                model.fit(
                    train_x, train_y, eval_metric=eval_metric,
                    eval_set=[(valid_x, valid_y)],
                    early_stopping_rounds=early_stopping_rounds, verbose=-1,
                )
            else:
                model.fit(features, labels)
            
            # ax = lgb.plot_metric(evals_result, metric='auc')  # metric的值与之前的params里面的值对应
            # plt.show()
            
            if importance_type == 'shap':
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(valid_x)
                shap_values = np.sum(np.mean(abs(np.array(shap_values)), axis = 1), axis=0)
                feature_importance_values += shap_values / n_iterations
            else:
                feature_importance_values += model.feature_importances_ / n_iterations
            
            # 打印每轮次信息
            # logger.info('第{}次迭代, 训练集auc = {}'.format(
            #     i+1, roc_auc_score(train_y, model.predict_proba(train_x)[:, 1])
            # ))
            # logger.info('第{}次迭代, 验证集auc = {}'.format(
            #     i+1, roc_auc_score(valid_y, model.predict_proba(valid_x)[:, 1])
            # ))

        # 清理内存
        gc.enable()
        del train_x, train_y, valid_x, valid_y
        gc.collect()
        
        feature_importances = pd.DataFrame({
            "feature": feature_names, 
            "importance": feature_importance_values
        })
        
        # 根据特征重要性排序
        feature_importances = feature_importances.sort_values("importance", ascending=False).reset_index(drop=True)

        # 归一化特征重要性
        postive_features_sum = (feature_importances["importance"] * (feature_importances["importance"] >= 0)).sum()
        feature_importances["normalized_importance"] = (feature_importances["importance"] / postive_features_sum)
        feature_importances["cumulative_importance"] = np.cumsum(feature_importances["normalized_importance"])
            
        record_low_importance = feature_importances[feature_importances['cumulative_importance'] > cumulative_importance]
        features_to_drop = list(record_low_importance['feature'])

        self.feature_importances = feature_importances
        self.record_low_importance = record_low_importance
        self.remove_features['feat_importance_filter'] = features_to_drop
        logger.info("过滤累积特征重要性>{}的特征，共{}个特征".format(cumulative_importance, len(features_to_drop)))
    
    def all_filter(self, selection_params=None, feat_params=None):
        """用上述几个特征选择方法筛选特征
        
        参数：
            - selection_params : dict, 支持以下几种特征筛选的参数
                'missing_threshold'缺失值占比过滤 'variance_threshold'方差过滤
                'corr_threshold'相关系数过滤 'cum_importance_threshold' 特征重要性过滤
            - feat_params: dict, 利用树模型特征重要性作为特征筛选时需要传入的参数
                'task': classification or regression 必填
                'n_iterations': 迭代次数, 训练几次树模型
                'early_stop': 是否早停
                'early_stopping_rounds': 早停参数
                'importance_type': 特征重要性参数'split'、'gain' or 'shap'(采用shap value方法)
                'eval_metric': 验证集评估指标
                'n_estimators': 树的个数
                'learning_rate': 步长学习率
                'max_depth': 每棵树的深度
        
        """
        if selection_params:
            for key, val in selection_params.items():
                if key == 'missing_threshold':
                    self.missing_filter(missing_threshold=val)
                elif key == 'variance_threshold':
                    self.variance_filter(threshold=val)
                elif key == 'corr_threshold':
                    self.corr_filter(threshold=val)
                elif key == 'cum_importance_threshold':
                    feat_params['cumulative_importance'] = val
                    self.feature_importance_filter(**feat_params)

            # 删除掉需要过滤的特征
            self.feats_to_drop = list(set(list(chain(*list(self.remove_features.values())))))
            df = self.data.drop(columns = self.feats_to_drop)
            return df
        else:
            raise ValueError('请先传入参数')


if __name__ == '__main__':
    train = pd.read_csv('/Users/chenzhao/Desktop/龙盈智达/基于订单数据的目标客群模块通用化/project-name/data/raw/credit_example.csv')
    train_labels = train['TARGET']
    train = train.drop(columns = ['TARGET'])
    
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(curr_dir)
    import sys
    sys.path.append(parent_dir)
    print(sys.path)
    from preprocess import DataPreprocessor
    pre = DataPreprocessor()
    train = pre.encode_categorical_features(train, 'onehot')
    print(train.shape)
    
    fs = FeatureSelector(data = train, labels = train_labels)
    # 1 缺失值过滤
    # fs.missing_filter(0.6)
    # 2 方差过滤
    # fs.variance_filter(0)
    # 3 相关性过滤
    # fs.corr_filter(0.975)
    # 4 特征重要性过滤
    # fs.feature_importance_filter(task='classification', early_stop=True, eval_metric='auc', cumulative_importance=0.99)
    # 5 all
    selection_params = {'missing_threshold': 0.6, 'variance_threshold': 0, 'corr_threshold': 0.975, 'cum_importance_threshold': 0.99}
    feat_params = {'task': 'classification', 'early_stop': True, 'eval_metric': 'auc'}
    df = fs.all_filter(selection_params=selection_params, feat_params=feat_params)
    print(df.shape)

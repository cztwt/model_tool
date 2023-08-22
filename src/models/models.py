import numpy as np
import pandas as pd
# from utils import *

import optuna

from lightgbm import LGBMClassifier, LGBMRegressor

from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold


# class BayesOptimizer:
#     '''实现贝叶斯优化器
    
#     参数：
#         - model: 模型
#         - model_params_space: 模型参数空间
#         - cv_params: 交叉验证参数
#     '''
#     def __init__(
#         self, 
#         model = None, 
#         model_params_space: dict = None, 
#         cv_params: dict = None
#     ):
#         self.model = model
#         self.model_params_space = model_params_space
#         self.cv_params = cv_params
    
#     def objective(self, trial):
#         params = {}
#         for param_name, param_values in self.model_params_space.items():
#             if isinstance(param_values[0], int):
#                 params[param_name] = trial.suggest_int(param_name, *param_values)
#             elif isinstance(param_values[0], float):
#                 params[param_name] = trial.suggest_float(param_name, *param_values)
#             else:
#                 raise ValueError('不支持的参数类型.')

#         self.model.set_params(**params)
#         scores = cross_val_score(self.model, X_train, y_train, cv=self.cv_params['cv'], scoring=self.cv_params['scoring'])
#         return scores.mean()
    
#     def optimize(self, n_trials):
#         study = optuna.create_study(direction='maximize')
#         study.optimize(self.objective, n_trials=n_trials)

#         best_params = study.best_params
#         best_score = study.best_value
#         return best_params, best_score

class LightGBMModel:
    '''建立lightgbm模型, 利用贝叶斯优化以及交叉验证优化超参数, 并利用最佳参数进行建模.
    
    参数：
        - X: 特征数据集
        - y: 标签数据集
        - task: 模型任务类型 'classification'分类 'regression'回归
        - cv_params: 交叉验证参数, 'cv'分割策略 'scoring'评估指标(
            参考: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        )
        - bayes_params: 贝叶斯优化参数, 'n_trials'迭代次数, 'direction'(maximize, minimize)优化方向,需要与交叉验证中的评估指标对应
            注意是最大还是最小
        - model_params: 模型需要优化的超参数
    
    属性：
        - num_classes: 分类问题中标签数量, 用来判断是二分类还是多分类
        - best_score: 模型优化最好的分数
        - best_params: 模型优化最好的参数
    """
    '''
    def __init__(
        self, 
        X: pd.DataFrame = None, 
        y: pd.Series = None, 
        task: str = 'classification', 
        cv_params: dict = {'cv': 5, 'scoring': 'roc_auc'},
        bayes_params: dict = {'n_trials': 100, 'direction': 'maximize'},
        model_params_space: dict = None
    ):
        self.X = X
        self.y = y
        self.task = task
        self.cv_params = cv_params
        self.bayes_params = bayes_params
        self.model_params_space = model_params_space
        
        # 存储贝叶斯优化最好的分数和最好的参数集合
        self.best_params = None
        self.best_score = None
        
        # 判断多分类还是二分类
        if self.task == 'classification':
            self.num_classes = len(np.unique(self.y))
    
    def objective(self, trial):
        if not self.model_params_space:
            model_params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.1, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.1, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
            }
        else:
            model_params = {}
            for param_name, param_values in self.model_params_space.items():
                if isinstance(param_values[0], int):
                    model_params[param_name] = trial.suggest_int(param_name, *param_values)
                elif isinstance(param_values[0], float):
                    model_params[param_name] = trial.suggest_float(param_name, *param_values)
                else:
                    raise ValueError("不支持该类型的参数, 请输入int或者float.")
        
        if self.task == 'classification':
            if self.num_classes == 2:
                model = LGBMClassifier(objective='binary', metric='binary_logloss', verbose=-1, **model_params)
            else:
                model = LGBMClassifier(objective='multiclass', metric='multi_logloss', num_class=self.num_classes, verbose=-1, **model_params)
                
            cv = StratifiedKFold(n_splits=self.cv_params['cv'], shuffle=True, random_state=42)
            scores = cross_val_score(model, self.X, self.y, cv=cv, scoring=self.cv_params['scoring'])
            return np.mean(scores)
        elif self.task == 'regression':
            model = LGBMRegressor(objective='regression', metric='mse', **model_params)
            
            cv = KFold(n_splits=self.cv_params['cv'], shuffle=True, random_state=42)
            scores = cross_val_score(model, self.X, self.y, cv=cv, scoring=self.cv_params['scoring'])
            return -np.mean(scores)
        else:
            raise ValueError("无效的task, 请选择 'classification' 或者 'regression'.")
    
    def optimize_params(self):
        
        study = optuna.create_study(direction=self.bayes_params['direction'])
        study.optimize(self.objective, n_trials=self.bayes_params['n_trials'])
        
        self.best_params = study.best_params
        self.best_score = study.best_value
        # logger.info('贝叶斯优化最优参数：{}'.format(self.best_params))
        # logger.info('贝叶斯优化最优分数：{}'.format(self.best_score))
    
    def train(self):
        
        self.optimize_params()
        
        if self.task == 'classification':
            if self.num_classes == 2:
                model = LGBMClassifier(objective='binary', metric='binary_logloss', verbose=-1, **self.best_params)
            else:
                model = LGBMClassifier(objective='multiclass', metric='multi_logloss', num_class=self.num_classes, verbose=-1, **self.best_params)
        elif self.task == 'regression':
            model = LGBMRegressor(objective='regression', metric='mse', verbose=-1, **self.best_params)
        
        model.fit(self.X, self.y)
        return model



# 加载数据集
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
data = load_iris()
X, y = data.data, data.target

# 划分训练集和测试集
task = 'classification'
cv_params = {'cv': 5, 'scoring': 'roc_auc_ovr'}
bayes_params = {'n_trials': 100, 'direction': 'maximize'}
model_params_space = {
    'max_depth': (3, 10),
    'n_estimators': (100, 1000)
}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lgb = LightGBMModel(X=X_train, y=y_train, task = task, cv_params=cv_params, bayes_params=bayes_params, model_params_space=model_params_space)
lgb.train()
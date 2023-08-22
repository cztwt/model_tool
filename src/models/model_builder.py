import numpy as np
import pandas as pd

import optuna

from lightgbm import LGBMClassifier, LGBMRegressor

from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold


class LightGBMModel:
    """建立lightgbm模型, 利用贝叶斯优化方法优化超参数, 并选取最佳的参数进行建模
    
    参数：
        - X_train: 训练特征数据集
        - y_train: 训练标签数据集
        - X_vaild: 验证特征数据集
        - y_valid: 验证标签数据集
        - task: 模型任务类型 'classification'分类 'regression'回归
        - bayes_params: 贝叶斯优化参数
        - model_params: 模型需要优化的超参数
    
    属性：
        - num_classes: 分类问题中标签数量, 用来判断是二分类还是多分类
        - best_score: 模型优化最好的分数
        - best_params: 模型优化最好的参数
    """
    def __init__(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series, 
        X_vaild: pd.DataFrame, 
        y_valid: pd.Series, 
        task: str = 'classification', 
        bayes_params: dict = {'n_trials': 100},
        model_params: dict = None
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_vaild
        self.y_valid = y_valid
        self.task = task
        self.bayes_params = bayes_params
        self.model_params = model_params
        
        # 存储贝叶斯优化最好的分数和最好的参数集合
        self.best_params = None
        self.best_score = None
        
        # 判断多分类还是二分类
        if self.task == 'classification':
            self.num_classes = len(np.unique(self.y_train))
        
    def objective_classification(self, trial):
        if not self.model_params:
            model_params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.1, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.1, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
            }
        else:
            model_params = self.model_params
        
        if self.num_classes == 2:
            model = LGBMClassifier(objective='binary', metric='binary_logloss', **model_params)
            scoring = 'roc_auc'
        else:
            model = LGBMClassifier(objective='multiclass', metric='multi_logloss', num_class=self.num_classes, **model_params)
            scoring = 'roc_auc_ovr'
        
        # cv = KFold(n_splits=5, shuffle=True, random_state=42)
        # scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring=scoring)
        
        # return np.mean(scores)
        
        model.fit(self.X_train, self.y_train)
    
        if self.num_classes == 2:
            y_pred = model.predict_proba(self.X_valid)[:, 1]
            auc = roc_auc_score(self.y_valid, y_pred)
            return auc
        else:
            y_pred = model.predict_proba(self.X_valid)
            auc = roc_auc_score(self.y_valid, y_pred, multi_class='ovr')
            return auc
    
    def objective_regression(self, trial):
        if self.model_params is None:
            model_params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.1, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.1, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
            }
        else:
            model_params = self.model_params
    
        model = LGBMRegressor(objective='regression', metric='mse', **model_params)
        # scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='neg_mean_squared_error')
        # return -np.mean(scores)
    
        model.fit(self.X_train, self.y_train)
    
        y_pred = model.predict(self.X_valid)
        mse = mean_squared_error(self.y_valid, y_pred)
    
        return mse
    
    def optimize_params(self):
        
        if self.task == 'classification':
            objective = self.objective_classification
            direction = 'maximize'
        elif self.task == 'regression':
            objective = self.objective_regression
            direction = 'minimize'
        else:
            raise ValueError("无效的task, 请选择 'classification' 或者 'regression'.")
        
        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=self.bayes_params['n_trials'])
        
        self.best_params = study.best_params
        self.best_score = study.best_value
    
    def train(self):
        
        self.optimize_params()
        
        if self.task == 'classification':
            if self.num_classes == 2:
                model = LGBMClassifier(objective='binary', metric='binary_logloss', **self.best_params)
            else:
                model = LGBMClassifier(objective='multiclass', metric='multi_logloss', num_class=self.num_classes, **self.best_params)
        elif self.task == 'regression':
            model = LGBMRegressor(objective='regression', metric='mse', **self.best_params)
        
        model.fit(self.X_train, self.y_train)
        return model
        
    
    
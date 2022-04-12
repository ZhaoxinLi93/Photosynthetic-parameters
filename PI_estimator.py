# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 15:32:50 2022

@author: Li Zhaoxin
Estimator of phytoplankton photosynthetic parameters
---- Enhanced Random Forest Regressor ----
Multivariate linear regression + Random forest regression
"""

import numpy as np
from pathlib import Path
from joblib import dump, load
from datetime import datetime
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

#%%
class PI_Estimator(BaseEstimator):
    def __init__(self, model_1 = LinearRegression(), model_2 = RandomForestRegressor(), 
                 dict_params_1 = {'fit_intercept': True},
                 dict_params_2 = {'n_estimators': 300, 'oob_score': True, 'random_state': 0},
                 scaley_log10_flag = {'y0': True, 'y1': True}, 
                 scaler = [('sc1', RobustScaler()), ('sc2', StandardScaler())],
                 do_restore = False, restore_path = Path.cwd() / 'Model',
                 do_save = False, save_path = Path.cwd() / 'Model',
                 model_name = 'PI_Estimator_PBmax_Ek', **kwargs):
        super().__init__()
        self.dict_params_1 = dict_params_1
        self.dict_params_2 = dict_params_2
        self.scaley_log10_flag = scaley_log10_flag
        self.model_1 = model_1.set_params(**self.dict_params_1)
        self.model_2 = model_2.set_params(**self.dict_params_2)
        self.scaler = scaler
        self.do_restore = do_restore
        self.do_save = do_save
        self.restore_path = restore_path
        self.save_path = save_path
        self.model_name = model_name
        
        self.restore_model()
        
    def fit(self, x_input, y_input = None, **kwargs):
        self.reg_1 = Pipeline(self.scaler + [('model', self.model_1)])
        self.reg_2 = Pipeline([('model', self.model_2)])

        y_input_ = y_input.copy()
        for i, iflag in enumerate(self.scaley_log10_flag.values()):
            if iflag == True:
                y_input_[:, i] = np.log10(y_input_[:, i])
        self.reg_1.fit(x_input, y_input_)
        y_res = y_input_ - self.reg_1.predict(x_input)
        self.reg_2.fit(x_input, y_res)
        
        self.save_model()
        return self
    
    def predict(self, x_input, y_input = None):
        y_output = self.reg_1.predict(x_input) + self.reg_2.predict(x_input)
        for i, iflag in enumerate(self.scaley_log10_flag.values()):
            if iflag == True:
                y_output[:, i] = np.power(10, y_output[:, i])
        return y_output
    
    def score(self, x_input, y_input = None):
        score = r2_score(y_input, self.predict(x_input))
        return score
    
    def oob_score_(self):
        return self.reg_2[-1].oob_score_
    
    def feature_importances_(self):
        return self.reg_2[-1].feature_importances_
    
    def coef_(self):
        coef = self.reg_1[-1].coef_
        return coef
    
    def intercept_(self):
        intercept = self.reg_1[-1].intercept_
        return intercept
    
    def restore_model(self):
        if self.do_restore == True:
            Model_restore_path = str(Path(self.restore_path) / self.model_name)
            self.__dict__.update(load(Model_restore_path).__dict__)
            print('\n---------PI-estimator restored---------\n')

    def save_model(self):
        if self.do_save == True:
            self.save_path.mkdir(parents = True, exist_ok = True)
            model_name = self.model_name + datetime.now().strftime('_%Y%m%d_%H%M%S') + '_v1.0.joblib'
            Model_save_path = str(Path(self.save_path) / model_name)
            dump(self, Model_save_path, compress = 'zlib')
            print('\n---------PI-estimator saved---------\n')

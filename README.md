# Estimation of phytoplankton photosynthetic parameters
## Description
The pre-trained PI-estimator can be used to estimate photosynthetic (photosynthesis-irradiance) parameters of phytoplankton in clear to turbid waters.
#### Input
* Water temperature (T, °C)
* Chlorophyll-_a_ concentration (Chl-_a_, mg/m3)
* Depth-averaged diffuse attenuation coefficient of PAR (KdPAR, 1/m) 
* Light transmittance of downwelling PAR (rPAR, unitless)
#### Output
* Assimilation number (PBmax, mg C/mg Chl/h)
* Light saturation parameter (Ek, μmol photons/m2/s)

## Reference
["Remote estimation of phytoplankton primary production in clear to turbid waters by integrating a semi-analytical model with a machine learning algorithm"](https://www.sciencedirect.com/science/article/pii/S0034425722001419). Zhaoxin Li, et al. (2022). Remote Sensing of Environment. 113027. 10.1016/j.rse.2022.113027.

## Usage
```
from PI_estimator import PI_Estimator
Model = PI_Estimator(do_restore = True, restore_path = './Model', model_name = 'PI_Estimator_PBmax_Ek_20220412_v1.0.joblib')
Y = Model.predict(X)
PBmax = Y[:, 0]
Ek = Y[:, 1]
```
The user can set "n_jobs" to enable parallel computation:
```
Model.reg_2[-1].n_jobs = -1
```
(see [sklearn.ensemble.RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html))

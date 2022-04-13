# Estimation of phytoplankton photosynthetic parameters
## Description
The pre-trained PI-estimator can be used to estimate photosynthetic (photosynthesis-irradiance) parameters of phytoplankton in clear to turbid waters.
#### Input
* Water temperature (T)
* Chlorophyll-_a_ concentration (Chl-_a_)
* Depth-averaged diffuse attenuation coefficient of PAR (KdPAR) 
* Light transmittance of downwelling PAR (rPAR)
#### Output
* Assimilation number (PBmax)
* Light saturation parameter (Ek)

## Reference
["Remote estimation of phytoplankton primary production in clear to turbid waters by integrating a semi-analytical model with a machine learning algorithm"](https://www.sciencedirect.com/science/article/pii/S0034425722001419). Zhaoxin Li, et al. (2022). Remote Sensing of Environment. 113027. 10.1016/j.rse.2022.113027.

## Usage
```
from PI-estimator import PI-Estimator
Model = PI-Estimator(do_restore = True, restore_path = './Model', model_name = '...')
Y = Model.predict(X)
PBmax = Y[:, 0]
Ek = Y[:, 1]
```

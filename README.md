# MFA_testing
This repository has the goal of testing whether a Mixture of Factor Analyzers (MFA) is a good model-choise for hyperspectral data. 

The notebook aims to test:
-   Model Complexity - Will the model be able to get a good BIC score and out-of-sample likelyhood
-   Reconstruction Fidelity - Calculate the reconstruction error in RMSE and SAM (Spectral Angle Mapper)
-   Interpretability - Does the components represent different and new signals
-   Class Interpretability - Assign a colour to each FA, and map the Hyperspectral image next to its rgb-image, do we see the same structures? 


Includes: 
Hypso package (GDAL wheel is necessery for it to run)
torch
glob

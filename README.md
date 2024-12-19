Prediction of melting points

This git repository contains the Supplementary Information to the publication “Prediction of melting points of chemicals with a data augmentation-based neural network approach”:

• main.py

• mygraphconvmodel.py

• Data folder containing all datasets used in the study and the corresponding prediction outcomes

We used DeepChem library for model development. The full code for the NN development is given at the GIT repository of DeepChem https://github.com/deepchem/deepchem. Our adapted code is provided here in main.py. The keras model implementation is given in mygraphconvmodel.py.
The dataset for model development was taken from Bradley et al. [1].

All corrections made during the data curation procedure are included in the dataset (data_annotated.csv).

[1] Jean-Claude Bradley Open Melting Point Dataset. figshare Dataset. Bradley, J.-C. W., Antony; Lang, Andrew (2014)., Ed.; 2014: https://doi.org/10.6084/m9.figshare.1031637.v2


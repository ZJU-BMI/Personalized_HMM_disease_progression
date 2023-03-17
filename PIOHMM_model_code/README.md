# Personlized_Hmm
Our personalized_HMM model is part of the PIOHMM model. The PIOHMM model can refer to the paper "Discovery of Parkinson’s disease states and disease 
progression modelling: a longitudinal data study using machine learning".

 The PIOHMM model class is in `piohmm.py`.

## Running the code
See the jupyter notebook 'Sample Model' for a simple example of the model. There are three primary components for using a PIOHMM: 
* `HMM` to specify the particular model; see `__init__` for a description of the options
* `learn_model` to perform inference
* `predict_sequence` to use the Viterbi algorithm to make state predictions


## Machine and Deep learning algorithms of the project
1. CPA.py: Script to train and build a CPA-based model for the serology data. **It uses trainable vectors** to add the species effect.
2. CPA_intermediate_encoders.py: Script to train and build a CPA-based model for the serology data. **It uses ANNs** to add the species effect. It also generates Figure 6b-6c.
4. CrossValSplit.py: Script to create folds of the data for 10-fold cross-validation.
5. embeedProfiles.py: Script to embeed serology profiles into the latent spaces.
6. models.py: Contains classes to define layers for the autoencoder and the encoder and decoder models.
7. evaluationUtils.py: Contains functions for evaluation of the models.
8. importanceScores.py: Script to generate all importance scores that are needed for various tasks in the serology study.
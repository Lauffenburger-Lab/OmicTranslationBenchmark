## Code and data for the fibrosis case study

## Data
The data required to run this script are coming from 4 single-cell RNA sequencing datasets:
1. Human lung fibrosis[^1]
2. Mouse lung fibrosis[^2]
3. Mouse lung fibrosis[^3] external test set
4. Human liver cirrhosis[^4]

## Folder structure
1. **data** : Folder that should contain the raw data of this specific case study.
2. **models** : Folder that contains trained models for that study.
3. **pre_models** : Folder containing pre-trained models of the adverse classifier and encoders. 
3. **results** : Folder to save results of downstream analysis of the study.

## Scripts
1. TranslationModel_lung.py : Code to train a CPA-like[^5] model to translate mouse lung fibrosis to human lung fibrosis and vice-versa.
2. PretrainTranslationModel_lung.py : Code to for the pre-training phase of the CPA-like[^5] model to translate mouse lung fibrosis to human lung fibrosis and vice-versa.
3. TranslationModel_lung_homologues.py: Code to train a CPA-like[^5] model to translate mouse lung fibrosis to human lung fibrosis and vice-versa, but only using homolog genes.
4. PretrainTranslationModel_lung_homologues.py : Code to for the pre-training phase of the CPA-like[^5] model to translate mouse lung fibrosis to human lung fibrosis and vice-versa, but only using homolog genes.
5. PretrainDCSv2_lung_homologues.py : Code to for the pre-training phase of the DCS model[^6] (DCS modified v2) using homolog genes.
6. DCSv2_lung_homologues.py : Code to train a modified version of the DCS model[^6] (DCS modified v2) using homolog genes.
7. LiverDataInLungModel.py : Code to make predictions on the external mouse lung and liver cirrhosis datasets.
8. trainingUtils.py : Script containing utility functions to perform model training.
9. models.py : Script containing functions and classes used to define models or different parts of them
10. evaluationUtils.py: Script containing code to evaluate predictions of the models.
11. createFolds.py: Script to split data into 10-fold cross-validation splits.
12. EmbsEval.R: Script used to evaluate embeddings and latent space separation.
13. FindOrthologs.R: Script used to create a human-mouse homolog genes mapping.
14. InterDatasetEvaluation.R: Script to analyze and evaluate the results on the external test-set datasets (mouse lung new, human liver cirrhosis)
15. performanceEval.R: Script to evaluate the performance of models in the inter-species lung fibrosis translation
16. SpeciesSeparateAnalysis.R: Analysis to find fibrosis relevent genes in specific species seperately, by using simple PCA and linear regression models.

## References
[^1]: Habermann, A. C. et al. Single-cell RNA sequencing reveals profibrotic roles of distinct epithelial and mesenchymal lineages in pulmonary fibrosis. Science Advances 6, eaba1972 (2020).
[^2]: Strunz, M. et al. Alveolar regeneration through a Krt8+ transitional stem cell state that persists in human lung fibrosis. Nat Commun 11, 3559 (2020).
[^3]: Aran, D. et al. Reference-based analysis of lung single-cell sequencing reveals a transitional profibrotic macrophage. Nat Immunol 20, 163–172 (2019).
[^4]: Ramachandran, P. et al. Resolving the fibrotic niche of human liver cirrhosis at single-cell level. Nature 575, 512–518 (2019).
[^5]: Lotfollahi, Mohammad, et al. "Predicting cellular responses to complex perturbations in high‐throughput screens." Molecular Systems Biology (2023): e11517.
[^6]: Umarov, Ramzan, Yu Li, and Erik Arner. "DeepCellState: An autoencoder-based framework for predicting cell type specific transcriptional states induced by drug treatment." PLoS Computational Biology 17.10 (2021): e1009465.

## Code and data for the serology case study

## Data
The data required to run the serology case study:
1. human dataset : https://pubmed.ncbi.nlm.nih.gov/26544943/
2. non-human primates dataset : https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6192527/

## Folder structure
1. **data** : Folder that should contain the raw data of this specific case study. The authors of the original studies were contacted to retrieve the data.
2. **learning** : Folder containing scripts to train machine learning models.
3. **results** : Folder containing the results of this study, **when using the CPA-based approach with trainable vectors** to add the species effect. Here you should save also your own results.
4. **results_intermediate_encoders** : Folder containing the results of this study, **when using the CPA-based approach with ANNs** to add the species effect. Here you should save also your own results.
5. **LRT_results** : Folder containing the results from the Likelihood Ratio Tests (LRT), when using the CPA-based approach with ANNs.
6. **LRT_results_cpa** : Folder containing the results from the Likelihood Ratio Tests (LRT), when using the CPA-based approach with trainable vectors.
7. **importance_results** : Folder containing the results from importance analysis using integradient gradients, when using the CPA-based approach with ANNs.
8. **importance_results_cpa** : Folder containing the results from importance analysis using integradient gradients, when using the CPA-based approach with trainable vectors.

## Scripts present here
1. preprocessSerology.R: Script to pre-process the serology datasets.
2. embsEval.R: Script to generate Supplementary Figure 31 and evaluate embeddings separation in the latent space.
3. evalClassification.R: Evaluate the performance in classifying species, protection, vaccination status, and species translation (Figure 6b)
4. feature_importance.R: Script to perform the whole importance analysis for this study (after getting the results from LRT and gradient scores). It is used for Supplementary Figures 33-34.
5. lrt_analysis.R: Script to perform LRT (Supplementary Figure 32)
6. NHP_feature_importance_interpretation.R: Script to create Figure 6 c-d.

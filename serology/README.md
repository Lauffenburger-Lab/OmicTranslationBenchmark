## Code and data for the serology case study

## Data
The data required to run the serology case study:
	* **human dataset** : https://pubmed.ncbi.nlm.nih.gov/26544943/
	* **non-human primates dataset** : https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6192527/

## Folder structure
1. **data** : Folder that should contain the raw data of this specific case study. The authors of the original studies were contacred to retriece the data.
2. **learning** : Folder containing scripts to train machine learining models.
3. **results** : Folder containing the results of this study, **when using the CPA-based approach with trainable vectors** to add the species effect. Here you should save also your own results.
4. **results_intermediate_encoders** : Folder containing the results of this study, **when using the CPA-based approach with ANNs** to add the species effect. Here you should save also your own results.
5. **LRT_results** : Folder containing the results from the Likelihood Ratio Tests (LRT).

## Scripts present here
1. embsEval.R: Script to generate Supplementary Figure 15 and evaluate embeddings separation in the latent space.
2. evalClassification.R: Evaluate the performance in classifying species, protection, vaccination status, species translation (Figure 6d)
3. feature_importance.R: Script to perform the whole importance analysis for this study (after getting the results from LRT and gradient scores). It is used for Figure 6e,6f, Supplementary Figure 17-18.
4. lrt_analysis.R: Script to perform LRT (Supplementary Figure 16)

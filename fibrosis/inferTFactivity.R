# Load requeired packages
library("dorothea")
library(tidyverse)
library(Seurat)

### Load preprocessed human gene expression-------------
cell_human_info_all <- data.table::fread('data/human_cells_info.csv')
meta_data_all <- data.table::fread('../../../Fibrosis Species Translation/human lung fibrosis/Habermann_metadata.csv')

gene_data <- data.table::fread('../../../Fibrosis Species Translation/human lung fibrosis/Habermann_filtered_scaled_genes.csv')
meta_data_all <- meta_data_all %>% filter(V1 %in% gene_data$V1)
meta_data_all <- meta_data_all %>% group_by(Cell_Type) %>% mutate(cell_type_counts = n()) %>% ungroup() %>%
  filter(cell_type_counts>10)
cell_human_info_all <- cell_human_info_all %>% filter(specific_cell %in% meta_data_all$Cell_Type)
gene_data <- gene_data %>% filter(V1 %in% meta_data_all$V1)
gene_data <- gene_data %>% column_to_rownames('V1')
gene_data <- t(gene_data)
hist(as.matrix(gene_data))

### Use data with dorothea------------------
standarized_genes <- gene_data - rowMeans(gene_data)
hist(standarized_genes)

minNrOfGenes = 5
dorotheaData = read.table('../../../Artificial-Signaling-Network/TF activities/annotation/dorothea.tsv', sep = "\t", header=TRUE)
confidenceFilter = is.element(dorotheaData$confidence, c('A', 'B'))
dorotheaData = dorotheaData[confidenceFilter,]

# Estimate TF activities
settings = list(verbose = TRUE, minsize = minNrOfGenes)
TF_activities = run_viper(standarized_genes, dorotheaData, options =  settings)
TF_activities <- t(TF_activities)
### save data for future machine learning-------------
data.table::fwrite(meta_data_all,
                   '../../../Fibrosis Species Translation/human lung fibrosis/meta_data_human_filtered.csv')
data.table::fwrite(as.data.frame(TF_activities),
                   '../../../Fibrosis Species Translation/human lung fibrosis/TF_activities_human_filtered.csv',row.names = T)

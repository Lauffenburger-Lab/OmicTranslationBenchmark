library(tidyverse)
library(cmapR)
library(doFuture)
# parallel: set number of workers
cores <- 16
registerDoFuture()
plan(multisession,workers = cores)
library(doRNG)
library(caret)
########## The whole pre-processing analysis is in the L1000 folder of the new data ###############

### Load data and keep only well-inferred and landmark genes----------------------------------------------------
geneInfo <- read.delim('../../../L1000_2021_11_23/geneinfo_beta.txt')
# geneInfo <-  geneInfo %>% filter(feature_space != "inferred")
geneInfo <-  geneInfo %>% filter(feature_space == "landmark") # keep landmarks
# Keep only protein-coding genes
geneInfo <- geneInfo %>% filter(gene_type=="protein-coding")

# Load signature info and split data to high quality replicates and low quality replicates
sigInfo <- read.delim('../../../L1000_2021_11_23/siginfo_beta.txt')
sigInfo <- sigInfo %>% mutate(quality_replicates = ifelse(qc_pass==1 & nsample>=3,1,0)) # no exempler controls so I just remove that constraint
sigInfo <- sigInfo %>% filter(is_exemplar_sig==1)
sigInfo <- sigInfo %>% filter(pert_type=='trt_cp')
# sigInfo <- sigInfo %>% filter(pert_type=='ctl_untrt')
sigInfo <- sigInfo %>% filter(quality_replicates==1)
sigInfo <- sigInfo %>% filter(tas>=0.3)
sigInfo <- sigInfo %>% group_by(cell_iname) %>% mutate(per_cell_sigs = n_distinct(sig_id)) %>% ungroup()

# Duplicate information
sigInfo <- sigInfo %>% mutate(duplIdentifier = paste0(cmap_name,"_",pert_idose,"_",pert_itime,"_",cell_iname))
sigInfo <- sigInfo %>% group_by(duplIdentifier) %>% mutate(dupl_counts = n()) %>% ungroup()

# Drug condition information
sigInfo <- sigInfo  %>% mutate(conditionId = paste0(cmap_name,"_",pert_idose,"_",pert_itime))
conditions <- sigInfo %>%  group_by(cell_iname) %>% summarise(conditions_per_cell = n_distinct(conditionId)) %>% ungroup()

### Load gene expression data -----------------------------------------------------------------------------------

# Split sigs to run in parallel
sigIds <- unique(sigInfo$sig_id)
sigList <-  split(sigIds, ceiling(seq_along(sigIds)/ceiling(length(sigIds)/cores)))

# # Parallelize parse_gctx
# ds_path <- '../../../L1000_2021_11_23/level5_beta_ctl_n58022x12328.gctx'
# parse_gctx_parallel <- function(path ,rid,cid){
#   gctx_file <- parse_gctx(path ,rid = rid,cid = cid)
#   return(gctx_file@mat)
# }
# cmap_gctx <- foreach(sigs = sigList) %dopar% {
#   parse_gctx_parallel(ds_path ,rid = unique(as.character(geneInfo$gene_id)),cid = sigs)
# }
# cmap <-do.call(cbind,cmap_gctx)
# saveRDS(cmap,'preprocessed_data/baselineCell/cmap_all_baselines_q1.rds')
# cmap <- t(cmap)
# cmap <- as.data.frame(cmap)
# data.table::fwrite(cmap,'preprocessed_data/baselineCell/cmap_all_baselines_q1.csv',row.names = T)

### Clean to keep cell lines with enough data for translation -----------------------------------------------------------------------------------
# cmap <- data.table::fread('preprocessed_data/all_cmap_landmarks.csv',header = T) %>% column_to_rownames('V1')
sigInfo <- sigInfo %>% filter(sig_id %in% colnames(cmap))
sigInfo <- sigInfo %>% filter(per_cell_sigs>=100)
print(paste0('Unique cell lines : ',length(unique(sigInfo$cell_iname))))


## Save and create folder per cell line
num_folds <- 5
for (cell in unique(sigInfo$cell_iname)){
  sampleInfo <- sigInfo %>% filter(cell_iname==cell)
  dir.create(path = paste0('preprocessed_data/SameCellimputationModel/',cell))
  folds <- createFolds(sampleInfo$sig_id, k = num_folds, list = TRUE, returnTrain = FALSE)
  i <- 0
  for (fold in folds){
    val <- sampleInfo[fold,]
    train <- sampleInfo[-fold,]
    
    data.table::fwrite(train,paste0('preprocessed_data/SameCellimputationModel/',cell,'/train_',i,'.csv'))
    data.table::fwrite(val,paste0('preprocessed_data/SameCellimputationModel/',cell,'/val_',i,'.csv'))
    
    i <- i+1
  }
  print(paste0('Finished cell-line : ',cell))
}

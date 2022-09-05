library(tidyverse)
library(cmapR)

library(doFuture)

# parallel set number of workers
cores <- 15
registerDoFuture()
plan(multiprocess,workers = cores)

### Load data and keep only well-inferred and landmark genes----
# Check L1000 documentation for information.
geneInfo <- read.delim('../../../L1000_2021_11_23/geneinfo_beta.txt')
geneInfo <-  geneInfo %>% filter(feature_space != "inferred")
# Keep only protein-coding genes
geneInfo <- geneInfo %>% filter(gene_type=="protein-coding")

#Load signature info and split data to high quality replicates and low quality replicates
sigInfo <- read.delim('../../../L1000_2021_11_23/siginfo_beta.txt')

# Create a proxy for quality of replicates
# Keep only samples with at least 3 replicates and that satisfy specific conditions.
# Check the LINCS2020 Release Metadata Field Definitions.xlsx file for 
# a complete description of each argument. It can be accessed online 
# or in the data folder.

sigInfo <- sigInfo %>% 
  mutate(quality_replicates = ifelse(is_exemplar_sig==1 & qc_pass==1 & nsample>=3,1,0))

# Filter drugs
sigInfo <- sigInfo %>% filter(pert_type=='trt_cp')
sigInfo <- sigInfo %>% filter(quality_replicates==1)
sigInfo <- sigInfo %>% filter(tas>=0.3)

# Create identifier to signify duplicate
# signatures: meaning same drug, same dose,
# same time duration, same cell-type
sigInfo <- sigInfo %>% 
  mutate(duplIdentifier = paste0(cmap_name,"_",pert_idose,"_",pert_itime,"_",cell_iname))
sigInfo <- sigInfo %>% group_by(duplIdentifier) %>%
  mutate(dupl_counts = n()) %>% ungroup()

### Load gene expression data----

# rid is the gene entrez_id to find the gene in the data
# cid is the sig_id, meaning the sampe id
# path is the path to the data
parse_gctx_parallel <- function(path ,rid,cid){
  gctx_file <- parse_gctx(path ,rid = rid,cid = cid)
  return(gctx_file@mat)
}

# Read GeX for drugs
# Split sig_ids to run in parallel
sigIds <- unique(sigInfo$sig_id)
sigList <-  split(sigIds, 
                  ceiling(seq_along(sigIds)/ceiling(length(sigIds)/cores)))

# Parallelize parse_gctx function

# Path to raw data
ds_path <- '../../../L1000_2021_11_23/level5_beta_trt_cp_n720216x12328.gctx'

# Parse the data file in parallel
cmap_gctx <- foreach(sigs = sigList) %dopar% {
  parse_gctx_parallel(ds_path ,
                      rid = unique(as.character(geneInfo$gene_id)),
                      cid = sigs)
}
cmap <-do.call(cbind,cmap_gctx)

### Load CCLE data----
ccle <- t(data.table::fread('../data/CCLE/CCLE_expression.csv') %>% column_to_rownames('V1'))
ccle <- as.data.frame(ccle) %>% rownames_to_column('V1') %>% separate(V1,c('gene_id','useless'),sep=" ") %>%
  dplyr::select(-useless) %>% column_to_rownames('gene_id')
ccle <- as.data.frame(t(ccle)) %>% rownames_to_column('DepMap_ID')
sample_info <- data.table::fread('../data/CCLE/sample_info.csv') %>% dplyr::select(DepMap_ID,stripped_cell_line_name) %>%
  unique()
ccle <- left_join(ccle,sample_info) %>% dplyr::select(-DepMap_ID) %>%
  column_to_rownames('stripped_cell_line_name')
ccle <- ccle[which(rownames(ccle) %in% unique(sigInfo$cell_iname)),]

###CCLE based similarity and use that to make predictions----
ccle_cor  <- cor(t(ccle))
fold_change <- function(x,y){
  return(x/y)
}

ccle_fc <- NULL
for (i in 1:nrow(ccle)){
  k <- 1
  for (j in i:nrow(ccle)){
    if (k==1){
      mat <- ccle[i,]/ccle[j,]
    }else{
      mat <- rbind(mat,ccle[i,]/ccle[j,])
    }
    k <- k+1
  }
  mat <- as.matrix(mat)
  rownames(mat) <-  rownames(ccle)[i:nrow(ccle)] 
  mat[which(is.na(mat) | mat==Inf)] <- 0
  ccle_fc[[i]] <- mat
}
print('Finished FC')
saveRDS(ccle_fc,'../results/ccle_fc.rds')
### CV for similarity based approach using L1000 z-scores----

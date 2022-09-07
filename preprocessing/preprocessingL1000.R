library(tidyverse)
library(cmapR)

library(doFuture)

# parallel set number of workers
cores <- 15
registerDoFuture()
plan(multiprocess,workers = cores)

########## The whole pre-processing analysis is in the L1000 folder of the new data ###############

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

# Drug condition information
sigInfo <- sigInfo  %>% mutate(conditionId = paste0(cmap_name,"_",pert_idose,"_",pert_itime))
conditions <- sigInfo %>%  group_by(cell_iname) %>% summarise(conditions_per_cell = n_distinct(conditionId)) %>% ungroup()
write.csv(sigInfo %>% select(sig_id,conditionId,cell_iname) %>% unique(),'preprocessed_data/all_conditions_tas03.csv')
# Take top 5 cell-lines and keep the two with the most common data
#cells <-conditions$cell_iname[order(conditions$conditions_per_cell,decreasing = T)][1:100] 
#print(cells)
cells <- unique(sigInfo$cell_iname)

common <- matrix(nrow = length(cells),ncol = length(cells))
colnames(common) <- cells
rownames(common) <- cells
for (i in 1:length(cells)){
  cell1 <- sigInfo %>% filter(cell_iname==cells[i])
  cell1 <- unique(cell1$conditionId)
  for (j in i:length(cells)){
    cell2 <- sigInfo %>% filter(cell_iname==cells[j])
    cell2 <- unique(cell2$conditionId)
    common[i,j] <- length(intersect(cell1,cell2))
  }
}
common <- reshape2::melt(common)
common <- common %>% filter(!is.na(value))
common <- common %>% filter(Var1!=Var2)

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

### Load GeX for the above filtered pairs of cell-lines----
sigInfo <- sigInfo %>% filter(cell_iname %in% unique(c(common$Var1,common$Var2)))
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
gc()
cmap <-do.call(cbind,cmap_gctx)
#cmap_cor_all <- cor(cmap)

#Cell-line based similarity----
#common <- common %>% filter(Var1 %in% rownames(ccle) & Var2 %in% rownames(ccle))
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
#saveRDS(ccle_fc,'../results/ccle_fc.rds')
ccle_fc <- readRDS('../results/ccle_fc.rds')

### ADD CCLE GSEA DISTANCE
library(doRNG)
# Specify the thresholds to check
# bottom and top regulated genes
thresholds <- c(30,50,100,200,300,400,
                500,600,700,800,900,1000)

# Initialize empty list for the results:
# Each element of the list (for each threshold)
# contains an NxN matrix with comparing all these
# samples. Each element of the matrix is the
# GSEA distance.
dist_all_ccle <- NULL
### SOS:
### RUN FIRST THE distance_scores.R
### SCRIPT TO LOAD THE FUNCTION!!!
### calculate distances: SEE distance_scores.R
# for information about the function inputs
dist_all_ccle <- foreach(thres = thresholds) %dorng% {
  distance_scores(num_table = t(ccle) ,
                  threshold_count = thres,names = colnames(t(ccle)))
}
# Transform list to array
distance <- do.call(cbind,dist_all_ccle)
distance <- array(distance,
                  c(dim=dim(dist_all_ccle[[1]]),length(dist_all_ccle)))
# Get the average distance across thresholds
mean_dist <- apply(distance, c(1,2), mean, na.rm = TRUE)
colnames(mean_dist) <- colnames(t(ccle))
rownames(mean_dist) <- colnames(t(ccle))
### Convert matrix into data frame
# Keep only unique (non-self) pairs
mean_dist[lower.tri(mean_dist,diag = T)] <- -100
dist <- reshape2::melt(mean_dist)
dist <- dist %>% filter(value != -100)
colnames(dist)[3] <- 'ccle_gsea'

dist <- left_join(dist,duplicateSigs,by = c("Var1"="sig_id"))
dist <- left_join(dist,duplicateSigs,by = c("Var2"="sig_id"))
dist <- dist %>% mutate(is_duplicate = (duplIdentifier.x==duplIdentifier.y))

gc()
common <- common %>% filter(value>0)
common <- left_join(common,dist,by=c('Var1','Var2'))
common$ccle_gsea <- 1 - common$ccle_gsea
common$cmap_cor <- 0
common$ccle_cor <- 0
common$ccle_avg_fc <- 0
for (i in 1:nrow(common)){
  cell1 <- common$Var1[i]
  cell2 <- common$Var2[i]
  cell_info1 <- sigInfo %>% filter(cell_iname==cell1)
  cell_info2 <- sigInfo %>% filter(cell_iname==cell2)
  sig1 <- unique(cell_info1$sig_id)
  sig2 <- unique(cell_info2$sig_id)
  ind1 <- which(rownames(ccle_cor)==cell1)
  ind2 <- which(rownames(ccle_cor)==cell2)
  
  #cmap_cor <- cmap_cor_all[sig1,sig2]
  
  paired <- left_join(cell_info1 %>% select(c('sig_id.x'='sig_id'),conditionId) %>% unique(), 
                      cell_info2 %>% select(c('sig_id.y'='sig_id'),conditionId) %>% unique()) %>% 
    filter(!is.na(sig_id.x) & !is.na(sig_id.y)) %>% unique()
  
  cmap1 <- t(cmap[,paired$sig_id.x])
  cmap2 <- t(cmap[,paired$sig_id.y])
  
  #cmap_cor <- cmap_cor[paired$sig_id.x,paired$sig_id.y]
  
  if (is_empty(ind1) | is_empty(ind2)){
    common$ccle_cor[i] <- NA
    common$ccle_avg_fc[i] <- NA
  } else{
    ccle_cor_pair <- ccle_cor[ind1,ind2]
    
    fc <- ccle_fc[[ind1]]
    ind2 <- which(rownames(fc)==cell2)
    fc <- fc[ind2,]
    
    common$ccle_cor[i] <- ccle_cor_pair
    common$ccle_avg_fc[i] <- mean(fc)
  }
  
  common$cmap_cor[i] <- cor(c(cmap1),c(cmap2))
  
  print(paste0('Finished:',i))
  
}
saveRDS(common,'../results/cell_pairs_similarity.rds')

common <- readRDS('../results/cell_pairs_similarity.rds')
common_filtered <- common %>% filter(!is.na(ccle_gsea))
ggplot(common_filtered,aes(cmap_cor,ccle_gsea)) + geom_point()  + ylim(c(0,1)) + geom_smooth()
cor(common_filtered$ccle_gsea,common_filtered$cmap_cor)

### For the pair with the most common datasets split it in smaller datasets to see the effect of training size----
### Then do it for other 2 pairs of cells too.
ind <- which(common$value==max(common$value))
cell1 <- as.character(common$Var1[ind])
cell2 <- as.character(common$Var2[ind])

### We need to sample diverse conditions.
### Basicaly, we sample a number of drugs (n1) and randomly then select conditions (n1) for these drugs
### Then find the corresponding condition in each cell-line.
### We randomly select conditions (n2) and other conditons from the same drugs (n3) 
### such as that n1+n2+n3=n the number of paired samples.
### Finally each time we sample conditions with the minimum GeX correlation.
sigInfo <- sigInfo %>% filter(cell_iname %in% unique(c(common$Var1,common$Var2)))
drugs <- unique(sigInfo$cmap_name)
pairedInfo <- left_join(sigInfo %>% filter(cell_iname=='A375') %>% 
                          select(c('sig_id.x'='sig_id'),conditionId,cmap_name) %>% unique(), 
                        sigInfo %>% filter(cell_iname=='HT29') %>% 
                          select(c('sig_id.y'='sig_id'),conditionId,cmap_name) %>% unique()) %>% 
  filter(!is.na(sig_id.x) & !is.na(sig_id.y)) %>% 
  mutate(pair_id=paste0(sig_id.x,'_',sig_id.y))%>% unique()
conditionInfo <- sigInfo %>% select(cmap_name,conditionId) %>% unique()

# Lets assume for now that since the maximum percentage of paired conditions in each cell-line is <40%.
# That n1 is 40% of n and n2,n3 are 40%,20%.
max_n1 <- nrow(pairedInfo)
n <- c(seq(50,250,50),400,500,750,1000) # the number of samples per cell-line
n1 <- sapply(0.4*n,ceiling)
n2 <- sapply(0.4*n,ceiling)
n3 <- sapply(0.2*n,ceiling)
print(paste0('The sum of n1+n2+n3 is equal to n: ',all(n1+n2+n3==n)))


cmap_cor <- cor(cmap)
gc()
cmap_cor <- reshape2::melt(cmap_cor)
gc()
cmap_cor <- cmap_cor %>% filter(Var1!=Var2)
gc()

for (i in 1:length(n)) {
  sampled_conditions <- createSample(sigInfo,pairedInfo,n1[i],2*n2[i],2*n3[i],c('A375','HT29'),cmap_cor,maxIter=100)
  saveRDS(sampled_conditions,paste0('preprocessed_data/sampledDatasetes/sample_len',nrow(sampled_conditions),'.rds'))
  print(paste0('Finished sample: ',i,'/',length(n)))
}

### Ratio splitting
cells <- c('A375','HT29')
ratios <- seq(0.1,1,0.1)
for (cell in cells){
  fullCell <- sigInfo %>% filter(cell_iname!=cell & cell_iname %in% cells) %>% 
    filter(!(sig_id %in% unique(c(pairedInfo$sig_id.x,pairedInfo$sig_id.y)))) %>% unique()
  cellInfo <- sigInfo %>% filter(cell_iname==cell) %>% 
    filter(!(sig_id %in% unique(c(pairedInfo$sig_id.x,pairedInfo$sig_id.y)))) %>% unique()
  paired <- sigInfo %>% filter(sig_id %in% unique(c(pairedInfo$sig_id.x,pairedInfo$sig_id.y)))
  for (ratio in ratios){
    n <- floor(ratio * nrow(fullCell))
    if (n>nrow(cellInfo)){
      n <- nrow(cellInfo)
    }
    cellInfo_percentage <- sample_n(cellInfo,n)
    data <- rbind(fullCell,paired,cellInfo_percentage)
    n1 <- nrow(data %>% filter(cell_iname!=cell))
    n2 <- nrow(data %>% filter(cell_iname==cell))
    r <- n2/n1
    saveRDS(data,paste0('preprocessed_data/sampledDatasetes/ratios',cell,'/sample_ratio',r,'.rds'))
  }
}

### Paired percentage
cells <- c('A375','HT29')
data <- sigInfo %>% filter(cell_iname %in% cells) %>% 
  filter(!(sig_id %in% unique(c(pairedInfo$sig_id.x,pairedInfo$sig_id.y)))) %>% unique()
paired <- sigInfo %>% filter(sig_id %in% unique(c(pairedInfo$sig_id.x,pairedInfo$sig_id.y)))
ns <- ceiling(seq(0.05,0.3,0.05) * (nrow(paired)+nrow(data)))
for (n in ns){
  data_paired <- sample_n(paired,n)
  data_all <- rbind(data,data_paired)
  saveRDS(data_all,paste0('preprocessed_data/sampledDatasetes/pairedPercs/sample_ratio_',n,'.rds'))
}

### For cell-line based similarity vs performance analysis we will use the existing pairs

### Create 5fold validation splits for all cases-----
library(caret)
folders <- c('A375_HT29','A375_PC3','HA1E_VCAP',
             'HT29_MCF7','HT29_PC3','MCF7_HA1E',
             'PC3_HA1E','MCF7_PC3','pairedPercs',
             'ratiosA375','ratiosHT29')

for (folder in folders){
  files <- list.files(paste0('preprocessed_data/sampledDatasetes/',folder),recursive = T,full.names = T)
  files <- as.data.frame(files)
  files <- files %>% 
    mutate(new_dirs=strsplit(files,'.rds')) %>% 
    unnest(new_dirs) %>% filter(new_dirs!='')
  
  for (i in 1:nrow(files)){
    new_dir <- files$new_dirs[i]
    data <- readRDS(files$files[i])
    if (folder %in% c('pairedPercs','ratiosA375','ratiosHT29')){
      cells <- c('A375','HT29')
    }else{
      cells <- strsplit(folder,'_')[[1]]
    }
    pairedInfo <- left_join(data %>% filter(cell_iname==cells[1]) %>% 
                              select(c('sig_id.x'='sig_id'),conditionId,cmap_name) %>% unique(), 
                            data %>% filter(cell_iname==cells[2]) %>% 
                              select(c('sig_id.y'='sig_id'),conditionId,cmap_name) %>% unique()) %>% 
      filter(!is.na(sig_id.x) & !is.na(sig_id.y))
    
    cellInfo_1 <- data %>% filter(cell_iname==cells[1]) %>% 
      filter(!(sig_id %in% unique(c(pairedInfo$sig_id.x,pairedInfo$sig_id.y))))
    cellInfo_2 <- data %>% filter(cell_iname==cells[2]) %>% 
      filter(!(sig_id %in% unique(c(pairedInfo$sig_id.x,pairedInfo$sig_id.y))))
    
    dir.create(new_dir,showWarnings = F)
    
    paired_folds <- createFolds(pairedInfo$conditionId, k = 5, list = TRUE, returnTrain = TRUE)
    cell1_folds <- createFolds(cellInfo_1$sig_id, k = 5, list = TRUE, returnTrain = TRUE)
    cell2_folds <- createFolds(cellInfo_2$sig_id, k = 5, list = TRUE, returnTrain = TRUE)
    
    for (j in 1:length(paired_folds)){
      data.table::fwrite(pairedInfo[paired_folds[[j]],],paste0(new_dir,'/train_paired_',j,'.csv'))
      data.table::fwrite(pairedInfo[-paired_folds[[j]],],paste0(new_dir,'/val_paired_',j,'.csv'))
      
      data.table::fwrite(cellInfo_1[cell1_folds[[j]],],paste0(new_dir,'/train_',cells[1],'_',j,'.csv'))
      data.table::fwrite(cellInfo_1[-cell1_folds[[j]],],paste0(new_dir,'/val_',cells[1],'_',j,'.csv'))
      
      data.table::fwrite(cellInfo_2[cell2_folds[[j]],],paste0(new_dir,'/train_',cells[2],'_',j,'.csv'))
      data.table::fwrite(cellInfo_2[-cell2_folds[[j]],],paste0(new_dir,'/val_',cells[2],'_',j,'.csv'))
    }
  }
             
}


### These next is for only 2 hand-picked cell-lines------
ind <- which(common$value==max(common$value))
cell1 <- as.character(common$Var1[ind])
cell2 <- as.character(common$Var2[ind])

#cell3 <- 'PC3'
#cell4 <- 'MCF7'

sigInfo <- sigInfo %>% filter(cell_iname==cell1 | cell_iname==cell2 | cell_iname==cell3 | cell_iname==cell4)

# Split the data of the two cell-lines into:
# paired: 1 dataframe with paired conditions
# unpaired: 2 datasets one for each celline

a375 <- sigInfo %>% filter(cell_iname=='A375') %>% select(conditionId,sig_id,cell_iname) %>% unique()
ht29 <- sigInfo %>% filter(cell_iname=='HT29') %>% select(conditionId,sig_id,cell_iname) %>% unique()
paired <- merge(a375,ht29,by="conditionId") %>% filter((!is.na(sig_id.x) & !is.na(sig_id.y))) %>% unique()
#write.csv(paired,'preprocessed_data/10fold_validation_spit/alldata/paired_pc3_ha1e.csv')

sigInfo <- sigInfo %>% select(sig_id,cell_iname,conditionId) %>% unique() %>%
  filter(!(sig_id %in% unique(c(paired$sig_id.x,paired$sig_id.y)))) %>% unique()
a375 <- sigInfo %>% filter(cell_iname=='A375') %>% filter(!(sig_id %in% paired$sig_id.x)) %>% unique()
ht29 <- sigInfo %>% filter(cell_iname=='HT29') %>% filter(!(sig_id %in% paired$sig_id.y)) %>% unique()
#write.csv(a375,'preprocessed_data/10fold_validation_spit/alldata/a375_unpaired.csv')
#write.csv(ht29,'preprocessed_data/10fold_validation_spit/alldata/ht29_unpaired.csv')

#write.csv(sigInfo,'preprocessed_data/conditions_HT29_A375.csv')

### Load gene expression data -----------------------------------------------------------------------------------

# Split sigs to run in parallel
#sigInfo <- sigInfo %>% filter(cell_iname=='PC3' | cell_iname=='HA1E')
sigIds <- unique(sigInfo$sig_id)
sigList <-  split(sigIds, ceiling(seq_along(sigIds)/ceiling(length(sigIds)/cores)))

# Parallelize parse_gctx
ds_path <- '../L1000_2021_11_23/level5_beta_trt_cp_n720216x12328.gctx'
parse_gctx_parallel <- function(path ,rid,cid){
  gctx_file <- parse_gctx(path ,rid = rid,cid = cid)
  return(gctx_file@mat)
}
cmap_gctx <- foreach(sigs = sigList) %dopar% {
  parse_gctx_parallel(ds_path ,rid = unique(as.character(geneInfo$gene_id)),cid = sigs)
}
cmap <-do.call(cbind,cmap_gctx)
#write.table(t(cmap), file = 'preprocessed_data/cmap_HT29_A375.tsv', quote=FALSE, sep = "\t", row.names = TRUE, col.names = NA)
write.csv(t(cmap), 'preprocessed_data/cmap_landmarks_HT29_A375.csv')

library(tidyverse)
library(cmapR)

library(doFuture)

# parallel set number of workers
cores <- 15
registerDoFuture()
plan(multiprocess,workers = cores)

########## The whole pre-processing analysis is in the L1000 folder of the new data ###############

### Load data and keep only well-inferred and landmark genes----------------------------------------------------
geneInfo <- read.delim('../../../L1000_2021_11_23/geneinfo_beta.txt')
geneInfo <-  geneInfo %>% filter(feature_space != "inferred")
# Keep only protein-coding genes
geneInfo <- geneInfo %>% filter(gene_type=="protein-coding")

# Load signature info and split data to high quality replicates and low quality replicates
sigInfo <- read.delim('../../../L1000_2021_11_23/siginfo_beta.txt')
sigInfo <- sigInfo %>% mutate(quality_replicates = ifelse(is_exemplar_sig==1 & qc_pass==1 & nsample>=3,1,0))
sigInfo <- sigInfo %>% filter(pert_type=='trt_cp')
sigInfo <- sigInfo %>% filter(quality_replicates==1)

# Filter based on TAS
sigInfo <- sigInfo %>% filter(tas>=0.3)

# Duplicate information
sigInfo <- sigInfo %>% mutate(duplIdentifier = paste0(cmap_name,"_",pert_idose,"_",pert_itime,"_",cell_iname))
sigInfo <- sigInfo %>% group_by(duplIdentifier) %>% mutate(dupl_counts = n()) %>% ungroup()

# Drug condition information
sigInfo <- sigInfo  %>% mutate(conditionId = paste0(cmap_name,"_",pert_idose,"_",pert_itime))
conditions <- sigInfo %>%  group_by(cell_iname) %>% summarise(conditions_per_cell = n_distinct(conditionId)) %>% ungroup()
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

ind <- which(common$value==max(common$value))
cell1 <- as.character(common$Var1[ind])
cell2 <- as.character(common$Var2[ind])

cell3 <- 'PC3'
cell4 <- 'MCF7'

sigInfo <- sigInfo %>% filter(cell_iname==cell1 | cell_iname==cell2 | cell_iname==cell3 | cell_iname==cell4)

a375 <- sigInfo %>% filter(cell_iname=='A375') %>% select(conditionId,sig_id,cell_iname) %>% unique()
ht29 <- sigInfo %>% filter(cell_iname=='HT29') %>% select(conditionId,sig_id,cell_iname) %>% unique()
pc3 <- sigInfo %>% filter(cell_iname==cell3) %>% select(conditionId,sig_id,cell_iname) %>% unique()
mcf7 <- sigInfo %>% filter(cell_iname==cell4) %>% select(conditionId,sig_id,cell_iname) %>% unique()
paired <- read.csv('preprocessed_data/10fold_validation_spit/alldata/paired_a375_ht29.csv',row.names = 'X') 
paired_with_pc3 <- merge(paired,pc3,by="conditionId") %>% filter((!is.na(sig_id.x) & !is.na(sig_id.y) & !is.na(sig_id))) %>% unique()

## Load all validation sets and add to them pc3 data
for (i in 1:10){
  valPaired <- read.csv(paste0('preprocessed_data/10fold_validation_spit/val_paired_',i-1,'.csv'),row.names = 'X')
  val_paired_with_pc3 <- left_join(valPaired,paired_with_pc3,
                                   by=c("conditionId","sig_id.x","sig_id.y","cell_iname.x","cell_iname.y")) %>% 
    filter(!is.na(sig_id))
  print(nrow(val_paired_with_pc3))
  write.csv(val_paired_with_pc3,paste0('preprocessed_data/10fold_validation_spit/add_pc3/val_paired_',i-1,'.csv'))
  paired_with_pc3 <- anti_join(paired_with_pc3,valPaired,
                               by=c("conditionId","sig_id.x","sig_id.y","cell_iname.x","cell_iname.y"))
}
paired <- read.csv('preprocessed_data/10fold_validation_spit/alldata/paired_a375_ht29.csv',row.names = 'X') 
paired <- merge(paired,pc3,by="conditionId") %>% filter((!is.na(sig_id.x) & !is.na(sig_id.y) & !is.na(sig_id))) %>% unique()


#MAKE PAIRED TRAIN SETS SEPERATELY FOR HT29-PC3 AND A375-PC3
potential_sig_ids <- unique(pc3$sig_id)
for (i in 1:10){
  val_paired_with_pc3 <- read.csv(paste0('preprocessed_data/10fold_validation_spit/add_pc3/val_paired_',i-1,'.csv'),row.names='X')
  valPaired <- read.csv(paste0('preprocessed_data/10fold_validation_spit/val_paired_',i-1,'.csv'),row.names = 'X')
  pc3_train <- pc3 %>% filter(!(sig_id %in% val_paired_with_pc3$sig_id))
  
  paired_a375 <- merge(a375,pc3_train,by="conditionId") %>% filter((!is.na(sig_id.x) & !is.na(sig_id.y))) %>% unique()
  paired_ht29 <- merge(ht29,pc3_train,by="conditionId") %>% filter((!is.na(sig_id.x) & !is.na(sig_id.y))) %>% unique()
  
  paired_a375 <- paired_a375 %>% filter(!(sig_id.x %in% valPaired$sig_id.x))
  paired_ht29 <- paired_ht29 %>% filter(!(sig_id.x %in% valPaired$sig_id.y))
  
  pc3_train <- pc3_train %>% select(sig_id,cell_iname,conditionId) %>% unique() %>%
    filter(!(sig_id %in% unique(c(paired_a375$sig_id.y,paired_ht29$sig_id.y)))) %>% unique()
  
  no <- round(nrow(pc3_train)*0.1)
  
  sig_ids <- potential_sig_ids[which(potential_sig_ids %in% pc3_train$sig_id)]
  sig_ids <- sample(sig_ids,no)
  potential_sig_ids <- potential_sig_ids[which(!(potential_sig_ids %in% sig_ids))]
  
  pc3_upaired_train <- pc3_train %>% filter(!(sig_id %in% sig_ids))
  pc3_upaired_val <- pc3_train %>% filter(sig_id %in% sig_ids)
  
  print(c(nrow(pc3_upaired_train),nrow(pc3_upaired_val)))
  
  write.csv(paired_a375,paste0('preprocessed_data/10fold_validation_spit/add_pc3/train_paired_a375_',i-1,'.csv'))
  write.csv(paired_ht29,paste0('preprocessed_data/10fold_validation_spit/add_pc3/train_paired_ht29_',i-1,'.csv'))
  
  write.csv(pc3_upaired_train,paste0('preprocessed_data/10fold_validation_spit/add_pc3/train_pc3_',i-1,'.csv'))
  write.csv(pc3_upaired_val,paste0('preprocessed_data/10fold_validation_spit/add_pc3/val_pc3_',i-1,'.csv'))
}

### Load gene expression data -----------------------------------------------------------------------------------

# Split sigs to run in parallel
sigIds <- unique(c(pc3$sig_id,mcf7$sig_id))
sigList <-  split(sigIds, ceiling(seq_along(sigIds)/ceiling(length(sigIds)/cores)))

# Parallelize parse_gctx
ds_path <- '../L1000_2021_11_23/level5_beta_trt_cp_n720216x12328.gctx'
parse_gctx_parallel <- function(path ,rid,cid){
  gctx_file <- parse_gctx(path ,rid = rid,cid = cid)
  return(gctx_file@mat)
}
geneInfo <- geneInfo %>% filter(feature_space=='landmark')
cmap_gctx <- foreach(sigs = sigList) %dopar% {
  parse_gctx_parallel(ds_path ,rid = unique(as.character(geneInfo$gene_id)),cid = sigs)
}
cmap <-do.call(cbind,cmap_gctx)
#write.table(t(cmap), file = 'preprocessed_data/cmap_HT29_A375.tsv', quote=FALSE, sep = "\t", row.names = TRUE, col.names = NA)
write.csv(t(cmap), 'preprocessed_data/cmap_PC3_MCF7_landmarks.csv')

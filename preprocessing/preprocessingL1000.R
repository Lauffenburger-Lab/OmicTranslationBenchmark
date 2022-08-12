library(tidyverse)
library(cmapR)

library(doFuture)

# parallel set number of workers
cores <- 15
registerDoFuture()
plan(multiprocess,workers = cores)

########## The whole pre-processing analysis is in the L1000 folder of the new data ###############

### Load data and keep only well-inferred and landmark genes----------------------------------------------------
geneInfo <- read.delim('../L1000_2021_11_23/geneinfo_beta.txt')
geneInfo <-  geneInfo %>% filter(feature_space != "inferred")
# Keep only protein-coding genes
geneInfo <- geneInfo %>% filter(gene_type=="protein-coding")

# Load signature info and split data to high quality replicates and low quality replicates
sigInfo <- read.delim('../L1000_2021_11_23/siginfo_beta.txt')
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
write.csv(sigInfo %>% select(sig_id,conditionId,cell_iname) %>% unique(),'all_conditions_tas03.csv')
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

#cell3 <- 'PC3'
#cell4 <- 'MCF7'

sigInfo <- sigInfo %>% filter(cell_iname==cell1 | cell_iname==cell2 | cell_iname==cell3 | cell_iname==cell4)

# Split the data of the two cell-lines into:
# paired: 1 dataframe with paired conditions
# unpaired: 2 datasets one for each celline

a375 <- sigInfo %>% filter(cell_iname=='A375') %>% select(conditionId,sig_id,cell_iname) %>% unique()
ht29 <- sigInfo %>% filter(cell_iname=='HT29') %>% select(conditionId,sig_id,cell_iname) %>% unique()
paired <- merge(a375,ht29,by="conditionId") %>% filter((!is.na(sig_id.x) & !is.na(sig_id.y))) %>% unique()
#write.csv(paired,'10fold_validation_spit/alldata/paired_pc3_ha1e.csv')

sigInfo <- sigInfo %>% select(sig_id,cell_iname,conditionId) %>% unique() %>%
  filter(!(sig_id %in% unique(c(paired$sig_id.x,paired$sig_id.y)))) %>% unique()
a375 <- sigInfo %>% filter(cell_iname=='A375') %>% filter(!(sig_id %in% paired$sig_id.x)) %>% unique()
ht29 <- sigInfo %>% filter(cell_iname=='HT29') %>% filter(!(sig_id %in% paired$sig_id.y)) %>% unique()
#write.csv(a375,'10fold_validation_spit/alldata/a375_unpaired.csv')
#write.csv(ht29,'10fold_validation_spit/alldata/ht29_unpaired.csv')

#write.csv(sigInfo,'conditions_HT29_A375.csv')

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
#write.table(t(cmap), file = 'cmap_HT29_A375.tsv', quote=FALSE, sep = "\t", row.names = TRUE, col.names = NA)
write.csv(t(cmap), 'cmap_landmarks_HT29_A375.csv')

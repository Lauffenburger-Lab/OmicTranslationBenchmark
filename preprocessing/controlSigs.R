library(tidyverse)
library(cmapR)
library(ggplot2)
library(ggpubr)
library(doFuture)
# parallel: set number of workers
cores <- 11
registerDoFuture()
plan(multisession,workers = cores)
library(doRNG)

########## The whole pre-processing analysis is in the L1000 folder of the new data ###############

### Load data and keep only well-inferred and landmark genes----------------------------------------------------
geneInfo <- read.delim('../../../L1000_2021_11_23/geneinfo_beta.txt')
geneInfo <-  geneInfo %>% filter(feature_space != "inferred")
# geneInfo <-  geneInfo %>% filter(feature_space == "landmark") # keep landmarks
# Keep only protein-coding genes
geneInfo <- geneInfo %>% filter(gene_type=="protein-coding")

# Load signature info and split data to high quality replicates and low quality replicates
sigInfo <- read.delim('../../../L1000_2021_11_23/siginfo_beta.txt')
sigInfo <- sigInfo %>% mutate(quality_replicates = ifelse(qc_pass==1 & nsample>=3,1,0)) # no exempler controls so I just remove that constraint
sigInfo <- sigInfo %>% filter(pert_type=='ctl_untrt')
sigInfo <- sigInfo %>% filter(quality_replicates==1)
sigInfo <- sigInfo %>% filter(pert_time<=24)

# Filter based on TAS
# sigInfo <- sigInfo %>% filter(tas<=0.3)

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

# Parallelize parse_gctx
ds_path <- '../../../L1000_2021_11_23/level5_beta_ctl_n58022x12328.gctx'
parse_gctx_parallel <- function(path ,rid,cid){
  gctx_file <- parse_gctx(path ,rid = rid,cid = cid)
  return(gctx_file@mat)
}
cmap_gctx <- foreach(sigs = sigList) %dopar% {
  parse_gctx_parallel(ds_path ,rid = unique(as.character(geneInfo$gene_id)),cid = sigs)
}
cmap <-do.call(cbind,cmap_gctx)
saveRDS(cmap,'preprocessed_data/cmap_all_controls_untreated_q1.rds')

sigInfo <- sigInfo %>% filter(sig_id %in% colnames(cmap))
conditions <- sigInfo %>%  group_by(cell_iname) %>% summarise(conditions_per_cell = n_distinct(conditionId)) %>% ungroup()

### Check correlation of TAS with duplicates correlation and noise-------------------------------------------------
# Calculate gsea distances 
library(doRNG)
# run distances
thresholds <- c(30,50,100,200,300,400,500,600,700,800,900,1000)
dist_all_dupls <- NULL
print('Begin calculating GSEA distance...')
### calculate distances
dist_all_dupls <- foreach(thres = thresholds) %dorng% {
  distance_scores(num_table = cmap ,threshold_count = thres,names = colnames(cmap))
}
distance <- do.call(cbind,dist_all_dupls)
distance <- array(distance,c(dim=dim(dist_all_dupls[[1]]),length(dist_all_dupls)))
mean_dist <- apply(distance, c(1,2), mean, na.rm = TRUE)
colnames(mean_dist) <- colnames(cmap)
rownames(mean_dist) <- colnames(cmap)
print('Begin saving GSEA distance...')
#saveRDS(mean_dist,'preprocessed_data/cmap_mean_dist_controls_untreated.rds')

### Convert matrix into data frame
# Keep only unique (non-self) pairs
mean_dist[lower.tri(mean_dist,diag = T)] <- -100
dist <- reshape2::melt(mean_dist)
dist <- dist %>% filter(value != -100)


# Merge meta-data info and distances values
dist <- left_join(dist,sigInfo,by = c("Var1"="sig_id"))
dist <- left_join(dist,sigInfo,by = c("Var2"="sig_id"))
dist <- dist %>% filter(!is.na(value))
dist <- dist %>% mutate(is_duplicate = (duplIdentifier.x==duplIdentifier.y))

# Thresholds of tas numbers to split dataset.
# TAS number is a metric given by the L1000 platform,
# which signifys the strength of the signal and
# the quality of the data. The higher it is
# the higher the quality of the data.

tas_thresholds_lower <- c(0.0,0.2,0.25,0.3,0.35,0.4,0.45,0.5)
tas_thresholds_upper <- c(0.99,0.25,0.3,0.35,0.4,0.45,0.5,0.55)

# Initialize empty lists to store plots
# and number of data after filtering 
# based on TAS number
plotList <- NULL
num_of_sigs <- NULL
p_vals <- NULL
for (i in 1:length(tas_thresholds_upper)){
  
  # Calculate number of data remaining
  num_of_sigs[i] <- nrow(sigInfo %>% filter(tas>=tas_thresholds_lower[i]))
  
  # Get the distances for these TAS numbers
  df_dist <- dist %>%  
    filter((tas.x>=tas_thresholds_lower[i] & tas.y>=tas_thresholds_lower[i])) %>%
    mutate(is_duplicate=ifelse(is_duplicate==T,
                               'Duplicate Signatures','Random Signatures')) %>%
    mutate(is_duplicate = factor(is_duplicate,
                                 levels = c('Random Signatures',
                                            'Duplicate Signatures')))
  # The distance metric ranges from 0-2
  # Normalize it between 0-1
  df_dist$value <- df_dist$value/2
  
  # Perform t-test
  random <- df_dist %>% filter(is_duplicate=='Random Signatures')
  dupl <- df_dist %>% filter(is_duplicate!='Random Signatures')
  p_vals[i] <- t.test(random$value,dupl$value,'greater')$p.value
  
  # Plot the distributions and store them forn now in the list using ggplot
  plotList[[i]] <- ggplot(df_dist,aes(x=value,color=is_duplicate,fill=is_duplicate)) +
    geom_density(alpha=0.2) +
    labs(col = 'Type',fill='Type',title="",x="GSEA Distance", y = "Density")+
    xlim(c(0,1))+#xlim(c(min(df_dist$value),max(df_dist$value)))+
    theme_classic() + theme(text = element_text(size=10))
}

# Save all subplots/distributions into one common figure
png(file="duplicate_vs_random_distribution_untreated.png",width=16,height=8,units = "in",res=300)

p <- ggarrange(plotlist=plotList,ncol=4,nrow=2,common.legend = TRUE,legend = 'right',
               labels = paste0(c('TAS>=0.15','TAS>=0.2','TAS>=0.25','TAS>=0.3','TAS>=0.35',
                                 'TAS>=0.4','TAS>=0.45','TAS>=0.5'),',Number of signatures:',num_of_sigs),
               font.label = list(size = 10, color = "black", face = "plain", family = NULL),
               hjust=-0.15)

annotate_figure(p, top = text_grob("Distributions of GSEA distances for different TAS thesholds", 
                                   color = "black",face = 'plain', size = 14))
dev.off()

# Save the line-plot of how the number of available data
# changes with varying TAS thresholds

#Dummy dataframe to plot
df = data.frame(tas_thresholds_lower,num_of_sigs)

# Plot the decrease of aavailable data
# as we become more strict in their quality
png(file="numberOfSignatures_vs_TASthresholf.png",width=9,height=6,units = "in",res=300)
ggplot(df,aes(x=tas_thresholds_lower,y=num_of_sigs)) +
  labs(x="TAS number threshold", y = "Number of data points")+
  geom_point() +geom_line(linetype='dashed')
dev.off()

# Adjust p-values with Bonferroni and plot them
p.adj <- p.adjust(p_vals,"bonferroni")
hist(p.adj)

## Clean save conditions ------
cmap <- readRDS('preprocessed_data/cmap_all_controls_untreated_q1.rds')
write.csv(t(cmap), 'preprocessed_data/cmap_untreated_untreated_q1.csv')

# Drug condition information
# sigInfo <- sigInfo  %>% filter(tas<=0.3)
sigInfo <- sigInfo %>% filter(pert_time<=24)
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

# For now get A375 and HT29
sigInfo <- sigInfo %>% filter(cell_iname=='A375' | cell_iname=='HT29')

# Split the data of the two cell-lines into:
# paired: 1 dataframe with paired conditions
# unpaired: 2 datasets one for each celline

a375 <- sigInfo %>% filter(cell_iname=='A375') %>% select(conditionId,sig_id,cell_iname) %>% unique()
ht29 <- sigInfo %>% filter(cell_iname=='HT29') %>% select(conditionId,sig_id,cell_iname) %>% unique()
paired <- merge(a375,ht29,by="conditionId") %>% filter((!is.na(sig_id.x) & !is.na(sig_id.y))) %>% unique()
write.csv(paired,'preprocessed_data/10fold_validation_spit/alldata/paired_untreated_a375_ht29.csv')

sigInfo <- sigInfo %>% select(sig_id,cell_iname,conditionId) %>% unique() %>%
  filter(!(sig_id %in% unique(c(paired$sig_id.x,paired$sig_id.y)))) %>% unique()
a375 <- sigInfo %>% filter(cell_iname=='A375') %>% filter(!(sig_id %in% paired$sig_id.x)) %>% unique()
ht29 <- sigInfo %>% filter(cell_iname=='HT29') %>% filter(!(sig_id %in% paired$sig_id.y)) %>% unique()
write.csv(a375,'preprocessed_data/10fold_validation_spit/alldata/pc3_unpaired_untreated.csv')
write.csv(ht29,'preprocessed_data/10fold_validation_spit/alldata/ha1e_unpaired_untreated.csv') #zero

#write.csv(sigInfo,'preprocessed_data/conditions_HA1E_PC3.csv')

# Check correlations in baseline-----
cmap <- readRDS('preprocessed_data/cmap_all_controls_untreated_q1.rds')
paired <- read.csv('preprocessed_data/10fold_validation_spit/alldata/paired_untreated_a375_ht29.csv') %>% column_to_rownames('X')

a375_untreated <- cmap[,paired$sig_id.x]
ht29_untreated <- cmap[,paired$sig_id.y]
corr_spear <- mapply(cor, as.data.frame(a375_untreated),as.data.frame(ht29_untreated),method='spearman')
hist(corr_spear)
print(mean(corr_spear))

a375_sign <- a375_untreated
a375_sign[which(abs(a375_sign)<=0.0001)] <- 0
a375_sign[which(a375_sign<0)] <- -1
a375_sign[which(a375_sign>0)] <- 1
ht29_sign <- ht29_untreated
ht29_sign[which(abs(ht29_sign)<=0.0001)] <- 0
ht29_sign[which(ht29_sign<0)] <- -1
ht29_sign[which(ht29_sign>0)] <- 1
Accuracy <- function(x,y){
  trues <- sum(x==y)
  return(trues/length(x))
}
acc <- mapply(Accuracy,as.data.frame(a375_sign),as.data.frame(ht29_sign))
hist(acc)
print(mean(acc))

a375_untreated <- t(a375_untreated)
ht29_untreated <- t(ht29_untreated)
corr_pear <- mapply(cor, as.data.frame(a375_untreated),as.data.frame(ht29_untreated))
hist(corr_pear)
print(mean(corr_pear))

## Load ccle and keep only genes of L1000-----
geneInfo <- read.delim('../../../L1000_2021_11_23/geneinfo_beta.txt')
geneInfo <-  geneInfo %>% filter(feature_space != "inferred")
# geneInfo <-  geneInfo %>% filter(feature_space == "landmark") # keep only landmark
# Keep only protein-coding genes
geneInfo <- geneInfo %>% filter(gene_type=="protein-coding")

# Load signature info and split data to high quality replicates and low quality replicates
sigInfo <- read.delim('../../../L1000_2021_11_23/siginfo_beta.txt')
sigInfo <- sigInfo %>% mutate(quality_replicates = ifelse(qc_pass==1 & nsample>=3,1,0)) # no exempler controls so I just remove that constraint
sigInfo <- sigInfo %>% filter(pert_type=='ctl_untrt')
sigInfo <- sigInfo %>% filter(quality_replicates==1)
sigInfo <- sigInfo %>% filter(pert_time<=24)

# Filter based on TAS
# sigInfo <- sigInfo %>% filter(tas<=0.3)

# Duplicate information
sigInfo <- sigInfo %>% mutate(duplIdentifier = paste0(cmap_name,"_",pert_idose,"_",pert_itime,"_",cell_iname))
sigInfo <- sigInfo %>% group_by(duplIdentifier) %>% mutate(dupl_counts = n()) %>% ungroup()

# Drug condition information
sigInfo <- sigInfo  %>% mutate(conditionId = paste0(cmap_name,"_",pert_idose,"_",pert_itime))
conditions <- sigInfo %>%  group_by(cell_iname) %>% summarise(conditions_per_cell = n_distinct(conditionId)) %>% ungroup()

ccle <- t(data.table::fread('../data/CCLE/CCLE_expression.csv') %>% column_to_rownames('V1'))
ccle <- as.data.frame(ccle) %>% rownames_to_column('V1') %>% separate(V1,c('gene_id','useless'),sep=" ") %>%
  dplyr::select(-useless) %>% column_to_rownames('gene_id')
ccle <- as.data.frame(t(ccle)) %>% rownames_to_column('DepMap_ID')
sample_info <- data.table::fread('../data/CCLE/sample_info.csv') %>% dplyr::select(DepMap_ID,stripped_cell_line_name) %>%
  unique()
ccle <- left_join(ccle,sample_info) %>% dplyr::select(-DepMap_ID) %>%
  column_to_rownames('stripped_cell_line_name')
ccle <- ccle[which(rownames(ccle) %in% unique(sigInfo$cell_iname)),]
ind <- ncol(ccle)

genes_missing <- geneInfo$gene_symbol[which(!(geneInfo$gene_symbol %in% colnames(ccle)))]

ccle <- cbind(ccle,data.frame(matrix(0,nrow(ccle),length(genes_missing))))
colnames(ccle)[(ind+1):ncol(ccle)] <- genes_missing
print(all(rownames(cmap)==geneInfo$gene_id))
df_genes <- left_join(data.frame(gene_symbol=rownames(cmap)),
                      geneInfo %>% select(gene_symbol,gene_id) %>% unique())
print(all(rownames(cmap)==geneInfo$gene_id))
rownames(cmap) <- geneInfo$gene_symbol  
print(all(rownames(cmap)==geneInfo$gene_symbol))

ccle <- ccle %>% select(rownames(cmap))
ccle <- t(ccle)
write.csv(ccle,'preprocessed_data/ccle_l1000genes.csv')

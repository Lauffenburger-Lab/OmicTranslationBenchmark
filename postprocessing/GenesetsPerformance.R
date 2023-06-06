library(tidyverse)
library(ggplot2)
library(ggsignif)
library(ggpubr)

## Load all cmap data---------
genespace <- 'best_infered plus landmarks'
cmap <- data.table::fread('../preprocessing/preprocessed_data/cmap_all_genes_q1_tas03.csv',header = T) %>% column_to_rownames('V1')
cmap <- t(cmap)

geneInfo <- read.delim('../../../L1000_2021_11_23/geneinfo_beta.txt')
if (genespace=='landmark'){
  geneInfo <-  geneInfo %>% filter(feature_space == "landmark")
}else{
  geneInfo <-  geneInfo %>% filter(feature_space != "inferred")
}

# Keep only protein-coding genes
geneInfo <- geneInfo %>% filter(gene_type=="protein-coding")

#Load signature info and split data to high quality replicates and low quality replicates
sigInfo <- read.delim('../../../L1000_2021_11_23/siginfo_beta.txt')
sigInfo <- sigInfo %>% 
  mutate(quality_replicates = ifelse(is_exemplar_sig==1 & qc_pass==1 & nsample>=3,1,0))
sigInfo <- sigInfo %>% filter(pert_type=='trt_cp')
sigInfo <- sigInfo %>% filter(quality_replicates==1)
sigInfo <- sigInfo %>% filter(tas>=0.3)
sigInfo <- sigInfo %>% 
  mutate(duplIdentifier = paste0(cmap_name,"_",pert_idose,"_",pert_itime,"_",cell_iname))
sigInfo <- sigInfo %>% group_by(duplIdentifier) %>%
  mutate(dupl_counts = n()) %>% ungroup()
sigInfo <- sigInfo  %>% mutate(conditionId = paste0(cmap_name,"_",pert_idose,"_",pert_itime))

## Perfome Genesets perfromance analysis for a specific pair of cell-lines---------
cells <- c('A375','HT29')
sigInfo <-  sigInfo %>% filter(cell_iname %in% cells)
sig1 <- c(as.matrix(sigInfo %>% filter(cell_iname==cells[1]) %>% dplyr::select(sig_id)))
sig2 <- c(as.matrix(sigInfo %>% filter(cell_iname==cells[2]) %>% dplyr::select(sig_id)))

# paired conditions
all_paired <- data.table::fread('../preprocessing/preprocessed_data/10fold_validation_spit/alldata/paired_a375_ht29.csv',
                                header = T) %>% column_to_rownames('V1')

cmap <- cmap[as.character(geneInfo$gene_id),sigInfo$sig_id]

# genesets_results <- fastenrichment(sigInfo$sig_id,
#                                    geneInfo$gene_id,
#                                    cmap,
#                                    enrichment_space = c("go_bp","kegg","msig_db_h",
#                                                         "msig_db_c1","msig_db_c2","msig_db_c3",
#                                                         "msig_db_c4","msig_db_c5","msig_db_c6","msig_db_c7"),
#                                    pval_adjustment=F,
#                                    n_permutations=1000)
# saveRDS(genesets_results,'../results/all_gensetsEnrich_results_landmarks_drugs_pc3_ha1e.rds')


genesets_results <- readRDS('../results/all_gensetsEnrich_results_allgenes_drugs_a375_ht29.rds')

# Choose Genesets level to perform analysis
# "NES KEGG"
# "NES GO BP"

#sets <- c("NES KEGG","NES GO BP","NES MSIG Hallmark")
sets <- names(genesets_results$NES)[which(names(genesets_results$NES)!="NES TF")]

#set <-  "NES GO BP"
# base_geneset_corr <- NULL
# for (set in names(genesets_results$NES)[which(names(genesets_results$NES)!="NES TF")]){
#   NES <- genesets_results$NES[[set]]
#   
#   nes1 <- t(NES[,all_paired$sig_id.x])
#   nes2 <- t(NES[,all_paired$sig_id.y])
#   
#   base_geneset_corr[[set]] <- as.numeric(cor.test(c(nes1),c(nes2))$estimate)
#   
# }

#geneset_train_corr <- NULL
base_geneset_corr <- NULL
model_geneset_corr_cell2_to_cell1 <- NULL
model_geneset_corr_cell1_to_cell2 <- NULL
sets_to_enr <- c("go_bp","kegg",'msig_db_h',"msig_db_c1","msig_db_c2","msig_db_c3",
                 "msig_db_c4","msig_db_c5","msig_db_c6","msig_db_c7")
k <- 1
for (set in sets){
  geneset_val_corr_cell2_to_cell1 <- NULL
  geneset_val_corr_cell1_to_cell2 <- NULL
  base_geneset_val_corr <- NULL
  for (i in 0:9){
    # validation info
    # valPaired = data.table::fread(paste0('../preprocessing/preprocessed_data/10fold_validation_spit/val_paired_',
    #                                      tolower(cells[1]),'_',
    #                                      tolower(cells[2]),'_',i,'.csv'),header = T) %>% column_to_rownames('V1')
    valPaired = data.table::fread(paste0('../preprocessing/preprocessed_data/10fold_validation_spit/val_paired_',
                                         i,'.csv'),header = T) %>% column_to_rownames('V1')
    valInfo <- rbind(valPaired %>% dplyr::select(c('sig_id'='sig_id.x'),c('cell_iname'='cell_iname.x'),conditionId),
                     valPaired %>% dplyr::select(c('sig_id'='sig_id.y'),c('cell_iname'='cell_iname.y'),conditionId))
    valInfo <- valInfo %>% unique()
    valInfo <- valInfo %>% dplyr::select(sig_id,conditionId,cell_iname)
    
    # Load predictions # RUN SATORI TO GET PREDICTIONS
    Trans_val <- distinct(rbind(data.table::fread(paste0('../results/MI_results/embs/CPA_approach_10K/validation/valTrans_',i,'_',tolower(cells[1]),'.csv'),header = T),
                       data.table::fread(paste0('../results/MI_results/embs/CPA_approach_10K/validation/valTrans_',i,'_',tolower(cells[2]),'.csv'),header = T)))
    sigs <- Trans_val$V1
    Trans_val <- t(Trans_val %>% dplyr::select(-V1))
    rownames(Trans_val) <- rownames(cmap)
    #colnames(Trans_val) <- sigs
    #Trans_val <- Trans_val[,valInfo$sig_id]
    
    Trans_gsea <- fastenrichment(sigs,
                                 geneInfo$gene_id,
                                 Trans_val,
                                 enrichment_space = sets_to_enr[k],
                                 order_columns = F,
                                 pval_adjustment=F,
                                 n_permutations=100)
    
    NEShat <- Trans_gsea$NES[[set]]
    NES <- genesets_results$NES[[set]]
    
    if (nrow(NES)>nrow(NEShat)){
      NEShat <- NEShat[which(rownames(NEShat) %in% rownames(NES)),]
      NES <- NES[rownames(NEShat),]
      warning(paste0(set,' in Fold ',i,' has now fewer genesets infered'))
    }
    val_sig1 <- valPaired$sig_id.x
    val_sig2 <- valPaired$sig_id.y
    
    # Baseline correlation
    nes1 <- t(NES[,val_sig1])
    nes2 <- t(NES[,val_sig2])
    base_geneset_val_corr[i+1] <- as.numeric(cor.test(c(nes1),c(nes2))$estimate)
    
    # Predicted geneset enrichment
    nes1_hat <- t(NEShat[,which(sigs==val_sig1)])
    nes2_hat <- t(NEShat[,which(sigs==val_sig2)])
    geneset_val_corr_cell2_to_cell1[i+1] <- as.numeric(cor.test(c(nes1_hat),c(nes1))$estimate)
    geneset_val_corr_cell1_to_cell2[i+1] <- as.numeric(cor.test(c(nes2_hat),c(nes2))$estimate)
    
  }
  base_geneset_corr[[set]] <- base_geneset_val_corr
  model_geneset_corr_cell2_to_cell1[[set]] <- geneset_val_corr_cell2_to_cell1
  model_geneset_corr_cell1_to_cell2[[set]] <- geneset_val_corr_cell1_to_cell2
  k <- k+1
  print(paste0('Finished ',set))
}

# saveRDS(base_geneset_corr,'../results/base_genesets_correlation.rds')
# saveRDS(model_geneset_corr_cell1_to_cell2,'../results/model_A375_to_HT29_genesets_correlation.rds')
# saveRDS(model_geneset_corr_cell2_to_cell1,'../results/model_HT29_to_A375_genesets_correlation.rds')

base_geneset_corr <- readRDS('../results/base_genesets_correlation.rds')
model_geneset_corr_cell1_to_cell2 <- readRDS('../results/model_A375_to_HT29_genesets_correlation.rds')
model_geneset_corr_cell2_to_cell1 <- readRDS('../results/model_HT29_to_A375_genesets_correlation.rds')

### Visualize results ------
baseline_res <- do.call(cbind.data.frame, base_geneset_corr)
baseline_res <- baseline_res %>% rownames_to_column('fold') %>% gather('geneset','cor',-fold)
baseline_res <- baseline_res %>% mutate(model = 'direct translation')

model_res <- do.call(cbind.data.frame, model_geneset_corr_cell1_to_cell2)
model_res <- model_res %>% rownames_to_column('fold') %>% gather('geneset','cor',-fold)
model_res <- model_res %>% mutate(model = 'model')

results <- rbind(baseline_res,model_res)
results <- results %>% mutate(geneset=strsplit(geneset,'NES')) %>% unnest(geneset) %>% filter(geneset!='') %>% unique()

p <- ggboxplot(results, x = "geneset", y = 'cor',color = "model",add='jitter')+
  ggtitle(paste0('Genestet performance for translating from ',cells[1],' to ',cells[2]))+ ylab('pearson`s r')+ ylim(c(0,0.85))+
  theme_minimal(base_family = "Arial",base_size = 16)+
  theme(plot.title = element_text(hjust = 0.5,size=16),
        axis.text.x = element_text(size=14))
p <- p + stat_compare_means(aes(group = model),label='p.signif')
print(p)
png(paste0('../figures/genesets_model_vs_direct_',tolower(cells[1]),'_',tolower(cells[2]),'.png')
    ,width=16,height=8,units = "in",res = 600)
print(p)
dev.off()
# setEPS()
# postscript(paste0('../figures/genesets_model_vs_direct_',tolower(cells[1]),'_',tolower(cells[2]),'.eps'))
# print(p)
# dev.off()


model_res <- do.call(cbind.data.frame, model_geneset_corr_cell2_to_cell1)
model_res <- model_res %>% rownames_to_column('fold') %>% gather('geneset','cor',-fold)
model_res <- model_res %>% mutate(model = 'model')

results <- rbind(baseline_res,model_res)
results <- results %>% mutate(geneset=strsplit(geneset,'NES')) %>% unnest(geneset) %>% filter(geneset!='') %>% unique()

p <- ggboxplot(results, x = "geneset", y = 'cor',color = "model",add='jitter')+
  ggtitle(paste0('Genestet performance for translating from ',cells[2],' to ',cells[1]))+ ylab('pearson`s r')+ ylim(c(0,0.85))+
  theme_minimal(base_family = "Arial",base_size = 16)+
  theme(plot.title = element_text(hjust = 0.5,size=16),
        axis.text.x = element_text(size=14))
p <- p + stat_compare_means(aes(group = model),label='p.signif')
print(p)
png(paste0('../figures/genesets_model_vs_direct_',tolower(cells[2]),'_',tolower(cells[1]),'.png')
    ,width=16,height=8,units = "in",res = 600)
print(p)
dev.off()

### Compare with pearson correlation of gene expression-----
k <- 1
model_gex_corr_cell2_to_cell1 <- NULL
model_gex_corr_cell1_to_cell2 <- NULL
for (set in sets){
  gex_corr_cell1_cell2 <- NULL
  gex_corr_cell2_cell1 <- NULL
  for (i in 0:9){
    # validation info
    # valPaired = data.table::fread(paste0('../preprocessing/preprocessed_data/10fold_validation_spit/val_paired_',
    #                                      tolower(cells[1]),'_',
    #                                      tolower(cells[2]),'_',i,'.csv'),header = T) %>% column_to_rownames('V1')
    valPaired = data.table::fread(paste0('../preprocessing/preprocessed_data/10fold_validation_spit/val_paired_',
                                         i,'.csv'),header = T) %>% column_to_rownames('V1')
    valInfo <- rbind(valPaired %>% dplyr::select(c('sig_id'='sig_id.x'),c('cell_iname'='cell_iname.x'),conditionId),
                     valPaired %>% dplyr::select(c('sig_id'='sig_id.y'),c('cell_iname'='cell_iname.y'),conditionId))
    valInfo <- valInfo %>% unique()
    valInfo <- valInfo %>% dplyr::select(sig_id,conditionId,cell_iname)
    
    # Load predictions # RUN SATORI TO GET PREDICTIONS
    Trans_val <- distinct(rbind(data.table::fread(paste0('../results/MI_results/embs/CPA_approach_10K/validation/valTrans_',i,'_',tolower(cells[1]),'.csv'),header = T),
                                data.table::fread(paste0('../results/MI_results/embs/CPA_approach_10K/validation/valTrans_',i,'_',tolower(cells[2]),'.csv'),header = T)))
    sigs <- Trans_val$V1
    Trans_val <- t(Trans_val %>% dplyr::select(-V1))
    rownames(Trans_val) <- rownames(cmap)
    #colnames(Trans_val) <- sigs
    #Trans_val <- Trans_val[,valInfo$sig_id]
    
    val_sig1 <- valPaired$sig_id.x
    val_sig2 <- valPaired$sig_id.y
    
    #Base ground truth
    camp_tmp <- cmap[,which(colnames(cmap) %in% c(val_sig1,val_sig2))]
    cmap1 <- camp_tmp[,val_sig1]
    cmap2 <- camp_tmp[,val_sig2]
    
    # Predicted geneset enrichment
    cmap1_hat <- Trans_val[,which(sigs==val_sig1)]
    cmap2_hat <- Trans_val[,which(sigs==val_sig2)]
    gex_corr_cell2_cell1[i+1] <- as.numeric(cor.test(c(cmap1_hat),c(cmap1))$estimate)
    gex_corr_cell1_cell2[i+1] <- as.numeric(cor.test(c(cmap2_hat),c(cmap2))$estimate)
    
  }
  model_gex_corr_cell2_to_cell1[[set]] <- gex_corr_cell2_cell1
  model_gex_corr_cell1_to_cell2[[set]] <- gex_corr_cell1_cell2
  k <- k+1
  print(paste0('Finished ',set))
}

# Visualize cell2 to cell1
model_gsea <- do.call(cbind.data.frame, model_geneset_corr_cell2_to_cell1)
model_gsea <- model_gsea %>% rownames_to_column('fold') %>% gather('geneset','cor',-fold)
model_gsea <- model_gsea %>% mutate(model = 'genesets performance')
model_gex <- do.call(cbind.data.frame, model_gex_corr_cell2_to_cell1)
model_gex <- model_gex %>% rownames_to_column('fold') %>% gather('geneset','cor',-fold)
model_gex <- model_gex %>% mutate(geneset = 'Genes Expr.') %>% unique()
model_gex <- model_gex %>% mutate(model = 'genes performance')

results <- rbind(model_gsea,model_gex)
results <- results %>% mutate(geneset=strsplit(geneset,'NES')) %>% unnest(geneset) %>% filter(geneset!='') %>% unique()
results$geneset <- factor(results$geneset,levels = c('Genes Expr.',str_split_fixed(sets,'NES',n=2)[,2]))
results <- results %>% mutate(type=model)
  
comparisons <- NULL
p.values <- NULL
k <- 1
for (set in str_split_fixed(sets,'NES',n=2)[,2]){
  comparisons[[k]] <- c('Genes Expr.',set)
  p.values[k] <- wilcox.test(as.matrix(results %>% filter(geneset==comparisons[[k]][1]) %>% dplyr::select('cor')),
                             as.matrix(results %>% filter(geneset==comparisons[[k]][2]) %>% dplyr::select('cor')))$p.value
  k <- k+1
}
p <- ggboxplot(results, x = "geneset", y = 'cor',color='type',add='jitter')+
  ggtitle(paste0('Genestet performance for translating from ',cells[2],' to ',cells[1]))+ ylab('pearson`s r')+
  theme_minimal(base_family = "Arial",base_size = 16)+
  theme(plot.title = element_text(hjust = 0.5,size=16),legend.position='bottom')
p <- p + stat_compare_means(comparisons=comparisons[which(p.values<0.05)],method = 'wilcox.test',label='p.signif')
print(p)

png(paste0('../figures/genesets_vs_genes_',tolower(cells[2]),'_',tolower(cells[1]),'.png')
    ,width=12,height=8,units = "in",res = 600)
print(p)
dev.off()

# Visualize cell1 to cell2
model_gsea <- do.call(cbind.data.frame, model_geneset_corr_cell1_to_cell2)
model_gsea <- model_gsea %>% rownames_to_column('fold') %>% gather('geneset','cor',-fold)
model_gsea <- model_gsea %>% mutate(model = 'genesets performance')
model_gex <- do.call(cbind.data.frame, model_gex_corr_cell1_to_cell2)
model_gex <- model_gex %>% rownames_to_column('fold') %>% gather('geneset','cor',-fold)
model_gex <- model_gex %>% mutate(geneset = 'Genes Expr.') %>% unique()
model_gex <- model_gex %>% mutate(model = 'genes performance')

results <- rbind(model_gsea,model_gex)
results <- results %>% mutate(geneset=strsplit(geneset,'NES')) %>% unnest(geneset) %>% filter(geneset!='') %>% unique()
results$geneset <- factor(results$geneset,levels = c('Genes Expr.',str_split_fixed(sets,'NES',n=2)[,2]))
results <- results %>% mutate(type=model)

comparisons <- NULL
p.values <- NULL
k <- 1
for (set in str_split_fixed(sets,'NES',n=2)[,2]){
  comparisons[[k]] <- c('Genes Expr.',set)
  p.values[k] <- wilcox.test(as.matrix(results %>% filter(geneset==comparisons[[k]][1]) %>% dplyr::select('cor')),
                             as.matrix(results %>% filter(geneset==comparisons[[k]][2]) %>% dplyr::select('cor')))$p.value
  k <- k+1
}
p <- ggboxplot(results, x = "geneset", y = 'cor',color='type',add='jitter')+
  ggtitle(paste0('Genestet performance for translating from ',cells[1],' to ',cells[2]))+ ylab('pearson`s r')+
  theme_minimal(base_family = "Arial",base_size = 16)+
  theme(plot.title = element_text(hjust = 0.5,size=16),legend.position='bottom')
p <- p + stat_compare_means(comparisons=comparisons[which(p.values<0.05)],method = 'wilcox.test',label='p.signif')
print(p)

png(paste0('../figures/genesets_vs_genes_',tolower(cells[1]),'_',tolower(cells[2]),'.png')
    ,width=12,height=8,units = "in",res = 600)
print(p)
dev.off()

### TFs dorothea performance--------------
library(dorothea)
minNrOfGenes = 5
dorotheaData = read.table('../../../Artificial-Signaling-Network/TF activities/annotation/dorothea.tsv', sep = "\t", header=TRUE)
confidenceFilter = is.element(dorotheaData$confidence, c('A', 'B'))
dorotheaData = dorotheaData[confidenceFilter,]

print(all(rownames(cmap)==geneInfo$gene_id))
rownames(cmap) <- geneInfo$gene_symbol
# Estimate TF activities of ground truth
settings = list(verbose = TRUE, minsize = minNrOfGenes)
TF_activities = run_viper(cmap, dorotheaData, options =  settings)

tfs_corr_cell1_cell2 <- NULL
tfs_corr_cell2_cell1 <- NULL
base_tfs_cor <- NULL
gex_corr_cell2_cell1 <- NULL
gex_corr_cell1_cell2 <- NULL
for (i in 0:9){
  # validation info
  # valPaired = data.table::fread(paste0('../preprocessing/preprocessed_data/10fold_validation_spit/val_paired_',
  #                                      tolower(cells[1]),'_',
  #                                      tolower(cells[2]),'_',i,'.csv'),header = T) %>% column_to_rownames('V1')
  valPaired = data.table::fread(paste0('../preprocessing/preprocessed_data/10fold_validation_spit/val_paired_',
                                       i,'.csv'),header = T) %>% column_to_rownames('V1')
  valInfo <- rbind(valPaired %>% dplyr::select(c('sig_id'='sig_id.x'),c('cell_iname'='cell_iname.x'),conditionId),
                   valPaired %>% dplyr::select(c('sig_id'='sig_id.y'),c('cell_iname'='cell_iname.y'),conditionId))
  valInfo <- valInfo %>% unique()
  valInfo <- valInfo %>% dplyr::select(sig_id,conditionId,cell_iname)
  
  # Load predictions # RUN SATORI TO GET PREDICTIONS
  Trans_val <- distinct(rbind(data.table::fread(paste0('../results/MI_results/embs/CPA_approach_10K/validation/valTrans_',i,'_',tolower(cells[1]),'.csv'),header = T),
                              data.table::fread(paste0('../results/MI_results/embs/CPA_approach_10K/validation/valTrans_',i,'_',tolower(cells[2]),'.csv'),header = T)))
  sigs <- Trans_val$V1
  Trans_val <- t(Trans_val %>% dplyr::select(-V1))
  rownames(Trans_val) <- rownames(cmap)
  TF_activities_hat = run_viper(Trans_val, dorotheaData, options =  settings)
  
  val_sig1 <- valPaired$sig_id.x
  val_sig2 <- valPaired$sig_id.y
  
  ## Genes analysis
  
  #Base ground truth genes
  camp_tmp <- cmap[,which(colnames(cmap) %in% c(val_sig1,val_sig2))]
  cmap1 <- camp_tmp[,val_sig1]
  cmap2 <- camp_tmp[,val_sig2]
  
  # Predicted genes
  cmap1_hat <- Trans_val[,which(sigs==val_sig1)]
  cmap2_hat <- Trans_val[,which(sigs==val_sig2)]
  gex_corr_cell2_cell1[i+1] <- as.numeric(cor.test(c(cmap1_hat),c(cmap1))$estimate)
  gex_corr_cell1_cell2[i+1] <- as.numeric(cor.test(c(cmap2_hat),c(cmap2))$estimate)
  
  ## TFs analysis
  
  #Base ground truth TFS
  tfs_tmp <- TF_activities[,which(colnames(TF_activities) %in% c(val_sig1,val_sig2))]
  cmap1 <- tfs_tmp[,val_sig1]
  cmap2 <- tfs_tmp[,val_sig2]
  base_tfs_cor[i+1] <- as.numeric(cor.test(c(cmap1),c(cmap2))$estimate)
  
  # Predicted tfs enrichment
  cmap1_hat <- TF_activities_hat[,which(sigs==val_sig1)]
  cmap2_hat <- TF_activities_hat[,which(sigs==val_sig2)]
  tfs_corr_cell2_cell1[i+1] <- as.numeric(cor.test(c(cmap1_hat),c(cmap1))$estimate)
  tfs_corr_cell1_cell2[i+1] <- as.numeric(cor.test(c(cmap2_hat),c(cmap2))$estimate)
  
  print(paste0('Finished ',i))
  
}

model_tfs <- data.frame('HT29 to A375'=tfs_corr_cell2_cell1,
                      'A375 to HT29'=tfs_corr_cell1_cell2,
                      'direct translation'=base_tfs_cor)
model_tfs <- model_tfs %>% rownames_to_column('fold') %>% gather('translation','cor',-fold)
model_tfs <- model_tfs %>% mutate(translation=str_replace(translation,'[.]',' '))
model_tfs <- model_tfs %>% mutate(translation=str_replace(translation,'[.]',' '))
model_tfs <- model_tfs %>% mutate(level='TFs')

model_genes <- data.frame('HT29 to A375'=gex_corr_cell2_cell1,
                        'A375 to HT29'=gex_corr_cell1_cell2)
model_genes <- model_genes %>% rownames_to_column('fold') %>% gather('translation','cor',-fold)
model_genes <- model_genes %>% mutate(translation=str_replace(translation,'[.]',' '))
model_genes <- model_genes %>% mutate(translation=str_replace(translation,'[.]',' '))
model_genes <- model_genes %>% mutate(level='Genes')

results <- rbind(model_tfs,model_genes)

p1 <- ggboxplot(results %>% filter(translation!='direct translation'),
               x = "translation", y = 'cor',color='level',add='jitter',
               add.params = list(size = 2),size = 1)+
  ggtitle('')+ ylab('pearson`s r')+ ylim(c(0,0.85))+
  theme_minimal(base_family = "Arial",base_size = 42)+
  theme(text = element_text("Arial",size = 42),
        axis.title = element_text("Arial",size = 36,face = "bold"),
        axis.title.x = element_blank(),
        axis.text = element_text("Arial",size = 40,face = "bold"),
        axis.text.x = element_text("Arial",angle = 0,size = 26,face = "bold"),
        plot.title = element_text(hjust = 1.15,size=40),legend.position='bottom')
p1 <- p1 + stat_compare_means(aes(group=level),method = 'wilcox.test',label='p.signif',size=10)
print(p1)
 
# png('../figures/tfs_vs_genes_translating_comparison.png',width=14,height=15.16,units = "in",res = 600)
# print(p1)
# dev.off()
# setEPS()
# postscript('../figures/tfs_vs_genes_translating_comparison.ps',width=14,height=15.16)
# print(p1)
# dev.off()

comparisons <- list(c('direct translation','A375 to HT29'),c('direct translation','HT29 to A375'))
p2 <- ggboxplot(results %>% filter(level!='Genes'),
               x = "translation", y = 'cor',color='translation',add='jitter',
               add.params = list(size = 2),size = 1,width = 0.6) + 
  ggtitle('')+ ylab('pearson`s r')+ ylim(c(0,0.85))+
  theme_minimal(base_family = "Arial",base_size = 42)+
  theme(text = element_text("Arial",size = 42),
        axis.title = element_text("Arial",size = 36,face = "bold"),
        axis.text = element_text("Arial",size = 40,face = "bold"),
        axis.title.x = element_blank(),
        axis.text.x = element_text("Arial",angle = 0,size = 20,face = "bold"),
        plot.title = element_text(hjust = 1.15,size=40),
        legend.position='')
p2 <- p2 + stat_compare_means(comparisons=comparisons,method = 'wilcox.test',label='p.signif',
                              size=8,bracket.size = 0.7)
p2 <- p2 + scale_x_discrete(expand = c(0.2, 0))
print(p2)

# png('../figures/tfs_vs_direct_translating_comparison.png',width=14,height=15.16,units = "in",res = 600)
# print(p2)
# dev.off()
# setEPS()
# postscript('../figures/tfs_vs_direct_translating_comparison.ps',width=14,height=15.16)
# print(p2)
# dev.off()

# combine 2 subplots
p <- ggarrange(plotlist=list(p1,p2),ncol=2,nrow=1,common.legend = FALSE)
annotate_figure(p, top = text_grob("TFs performance from translating predicted gene expression", 
                                   family='Arial',color = "black",face = 'bold', size = 34))
ggsave(
  '../figures/tfs_analysis_figure1d.eps', 
  device = cairo_ps,
  scale = 1,
  width = 16,
  height = 8.7,
  units = "in",
  dpi = 600,
)


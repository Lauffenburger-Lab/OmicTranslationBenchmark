### Load dependencies------------------------------------------------------------------------
library(tidyverse)
library(caret)
library(ggplot2)
library(ggpubr)
library(ggpattern)
library(corrplot)
library(reshape2)
library(AnnotationDbi)
library(limma)
library(dorothea)
minNrOfGenes = 5
dorotheaData = read.table('../../../Artificial-Signaling-Network/TF activities/annotation/dorothea.tsv', sep = "\t", header=TRUE)
confidenceFilter = is.element(dorotheaData$confidence, c('A', 'B'))
dorotheaData = dorotheaData[confidenceFilter,]

### Load total data------------------------------------------------------------------------
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
sigInfo <- sigInfo %>% filter(quality_replicates==1)
sigInfo <- sigInfo %>% filter(tas>=0.3)
sigInfo <- sigInfo %>% group_by(cell_iname) %>% mutate(per_cell_sigs = n_distinct(sig_id)) %>% ungroup()
# Duplicate information
sigInfo <- sigInfo %>% mutate(duplIdentifier = paste0(cmap_name,"_",pert_idose,"_",pert_itime,"_",cell_iname))
sigInfo <- sigInfo %>% group_by(duplIdentifier) %>% mutate(dupl_counts = n()) %>% ungroup()
# Drug condition information
sigInfo <- sigInfo  %>% mutate(conditionId = paste0(cmap_name,"_",pert_idose,"_",pert_itime))
conditions <- sigInfo %>%  group_by(cell_iname) %>% summarise(conditions_per_cell = n_distinct(conditionId)) %>% ungroup()
cmap <- data.table::fread('../preprocessing/preprocessed_data/all_cmap_landmarks.csv',header = T) %>% column_to_rownames('V1')
sigInfo <- sigInfo %>% filter(sig_id %in% rownames(cmap))
sigInfo <- sigInfo %>% filter(per_cell_sigs>=100)
cells <- unique(sigInfo$cell_iname)

### GO-term enrichment analysis------------------------------------------------------------------------
all_cmap_go <-  fastenrichment(sigInfo$sig_id,
                               geneInfo$gene_id,
                               t(cmap),
                               enrichment_space = "go_bp",
                               pval_adjustment=T,
                               n_permutations=1000)
folds <- 5
random_iters <- 5
all_go_results <- data.frame()
for (j in 1:radom_iters){
  predictions <- data.frame()
  for (cell in cells){
    for (i in 1:folds){
      files_path <- paste0('../results/SameCellimputationModel/CPA/preds_AutoTransOp/',cell,'/')
      predictions <- rbind(predictions,
                           data.table::fread(paste0(files_path,'imputed_genes_fold',i-1,'_iter',j-1,'.csv'),
                                             header = T) %>% 
                             column_to_rownames('V1'))
    }
  }
  gos <- fastenrichment(rownames(predictions),
                               geneInfo$gene_id,
                               t(predictions),
                               enrichment_space = "go_bp",
                               pval_adjustment=T,
                               n_permutations=10)
  nes <- gos[["NES"]]$`NES GO BP`
  nes <- as.data.frame(nes) %>% rownames_to_column('GO BP') %>% gather('sig_id','NES',-`GO BP`)
  padj  <- gos[["Pval"]]$`Pval GO BP`
  padj <- as.data.frame(padj) %>% rownames_to_column('GO BP') %>% gather('sig_id','p.adj',-`GO BP`)
  gos <- left_join(nes,padj)
  gos <- gos %>% mutate(radom_iteration=j)
  all_go_results <- rbind(all_go_results,gos)
}

### KEGG pathway enrichment analysis------------------------------------------------------------------------
all_cmap_kegg <-  fastenrichment(sigInfo$sig_id,
                               geneInfo$gene_id,
                               t(cmap),
                               enrichment_space = "kegg",
                               pval_adjustment=T,
                               n_permutations=1000)

folds <- 5
random_iters <- 5
all_kegg_results <- data.frame()
for (j in 1:radom_iters){
  predictions <- data.frame()
  for (cell in cells){
    for (i in 1:folds){
      files_path <- paste0('../results/SameCellimputationModel/CPA/preds_AutoTransOp/',cell,'/')
      predictions <- rbind(predictions,
                           data.table::fread(paste0(files_path,'imputed_genes_fold',i-1,'_iter',j-1,'.csv'),
                                             header = T) %>% 
                             column_to_rownames('V1'))
    }
  }
  kegg_paths <- fastenrichment(rownames(predictions),
                               geneInfo$gene_id,
                               t(predictions),
                               enrichment_space = "kegg",
                               pval_adjustment=T,
                               n_permutations=1000)
  nes <- kegg_paths[["NES"]]$`NES KEGG`
  nes <- as.data.frame(nes) %>% rownames_to_column('KEGG') %>% gather('sig_id','NES',-KEGG)
  padj  <- kegg_paths[["Pval"]]$`Pval KEGG`
  padj <- as.data.frame(padj) %>% rownames_to_column('KEGG') %>% gather('sig_id','p.adj',-KEGG)
  kegg_paths <- left_join(nes,padj)
  kegg_paths <- kegg_paths %>% mutate(radom_iteration=j)
  all_kegg_results <- rbind(all_kegg_results,kegg_paths)
}


### Differential TF activity analysis------------------------------------------------------------------------
# Estimate TF activities of ground truth
print(all(colnames(cmap)==geneInfo$gene_id))
colnames(cmap) <- geneInfo$gene_symbol
settings = list(verbose = TRUE, minsize = minNrOfGenes)
TF_activities_cmap_all = run_viper(t(cmap), dorotheaData, options =  settings)

folds <- 5
random_iters <- 5
all_tf_results <- data.frame()
plotsList <- NULL
df_corr_all <- data.frame()
for (j in 1:random_iters){
  predictions <- data.frame()
  for (cell in cells){
    for (i in 1:folds){
      files_path <- paste0('../results/SameCellimputationModel/CPA/preds_AutoTransOp/',cell,'/')
      predictions <- rbind(predictions,
                           data.table::fread(paste0(files_path,'imputed_genes_fold',i-1,'_iter',j-1,'.csv'),
                                             header = T) %>% 
                             column_to_rownames('V1'))
    }
  }
  predictions <- predictions[,as.character(geneInfo$gene_id)]
  colnames(predictions) <- geneInfo$gene_symbol
  TF_activities_predicted = run_viper(t(predictions), dorotheaData, options =  settings)
  
  # Perform differential TF analysis
  TF_activities_cmap_subset <- TF_activities_cmap_all[,colnames(TF_activities_predicted)]
  condition = factor(c(rep("true", times =ncol(TF_activities_predicted)),
                       rep("predicted",times=ncol(TF_activities_predicted))))
  design_matrix <- model.matrix(~condition)
  fit <- lmFit(object = cbind(TF_activities_cmap_subset,TF_activities_predicted) ,
               design = design_matrix, method = "ls")
  fit <- eBayes(fit)
  TFanalysis_results <- topTable(fit,
                                 coef=2,
                                 number = nrow(TF_activities_cmap_all),
                                 genelist = rownames(TF_activities_cmap_all))
  TFanalysis_results <- TFanalysis_results %>% mutate(iteration=j)
  TFanalysis_results <- TFanalysis_results %>% mutate(iteration_adj_pval=random_iters*adj.P.Val)
  plotsList[[j]] <- ggplot(TFanalysis_results,aes(x=2^logFC,y=-log10(adj.P.Val))) + geom_point() +
    xlab('Fold Change') + ylab('-log10(P-value adjusted)') +
    geom_hline(yintercept=-log10(0.05), linetype="dashed",color = "#525252", size=0.5) +
    ggtitle(paste0('Random iteration ',j))+
    annotate("text",x=1.1,y=5,label="adjusted P-Value=0.05",size=6)+
    theme_pubr(base_size = 24,base_family = 'Arial')+
    theme(text = element_text(family = 'Arial',size = 24),
          plot.title = element_text(size = 14,hjust = 0.5))
    
  all_tf_results <- rbind(all_tf_results,TFanalysis_results)
  
  #Calculate correlations
  TF_activities_cmap_subset <- t(TF_activities_cmap_subset)
  TF_activities_predicted <- t(TF_activities_predicted)
  corr <- cor(TF_activities_cmap_subset,TF_activities_predicted)
  corr <- diag(corr)
  df_corr <- data.frame(corr) 
  df_corr <- df_corr %>% rownames_to_column('TF') %>% mutate(iteration=j)
  df_corr_all <- rbind(df_corr_all,df_corr)
  # corr_original <- cor(TF_activities_cmap_subset,TF_activities_cmap_subset)
  # corrplot(corr, method = 'color',
  #          order = 'hclust')
  # corrplot(corr_original, method = 'color',
  #          order = 'hclust')
  
}
p <- ggarrange(plotlist=plotsList,
                   ncol=2,nrow=3)
annotate_figure(p, top = text_grob("Differential TF activity", 
                                       color = "black",face = 'plain', size = 24))
ggsave(filename = '../article_supplementary_info/differential_tf_activity_cpa.eps',
       device = cairo_ps,
       width = 16,
       height = 16,
       units = 'in',
       dpi = 600)

# Plot TFs correlation
mean_tf_values <- aggregate(corr ~ TF, df_corr_all, mean)
df_corr_all$TF <- factor(df_corr_all$TF, levels = mean_tf_values$TF[order(mean_tf_values$corr)])
ggboxplot(df_corr_all,x='TF',y='corr',color ='TF' ,add = 'jitter') + 
  ylim(c(0,0.85))+ylab('Pearson`s r')+
  ggtitle('Performance in predicting TF activity')+
  theme(text = element_text(family = 'Arial',size=18),
        legend.position = 'none',
        plot.title = element_text(hjust = 0.5)) +
  coord_flip()
ggsave(filename = '../article_supplementary_info/correlation_tf_activity_samecell_imputation_cpa.eps',
       device = cairo_ps,
       width = 12,
       height = 16,
       units = 'in',
       dpi = 600)  

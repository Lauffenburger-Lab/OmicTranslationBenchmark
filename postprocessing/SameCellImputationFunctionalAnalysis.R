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
library(rstatix)
library(dorothea)
minNrOfGenes = 5
dorotheaData = read.table('../../../Artificial-Signaling-Network/TF activities/annotation/dorothea.tsv', sep = "\t", header=TRUE)
confidenceFilter = is.element(dorotheaData$confidence, c('A', 'B'))
dorotheaData = dorotheaData[confidenceFilter,]

confusionFun <- function(predicted, true,metric=c('accuracy','pvalue')){
  predicted <- factor(predicted,levels = c('up-regulated','down-regulated'))
  true <- factor(true,levels = c('up-regulated','down-regulated'))
  conf <- confusionMatrix(data=predicted, 
                          reference = true)
  if (metric=='accuracy'){
    val <- conf$overall['Accuracy']
  }else{
    val <- conf$overall['AccuracyPValue']
  }
  return(val)
}

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
#all_cmap_go <- readRDS('../results/SameCellimputationModel/go_enrichment_same_cell_imputation_original.rds')
all_cmap_go_nes <- all_cmap_go$NES[["NES GO BP"]]
all_cmap_go_nes <- as.data.frame(all_cmap_go_nes) %>% rownames_to_column('GO BP') %>% gather('sig_id','true NES',-`GO BP`)
all_cmap_go_padj <- all_cmap_go$Pval[["Pval GO BP"]]
all_cmap_go_padj <- as.data.frame(all_cmap_go_padj) %>% rownames_to_column('GO BP') %>% gather('sig_id','true p.adj',-`GO BP`)
all_cmap_go <- left_join(all_cmap_go_nes,all_cmap_go_padj)

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
#all_go_results <- readRDS('../results/SameCellimputationModel/all_go_results_same_cell_imputated.rds')

all_go_results <- left_join(all_cmap_go,all_go_results)
all_go_results <- all_go_results %>% mutate(log_padj = -log10(p.adj)) %>% mutate(log_padj_true = -log10(`true p.adj`))

all_cmap_go_nes <- NULL
all_cmap_go_padj <- NULL
all_cmap_go <- NULL
gc()

all_go_results <- all_go_results %>% filter(!is.na(NES)) %>% filter(!is.na(`true NES`))
gc()

# all_go_results <- readRDS('go_plotting_df.rds')
# gc()
all_go_results <- all_go_results %>% mutate(significant = ifelse(p.adj<0.05,
                                                                 ifelse(`true p.adj`<0.05,
                                                                        'both','imputed data'),
                                                                 ifelse(`true p.adj`<0.05,'true data',
                                                                        'none')))
gc()
all_go_confusion <- all_go_results %>% filter((p.adj<0.05) | (`true p.adj`<0.05)) %>%
  mutate(true_sig = ifelse(`true NES`<0,'down-regulated','up-regulated')) %>%
  mutate(pred_sig = ifelse(NES>0,'up-regulated','down-regulated')) %>%
  group_by(radom_iteration) %>% mutate(acc = confusionFun(pred_sig,true_sig))  %>% 
  mutate(acc_pval = confusionFun(pred_sig,true_sig,metric='pvalue')) %>%
  ungroup()
cof_results <- all_go_confusion %>% dplyr::select(acc,radom_iteration) %>% unique()
cof_results <- cof_results %>% mutate(`true NES`=-3,NES=1)
cof_results <-  cof_results %>% mutate(class_perform = paste0('Enrichment sign:','\n','accuracy:',100*round(acc,4),'%'))
p <- ggscatter(all_go_confusion,x="true NES", y = "NES",
               size=1,alpha=0.8,
               cor.coef = FALSE,
               cor.coeff.args = list(method = "pearson", label.x = -2.5, label.sep = "\n",size=6)) +
  geom_segment(aes(x=-3,xend = -0.5,y=-3,yend=-0.5),color='red',linetype='dashed')+
  geom_segment(aes(x=0.5,xend = 3,y=0.5,yend=3),color='red',linetype='dashed')+
  geom_hline(yintercept = 0,color='black',linetype='dashed',linewidth=1)+
  geom_vline(xintercept = 0,color='black',linetype='dashed',linewidth=1)+
  geom_text(aes(x=`true NES`,y=NES, label=class_perform),
            data=cof_results ,inherit.aes = FALSE,size=5, hjust = 0)+
  scale_x_continuous(breaks = c(-2,-1,0,1,2))+
  scale_y_continuous(breaks = c(-2,-1,0,1,2))+
  theme(text = element_text(family = 'Arial',size=24))+
  facet_wrap(~radom_iteration,ncol = 2)
print(p)
ggsave('../results/SameCellimputationModel/imputed_genes_go_term_analysis_nes_significant_colored.png',
       plot = p,
       width = 12,
       height = 12,
       units = 'in',
       dpi = 600)

### KEGG pathway enrichment analysis------------------------------------------------------------------------
all_cmap_kegg <-  fastenrichment(sigInfo$sig_id,
                               geneInfo$gene_id,
                               t(cmap),
                               enrichment_space = "kegg",
                               pval_adjustment=T,
                               n_permutations=1000)
# all_cmap_kegg <- readRDS('../results/SameCellimputationModel/kegg_erichment_same_cell_imputation_original.rds')
all_cmap_kegg_nes <- all_cmap_kegg$NES[["NES KEGG"]]
all_cmap_kegg_nes <- as.data.frame(all_cmap_kegg_nes) %>% rownames_to_column('KEGG') %>% gather('sig_id','true NES',-KEGG)
all_cmap_kegg_padj <- all_cmap_kegg$Pval[["Pval KEGG"]]
all_cmap_kegg_padj <- as.data.frame(all_cmap_kegg_padj) %>% rownames_to_column('KEGG') %>% gather('sig_id','true p.adj',-KEGG)
all_cmap_kegg <- left_join(all_cmap_kegg_nes,all_cmap_kegg_padj)

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
# all_kegg_results <- readRDS('../results/SameCellimputationModel/all_kegg_results_same_cell_imputated.rds')
all_kegg_results <- left_join(all_cmap_kegg,all_kegg_results)
all_kegg_results <- all_kegg_results %>% mutate(log_padj = -log10(p.adj)) %>% mutate(log_padj_true = -log10(`true p.adj`))

all_cmap_kegg_nes <- NULL
all_cmap_kegg_padj <- NULL
all_cmap_kegg <- NULL
gc()
all_kegg_results <- all_kegg_results %>% filter(!is.na(NES)) %>% filter(!is.na(`true NES`))
gc()
all_kegg_results <- all_kegg_results %>% mutate(significant = ifelse(p.adj<0.05,
                                                                 ifelse(`true p.adj`<0.05,
                                                                        'both','imputed data'),
                                                                 ifelse(`true p.adj`<0.05,'true data',
                                                                        'none')))
gc()
# all_kegg_results <-all_kegg_results %>% filter((p.adj<0.05) | (`true p.adj`<0.05))
# gc()
all_kegg_confusion <- all_kegg_results %>% filter((p.adj<0.05) | (`true p.adj`<0.05)) %>%
  mutate(true_sig = ifelse(`true NES`<0,'down-regulated','up-regulated')) %>%
  mutate(pred_sig = ifelse(NES>0,'up-regulated','down-regulated')) %>%
  group_by(radom_iteration) %>% mutate(acc = confusionFun(pred_sig,true_sig))  %>% 
  mutate(acc_pval = confusionFun(pred_sig,true_sig,metric='pvalue')) %>%
  ungroup()
# keggR2 <- all_kegg_confusion %>% filter(sign(`true NES`)==sign(NES)) %>% mutate(sign = ifelse(NES>0,'positive','negative')) %>%
#   group_by(radom_iteration) %>% mutate(R2 = cor(`true NES`,NES)^2) %>% mutate(r = cor(`true NES`,NES)) %>%
#   ungroup() %>% dplyr::select(R2,r,sign,radom_iteration) %>% unique()
cof_results <- all_kegg_confusion %>% dplyr::select(acc,radom_iteration) %>% unique()
cof_results <- cof_results %>% mutate(`true NES`=-3,NES=1)
cof_results <-  cof_results %>% mutate(class_perform = paste0('Enrichment sign:','\n','accuracy:',100*round(acc,4),'%'))
p <- ggscatter(all_kegg_confusion,x="true NES", y = "NES",
               color = "significant" ,size=1,alpha=0.8,
               cor.coef = FALSE,
               cor.coeff.args = list(method = "pearson", label.x = -2.5, label.sep = "\n",size=6)) +
  geom_segment(aes(x=-3,xend = -0.5,y=-3,yend=-0.5),color='red',linetype='dashed')+
  geom_segment(aes(x=0.5,xend = 3,y=0.5,yend=3),color='red',linetype='dashed')+
  geom_hline(yintercept = 0,color='black',linetype='dashed',linewidth=1)+
  geom_vline(xintercept = 0,color='black',linetype='dashed',linewidth=1)+
  geom_text(aes(x=`true NES`,y=NES, label=class_perform),
            data=cof_results ,inherit.aes = FALSE,size=5, hjust = 0)+
  scale_x_continuous(breaks = c(-2,-1,0,1,2))+
  scale_y_continuous(breaks = c(-2,-1,0,1,2))+
  theme(text = element_text(family = 'Arial',size=24))+
  facet_wrap(~radom_iteration,ncol = 2)
print(p)
ggsave('../results/SameCellimputationModel/imputed_genes_kegg_analysis_nes_signigicant_colored.png',
       plot = p,
       width = 12,
       height = 12,
       units = 'in',
       dpi = 600)
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
all_predicted_activities <- data.frame()
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
  if (j==1){
    all_predicted_activities <- as.data.frame(t(TF_activities_predicted)) %>% 
                                rownames_to_column('sig_id') %>%
                                mutate(iteration=j) 
  }else{
    all_predicted_activities <- rbind(all_predicted_activities,
                                      as.data.frame(t(TF_activities_predicted))%>% 
                                        rownames_to_column('sig_id') %>% 
                                        mutate(iteration=j))
  }
  
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
  ylim(c(0.55,0.92))+ylab('Pearson`s r')+
  annotate('text',x=40,y=0.65,label=paste0('Average r = ',round(mean(df_corr_all$corr),2)),size=8)+
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

mean_tf_values <- mean_tf_values[order(-mean_tf_values$corr),]
top3 <- mean_tf_values$TF[1:3]
bot3 <- mean_tf_values$TF[(nrow(mean_tf_values)-2):nrow(mean_tf_values)]

TF_combined <- left_join(as.data.frame(t(as.data.frame(TF_activities_cmap_all) %>% 
                                           dplyr::select(all_of(unique(all_predicted_activities$sig_id))))) %>%
                           rownames_to_column('sig_id') %>% gather('TF','activity',-sig_id) %>% filter(TF %in% c(bot3,top3)) %>%
                           mutate(type = 'true'),
                         all_predicted_activities %>% gather('TF','activity',-iteration,-sig_id) %>%
                         filter(TF %in% c(bot3,top3)) %>% mutate(type='predicted'),
                         by = c('sig_id','TF'))
TF_combined <- TF_combined %>% group_by(TF,iteration) %>% mutate(cohen_d =effectsize::cohens_d(activity.x,
                                                                                               activity.y,
                                                                                               paired = TRUE,
                                                                                               ci=0.95)$Cohens_d) %>%
  ungroup()
# TF_combined <- rbind(TF_combined %>% dplyr::select(sig_id,TF,c('activity'='activity.x'),c('type'='type.x'),iteration,cohen_d),
#                      TF_combined %>% dplyr::select(sig_id,TF,c('activity'='activity.y'),c('type'='type.y'),iteration,cohen_d))

stats_test = TF_combined %>% select(TF,cohen_d,iteration) %>% unique() %>% group_by(TF) %>%
  wilcox_test(cohen_d~1,mu = 0) %>% 
  adjust_pvalue(method = 'BH') %>% ungroup()
TF_combined <- TF_combined %>% select(TF,cohen_d,iteration) %>% unique()
mean_cohen_values <- aggregate(cohen_d ~ TF, TF_combined, mean)
TF_combined$TF <- factor(TF_combined$TF, levels = mean_cohen_values$TF[order(mean_cohen_values$cohen_d)])
p <- ggboxplot(TF_combined, x = 'TF', y = 'cohen_d',
         color = "TF", 
         point.size = 0.5,
         add='jitter') +
  xlab("transcription factor") +
  ylab('Cohen`s d') +
  geom_hline(yintercept = 0,linetype='dashed',color='black',linewidth=1)+
  geom_vline(xintercept = 3.5,linetype='dashed',color='black',linewidth=1)+
  annotate('text',x=2,y=0.02,label='3 worst performing TFs',size=5)+
  annotate('text',x=5,y=0.02,label='3 best performing TFs',size=5)+
  stat_pvalue_manual(stats_test,
                     label = "p = {p}",
                     x = "TF",
                     y.position = 0.35,
                     size = 6) +
  theme(text=element_text(family = 'Arial',size=24),
        legend.position = 'none')
print(p)
ggsave('../article_supplementary_info/effect_size_difference_imputed_tfs.eps',
       plot=p,
       device=cairo_ps,
       width = 12,
       height = 9,
       units = 'in',
       dpi=600)

tt <- TF_combined %>% select(TF,cohen_d,iteration) %>% unique() %>% filter(TF=='CTCF')
t.test(tt$cohen_d,mu = 0)

library(ggplot2)
library(lmtest)
library(tidyverse)
setEPS()

### Before all load gene names as they are in order in input data--------------
human_genes <- data.table::fread('data/10foldcrossval_lung/csvFiles/labeled_val_human_0.csv') %>%
  select(-diagnosis,-cell_type,-specific_cell)
human_genes <- colnames(human_genes)
mouse_genes <- data.table::fread('data/10foldcrossval_lung/csvFiles/labeled_val_mouse_0.csv') %>%
  select(-diagnosis,-cell_type,-specific_cell)
mouse_genes <- colnames(mouse_genes)

### First identify latent variables based on LRTs-----------------------
dim <- 512
for (i in 1:10){
  latent_embeddings_human <-data.table::fread(paste0("../../../Fibrosis Species Translation/human lung fibrosis/embs/train/trainEmbs_",i-1,"_human.csv"))
  latent_embeddings_human <- latent_embeddings_human %>% select(-V1)
  latent_embeddings_human <- latent_embeddings_human %>% mutate(species=1)
  latent_embeddings_mouse <-data.table::fread(paste0("../../../Fibrosis Species Translation/human lung fibrosis/embs/train/trainEmbs_",i-1,"_mouse.csv"))
  latent_embeddings_mouse <- latent_embeddings_mouse %>% select(-V1)
  latent_embeddings_mouse <- latent_embeddings_mouse %>% mutate(species=0)
  latent_embeddings <- rbind(latent_embeddings_human,latent_embeddings_mouse)
  data<-latent_embeddings[,1:dim]
  colnames(data) <- paste0('z',seq(0,511))
  outcomes<-latent_embeddings[,(dim+1):ncol(latent_embeddings)]
  
  center_apply <- function(x) {
    apply(x, 2, function(y) y - mean(y))
  }
  centered_data <- center_apply(data)
  centered_data<-data.frame(centered_data)
  
  #https://www.statology.org/likelihood-ratio-test-in-r/
  
  results <- data.frame(matrix(ncol = dim, nrow = 2))
  
  full<-cbind(centered_data,outcomes)
  
  for (j in 1:dim){
    
    full_model<-lm(full[,j]~diagnosis+cell_type+species,data = full)
    null_model<-lm(full[,j]~cell_type+species,data = full)
    kp<-lrtest(full_model,null_model)
    
    results[1,j]<-kp[2,5]
    results[2,j]<-summary(full_model)[["coefficients"]][, "t value"][2]
    
  }
  
  results<-data.frame(t(results))
  colnames(results)<-c("pvalue","tvalue")
  results$logp<-(-1*log(results$pvalue))
  row.names(results)<-colnames(centered_data)
  write.csv(results,paste0("results/LRT_results/LRT_latent_embeddings_",i,"_after_results.csv"))
  
  results$color <- ifelse(results$logp > (-log(0.01/dim)) & abs(results$tvalue)>1, "selected", NA_character_)
  results$label<-colnames(centered_data)
  
  setEPS()
  postscript(paste0("results/LRT_results/LRT_latent_embeddings_global_",i,".eps"))
  ggplot(results, aes(x=tvalue, y=logp,color = color)) + geom_point()+geom_hline(yintercept= (-log(0.05/dim)),linetype='dashed')+theme_bw()+  
    geom_text(data=subset(results, logp >  (-log(0.05/dim))),
              aes(x=tvalue, y=logp,label=label),nudge_x=0.5, nudge_y=0.5)
  dev.off()
  
  print(paste0('Finished fold ',i-1))
}
## Get consensus
all_selected <- data.frame()
for (i in 1:10){
  fold_lrt_res <- data.table::fread(paste0('results/LRT_results/LRT_latent_embeddings_',i,'_after_results.csv'))
  fold_lrt_res$selected <- ifelse(fold_lrt_res$logp > (-log(0.01/dim)) & abs(fold_lrt_res$tvalue)>1, 1,0)
  fold_lrt_res <- fold_lrt_res %>% dplyr::select(V1,selected) %>% mutate(Fold=i) %>% spread(key = V1,value = selected)
  fold_lrt_res$lrt_counts <- sum(fold_lrt_res[,2:ncol(fold_lrt_res)])
  all_selected <- rbind(all_selected,fold_lrt_res)
}
all_selected <- all_selected %>% filter(Fold!='')
colnames(all_selected)[ncol(all_selected)] <- 'lrt_counts'
all_selected <- all_selected %>% gather('latent_var','value',-Fold,-lrt_counts)
all_selected <- all_selected %>% filter(value>0) %>% select(-value) %>% mutate(lrt_importance='Important')
all_selected <- all_selected %>% mutate(Fold = paste0('fold ',as.numeric(Fold)-1))
data.table::fwrite(all_selected,'results/LRT_results/LRT_selected_latent_variables.csv')


### Get important variables for fibrosis from the classifier------------------------------------
LRT_results_cpa <- data.table::fread('results/LRT_results/LRT_selected_latent_variables.csv')

# Load classification and lrt t-values results
grad_results <- data.frame()
lrt_all_results <- data.frame()
for (i in 1:10){
  tmp <- data.table::fread(paste0('../../../Fibrosis Species Translation/human lung fibrosis/importance/important_scores_to_classify_human_fibrosis_',i-1,'.csv'))
  tmp <- tmp %>% select(-V1)
  mean_class_score <- colMeans(tmp)
  #mean_class_score <- apply(tmp,2,median)
  df_class_score <- data.frame(mean_class_score)
  colnames(df_class_score) <- 'mean_score'
  df_class_score <- df_class_score %>% rownames_to_column('latent_var')
  df_class_score <- df_class_score %>% mutate(Fold = paste0('fold ',i-1))
  grad_results <- rbind(grad_results,df_class_score)
  
  # load LRT results
  tmp <-  data.table::fread(paste0('results/LRT_results/LRT_latent_embeddings_',i,'_after_results.csv'))
  colnames(tmp)[1] <- 'latent_var'
  tmp <- tmp %>% mutate(Fold = paste0('fold ',i-1))
  tmp <- tmp %>% select(-logp)
  lrt_all_results <- rbind(lrt_all_results,tmp)
}
grad_results$latent_var <- factor(grad_results$latent_var,levels = paste0('z',seq(0,dim)))
grad_results$Fold <- factor(grad_results$Fold,levels = paste0('fold ',seq(0,9)))
grad_results <- grad_results %>% group_by(Fold) %>% mutate(mean_score=100*mean_score/max(abs(mean_score))) %>% ungroup()
grad_results <- left_join(LRT_results_cpa,grad_results)
lrt_all_results$latent_var <- factor(lrt_all_results$latent_var,levels = paste0('z',seq(0,dim)))
lrt_all_results$Fold <- factor(lrt_all_results$Fold,levels = paste0('fold ',seq(0,9)))
lrt_all_results <- left_join(LRT_results_cpa,lrt_all_results)
lrt_all_results <- lrt_all_results %>% filter(!is.na(pvalue)) %>% select(Fold,latent_var,tvalue) %>% unique()
grad_results <- left_join(grad_results,lrt_all_results)
grad_results <- grad_results %>% filter(!is.na(mean_score))
grad_results <- grad_results %>% mutate(agreement = ifelse(sign(mean_score)==sign(tvalue),'yes','no'))
p <- ggplot(grad_results,
            aes(x=reorder(str_split_fixed(latent_var,'z',n=2)[,2],as.numeric(str_split_fixed(latent_var,'z',n=2)[,2])),
                y=mean_score,
                fill=agreement))  + 
  scale_fill_manual(values = c('#b30024','#00b374')) + 
  geom_bar(stat='identity') + xlab('Latent variable z') + #position = position_dodge()
  scale_x_discrete(expand = c(0.001, 0))+
  facet_wrap(~Fold,ncol = 3)+
  ylab('% Importance score for classifying protection')+
  ggtitle('Important latent variables for protection') + 
  theme_minimal(base_family = "Arial",base_size = 28)+
  theme(plot.title = element_text(family = "Arial",size=28,hjust = 0.5),
        text = element_text(family = "Arial",size=28),
        axis.text.x = element_blank())
print(p)
#fill='#0077b3'
ggsave(
  'results/importance_results/figures/importance_scores_classification.eps', 
  device = cairo_ps,
  scale = 1,
  width = 18,
  height = 16,
  units = "in",
  dpi = 600,
)
png('results/importance_results/figures/importance_scores_classification.png',width=18,height=16,units = "in",res = 600)
print(p)
dev.off()

# Now keep only those in agreement and that mean score is more than 50%
grad_results <- grad_results %>% filter(agreement=='yes') %>% select(-agreement) %>%
  filter(abs(mean_score)>=50)
print(length(unique(grad_results$Fold)))

# Analysis for identifying features in each for that control the latent variables---------------
features_results <- data.frame()
cells <- c('Macrophage','Plasma cell','T cell',
           'AT2','Fibroblast','B cell',
           'Myofibroblast','AT1','NK cell')
for (cell in cells){
  print(paste0('Started cell : ',cell))
  for (i in 1:10){
    if (paste0("fold ",i-1) %in% grad_results$Fold){
      tmp <- data.table::fread(paste0('../../../Fibrosis Species Translation/human lung fibrosis/importance/',
                                      cell,'_important_genes_human_',i-1,'.csv'))
      tmp$V1 <- human_genes
      tmp <- tmp %>% column_to_rownames('V1')
      fold_variables <- grad_results %>% filter(Fold==paste0("fold ",i-1)) %>% unique()
      fold_variables <- fold_variables$latent_var
      tmp <- tmp %>% select(all_of(fold_variables))
      fold_variables_sign <- grad_results %>% filter(Fold==paste0("fold ",i-1)) %>% unique()
      fold_variables_sign <- sign(fold_variables_sign$mean_score)
      if(ncol(tmp)<2){
        tmp <- apply(tmp,1,'*',fold_variables_sign)
        mean_feature_score <- tmp
      }else{
        tmp <- apply(tmp,1,'*',fold_variables_sign)
        tmp <- t(tmp)
        mean_feature_score <- rowMeans(tmp)
      }
      
      df_feature_score <- data.frame(mean_feature_score)
      colnames(df_feature_score) <- 'mean_score'
      df_feature_score <- df_feature_score %>% mutate(cell_type = cell)
      df_feature_score <- df_feature_score %>% rownames_to_column('feature')
      df_feature_score <- df_feature_score %>% mutate(Fold = paste0("fold ",i-1))
      df_feature_score <- df_feature_score %>% mutate(percentage_score=100*mean_score/(max(abs(mean_score))))
      df_feature_score <- df_feature_score %>% select(Fold,cell_type,feature,mean_score,percentage_score)
      #df_feature_score <- df_feature_score %>%  filter(abs(percentage_score)>=70)
      features_results <- rbind(features_results,df_feature_score)
    }
    print(paste0('Finished fold : ',i))
  }
  print(paste0('Finished cell : ',cell))
}

features_results <- features_results %>% group_by(cell_type,feature) %>%
  mutate(mean_percentage_score = mean(percentage_score)) %>% ungroup()
features_results <- features_results %>% select(-Fold,-mean_score,-percentage_score) %>% unique()

data.table::fwrite(features_results,'results/importance_results/all_human_fibrosis_important_genes.csv')
data.table::fwrite(features_results %>% filter(abs(mean_percentage_score)>=50),
                   'results/importance_results/filtered_human_fibrosis_important_genes.csv')
##
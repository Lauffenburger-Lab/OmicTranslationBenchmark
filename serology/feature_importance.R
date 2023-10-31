library(tidyverse)
library(ggplot2)
library(ggpubr)
library(ggridges)
library(rstatix)
library(ggrepel)

# Analysis for classification of protection-------------
all_selected <- data.frame()
for (i in 1:10){
  fold_lrt_res <- data.table::fread(paste0('LRT_results_cpa/LRT_latent_embeddings_global',i,'_after_results.csv'))
  fold_lrt_res$selected <- ifelse(fold_lrt_res$logp > (-log(0.05/32)), 1,0)
  fold_lrt_res <- fold_lrt_res %>% dplyr::select(V1,selected) %>% mutate(Fold=paste0('global_',i)) %>% spread(key = V1,value = selected)
  fold_lrt_res$lrt_counts <- sum(fold_lrt_res[,2:ncol(fold_lrt_res)])
  all_selected <- rbind(all_selected,fold_lrt_res)
}
data.table::fwrite(all_selected,'LRT_results_cpa/LRT_selected_latent_variables.csv')

# Load LRT results
LRT_results_cpa <- data.table::fread('LRT_results_cpa/LRT_selected_latent_variables.csv') #%>% select(-z32)
LRT_results_cpa <- LRT_results_cpa %>% filter(Fold!='')
colnames(LRT_results_cpa)[ncol(LRT_results_cpa)] <- 'lrt_counts'
LRT_results_cpa <- LRT_results_cpa %>% gather('latent_var','value',-Fold,-lrt_counts)
LRT_results_cpa <- LRT_results_cpa %>% filter(value>0) %>% select(-value) %>% mutate(lrt_importance='Important')
LRT_results_cpa <- LRT_results_cpa%>% mutate(Fold = str_split(Fold,'_')) %>% unnest(Fold) %>% 
  filter(Fold!='global') %>% mutate(Fold = paste0('fold ',as.numeric(Fold)-1))

# Load classification and lrt t-values results
grad_results <- data.frame()
lrt_all_results <- data.frame()
for (i in 1:10){
  tmp <- data.table::fread(paste0('importance_results_cpa/important_scores_to_classify_human_protection_',i-1,'.csv'))
  tmp <- tmp %>% select(-V1)
  mean_class_score <- colMeans(tmp)
  #mean_class_score <- apply(tmp,2,median)
  df_class_score <- data.frame(mean_class_score)
  colnames(df_class_score) <- 'mean_score'
  df_class_score <- df_class_score %>% rownames_to_column('latent_var')
  df_class_score <- df_class_score %>% mutate(Fold = paste0('global_',i))
  grad_results <- rbind(grad_results,df_class_score)
  
  # load LRT results
  tmp <-  data.table::fread(paste0('LRT_results_cpa/LRT_latent_embeddings_global',i,'_after_results.csv'))
  colnames(tmp)[1] <- 'latent_var'
  tmp <- tmp %>% mutate(Fold = paste0('global_',i))
  tmp <- tmp %>% select(-logp)
  lrt_all_results <- rbind(lrt_all_results,tmp)
}
grad_results$latent_var <- factor(grad_results$latent_var,levels = paste0('z',seq(0,31)))
grad_results <- grad_results %>% mutate(Fold = str_split(Fold,'_')) %>% unnest(Fold) %>% 
  filter(Fold!='global') %>% mutate(Fold = paste0('fold ',as.numeric(Fold)-1))
grad_results$Fold <- factor(grad_results$Fold,levels = paste0('fold ',seq(0,9)))
grad_results <- grad_results %>% group_by(Fold) %>% mutate(mean_score=100*mean_score/max(abs(mean_score))) %>% ungroup()
grad_results <- left_join(LRT_results_cpa,grad_results)
lrt_all_results$latent_var <- factor(lrt_all_results$latent_var,levels = paste0('z',seq(0,31)))
lrt_all_results <- lrt_all_results %>% mutate(Fold = str_split(Fold,'_')) %>% unnest(Fold) %>% 
  filter(Fold!='global') %>% mutate(Fold = paste0('fold ',as.numeric(Fold)-1))
lrt_all_results$Fold <- factor(lrt_all_results$Fold,levels = paste0('fold ',seq(0,9)))
lrt_all_results <- left_join(LRT_results_cpa,lrt_all_results)
lrt_all_results <- lrt_all_results %>% select(Fold,latent_var,tvalue) %>% unique()
grad_results <- left_join(grad_results,lrt_all_results)
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
        axis.text.x = element_text(family = "Arial",size=10))
print(p)
#fill='#0077b3'
ggsave(
  'importance_results_cpa/figures/importance_scores_classification.eps', 
  device = cairo_ps,
  scale = 1,
  width = 18,
  height = 16,
  units = "in",
  dpi = 600,
)
png('importance_results_cpa/figures/importance_scores_classification.png',width=18,height=16,units = "in",res = 600)
print(p)
dev.off()

# # Load embeddings
# emb <- data.table::fread('results_intermediate_encoders/embs/combined_10fold/latent_embeddings_global_1.csv')
# emb <- emb %>% mutate(protected=ifelse(protected==1,'protected','non-protected'))
# ggplot(emb,aes(x=z0,y= z18,color=protected,shape=species)) + geom_point()
# ggplot(emb,aes(x=z18,fill=protected)) + geom_density()

# Now keep only those in agreement and that mean score is more than 10%
grad_results <- grad_results %>% filter(agreement=='yes') %>% select(-agreement) %>%
  filter(abs(mean_score)>=10)

# Analysis for identifying features in each for that control the latent variables---------------
features_results <- data.frame()
for (i in 1:10){
  if (paste0("fold ",i-1) %in% grad_results$Fold){
    tmp <- data.table::fread(paste0('importance_results_cpa/important_scores_human_features_',i-1,'.csv'))
    tmp <- tmp %>% column_to_rownames('V1')
    fold_variables <- grad_results %>% filter(Fold==paste0("fold ",i-1)) %>% unique()
    fold_variables <- fold_variables$latent_var
    tmp <- tmp %>% select(all_of(fold_variables))
    fold_variables_sign <- grad_results %>% filter(Fold==paste0("fold ",i-1)) %>% unique()
    fold_variables_sign <- sign(fold_variables_sign$mean_score)
    tmp <- apply(tmp,1,'*',fold_variables_sign)
    tmp <- t(tmp)
    mean_feature_score <- rowMeans(tmp)
    
    df_feature_score <- data.frame(mean_feature_score)
    colnames(df_feature_score) <- 'mean_score'
    df_feature_score <- df_feature_score %>% rownames_to_column('feature')
    df_feature_score <- df_feature_score %>% mutate(Fold = paste0("fold ",i-1))
    df_feature_score <- df_feature_score %>% mutate(percentage_score=100*mean_score/(max(abs(mean_score))))
    #df_feature_score <- df_feature_score %>%  filter(abs(percentage_score)>=70)
    features_results <- rbind(features_results,df_feature_score)
  }
}

features_results <- features_results %>% group_by(feature) %>%
  mutate(mean_percentage_score = mean(percentage_score)) %>% ungroup()
features_results <- features_results %>% select(-Fold,-mean_score,-percentage_score) %>% unique()
#  mutate(freq = n_distinct(Fold)/length(unique(grad_results$Fold)))
# Load features
feats <- data.table::fread('data/human_exprs.csv') %>% select(-V1)
labels <- data.table::fread('data/human_metadata.csv') %>% select(-V1)
feats$protection <- labels$infect
feats <- feats %>% mutate(protected=ifelse(protection==1,'protected','non-protected'))
feats$protected <- factor(feats$protected,levels = c('non-protected','protected'))
feats <- feats %>% gather('feature','value',-protection,-protected)
feats <- left_join(feats,features_results)
pairwise.test_human = feats %>% filter(mean_percentage_score>=50) %>% group_by(feature) %>%
  wilcox_test(value ~ protected) %>% 
  adjust_pvalue(method = 'bonferroni') %>% ungroup()
ggboxplot(feats %>% filter(mean_percentage_score>=50),
          x='protected',y='value',color='protected',add='jitter')+
  xlab('')+
  facet_wrap(~feature) +
  stat_pvalue_manual(pairwise.test_human,
                     label = "p.adj = {scales::pvalue(p.adj)}",
                     y.position = 2,
                     size=7)+
  theme(axis.title = element_text(family = 'Arial',size=24),
        axis.text = element_text(family = 'Arial',size=24),
        text = element_text(family = 'Arial',size=24),
        strip.text.x = element_text(size = 14))
#data.table::fwrite(pairwise.test_human,'results_intermediate_encoders/important_human_protection_features.csv')
ggsave(
  'importance_results_cpa/figures/importance_human_features_greather_than_20perc.eps', 
  device = cairo_ps,
  scale = 1,
  width = 18,
  height = 9,
  units = "in",
  dpi = 600,
)

# Repeat for primates
features_results_primates <- data.frame()
for (i in 1:10){
  if (paste0("fold ",i-1) %in% grad_results$Fold){
    tmp <- data.table::fread(paste0('importance_results_cpa/important_scores_primates_features_',i-1,'.csv'))
    tmp <- tmp %>% column_to_rownames('V1')
    fold_variables <- grad_results %>% filter(Fold==paste0("fold ",i-1)) %>% unique()
    fold_variables <- fold_variables$latent_var
    tmp <- tmp %>% select(all_of(fold_variables))
    fold_variables_sign <- grad_results %>% filter(Fold==paste0("fold ",i-1)) %>% unique()
    fold_variables_sign <- sign(fold_variables_sign$mean_score)
    tmp <- apply(tmp,1,'*',fold_variables_sign)
    tmp <- t(tmp)
    mean_feature_score <- rowMeans(tmp)
    
    df_feature_score <- data.frame(mean_feature_score)
    colnames(df_feature_score) <- 'mean_score'
    df_feature_score <- df_feature_score %>% rownames_to_column('feature')
    df_feature_score <- df_feature_score %>% mutate(Fold = paste0("fold ",i-1))
    df_feature_score <- df_feature_score %>% mutate(percentage_score=100*mean_score/(max(abs(mean_score))))
    #df_feature_score <- df_feature_score %>%  filter(abs(percentage_score)>=70)
    features_results_primates <- rbind(features_results_primates,df_feature_score)
  }
}

features_results_primates <- features_results_primates %>% group_by(feature) %>%
  mutate(mean_percentage_score = mean(percentage_score)) %>% ungroup()
features_results_primates <- features_results_primates %>% select(-Fold,-mean_score,-percentage_score) %>% unique()
#  mutate(freq = n_distinct(Fold)/length(unique(grad_results$Fold)))
# Load features
feats_primates <- data.table::fread('data/primates_exprs.csv') %>% select(-V1)
labels_primates <- data.table::fread('data/primates_metadata.csv') %>% select(-V1)
feats_primates$protection <- labels_primates$ProtectBinary
feats_primates <- feats_primates %>% mutate(protected=ifelse(protection==1,'protected','non-protected'))
feats_primates$protected <- factor(feats_primates$protected,levels = c('non-protected','protected'))
feats_primates <- feats_primates %>% gather('feature','value',-protection,-protected)
feats_primates <- left_join(feats_primates,features_results_primates)
pairwise.test_primates = feats_primates %>% filter(mean_percentage_score>=20) %>% group_by(feature) %>%
  wilcox_test(value ~ protected) %>% 
  adjust_pvalue(method = 'bonferroni') %>% ungroup()
ggboxplot(feats_primates %>% filter(mean_percentage_score>=20),
          x='protected',y='value',color='protected',add='jitter')+
  facet_wrap(~feature) +
  #ylim(c(-3.7,3))+
  stat_pvalue_manual(pairwise.test_primates,
                     label = "p.adj = {scales::pvalue(p.adj)}",
                     y.position = 1.25)


# filter human features not statistically significantly different between protected and non-protected
pairwise.test_human = feats %>% filter(mean_percentage_score>=20) %>% group_by(feature) %>%
  wilcox_test(value ~ protected) %>% 
  adjust_pvalue(method = 'bonferroni') %>% ungroup()
feats <- left_join(feats,pairwise.test_human %>% select(feature,p.adj))
feats_interesting <- rbind(feats_primates %>% filter(mean_percentage_score>=20) %>% 
  select('feature') %>% unique() %>% mutate(species='primates'),
  feats %>% filter(mean_percentage_score>=20) %>%
    filter(p.adj<0.1) %>% select(-p.adj) %>%
    select('feature') %>% unique()%>% mutate(species='human'))
data.table::fwrite(feats_interesting,'importance_results_cpa/interesting_features.csv',row.names = T)

# See the rank of importance of these primates important features for
# human important features
translation_importance <- data.frame()
interesting_human_feats <- feats_interesting %>% filter(species=='human') %>% select(feature)
interesting_human_feats <- unique(interesting_human_feats$feature)
for (i in 1:10){
  tmp <- data.table::fread(paste0('importance_results_cpa/translation/important_scores_primates_to_human_',i-1,'.csv'))
  tmp <- tmp %>% column_to_rownames('V1')
  mean_feature_score <- rowMeans(tmp)
  tmp$mean_score <- mean_feature_score
  tmp <- tmp %>% mutate(fold = i)
  tmp <- tmp %>% select(fold,all_of(interesting_human_feats),mean_score)
  data.table::fwrite(tmp,paste0('importance_results_cpa/translation/processed/important_scores_primates_to_human_',i-1,'.csv'),row.names = T)
  df_feature_score <- data.frame(mean_feature_score)
  colnames(df_feature_score) <- 'mean_score'
  df_feature_score <- df_feature_score %>% rownames_to_column('feature')
  df_feature_score <- df_feature_score %>% mutate(Fold = paste0("fold ",i-1))
  df_feature_score <- df_feature_score %>% mutate(percentage_score=100*mean_score/(max(abs(mean_score))))
  #tmp_ranked <- apply(tmp,2,rank)
  translation_importance <- rbind(translation_importance,df_feature_score)
}

translation_importance <- translation_importance %>% group_by(feature) %>%
  mutate(mean_percentage_score = median(percentage_score)) %>% ungroup()
translation_importance <- translation_importance %>% select(-Fold,-mean_score,-percentage_score) %>% unique()
# Load primates correlation in validation
pimates_performance <- data.table::fread('results/10foldvalidation_decoder_32dim2000ep_perFeature_primates.csv')
pimates_performance <- pimates_performance %>% select(-V1)
pimates_feature_performance <- colMeans(pimates_performance)
df_performance <- data.frame(r=pimates_feature_performance)
# df_performance <- df_performance %>% rownames_to_column('feature') %>%
#   mutate(quality=ifelse(r>=0.75,'r\u22650.75',
#                         ifelse(r>=0.5,'r\u22650.5',
#                                ifelse(r>=0.25,'r\u22650.25',
#                                       ifelse(r>0,'r>0','r\u22640')))))
# df_performance$quality <- factor(df_performance$quality,
#                                  levels = c('r\u22640','r>0',
#                                             'r\u22650.25','r\u22650.5',
#                                             'r\u22650.75'))
df_performance <- df_performance %>% rownames_to_column('feature') %>%
  mutate(quality=ifelse(r>=0.75,'high',
                        ifelse(r>=0.5,'good',
                               ifelse(r>=0.25,'medium',
                                      ifelse(r>0,'low','bad')))))
df_performance$quality <- factor(df_performance$quality,
                                 levels = c('bad','low',
                                            'medium','good',
                                            'high'))
translation_importance <- left_join(translation_importance,df_performance)
translation_importance <- translation_importance %>% mutate(annotation = ifelse(abs(mean_percentage_score)>=20 & quality %in% c('high','good'),
                                                                                feature,
                                                                                ''))
print(unique(translation_importance$quality))
# Make baplot and save
ggplot(translation_importance) +
  geom_bar(aes(x=reorder(feature,-mean_percentage_score), y=mean_percentage_score,fill=quality), stat="identity") +
  scale_fill_manual(values = c('#e67002','#d6e602','#00b374'))+
  #scale_fill_manual(values = c('#b30024','#e67002','#d6e602','#00b374'))+
  geom_hline(yintercept = 0,linewidth=1)+
  geom_hline(yintercept = 20,linetype='dashed',linewidth=1)+
  geom_hline(yintercept = -20,linetype='dashed',linewidth=1)+
  geom_label_repel(aes(x=feature,y=mean_percentage_score,label=annotation),
                   size = 5,
                   box.padding   = 0.75, 
                   point.padding = 0.1,
                   max.overlaps = 40,
                   segment.color = 'grey50')+
  xlab('primates serological features') + ylab('mean percentage score %') +
  ggtitle('Primates feature importance for protection associated human features')+
  theme_minimal(base_family = "Arial")+
  theme(panel.grid = element_blank(),
        axis.title = element_text("Arial",size = 36,face = "bold"),
        axis.text = element_text("Arial",size = 36,face = "bold"),
        legend.text = element_text("Arial",size = 36,face = "bold"),
        legend.title = element_text("Arial",size = 36,face = "bold"),
        axis.text.x = element_blank(),
        plot.title = element_text(hjust = 0.5,size=32))
ggsave(
  'importance_results_cpa/figures/important_translation.eps', 
  device = cairo_ps,
  scale = 1,
  width = 16,
  height = 9,
  units = "in",
  dpi = 600,
)  


### Supplementary analysis-----------------
#  mutate(freq = n_distinct(Fold)/length(unique(grad_results$Fold)))
# Load features
feats_primates <- data.table::fread('data/primates_exprs.csv') %>% select(-V1)
labels_primates <- data.table::fread('data/primates_metadata.csv') %>% select(-V1)
feats_primates$protection <- labels_primates$ProtectBinary
feats_primates <- feats_primates %>% mutate(protected=ifelse(protection==1,'protected','non-protected'))
feats_primates$protected <- factor(feats_primates$protected,levels = c('non-protected','protected'))
feats_primates <- feats_primates %>% gather('feature','value',-protection,-protected)
feats_primates <- left_join(feats_primates,translation_importance)
pairwise.test_primates = feats_primates %>% filter(mean_percentage_score>=20) %>% 
  filter(quality=='high' | quality=='good' ) %>% group_by(feature) %>%
  wilcox_test(value ~ protected) %>% 
  adjust_pvalue(method = 'bonferroni') %>% ungroup()
#pairwise.test_primates <- pairwise.test_primates %>% filter(p.adj<=0.05)
ggboxplot(feats_primates %>% filter(mean_percentage_score>=20) %>% 
            filter(quality=='high' | quality=='good' )%>%
            filter(feature %in% pairwise.test_primates$feature),
          x='protected',y='value',color='protected',add='jitter')+
  facet_wrap(~feature) +
  #ylim(c(-3.7,3))+
  stat_pvalue_manual(pairwise.test_primates,
                     label = "p.adj = {scales::pvalue(p.adj)}",
                     y.position = 1.25)
png('importance_results_cpa/figures/importance_primates_features_greather_than_30perctranslation_goodhigh.png',width=18,height=16,units = "in",res = 600)
ggboxplot(feats_primates %>% filter(mean_percentage_score>=20) %>% 
            filter(quality=='high' | quality=='good' )%>%
            filter(feature %in% pairwise.test_primates$feature),
          x='protected',y='value',color='protected',add='jitter')+
  facet_wrap(~feature) +
  #ylim(c(-3.7,3))+
  stat_pvalue_manual(pairwise.test_primates,
                     label = "p.adj = {scales::pvalue(p.adj)}",
                     y.position = 1.25)
dev.off()
# ggsave(
#   'importance_results_cpa/figures/importance_primates_features_greather_than_30perctranslation_goodhigh.eps', 
#   device = cairo_ps,
#   scale = 1,
#   width = 18,
#   height = 9,
#   units = "in",
#   dpi = 600,
# )
# compare importance score of primates for what is believed to be human protection
# and the translation
results <- left_join(translation_importance,features_results_primates,by='feature')
colnames(results)[2:3] <- c('translation_importance','encoding_importance')
enc <- results %>% filter(encoding_importance>=5)
trans <- results %>% filter(translation_importance>=5)
common <- Reduce(intersect,list(enc$feature,trans$feature))
union <- Reduce(union,list(enc$feature,trans$feature))
ggscatter(results ,x='encoding_importance',y='translation_importance',cor.coef = T,cor.method = 'spearman' ) +
  geom_hline(yintercept = 5,linetype='dashed',linewidth=2) + geom_vline(xintercept = 5,linetype='dashed',linewidth=2)+
  geom_hline(yintercept = -5,linetype='dashed',linewidth=2) + geom_vline(xintercept = -5,linetype='dashed',linewidth=2)+
  annotate('rect', xmin=5, xmax=max(results$encoding_importance), 
           ymin=5, ymax=max(results$translation_importance), 
           alpha=.1, fill='black')+
  annotate("text",x=20,y=3,label=paste0('Overlap ',100 * round(length(common)/length(union),4),' %'),size=5)

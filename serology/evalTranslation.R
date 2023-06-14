library(tidyverse)
library(ggplot2)
library(ggsignif)
library(ggrepel)
library(ggpubr)
library(ggridges)
library(rstatix)

# Load data--------------------------
human_expr <- data.table::fread('data/human_exprs.csv') %>% select(-V1)
human_hiv<- data.table::fread('data/human_metadata.csv') %>% select(-V1)
human <- cbind(human_expr,human_hiv %>% select(c('protected'='infect'),c('vaccinated'='trt')))
human_feature_means <- colMeans(human_expr)
human_feature_sds<- apply(human_expr,2,sd)
protection_feats_human <- data.table::fread('results_intermediate_encoders/important_human_protection_features.csv')
for (i in 1:10){
  translated_primate <- data.table::fread(paste0('results_intermediate_encoders/translated/primates2human/y_pred_train_',i-1,'.csv')) %>%
    select(-V1)
  features_means <- colMeans(translated_primate %>% select(-protected,-vaccinated))
  features_sds <- apply(translated_primate %>% select(-protected,-vaccinated),2,sd)
  df <- data.frame(human_feature_means,features_means)
  
  ggscatter(df,x='human_feature_means',y='features_means',cor.coef=T,cor.method = 'kendall')
  
  df_all <- rbind(human %>% mutate(species='human'),
                  translated_primate%>% mutate(species='translated primates'))
  df_all <- df_all %>% gather('feature','value',-protected,-vaccinated,-species)
  features_to_keep <- protection_feats_human$feature
  ggplot(df_all %>% filter(feature %in% features_to_keep) %>% filter(protected==1),
         aes(x=value,y=feature,fill=species)) +
    geom_density_ridges(alpha = 0.8,
                        color='black')
  # pairwise.test = df_all %>% filter(feature %in% features_to_keep) %>% filter(protected==1)%>%
  #   select(-protected,-vaccinated)  %>%
  #   group_by(feature) %>%
  #   wilcox_test(value ~ species) %>%
  #   adjust_pvalue(method = 'bonferroni') %>% ungroup()
  # ggboxplot(df_all %>% filter(feature %in% features_to_keep),
  #        x='species',y='value',color='species',add='jitter') +
  # facet_wrap(~feature) +
  # #stat_compare_means()+
  #   stat_pvalue_manual(pairwise.test,
  #                      label = "p.adj = {scales::pvalue(p.adj)}",y.position = 2.5)
}

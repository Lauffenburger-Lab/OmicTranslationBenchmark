library(tidyverse)
library(reshape2)
library(gg3D)
library(ggsignif)
library(ggpubr)
library(caret)

### Load performance results from different approaches---------
CPAbased_model <- data.table::fread('results/10foldvalidationResults_lungs.csv') %>% 
  select(-V1) %>%
  rownames_to_column('fold')
autoencoders_model <- data.table::fread('results/10foldvalidationResults_lungs_nocpa.csv') %>% 
  select(-V1)%>%
  rownames_to_column('fold')
CPAbased_homologues_model <- data.table::fread('results/10foldvalidationResults_lungs_homologues.csv') %>%
  select(-V1)%>%
  rownames_to_column('fold')
DCS_models <- data.table::fread('results/10foldvalidationResults_lungs_DCS_homologues.csv') %>% 
  select(-V1) %>%
  rownames_to_column('fold')
TransCompR_models <- data.table::fread('results/10foldvalidationResults_lungs_TransCompR.csv') %>% 
  select(-V1) %>%
  rownames_to_column('fold')

# Gather columns into long format
CPAbased_model <- CPAbased_model %>% gather('metric','value',-fold) %>% 
  mutate(task = ifelse(grepl('Translation',metric) | grepl('translation',metric) | grepl('_to_',metric) | grepl('traslation',metric),
                       'translation','other')) %>%
  mutate(task = ifelse(task=='other' & grepl('Species',metric),'species',
                       ifelse(task=='other' & grepl('cell',tolower(metric)),'cell-type',
                              ifelse(task=='other' & grepl('r2',metric),'reconstruction',
                                     ifelse(metric== 'F1' | metric== 'Accuracy' |metric== 'Precision' |metric== 'Recall',
                                            'fibrosis',task))))) %>%  
  mutate(task = ifelse(grepl('basal',tolower(metric)),'basal species',task)) %>%
  mutate(metric = ifelse(metric=='TranslationF1','KNN F1',ifelse(metric=='TranslationAccuracy','KNN accuracy',metric))) %>%
  mutate(metric = ifelse(grepl('species translation',metric),tolower(gsub(' species translation', "", metric)),metric)) %>%
  mutate(metric = ifelse(metric=='F1 species traslation','F1',metric)) %>%
  mutate(direction = ifelse(grepl('human_to_mouse',metric),'to mouse',
                            ifelse(grepl('mouse_to_human',metric),'to human',NA))) %>%
  mutate(direction = ifelse(metric=='r2_mu_human' |metric=='r2_var_human','to human',
                            ifelse(metric=='r2_mu_mouse' |metric=='r2_var_mouse','to mouse',direction))) %>%
  mutate(metric = ifelse(grepl('r2_mu',metric),'R2 of mean',metric)) %>%
  mutate(metric = ifelse(grepl('r2_var',metric),'R2 of variance',metric)) %>%
  mutate(metric = ifelse(task=='cell-type','F1',metric)) %>%
  mutate(metric = ifelse(task=='basal species','F1',metric)) %>%
  mutate(metric = ifelse(task=='fibrosis',tolower(metric),metric)) %>%
  mutate(metric = ifelse(metric=='f1','F1',metric)) 
CPAbased_homologues_model <- CPAbased_homologues_model %>% gather('metric','value',-fold) %>% 
  mutate(task = ifelse(grepl('Translation',metric) | grepl('translation',metric) | grepl('_to_',metric) | grepl('traslation',metric),
                       'translation','other')) %>%
  mutate(task = ifelse(task=='other' & grepl('Species',metric),'species',
                       ifelse(task=='other' & grepl('cell',tolower(metric)),'cell-type',
                              ifelse(task=='other' & grepl('r2',metric),'reconstruction',
                                     ifelse(metric== 'F1' | metric== 'Accuracy' |metric== 'Precision' |metric== 'Recall',
                                            'fibrosis',task))))) %>%  
  mutate(task = ifelse(grepl('basal',tolower(metric)),'basal species',task)) %>%
  mutate(metric = ifelse(metric=='TranslationF1','KNN F1',ifelse(metric=='TranslationAccuracy','KNN accuracy',metric))) %>%
  mutate(metric = ifelse(grepl('species translation',metric),tolower(gsub(' species translation', "", metric)),metric)) %>%
  mutate(metric = ifelse(metric=='F1 species traslation','F1',metric)) %>%
  mutate(direction = ifelse(grepl('human_to_mouse',metric),'to mouse',
                            ifelse(grepl('mouse_to_human',metric),'to human',NA))) %>%
  mutate(direction = ifelse(metric=='r2_mu_human' |metric=='r2_var_human','to human',
                            ifelse(metric=='r2_mu_mouse' |metric=='r2_var_mouse','to mouse',direction))) %>%
  mutate(metric = ifelse(grepl('r2_mu',metric),'R2 of mean',metric)) %>%
  mutate(metric = ifelse(grepl('r2_var',metric),'R2 of variance',metric)) %>%
  mutate(metric = ifelse(task=='cell-type','F1',metric)) %>%
  mutate(metric = ifelse(task=='basal species','F1',metric)) %>%
  mutate(metric = ifelse(task=='fibrosis',tolower(metric),metric)) %>%
  mutate(metric = ifelse(metric=='f1','F1',metric)) 
#autoencoders_model <- autoencoders_model
DCS_models <-  DCS_models %>% gather('metric','value',-fold) %>% 
  mutate(task = ifelse(grepl('Translation',metric) | grepl('translation',metric) | grepl('_to_',metric) | grepl('traslation',metric),
                       'translation','other')) %>%
  mutate(task = ifelse(task=='other' & grepl('Species',metric),'species',
                       ifelse(task=='other' & grepl('cell',tolower(metric)),'cell-type',
                              ifelse(task=='other' & grepl('r2',metric),'reconstruction',
                                     ifelse(metric== 'F1' | metric== 'Accuracy' |metric== 'Precision' |metric== 'Recall',
                                            'fibrosis',task))))) %>%  
  mutate(direction = ifelse(grepl('human_to_mouse',metric),'to mouse',
                            ifelse(grepl('mouse_to_human',metric),'to human',NA))) %>%
  mutate(direction = ifelse(metric=='r2_mu_human' |metric=='r2_var_human','to human',
                            ifelse(metric=='r2_mu_mouse' |metric=='r2_var_mouse','to mouse',direction))) %>%
  mutate(metric = ifelse(grepl('r2_mu',metric),'R2 of mean',metric)) %>%
  mutate(metric = ifelse(grepl('r2_var',metric),'R2 of variance',metric))
TransCompR_models <- TransCompR_models%>% gather('metric','value',-fold) %>% 
  mutate(task = ifelse(grepl('Translation',metric) | grepl('translation',metric) | grepl('_to_',metric) | grepl('traslation',metric),
                       'translation','other')) %>%
  mutate(task = ifelse(task=='other' & grepl('Species',metric),'species',
                       ifelse(task=='other' & grepl('cell',tolower(metric)),'cell-type',
                              ifelse(task=='other' & grepl('r2',metric),'reconstruction',
                                     ifelse(metric== 'F1' | metric== 'Accuracy' |metric== 'Precision' |metric== 'Recall',
                                            'fibrosis',task))))) %>%  
  mutate(task = ifelse(grepl('basal',tolower(metric)),'basal species',task)) %>%
  mutate(metric = ifelse(metric=='TranslationF1','KNN F1',ifelse(metric=='TranslationAccuracy','KNN accuracy',metric))) %>%
  mutate(metric = ifelse(grepl('species translation',metric),tolower(gsub(' species translation', "", metric)),metric)) %>%
  mutate(metric = ifelse(metric=='F1 species traslation','F1',metric)) %>%
  mutate(direction = ifelse(grepl('human_to_mouse',metric),'to mouse',
                            ifelse(grepl('mouse_to_human',metric),'to human',NA))) %>%
  mutate(direction = ifelse(metric=='r2_mu_human' |metric=='r2_var_human','to human',
                            ifelse(metric=='r2_mu_mouse' |metric=='r2_var_mouse','to mouse',direction))) %>%
  mutate(metric = ifelse(grepl('r2_mu',metric),'R2 of mean',metric)) %>%
  mutate(metric = ifelse(grepl('r2_var',metric),'R2 of variance',metric)) %>%
  mutate(metric = ifelse(task=='cell-type','F1',metric)) %>%
  mutate(metric = ifelse(task=='basal species','F1',metric)) %>%
  mutate(metric = ifelse(task=='fibrosis',tolower(metric),metric)) %>%
  mutate(metric = ifelse(metric=='f1','F1',metric)) 

# assign model-type label
CPAbased_homologues_model <- CPAbased_homologues_model %>% mutate(genes = 'homologues') %>% mutate(model='CPA-based homologues')
CPAbased_model <- CPAbased_model %>% mutate(genes = 'all genes') %>% mutate(model='CPA-based all genes')
DCS_models <- DCS_models%>% mutate(genes = 'homologues') %>% mutate(model='DCS modified v2')
TransCompR_models <- TransCompR_models %>% mutate(genes = 'homologues') %>% mutate(model='TransCompR')
  
# Visualize------------
results <- rbind(CPAbased_homologues_model,CPAbased_model,DCS_models,TransCompR_models)
results$model <- factor(results$model,levels = c('CPA-based all genes','CPA-based homologues','DCS modified v2','TransCompR'))

# Visualize performance for predicting counts
my_comparisons <- list( c('CPA-based all genes', 'CPA-based homologues'),
                        c('DCS modified v2', 'CPA-based homologues'),
                        c('CPA-based all genes', 'DCS modified v2'),
                        c('CPA-based all genes','TransCompR'))

p1 <- ggboxplot(results %>% filter(grepl('R2 of mean',metric)) %>% mutate(metric=gsub('R2 of','counts',metric)),
          x='model',y='value',color = 'model')  + xlab('')+ ylab(expression('R'^2))+
  ggtitle("Predicting the per gene mean of the counts distribution")+
  scale_y_continuous(breaks = seq(-1,1,0.2),minor_breaks = waiver())+
  geom_hline(yintercept = 0,lty='dashed',linewidth=1)+
  facet_wrap(vars(task,direction)) +
  stat_compare_means(label = 'p.format',
                     method = 'wilcox.test',
                     #step.increase = 0.02,
                     tip.length=0.05,
                     size =6,
                     aes(group=model),
                     comparisons = my_comparisons) +
  theme(panel.grid.major = element_line(color = "gray70", linewidth = 0.5, linetype = "dashed"),
        panel.grid.minor =  element_line(color = "gray70", linewidth = 0.5, linetype = "dashed"),
        text = element_text(family = 'Arial',size=23),
        plot.title = element_text(hjust = 0.5,size=23),
        legend.title = element_blank(),
        axis.text.x = element_blank())
print(p1)
ggsave('results/10fold_Rsquare_mean_performance_comparison.eps', 
       device = cairo_ps,
       scale = 1,
       width = 12,
       height = 12,
       units = "in",
       dpi = 600)
p2 <- ggboxplot(results %>% filter(grepl('R2 of variance',metric)) %>% mutate(metric=gsub('R2 of','counts',metric)),
               x='model',y='value',color = 'model')  +xlab('')+ ylab(expression('R'^2))+
  ggtitle("Predicting the per gene variance of the counts distribution")+
  scale_y_continuous(breaks = seq(-10,1,1),minor_breaks = waiver())+
  facet_wrap(vars(task,direction)) +
  stat_compare_means(label = 'p.format',
                     method = 'wilcox.test',
                     tip.length=0.05,
                     size =6,
                     aes(group=model),
                     comparisons = my_comparisons) +
  theme(panel.grid.major = element_line(color = "gray70", linewidth = 0.5, linetype = "dashed"),
        panel.grid.minor =  element_line(color = "gray70", linewidth = 0.5, linetype = "dashed"),
        text = element_text(family = 'Arial',size=23),
        plot.title = element_text(hjust = 0.5,size=23),
        legend.title = element_blank(),
        axis.text.x = element_blank())
print(p2)
ggsave('results/10fold_Rsquare_variance_performance_comparison.eps', 
       device = cairo_ps,
       scale = 1,
       width = 12,
       height = 12,
       units = "in",
       dpi = 600)

classification_results <- results %>% filter(grepl('F1',metric)) %>% filter(metric!='F1 cell-type species traslation') %>%
  mutate(task = ifelse(metric=='KNN F1','KNN translation',task)) %>%
  mutate(task = ifelse(task=='translation','species translation',task)) #%>% filter(task!='basal species')
classification_results$task <- factor(classification_results$task,
                                      levels = c('fibrosis','species translation','KNN translation',
                                                 'species','cell-type',
                                                 'basal species'))
my_class_comparisons <- list( c('CPA-based all genes', 'CPA-based homologues'),
                              c('CPA-based all genes','TransCompR'),
                              c('CPA-based homologues','TransCompR'))
p3 <- ggboxplot(classification_results %>% filter(task!='basal species'),
                x='model',y='value',color = 'model')  + xlab('')+ ylab('F1 score')+
  scale_y_continuous(breaks = seq(0.25,1,0.25),minor_breaks = waiver())+
  facet_wrap(~task,ncol = 5)+
  stat_compare_means(label = 'p.format',
                     method = 'wilcox.test',
                     tip.length=0.05,
                     size =6,
                     aes(group=model),
                     comparisons = my_class_comparisons) +
  theme(panel.grid.major = element_line(color = "gray70", size = 0.5, linetype = "dashed"),
        panel.grid.minor =  element_line(color = "gray70", size = 0.5, linetype = "dashed"),
        axis.text.x = element_blank(),
        text = element_text(family = 'Arial',size=24))
print(p3)
ggsave('results/10fold_classification_performance_comparison.eps', 
       device = cairo_ps,
       scale = 1,
       width = 12,
       height = 6,
       units = "in",
       dpi = 600)

# p <- ggarrange(plotlist=list(p1,p2),ncol=1,nrow=2,common.legend = TRUE,legend = 'top')
# annotate_figure(p, top = text_grob("10-fold cross validation performance", 
#                                    color = "black",face = 'plain', size = 14))
# ggsave('results/10fold_Rsquare_performance_comparison.eps', 
#        device = cairo_ps,
#        scale = 1,
#        width = 16,
#        height = 16,
#        units = "in",
#        dpi = 600)
# png(file="results/10fold_Rsquare_performance_comparison.png",width=18,height=9,units = "in",res=600)
# annotate_figure(p, top = text_grob("10-fold cross validation performance", 
#                                    color = "black",face = 'plain', size = 16))
# dev.off()

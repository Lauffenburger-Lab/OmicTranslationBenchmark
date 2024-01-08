library(tidyverse)
library(ggplot2)
library(ggpubr)
library(ggridges)
library(rstatix)
library(patchwork)

# Load results--------
cpa_intermediate <- data.table::fread('../serology/results_intermediate_encoders/10foldvalidation_wholeModel_32dim2000ep_serology.csv')
colnames(cpa_intermediate)[1] <- 'fold'
# cpa_intermediate <- cpa_intermediate[,1:17]
cpa_intermediate <- cpa_intermediate %>% mutate(model='CPA with intermediate encoders')

cpa <- data.table::fread('../serology/results/10foldvalidation_wholeModel_32dim2000ep_serology.csv')
colnames(cpa)[1] <- 'fold'
cpa <- cpa %>% mutate(model='CPA-based model')

all_results <- rbind(cpa_intermediate,cpa)
results_f1 <- all_results%>% select(F1Species,F1_global,F1Protection,KNNTranslationF1,ClassifierTranslationF1,model)
results_f1 <- results_f1 %>% gather('task','F1',-model) %>% 
  mutate(task = ifelse(grepl('Species',task),'species',
                       ifelse(grepl('Protection',task),'protection',
                              ifelse(grepl('global',task),'vaccination',
                                     ifelse(grepl('KNN',task),'KNN species translation',
                                            'species translation')))))
colnames(results_f1)[3] <- 'value'
results_f1 <- results_f1 %>% mutate(metric='F1')
results_f1$task <- factor(results_f1$task,
                          levels = c('protection',
                                     'vaccination',
                                     'species',
                                     'species translation',
                                     'KNN species translation'))
results_acc <- all_results %>% select(AccuracySpecies,Accuracy_global,AccProtection,model)
results_acc <- results_acc %>% gather('task','Accuracy',-model) %>% 
  mutate(task = ifelse(grepl('Species',task),'species',
                       ifelse(grepl('Protection',task),'protection',
                              ifelse(grepl('global',task),'vaccination',
                                     ifelse(grepl('KNN',task),'KNN species translation',
                                            'species translation')))))
colnames(results_acc)[3] <- 'value'
results_acc <- results_acc %>% mutate(metric='Accuracy')
results_acc$task <- factor(results_acc$task,
                           levels = c('protection',
                                      'vaccination',
                                      'species',
                                      'species translation',
                                      'KNN species translation'))  

all_results_class <- rbind(results_f1,results_acc)

all_results <- all_results %>% gather('metric','value',-fold,-model)
all_results$model <- factor(all_results$model,levels=c('CPA-based model',
                                                       'CPA with intermediate encoders'))
all_results_class$model <- factor(all_results_class$model,levels=c('CPA-based model',
                                                       'CPA with intermediate encoders'))

all_results_recon <- all_results %>% filter(grepl('recon',metric))
all_results_recon <- all_results_recon %>% mutate(species=ifelse(grepl('human',metric),'human','primates')) %>%
  mutate(metric=ifelse(grepl('pear',metric),'per feature pearson','per feature R\u00B2'))

# Visualize comparison
p1a <-  ggboxplot(all_results_recon %>% filter(metric=='per feature pearson'),
                x='model',y='value',color='model',add='jitter')+
  ylab('per feature pearson')+
  facet_wrap(~species) +
  theme(text = element_text(family = 'Arial',size=20),
        axis.text.x = element_blank(),
        axis.title.x=element_blank())+
  stat_compare_means(method = 'wilcox.test',
                     tip.length=0.05,
                     size =5,
                     aes(group=model))
p1b <-  ggboxplot(all_results_recon %>% filter(metric!='per feature pearson'),
                 x='model',y='value',color='model',add='jitter')+
  ylab('per feature R\u00B2')+
  facet_wrap(~species) +
  theme(text = element_text(family = 'Arial',size=20),
        axis.text.x = element_blank(),
        axis.title.x=element_blank(),
        legend.position = 'none')+
  stat_compare_means(method = 'wilcox.test',
                     tip.length=0.05,
                     size =5,
                     aes(group=model))
p1 <- p1a/p1b
print(p1)
ggsave(
  'suppl_fig28_1.eps', 
  plot=p1,
  device = cairo_ps,
  scale = 1,
  width = 16,
  height = 9,
  units = "in",
  dpi = 600,
)


p2a <-  ggboxplot(all_results_class %>% filter(metric=='F1'),
                  x='model',y='value',color='model',add='jitter')+
  ylim(c(0.5,1.1))+
  geom_hline(yintercept = 0.5,linetype='dashed',linewidth=1)+
  ylab('F1')+
  facet_wrap(~task) +
  theme(text = element_text(family = 'Arial',size=20),
        axis.text.x = element_blank(),
        axis.title.x=element_blank(),
        legend.position = 'none')+
  stat_compare_means(method = 'wilcox.test',label.y = 1,
                     tip.length=0.05,
                     size =5,
                     aes(group=model))

p2b <-  ggboxplot(all_results_class %>% filter(metric!='F1'),
                  x='model',y='value',color='model',add='jitter')+
  ylim(c(0.5,1))+
  geom_hline(yintercept = 0.5,linetype='dashed',linewidth=1)+
  ylab('Accuracy')+
  facet_wrap(~task) +
  theme(text = element_text(family = 'Arial',size=20),
        axis.text.x = element_blank(),
        axis.title.x=element_blank(),
        legend.position = 'none')+
  stat_compare_means(method = 'wilcox.test',
                     tip.length=0.05,
                     size =5,
                     aes(group=model))

p2 <- p2a/p2b
ggsave(
  'suppl_fig28_2.eps', 
  plot=p2,
  device = cairo_ps,
  scale = 1,
  width = 16,
  height = 9,
  units = "in",
  dpi = 600,
)
# png('../article_supplementary_info/suppl_fig12.png',width = 16,height = 9,units = 'in',res = 600)
# p1+p2
# dev.off()

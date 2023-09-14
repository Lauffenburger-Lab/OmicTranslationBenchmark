library(tidyverse)
library(caret)
library(ggplot2)
library(ggpubr)
library(ggpattern)
library(corrplot)
library(reshape2)

### Load results---------------
results <- data.table::fread('../results/SameCellimputationModel/translation_results.csv')
results <- results %>% filter(set=='validation') %>% select(-set)
results <- distinct(results)
results <- results %>% mutate(mean_r = 0.5*(model_pearson2to1+model_pearson1to2)) %>%
  mutate(mean_recon = 0.5*(recon_pear_1 +recon_pear_2)) %>% select(model,cell,mean_r,mean_recon,fold,iteration) %>%
  unique()

### Visualize results-------------
p <- ggboxplot(results,x='model',y='mean_r',color='model',outlier.shape = 19) +
  ylab('Average pearson`s correlation for translation')+
  scale_y_continuous(breaks = seq(0,0.8,0.1),limits = c(-0.1,0.8))+
  geom_hline(yintercept = 0,linewidth=0.75,color='black',linetype = 'dashed')+
  stat_compare_means(method = 'wilcox.test',label.y = 0.75,size=10)+
  facet_wrap(~cell,ncol = 5) +
  theme(text = element_text(size=30,family = 'Arial'),
        panel.grid.major.y = element_line(linewidth = 1,linetype = 'dashed'))
print(p)
ggsave('../results/SameCellimputationModel/fig1f_same_cell_imputation_v1.png',
       plot = p,
       width = 24,
       height = 24,
       units = 'in',
       dpi = 600)

median_cell_values <- aggregate(mean_r ~ cell, results, mean)
results$cell <- factor(results$cell, levels = median_cell_values$cell[order(-median_cell_values$mean_r)])
ggboxplot(results,x='model',y='mean_r',color='cell') +
  ylab('Average pearson`s correlation for translation')+
  scale_y_continuous(breaks = seq(0.0,0.8,0.1),limits = c(-0.1,0.8))+
  geom_hline(yintercept = 0,linewidth=0.75,color='black',linetype = 'dashed')+
  theme(text = element_text(size=24,family = 'Arial'),
        panel.grid.major.y = element_line(linewidth = 1,linetype = 'dashed'),
        legend.position = 'right')
ggsave('../results/SameCellimputationModel/fig1f_same_cell_imputation_v2.png',
       width = 12,
       height = 9,
       units = 'in',
       dpi = 600)

### All together
ggboxplot(results,x='model',y='mean_r',color = 'model') +
  ylab('Average pearson`s correlation for translation')+
  scale_y_continuous(breaks = seq(0.0,0.8,0.1),limits = c(-0.1,0.8))+
  geom_hline(yintercept = 0,linewidth=0.75,color='black',linetype = 'dashed')+
  stat_compare_means(comparisons = list(c('model','shuffled')),method = 'wilcox.test',size=6,label.y = 0.75)+
  theme(text = element_text(size=24,family = 'Arial'),
        panel.grid.major.y = element_line(linewidth = 1,linetype = 'dashed'),
        legend.position = 'none')
ggsave('../results/SameCellimputationModel/fig1f_same_cell_imputation_v0.png',
       width = 12,
       height = 9,
       units = 'in',
       dpi = 600)

### Same for reconstruction
median_cell_values <- aggregate(mean_recon ~ cell, results, mean)
results$cell <- factor(results$cell, levels = median_cell_values$cell[order(-median_cell_values$mean_recon)])
p <- ggboxplot(results,x='model',y='mean_recon',color='model',outlier.shape = 19) +
  ylab('Average pearson`s correlation for translation')+
  scale_y_continuous(breaks = seq(0,1,0.1),limits = c(-0.1,1.1))+
  geom_hline(yintercept = 0,linewidth=0.75,color='black',linetype = 'dashed')+
  stat_compare_means(method = 'wilcox.test',label.y = 0.85,size=10)+
  facet_wrap(~cell,ncol = 5) +
  theme(text = element_text(size=30,family = 'Arial'),
        panel.grid.major.y = element_line(linewidth = 1,linetype = 'dashed'))
print(p)
ggsave('../results/SameCellimputationModel/fig1f_same_cell_imputation_recon_v1.png',
       plot = p,
       width = 24,
       height = 18,
       units = 'in',
       dpi = 600)

median_cell_values <- aggregate(mean_recon ~ cell, results, mean)
results$cell <- factor(results$cell, levels = median_cell_values$cell[order(-median_cell_values$mean_recon)])
ggboxplot(results,x='model',y='mean_recon',color='cell') +
  ylab('Average pearson`s correlation for translation')+
  scale_y_continuous(breaks = seq(0.0,0.8,0.1),limits = c(-0.1,0.8))+
  geom_hline(yintercept = 0,linewidth=0.75,color='black',linetype = 'dashed')+
  theme(text = element_text(size=24,family = 'Arial'),
        panel.grid.major.y = element_line(linewidth = 1,linetype = 'dashed'),
        legend.position = 'right')
ggsave('../results/SameCellimputationModel/fig1f_same_cell_imputation_recon_v2.png',
       width = 12,
       height = 9,
       units = 'in',
       dpi = 600)

### All together
ggboxplot(results,x='model',y='mean_recon',color = 'model') +
  ylab('Average pearson`s correlation for translation')+
  scale_y_continuous(breaks = seq(0.0,1,0.1),limits = c(-0.1,1))+
  geom_hline(yintercept = 0,linewidth=0.75,color='black',linetype = 'dashed')+
  stat_compare_means(comparisons = list(c('model','shuffled')),method = 'wilcox.test',size=6,label.y = 0.9)+
  theme(text = element_text(size=24,family = 'Arial'),
        panel.grid.major.y = element_line(linewidth = 1,linetype = 'dashed'),
        legend.position = 'none')
ggsave('../results/SameCellimputationModel/fig1f_same_cell_imputation_recon_v0.png',
       width = 12,
       height = 9,
       units = 'in',
       dpi = 600)

library(tidyverse)
library(caret)
library(ggplot2)
library(ggpubr)
library(ggpattern)
library(corrplot)
library(reshape2)

# results_cpa <- data.frame()
# for(cell in c("PC3","HT29","MCF7","A549","NPC","HEPG2","A375","YAPC","U2OS","MCF10A","HA1E","HCC515","ASC","VCAP","HUVEC","HELA")){
#   tmp <- data.table::fread(paste0('../results/SameCellimputationModel/CPA/translation_results_',cell,'.csv'))
#   colnames(tmp)[1] <- 'fold'
#   results_cpa <- rbind(results_cpa,tmp)
# }
# results_cpa <- distinct(results_cpa)
# results_cpa <- results_cpa %>% mutate(model='AutoTransOp v2')
# data.table::fwrite(results_cpa,'../results/SameCellimputationModel/CPA/translation_results.csv')
## Load results---------------
# results <- data.frame()
# for(cell in c("PC3","HT29","MCF7","A549","NPC","HEPG2","A375","YAPC","U2OS","MCF10A","HA1E","HCC515","ASC","VCAP","HUVEC","HELA")){
#   tmp <- data.table::fread(paste0('../results/SameCellimputationModel/translation_results_',cell,'.csv'))
#   colnames(tmp)[1] <- 'fold'
#   results <- rbind(results,tmp)
# }
# results <- distinct(results)
# results <- results %>% mutate(model=ifelse(model=='shuffled','shuffled v1','AutoTransOp v1'))
# data.table::fwrite(results,'../results/SameCellimputationModel/translation_results.csv')
results <- data.table::fread('../results/SameCellimputationModel/translation_results.csv')
results <- results %>% filter(set=='validation') %>% select(-set)
results <- distinct(results)
results <- results %>% mutate(mean_r = 0.5*(model_pearson2to1+model_pearson1to2)) %>%
  mutate(mean_recon = 0.5*(recon_pear_1 +recon_pear_2)) %>% select(model,cell,mean_r,mean_recon,fold,iteration) %>%
  unique()

### Load results CPA---------------
results_cpa <- data.table::fread('../results/SameCellimputationModel/CPA/translation_results.csv')
results_cpa <- results_cpa %>% filter(set=='validation') %>% select(-set)
results_cpa <- distinct(results_cpa)
results_cpa <- results_cpa %>% mutate(mean_r = 0.5*(model_pearson2to1+model_pearson1to2)) %>%
  mutate(mean_recon = 0.5*(recon_pear_1 +recon_pear_2)) %>% select(model,cell,mean_r,mean_recon,fold,iteration) %>%
  unique()

all_results <- rbind(results,results_cpa)
# all_results <- all_results %>% mutate(model = ifelse(model=='shuffled','shuffled v1',model))
all_results$model <- factor(all_results$model,levels = c('AutoTransOp v1','AutoTransOp v2','shuffled v1'))

### Visualize results-------------
p <- ggboxplot(all_results,x='model',y='mean_r',color='model',outlier.shape = 19) +
  ylab('Average pearson`s correlation for translation')+
  scale_y_continuous(breaks = seq(0,0.8,0.1),limits = c(-0.1,0.8))+
  geom_hline(yintercept = 0,linewidth=0.75,color='black',linetype = 'dashed')+
  stat_compare_means(comparisons =list(c('AutoTransOp v1','AutoTransOp v2')) ,method = 'wilcox.test',label.y = 0.6,size=6)+
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

median_cell_values <- aggregate(mean_r ~ cell, all_results, mean)
all_results$cell <- factor(all_results$cell, levels = median_cell_values$cell[order(-median_cell_values$mean_r)])
# ggboxplot(all_results,x='model',y='mean_r',color='cell',
#           width = 2.5,add='jitter',
#           bxp.errorbar = T,bxp.errorbar.width = 1.25,size = 0.75,position = position_dodge(width = 1)) +
#   ylab('Average pearson`s correlation for translation')+
#   scale_y_continuous(breaks = seq(0.0,0.8,0.1),limits = c(-0.1,0.8))+
#   geom_hline(yintercept = 0,linewidth=0.75,color='black',linetype = 'dashed')+
#   theme(text = element_text(size=24,family = 'Arial'),
#         panel.grid.major.y = element_line(linewidth = 1,linetype = 'dashed'),
#         legend.position = 'right')
p_trans <- ggplot(all_results, aes(x = model, y = mean_r, color = cell)) +
  geom_boxplot(position = position_dodge(width = 1.05), width = 0.95,size=0.3,na.rm = T) + 
  geom_jitter(position = position_jitterdodge(dodge.width = 1.05, jitter.width = 0.1), size = 1) +
  ylab('average pearson`s r') +
  ggtitle('Translation')+
  scale_y_continuous(breaks = seq(0.0, 0.8, 0.1), limits = c(-0.1, 0.85)) +
  theme_pubr(base_size = 24,base_family = 'Arial')+
  theme(text = element_text(size = 24, family = 'Arial'),
        panel.grid.major.y = element_line(linewidth = 1, linetype = 'dashed'),
        legend.position = 'bottom',
        plot.title = element_text(hjust = 0.5)) + coord_flip()
ggsave('../results/SameCellimputationModel/fig1f_same_cell_imputation_v2.png',
       width = 16,
       height = 8,
       units = 'in',
       dpi = 600)
setEPS()
postscript('../results/SameCellimputationModel/fig1f_same_cell_imputation_v2.eps',width = 8,height = 8)
ggplot(all_results, aes(x = model, y = mean_r, color = cell)) +
  geom_boxplot(position = position_dodge(width = 1.05), width = 0.95,size=0.3,na.rm = T) + 
  geom_jitter(position = position_jitterdodge(dodge.width = 1.05, jitter.width = 0.1), size = 1) +
  ylab('average pearson`s r') +
  ggtitle('Translation')+
  scale_y_continuous(breaks = seq(0.0, 0.8, 0.1), limits = c(-0.1, 0.85)) +
  theme_pubr(base_size = 24,base_family = 'Arial')+
  theme(text = element_text(size = 24, family = 'Arial'),
        panel.grid.major.y = element_line(linewidth = 1, linetype = 'dashed'),
        legend.position = 'none',
        plot.title = element_text(hjust = 0.5)) + coord_flip()
dev.off()

### All together
ggboxplot(all_results,x='model',y='mean_r',color = 'model',add = 'jitter') +
  ylab('Average pearson`s correlation for translation')+
  scale_y_continuous(breaks = seq(0.0,0.8,0.1),limits = c(-0.1,0.85))+
  geom_hline(yintercept = 0,linewidth=0.75,color='black',linetype = 'dashed')+
  stat_compare_means(ref.group = 'shuffled v1',method = 'wilcox.test',size=6,label.y = 0.75)+
  theme(text = element_text(size=24,family = 'Arial'),
        panel.grid.major.y = element_line(linewidth = 1,linetype = 'dashed'),
        legend.position = 'none')
ggsave('../results/SameCellimputationModel/fig1f_same_cell_imputation_v0.png',
       width = 12,
       height = 9,
       units = 'in',
       dpi = 600)
res_trans <- all_results %>% mutate(type = 'Translation')

### Same for reconstruction------------------------------
all_results <- rbind(results,results_cpa)
# all_results <- all_results %>% mutate(model = ifelse(model=='shuffled','shuffled v1',model))
all_results$model <- factor(all_results$model,levels = c('AutoTransOp v1','AutoTransOp v2','shuffled v1'))
median_cell_values <- aggregate(mean_recon ~ cell, all_results, mean)
all_results$cell <- factor(all_results$cell, levels = median_cell_values$cell[order(-median_cell_values$mean_recon)])
p <- ggboxplot(all_results,x='model',y='mean_recon',color='model',outlier.shape = 19) +
  ylab('Average pearson`s correlation for translation')+
  scale_y_continuous(breaks = seq(0,1,0.1),limits = c(-0.1,1.1))+
  geom_hline(yintercept = 0,linewidth=0.75,color='black',linetype = 'dashed')+
  stat_compare_means(ref.group = 'shuffled',method = 'wilcox.test',label.y = 0.85,size=6)+
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

median_cell_values <- aggregate(mean_recon ~ cell, all_results, mean)
all_results$cell <- factor(all_results$cell, levels = median_cell_values$cell[order(-median_cell_values$mean_recon)])
p_recon <- ggplot(all_results, aes(x = model, y = mean_recon, color = cell)) +
  geom_boxplot(position = position_dodge(width = 1.05), width = 0.95,size=0.6,na.rm = T) + 
  geom_jitter(data= all_results, aes(x = model, y = mean_recon, color = cell),
              position = position_jitterdodge(dodge.width = 1.05, jitter.width = 0.3), size = 1) +
  ylab('average pearson`s r') +
  xlab('')+
  ggtitle('Reconstruction')+
  scale_y_continuous(breaks = seq(0.0, 1.0, 0.1), limits = c(0.0, 1.0)) +
  theme_pubr(base_size = 24,base_family = 'Arial')+
  theme(text = element_text(size = 24, family = 'Arial'),
        panel.grid.major.y = element_line(linewidth = 1, linetype = 'dashed'),
        legend.position = 'bottom',
        plot.title = element_text(hjust = 0.5)) + coord_flip()
ggsave('../results/SameCellimputationModel/fig1f_same_cell_imputation_recon_v2.png',
       plot = p_recon,
       width = 16,
       height = 8,
       units = 'in',
       dpi = 600)
setEPS()
postscript('../results/SameCellimputationModel/fig1f_same_cell_imputation_recon_v2.eps',width = 8,height = 8)
ggplot(all_results, aes(x = model, y = mean_recon, color = cell)) +
  geom_boxplot(position = position_dodge(width = 1.05), width = 0.95,size=0.6,na.rm = T) + 
  geom_jitter(data= all_results, aes(x = model, y = mean_recon, color = cell),
              position = position_jitterdodge(dodge.width = 1.05, jitter.width = 0.3), size = 1) +
  ylab('average pearson`s r') +
  xlab('')+
  ggtitle('Reconstruction')+
  scale_y_continuous(breaks = seq(0.0, 1.0, 0.1), limits = c(0.0, 1.0)) +
  theme_pubr(base_size = 24,base_family = 'Arial')+
  theme(text = element_text(size = 24, family = 'Arial'),
        panel.grid.major.y = element_line(linewidth = 1, linetype = 'dashed'),
        legend.position = 'right',
        plot.title = element_text(hjust = 0.5)) + coord_flip()
dev.off()

all_results <- rbind(all_results %>% mutate(type = 'Reconstruction') %>% mutate(r= mean_recon) %>% select(-mean_r,-mean_recon),
                     res_trans %>% mutate(r= mean_r) %>% select(-mean_r,-mean_recon))
setEPS()
postscript('../results/SameCellimputationModel/fig1f.eps',width = 16,height = 8)
ggplot(all_results, aes(x = model, y = r, color = cell)) +
  geom_boxplot(position = position_dodge(width = 1.05), width = 0.95,size=0.6,na.rm = T) + 
  geom_jitter(data= all_results, aes(x = model, y = r, color = cell),
              position = position_jitterdodge(dodge.width = 1.05, jitter.width = 0.3), size = 1) +
  ylab('average pearson`s r') +
  xlab('model')+
  scale_y_continuous(breaks = seq(0.0, 1.0, 0.1), limits = c(-0.1, 1.0)) +
  facet_wrap(~type)+
  theme_pubr(base_size = 24,base_family = 'Arial')+
  theme(text = element_text(size = 24, family = 'Arial'),
        panel.grid.major.y = element_line(linewidth = 1, linetype = 'dashed'),
        legend.position = 'top') + coord_flip()
dev.off()
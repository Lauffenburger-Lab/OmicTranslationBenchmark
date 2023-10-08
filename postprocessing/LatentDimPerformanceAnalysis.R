library(tidyverse)
library(ggplot2)
library(ggpubr)
library(patchwork)

### Load performance results------------
landmarks <- data.table::fread('../results/LatentDimAnalysis/validation_results.csv') 
colnames(landmarks)[1] <- 'fold'
landmarks <- landmarks %>% mutate(feature_space = 'landmarks')
allgenes <- data.table::fread('../results/LatentDimAnalysis_allgenes/validation_results.csv') 
colnames(allgenes)[1] <- 'fold'
allgenes <- allgenes %>% mutate(feature_space = 'all genes')
results <- rbind(landmarks,allgenes)
results <- results %>% mutate(translation = 0.5*(model_pearson1to2+model_pearson2to1)) %>%
  mutate(reconstraction = 0.5*(recon_pear_2+recon_pear_1))
results <- results %>% select(fold,latent_dim,translation,reconstraction,ClassF1,ClassAcc,feature_space) %>% unique()
results <- results %>% gather('task','r',-fold,-latent_dim,-ClassF1,-ClassAcc,-feature_space)
results <- results %>% group_by(feature_space,task,latent_dim) %>% mutate(mean_r = mean(r)) %>% mutate(sd_r = sd(r)) %>% ungroup()
results <- results %>% gather('metric','value',-mean_r,-sd_r,-fold,-latent_dim,-feature_space,-task,-r)
results <- results %>% group_by(feature_space,metric,latent_dim) %>% mutate(mean_class = mean(value)) %>% mutate(sd_class = sd(value)) %>% ungroup()
results <- results %>% mutate(metric=ifelse(grepl('Acc',metric),'Accuracy','F1'))

### Visualize in one plot--------------
p1 <- ggplot(results,aes(x=latent_dim,y=mean_r,color=task)) +
  geom_point() + geom_line() + 
  geom_errorbar(aes(ymin = mean_r - sd_r/sqrt(10) , ymax = mean_r + sd_r/sqrt(10) ))+
  xlab('latent dimension')+
  ylab('pearson`s r') +
  ylim(c(0.3,0.9))+
  scale_x_continuous(n.breaks = 10)+
  theme_pubr(base_family = 'Arial',base_size = 22)+
  facet_wrap(~feature_space,scales = 'free_x')
print(p1)

p2 <- ggplot(results,aes(x=latent_dim,y=100*mean_class,color=metric)) +
  geom_point() + geom_line() + 
  geom_errorbar(aes(ymin = 100*(mean_class - sd_class/sqrt(10)) , ymax = 100*(mean_class + sd_class/sqrt(10) )))+
  xlab('latent dimension')+
  ylab('value (%)') +
  ylim(c(40,100))+
  scale_x_continuous(n.breaks = 10)+
  theme_pubr(base_family = 'Arial',base_size = 22)+
  facet_wrap(~feature_space,scales = 'free_x')
print(p2)

p <- p1/p2
print(p)
ggsave('../article_supplementary_info/latent_dim_performance_ha1e_pc3.png',
       plot = p,
       height = 16,
       width = 16,
       units = 'in',
       dpi=600)

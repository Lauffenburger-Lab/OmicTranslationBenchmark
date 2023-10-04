library(tidyverse)
library(ggplot2)
library(ggpubr)
library(ggridges)

### Load performance results for different prior regularization--------------
results_stateLoss_normal <- data.table::fread('../results/PriorLossAnalysis/translation_results_normal_mselike_U2OS.csv')
colnames(results_stateLoss_normal)[1] <- 'fold'
results_stateLoss_normal <- results_stateLoss_normal %>% mutate(approach='stateLoss') %>% mutate(distribution='normal')

results_stateLoss_uniform <- data.table::fread('../results/PriorLossAnalysis/translation_results_uniform_mselike_U2OS.csv')
colnames(results_stateLoss_uniform)[1] <- 'fold'
results_stateLoss_uniform <- results_stateLoss_uniform %>% mutate(approach='stateLoss') %>% mutate(distribution='uniform')

results_kld_normal <- data.table::fread('../results/PriorLossAnalysis/translation_results_KLD_normal_U2OS.csv')
colnames(results_kld_normal)[1] <- 'fold'
results_kld_normal <- results_kld_normal %>% mutate(approach='KL-div.') %>% mutate(distribution='normal')

results_discr_normal <- data.table::fread('../results/PriorLossAnalysis/translation_results_discr_normal_U2OS.csv')
colnames(results_discr_normal)[1] <- 'fold'
results_discr_normal <- results_discr_normal %>% mutate(approach='discriminator') %>% mutate(distribution='normal')

results_discr_uniform <- data.table::fread('../results/PriorLossAnalysis/translation_results_discr_uniform_U2OS.csv')
colnames(results_discr_uniform)[1] <- 'fold'
results_discr_uniform <- results_discr_uniform %>% mutate(approach='discriminator') %>% mutate(distribution='uniform')

results_current <- data.table::fread('../results/PriorLossAnalysis/translation_results_current_U2OS.csv')
colnames(results_current)[1] <- 'fold'
results_current <- results_current %>% mutate(approach='current') %>% mutate(distribution='normal')

results <- rbind(results_stateLoss_normal,
                 results_stateLoss_uniform,
                 results_kld_normal,
                 results_discr_normal,
                 results_discr_uniform,
                 results_current)
results <- results %>% mutate(avg_trans = 0.5*(model_pearson2to1+model_pearson1to2)) %>% mutate(avg_recon = 0.5*(recon_pear_1+recon_pear_2))
results <- results %>% group_by(beta,approach,distribution) %>% 
  mutate(mean_r_trans=mean(avg_trans))  %>%
  mutate(mean_r_recon=mean(avg_recon)) %>%
  mutate(sd_r_trans=sd(avg_trans))  %>%
  mutate(sd_r_recon=sd(avg_recon)) %>%
  ungroup()

### Visualize results--------------
p <- ggplot(results,
       aes(x=beta,y=mean_r_trans,color = approach)) +
  geom_point(aes(shape=distribution))+
  geom_line(aes(linetype=distribution),linewidth=1)+
  geom_errorbar(aes(ymin=mean_r_trans-sd_r_trans/sqrt(5),
                    ymax=mean_r_trans+sd_r_trans/sqrt(5)),
                width = 0.1)+
  geom_hline(yintercept = 0,linetype='dashed',color='black',linewidth=1)+
  scale_x_log10()+
  ylim(c(-0.2,0.85)) + 
  ylab('pearson`s r')+
  xlab('prior loss regularization')+
  ggtitle('Average performance in translation')+
  theme_pubr(base_family = 'Arial',base_size = 20)+
  theme(plot.title = element_text(hjust=0.5))
print(p)
ggsave('../article_supplementary_info/prior_vs_performance.eps',
       plot=p,
       device=cairo_ps,
       height = 12,
       width = 12,
       units = 'in',
       dpi=600)

p_zoomed <- ggplot(results,
            aes(x=beta,y=mean_r_trans,color = approach)) +
  geom_point(aes(shape=distribution))+
  geom_line(aes(linetype=distribution),linewidth=1)+
  geom_errorbar(aes(ymin=mean_r_trans-sd_r_trans/sqrt(5),
                    ymax=mean_r_trans+sd_r_trans/sqrt(5)),
                width = 0.1)+
  geom_hline(yintercept = 0,linetype='dashed',color='black',linewidth=1)+
  scale_x_log10()+
  ylim(c(0.75,0.81)) + 
  ylab('pearson`s r')+
  xlab('prior loss regularization')+
  ggtitle('Average performance in translation')+
  theme_pubr(base_family = 'Arial',base_size = 20)+
  theme(plot.title = element_blank(),
        #axis.line.x = element_blank(),
        #axis.ticks.x = element_blank(),
        axis.title.x = element_blank(),
        axis.title.y = element_blank(),
        legend.position = 'none')
print(p_zoomed)
ggsave('../article_supplementary_info/prior_vs_performance_zoomed.eps',
       plot=p_zoomed,
       device=cairo_ps,
       height = 9,
       width = 9,
       units = 'in',
       dpi=600)





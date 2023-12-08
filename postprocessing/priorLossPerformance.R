library(tidyverse)
library(ggplot2)
library(ggpubr)
library(ggridges)
library(patchwork)

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

results_noprior <- data.table::fread('../results/PriorLossAnalysis/translation_results_noprior_U2OS.csv')
colnames(results_noprior)[1] <- 'fold'
results_noprior <- results_noprior %>% mutate(approach='no prior') %>% mutate(distribution='free')

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

results_noprior <- results_noprior %>% mutate(avg_trans = 0.5*(model_pearson2to1+model_pearson1to2)) %>% mutate(avg_recon = 0.5*(recon_pear_1+recon_pear_2))
results_noprior <- results_noprior %>% 
  mutate(mean_r_trans=mean(avg_trans))  %>%
  mutate(mean_r_recon=mean(avg_recon)) %>%
  mutate(sd_r_trans=sd(avg_trans))  %>%
  mutate(sd_r_recon=sd(avg_recon))
results_noprior <- do.call("rbind", replicate(length(unique(results$beta)), results_noprior, simplify = FALSE)) 
results_noprior <- results_noprior %>%
  mutate(beta = c(do.call("rbind",replicate(5, unique(results$beta), simplify = FALSE))))

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
# p + geom_line(data=results_noprior,aes(x=beta,y=mean_r_trans,color = approach))+
#   geom_line(data=results_noprior,aes(x=beta,y=mean_r_trans-sd_r_trans/sqrt(5),color = approach),linetype='dashed',linewidth=2)+
#   geom_line(data=results_noprior,aes(x=beta,y=mean_r_trans+sd_r_trans/sqrt(5),color = approach),linetype='dashed',linewidth=2)+
#   geom_ribbon(data=results_noprior,aes(x=beta,ymin=mean_r_trans-sd_r_trans/sqrt(5),ymax=mean_r_trans+sd_r_trans/sqrt(5),fill = approach),alpha=0.3)
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

## prior vs no-prior comparison--------------------------------
res_comp <- rbind(results %>% filter(approach=='current'),
                  results_noprior)
res_comp <- res_comp %>% mutate(approach= ifelse(approach=='current','prior loss','no prior loss'))
p <- ggplot(data=results %>% filter(approach=='current') %>% mutate(approach= ifelse(approach=='current','prior loss','no prior loss'))) + 
  geom_point(data=results %>% filter(approach=='current') %>% mutate(approach= ifelse(approach=='current','prior loss','no prior loss')) %>% 
               select(beta,mean_r_trans,approach) %>% unique(),
    aes(x = beta , y = mean_r_trans,color=approach),size=4,shape=18) +
  geom_ribbon(aes(x=beta,ymin=mean_r_trans-sd_r_trans/sqrt(5),ymax=mean_r_trans+sd_r_trans/sqrt(5),fill=approach),alpha=0.25)+
  geom_line(aes(x=beta,y=mean_r_trans,color=approach),linewidth=1.5)+
  geom_line(aes(x=beta,y=mean_r_trans-sd_r_trans/sqrt(5),color = approach),linetype='dashed',linewidth=1)+
  geom_line(aes(x=beta,y=mean_r_trans+sd_r_trans/sqrt(5),color = approach),linetype='dashed',linewidth=1)+
  geom_line(data=results_noprior,aes(x=beta,y=mean_r_trans,color = approach),linewidth=1.5)+
  geom_line(data=results_noprior,aes(x=beta,y=mean_r_trans-sd_r_trans/sqrt(5),color = approach),linetype='dashed',linewidth=1)+
  geom_line(data=results_noprior,aes(x=beta,y=mean_r_trans+sd_r_trans/sqrt(5),color = approach),linetype='dashed',linewidth=1)+
  geom_ribbon(data=results_noprior,aes(x=beta,ymin=mean_r_trans-sd_r_trans/sqrt(5),ymax=mean_r_trans+sd_r_trans/sqrt(5),fill = approach),alpha=0.25)+
  scale_x_log10() + 
  ylim(c(0.79,0.81))+
  ylab('pearson`s r')+
  xlab('prior loss regularization')+
  ggtitle('Average performance in translation')+
  theme_pubr(base_family = 'Arial',base_size = 18)+
  theme(plot.title = element_text(hjust=0.5,size=14))
print(p)

noprior <- unique(results_noprior$avg_trans)
prior <- results %>% filter(approach=='current') %>% mutate(approach= ifelse(approach=='current','prior loss','no prior loss'))
prior <- prior$avg_trans
stats <- wilcox.test(prior,noprior)
print(stats)

p1 <- p + annotate(geom = 'text',x = 2e-02, y=0.806,label=paste0('Wilcoxon test: p-value=',round(stats$p.value,4)),size=5)
print(p1)

### Compare distributions of current prior and using no-prior
# Use the first fold as an example
i <- 0 # 1st fold
embs_prior <- rbind(data.table::fread(paste0('../results/PriorLossAnalysis/embs_AutoTransOp/U2OS_current/train_embs1_fold',i,'_beta1.0.csv'),header = T),
                    data.table::fread(paste0('../results/PriorLossAnalysis/embs_AutoTransOp/U2OS_current/train_embs2_fold',i,'_beta1.0.csv'),header = T)) %>% unique()
embs_noprior <- rbind(data.table::fread(paste0('../results/PriorLossAnalysis/embs_AutoTransOp/U2OS_noprior/train_embs1_fold',i,'.csv'),header = T),
                    data.table::fread(paste0('../results/PriorLossAnalysis/embs_AutoTransOp/U2OS_noprior/train_embs2_fold',i,'.csv'),header = T)) %>% unique()

embs_prior <- embs_prior %>% gather('latent_var','embeddings',-sig_id)
embs_noprior <- embs_noprior %>% gather('latent_var','embeddings',-sig_id)

embs <- rbind(embs_prior %>% mutate(approach='prior loss'),
              embs_noprior %>% mutate(approach='no prior'))
p2 <- ggplot(embs,aes(x=embeddings)) + geom_histogram(aes(fill=approach),color='black',alpha=0.5,bins = 80,position = 'identity')+ #color='black'
  ylab('counts')+
  xlab('latent variables` values')+
  ggtitle('Distribution of embeddings in the latent space')+
  theme_pubr(base_family = 'Arial',base_size = 18)+
  theme(plot.title = element_text(hjust=0.5,size=14))
print(p2)

p3 <- p1 + p2
print(p3)

ggsave('../article_supplementary_info/prior_vs_noprior_loss.png',
       plot=p3,
       width = 12,
       height = 9,
       units = 'in',
       dpi = 600)

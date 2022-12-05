library(tidyverse)
library(ggplot2)
library(ggsignif)
library(ggpubr)

### Performance all different approaches----------------
### For all genes
res_dcs_noise <- data.table::fread('../results/deepcellstate_results_10kgenes/deepcellstate_allgenes_10foldvalidation_results1000ep_with_noise_10k.csv')
colnames(res_dcs_noise)[1] <- 'fold'
res_dcs_noise <- res_dcs_noise %>% mutate(model='DCS')
res_dcs_noise <- res_dcs_noise %>% dplyr::select(-Direct_pearson,-Direct_spearman,-DirectAcc_ht29,-DirectAcc_a375,-cross_corr_ht29,-cross_corr_a375)
res_dcs_nonoise <- data.table::fread('../results/deepcellstate_results_10kgenes/deepcellstate_allgenes_10foldvalidation_results1000ep_no_noise_10k.csv')
colnames(res_dcs_nonoise)[1] <- 'fold'
res_dcs_nonoise <- res_dcs_nonoise %>% mutate(model='DCS modified v1')
res_dcs_nonoise <- res_dcs_nonoise %>% dplyr::select(-Direct_pearson,-Direct_spearman,-DirectAcc_ht29,-DirectAcc_a375,-cross_corr_ht29,-cross_corr_a375)
res_notvae_2encs <- data.table::fread('../results/my_results_10kgenes/allgenes_10foldvalidation_notvae_results1000ep512bs.csv')
colnames(res_notvae_2encs)[1] <- 'fold'
res_notvae_2encs <- res_notvae_2encs %>% mutate(model='Two Autoencoders')
res_notvae_2encs <- res_notvae_2encs %>% dplyr::select(-Direct_pearson,-Direct_spearman,-DirectAcc_ht29,-DirectAcc_a375,-cross_corr_ht29,-cross_corr_a375)
res_dcs_direct <- data.table::fread('../results/deepcellstate_results_10kgenes/deepcellstate_allgenes_10foldvalidation_results1000ep_direct_andl2similarity_10k.csv')
colnames(res_dcs_direct)[1] <- 'fold'
res_dcs_direct <- res_dcs_direct %>% mutate(model='DCS modified v2')
res_dcs_direct <- res_dcs_direct %>% dplyr::select(-Direct_pearson,-Direct_spearman,-DirectAcc_ht29,-DirectAcc_a375,-cross_corr_ht29,-cross_corr_a375)
baseline <- data.table::fread('../results/baseline_evaluation_allgenes.csv') %>% dplyr::select(-V1) %>% unique()
baseline <- baseline %>% rownames_to_column('fold')
baseline <- baseline %>% mutate(model='direct translation')
baseline <- baseline %>% gather('metric','value',-fold,-model)
base1 <- baseline %>% filter(metric!='DirectAcc_ht29')
base1 <- base1 %>% mutate(translation='A375 to HT29') %>%
  mutate(metric=ifelse(grepl('Acc',metric),'accuracy',ifelse(grepl('spearman',metric),'spearman','pearson')))
base2 <- baseline %>% filter(metric!='DirectAcc_a375')
base2 <- base2 %>% mutate(translation='HT29 to A375') %>%
  mutate(metric=ifelse(grepl('Acc',metric),'accuracy',ifelse(grepl('spearman',metric),'spearman','pearson')))
res_cpa <- data.table::fread('../results/MI_results/allgenes_10foldvalidation_withCPA_1000ep512bs_a375_ht29.csv')
#res_cpa <- data.table::fread('../results/MI_results/allgenes_10foldvalidation_notpretrained_MIuniform_and_l2sim_2encs_1000ep512bs.csv')
colnames(res_cpa)[1] <- 'fold'
res_cpa <- res_cpa %>% mutate(model='CPA approach')
res_cpa <- res_cpa %>% dplyr::select(-F1_score,-ClassAccuracy,-Direct_pearson,-Direct_spearman,-DirectAcc_ht29,-DirectAcc_a375)

all_results <- rbind(res_dcs_direct,res_dcs_noise,res_dcs_nonoise,res_dcs_direct,res_notvae_2encs,res_cpa)
all_results <- all_results %>% gather('metric','value',-fold,-model)

all_results_reconstruction <- all_results %>% filter(grepl('rec',metric)) %>% unique()
all_results_reconstruction <- all_results_reconstruction %>% mutate(reconstruct = ifelse(grepl('ht29',metric),'HT29','A375')) %>%
  mutate(metric = ifelse(grepl('spear',metric),'spearman',ifelse(grepl('acc',metric),'accuracy','pearson')))


all_results <- all_results %>% filter(!grepl('rec',metric)) %>% unique()
all_results <- all_results %>% mutate(translation = ifelse(grepl('HT29',metric),'A375 to HT29','HT29 to A375')) %>%
  mutate(metric = ifelse(grepl('spear',metric),'spearman',ifelse(grepl('acc',metric),'accuracy','pearson')))
all_results <- rbind(all_results,base1,base2)
all_results$model <- factor(all_results$model,levels=c('direct translation',"DCS",
                                                       "DCS modified v1","DCS modified v2",
                                                       "Two Autoencoders","CPA approach"))

# comparisons <- NULL
# p.values <- NULL
# k <- 1
# models <- c('direct translation',"DCS","DCS modified v1","DCS modified v2","Two Autoencoders","CPA approach")
# groups <- c(paste0(models,'pearson'),paste0(models,'spearman',paste0(models,'accuracy')))
# k <- 1
# for (i in 2:length(models)){
#   comparisons[[k]] <- c('direct translation',models[i])
#   p.values[k] <- wilcox.test(as.matrix(all_results %>% filter(model==comparisons[[k]][1]) %>%
#                                          filter(metric=='pearson') %>%
#                                          dplyr::select('value')),
#                              as.matrix(all_results %>% filter(model==comparisons[[k]][2]) %>%
#                                          filter(metric=='pearson') %>%
#                                          dplyr::select('value')))$p.value
#     k <- k+1
# }
# p.values <- p.adjust(p.values,'bonferroni')
all_results <- all_results %>% filter()
p <- ggboxplot(all_results,x='model',y='value',color='metric',add='jitter') +
  theme_gray(base_family = "serif",base_size = 15)+
  theme(plot.title = element_text(hjust = 0.5,size=15),legend.position='bottom') +
  ggtitle('Performance in translation for ~10k genes') + ylim(c(0,0.75))+
  facet_wrap(~ translation)
p <- p+ stat_compare_means(aes(group=model),comparisons = list(c('DCS','direct translation')) ,method='wilcox.test',label='p.signif')
print(p)


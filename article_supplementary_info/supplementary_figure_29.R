library(tidyverse)
library(ggplot2)
library(ggpubr)
library(ggridges)
library(rstatix)

# Load results--------
results_intermediate_1dec <- data.table::fread('../results/MI_results/landmarks_10foldvalidation_withIntermediate_1000ep512bs_a375_ht29.csv')
colnames(results_intermediate_1dec)[1] <- 'fold'
results_intermediate_1dec <- results_intermediate_1dec%>%
  dplyr::select(-Direct_pearson,-Direct_spearman,-DirectAcc_ht29,-DirectAcc_a375) %>% 
  mutate(model='Intermediate encoders with 1 decoder')
  
results_intermediate_2dec <- data.table::fread('../results/MI_results/landmarks_10foldvalidation_withIntermediate2Dec_1000ep512bs_a375_ht29.csv')
colnames(results_intermediate_2dec)[1] <- 'fold'
results_intermediate_2dec <- results_intermediate_2dec%>%
  dplyr::select(-Direct_pearson,-Direct_spearman,-DirectAcc_ht29,-DirectAcc_a375) %>% 
  mutate(model='Intermediate encoders with 2 decoders')

results_cpa <- data.table::fread('../results/MI_results/landmarks_10foldvalidation_withCPA_1000ep512bs_a375_ht29.csv')
colnames(results_cpa)[1] <- 'fold'
results_cpa <- results_cpa%>%
  dplyr::select(-Direct_pearson,-Direct_spearman,-DirectAcc_ht29,-DirectAcc_a375) %>% 
  mutate(model='CPA-based model')

all_results <- rbind(results_cpa,results_intermediate_2dec,results_intermediate_1dec)
all_results <- all_results %>% gather('metric','value',-fold,-model)
all_results <- all_results %>% filter(!grepl('rec',metric)) %>% unique()
all_results <- all_results %>% filter(!grepl('Class',metric)) %>% unique()
all_results <- all_results %>% filter(grepl('model',metric)) %>% unique()
all_results <- all_results %>% mutate(translation = ifelse(grepl('HT29',metric),'A375 to HT29','HT29 to A375')) %>%
  mutate(metric = ifelse(grepl('spear',metric),'spearman',ifelse(grepl('acc',metric),'sign accuracy','pearson')))
all_results$model <- factor(all_results$model,levels=c('CPA-based model',
                                                       'Intermediate encoders with 1 decoder',
                                                       'Intermediate encoders with 2 decoders'))

# Visualize comparison
# pairwise.tests = all_results %>% group_by(translation,metric) %>%
#   wilcox_test(value ~ model) %>% 
#   adjust_pvalue(method = 'BH') %>% ungroup()
my_comparisons <- list( c('CPA-based model', 'Intermediate encoders with 2 decoders'),
                        c('Intermediate encoders with 2 decoders', 'Intermediate encoders with 1 decoder'))
p <-  ggboxplot(all_results,
          x='model',y='value',color='model',add='jitter')+
  facet_wrap(vars(translation,metric)) +
  theme(text = element_text(family = 'Arial',size=20),
        axis.text.x = element_blank())+
  stat_compare_means(label = 'p.format',
                     method = 'wilcox.test',
                     #step.increase = 0.02,
                     tip.length=0.05,
                     size =5,
                     aes(group=model),
                     comparisons = my_comparisons)
print(p)
png('../article_supplementary_info/suppl_fig29.png',width = 16,height = 12,units = 'in',res = 600)
print(p)
dev.off()

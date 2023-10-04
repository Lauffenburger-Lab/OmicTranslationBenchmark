library(tidyverse)
library(ggplot2)
library(ggpubr)
library(ggpattern)
library(patchwork)
library(lsa)
library(rstatix)

process_embeddings <- function(embbedings,dataInfo,sampleInfo){
  dataInfo <- dataInfo %>% select(sig_id,cmap_name,duplIdentifier) %>% unique()
  
  sampleInfo <- left_join(sampleInfo,dataInfo)
  
  embbedings <- embbedings %>% rownames_to_column('sig_id')
  
  embs_processed <- left_join(embbedings,sampleInfo)
  
  return(embs_processed)
}

samples_separation <- function(processed_embbedings,save_name,
                               compare_level=c('duplicates',
                                               'equivalent condition',
                                               'cell',
                                               'drug',
                                               'cell-drug'),
                               metric=c("euclidean", "maximum", "manhattan",
                                        "canberra", "binary","cosine"),
                               show_plot=TRUE){
  library(tidyverse)
  embs <- processed_embbedings %>% column_to_rownames('sig_id') %>%
    select(-conditionId,-duplIdentifier,-cell_iname,-cmap_name)
  sample_info <- processed_embbedings %>% select(sig_id,conditionId,duplIdentifier,cell_iname,cmap_name)
  
  
  # calculate distance matrix
  if (metric=='cosine'){
    library(lsa)
    mat <- t(embs)
    dist <- 1 - cosine(mat)
  } else{
    dist <- as.matrix(dist(embs, method = metric))
  }
  
  # Conver to long format data frame
  # Keep only unique (non-self) pairs
  dist[lower.tri(dist,diag = T)] <- NA
  dist <- reshape2::melt(dist)
  dist <- dist %>% filter(!is.na(value))
  
  # Merge meta-data info and distances values
  dist <- left_join(dist,sample_info,by = c("Var1"="sig_id"))
  dist <- left_join(dist,sample_info,by = c("Var2"="sig_id"))
  dist <- dist %>% filter(!is.na(value))
  
  if (compare_level=='duplicates'){
    dist <- dist %>% mutate(is_same = (duplIdentifier.x==duplIdentifier.y))
    label <- 'Duplicate Signatures'
  }else if (compare_level=='equivalent condition'){
    dist <- dist %>% mutate(is_same = (conditionId.x==conditionId.y))
    label <- 'Same condition in different cell-line'
  }else if (compare_level=='cell'){
    dist <- dist %>% mutate(is_same = (cell_iname.x==cell_iname.y))
    label <- 'Same cell-line'
  }else if (compare_level=='drug'){
    dist <- dist %>% mutate(is_same = (cmap_name.x==cmap_name.y))
    label <- 'Same drug'
  } else if (compare_level=='cell-drug'){
    dist <- dist %>% mutate(is_same = (paste0(cmap_name.x,cell_iname.x)==paste0(cmap_name.y,cell_iname.y)))
    label <- 'Same drug,same cell-line'
  }
  
  dist <-dist %>% mutate(is_same=ifelse(is_same==T,
                                        label,'Random Signatures')) %>%
    mutate(is_same = factor(is_same,
                            levels = c('Random Signatures',
                                       label)))
  p <- ggplot(dist,aes(x=value,color=is_same,fill=is_same)) +
    geom_density(alpha=0.2) +
    labs(col = 'Type',fill='Type',title="Distance distribution in the latent space",x=paste0(metric,' distance'), y = "Density")+
    theme_classic() + theme(text = element_text(size=10),plot.title = element_text(hjust = 0.5))
  # png(paste0(save_name,'_',compare_level,'_seperation_latent_space.png'),width=12,height=8,units = "in",res=600)
  # print(p)
  # dev.off()
  
  if (show_plot==T){
    print(p)
  } 
  return(dist)
}
# Load samples info-------------------
sigInfo <- read.delim('../../../L1000_2021_11_23/siginfo_beta.txt')
sigInfo <- sigInfo %>% mutate(quality_replicates = ifelse(is_exemplar_sig==1 & qc_pass==1 & nsample>=3,1,0))
sigInfo <- sigInfo %>% filter(pert_type=='trt_cp')
sigInfo <- sigInfo %>% filter(quality_replicates==1)
# Filter based on TAS
sigInfo <- sigInfo %>% filter(tas>=0.3)
# Duplicate information
sigInfo <- sigInfo %>% mutate(duplIdentifier = paste0(cmap_name,"_",pert_idose,"_",pert_itime,"_",cell_iname))
sigInfo <- sigInfo %>% group_by(duplIdentifier) %>% mutate(dupl_counts = n()) %>% ungroup()
# Drug condition information
sigInfo <- sigInfo  %>% mutate(conditionId = paste0(cmap_name,"_",pert_idose,"_",pert_itime))

### Load embeddings for DCS and calculate Cohen's D------------------ 
### between cell lines for each latent variable
emb_dcs_1 <- distinct(data.table::fread('../results/deepcellstate_results_10kgenes/allembs_a375_dcs_withnoise_10k.csv')) %>% column_to_rownames('V1')
dcs_mean_1 <- colMeans(emb_dcs_1)
dcs_sd_1 <- apply(emb_dcs_1,2,sd)

emb_dcs_2 <- distinct(data.table::fread('../results/deepcellstate_results_10kgenes/allembs_ht29_dcs_withnoise_10k.csv')) %>% column_to_rownames('V1')
dcs_mean_2 <- colMeans(emb_dcs_2)
dcs_sd_2 <- apply(emb_dcs_2,2,sd)

s_dcs <- sqrt(((nrow(emb_dcs_1)-1) * dcs_sd_1^2 + (nrow(emb_dcs_2)-1)* dcs_sd_2^2)/(nrow(emb_dcs_1) + nrow(emb_dcs_2)-2))
effect_size_dcs <- (dcs_mean_1-dcs_mean_2)/s_dcs
hist(effect_size_dcs)

### Load embeddings for AutoTransOp and calculate Cohen's D------------------ 
### between cell lines for each latent variable
emb_1 <- distinct(data.table::fread('../results/trained_embs_all/AllEmbs_MI_a375.csv')) %>% column_to_rownames('V1')
mean_1 <- colMeans(emb_1)
sd_1 <- apply(emb_1,2,sd)

emb_2 <- distinct(data.table::fread('../results/trained_embs_all/AllEmbs_MI_ht29.csv')) %>% column_to_rownames('V1')
mean_2 <- colMeans(emb_2)
sd_2 <- apply(emb_2,2,sd)

s <- sqrt(((nrow(emb_1)-1) * sd_1^2 + (nrow(emb_2)-1)* sd_2^2)/(nrow(emb_1) + nrow(emb_2)-2))
effect_size <- (mean_1-mean_2)/s
hist(effect_size)

### Visualize absolute effect size comparison and the latent variables with the largest effect-------------
effect_sizes <- rbind(data.frame(d=abs(effect_size_dcs), model = rep('DCS',length(s_dcs))),
                      data.frame(d=abs(effect_size), model = rep('AutoTransOp v1',length(s_dcs))))
p1 <- ggboxplot(effect_sizes,x='model',y='d',color='model',outlier.shape = NA) + geom_jitter(aes(color=model),width = 0.25,alpha=0.5,size=1)+
  ylab('absolute Cohen`s d') +
  ggtitle('Effect size comparison')+
  stat_compare_means(size=6,label.y = 0.47,label.x = 1.35,method = 'wilcox.test')+
  theme(text = element_text(family = 'Arial',size=24),
        plot.title = element_text(hjust = 0.5),
        legend.position = 'none')

largest_ind <- which.max(abs(effect_size))
print(names(largest_ind))
largest_ind_dcs <- which.max(abs(effect_size_dcs))
print(names(largest_ind_dcs))
latent_var <- rbind(rbind(emb_1 %>% select(c('latent variable with larger effect size'='z956')) %>% mutate(cell = 'A375'),
                    emb_2 %>% select(c('latent variable with larger effect size'='z956')) %>% mutate(cell = 'HT29')) %>% mutate(model='AutoTransOp v1'),
                    rbind(emb_dcs_1 %>% select(c('latent variable with larger effect size'='z_234')) %>% mutate(cell = 'A375'),
                          emb_dcs_2 %>% select(c('latent variable with larger effect size'='z_234')) %>% mutate(cell = 'HT29')) %>% mutate(model='DCS'))
latent_var <- latent_var %>% mutate(d=ifelse(model=='DCS',effect_size_dcs[largest_ind_dcs],effect_size[largest_ind])) %>%
  mutate(d = paste0('Cohen`s D : ',round(d,digits = 2)))
p2 <- ggboxplot(latent_var, x = "cell", y = "latent variable with larger effect size",
         color = "cell",outlier.shape = NA)+
  geom_jitter(aes(color=cell),width = 0.25,alpha=0.5,size=1)+
  geom_text(aes(x=1.5,y=5, label=d),
            data=latent_var %>% select(model,d) %>% unique(),inherit.aes = FALSE,
            size=6)+
  scale_color_manual(values = c("#C4961A","#293352"))+
  facet_wrap(~model)+
  ggtitle('Latent variable comparison')+
  stat_compare_means(comparisons = list(c('A375','HT29')),
                     method = 't.test',
                     size=6,label.y = 3.2)+
  theme(text = element_text(family = 'Arial',size=24),
        plot.title = element_text(hjust = 0.5),
        legend.position = 'none')

# ### Repeat for smallest effect size
# smallest_ind <- which.min(abs(effect_size))
# print(names(smallest_ind))
# smallest_ind_dcs <- which.min(abs(effect_size_dcs))
# print(names(smallest_ind_dcs))
# latent_var <- rbind(rbind(emb_1 %>% select(c('latent variable with larger effect size'='z665')) %>% mutate(cell = 'A375'),
#                           emb_2 %>% select(c('latent variable with larger effect size'='z665')) %>% mutate(cell = 'HT29')) %>% mutate(model='AutoTransOp v1'),
#                     rbind(emb_dcs_1 %>% select(c('latent variable with larger effect size'='z_888')) %>% mutate(cell = 'A375'),
#                           emb_dcs_2 %>% select(c('latent variable with larger effect size'='z_888')) %>% mutate(cell = 'HT29')) %>% mutate(model='DCS'))
# latent_var <- latent_var %>% mutate(d=ifelse(model=='DCS',effect_size_dcs[smallest_ind_dcs],effect_size[smallest_ind])) %>%
#   mutate(d = paste0('Cohen`s D : ',round(d,digits = 4)))
# p3 <- ggboxplot(latent_var, x = "cell", y = "latent variable with larger effect size",
#                 color = "cell",outlier.shape = NA)+
#   geom_jitter(aes(color=cell),width = 0.25,alpha=0.5,size=1)+
#   geom_text(aes(x=1.5,y=3.2, label=d),
#             data=latent_var %>% select(model,d) %>% unique(),inherit.aes = FALSE,
#             size=6)+
#   facet_wrap(~model)+
#   ggtitle('Latent variable comparison')+
#   stat_compare_means(comparisons = list(c('A375','HT29')),
#                      method = 't.test',
#                      size=6,label.y = 1)+
#   theme(text = element_text(family = 'Arial',size=24),
#         plot.title = element_text(hjust = 0.5),
#         legend.position = 'none')

p <- p1+p2
print(p)
ggsave('../figures/fig1_globality_comparison.eps',
       plot = p,
       device = cairo_ps,
       width = 12,
       height = 9,
       units = 'in',
       dpi = 600)

### Plot adjusted-Pvalues from univariate comparisons for DCS and AutoTransOp v1------------
pairwise.dcs = rbind(emb_dcs_1 %>% gather('latent_var','value') %>% mutate(cell='A375'),
                     emb_dcs_2 %>% gather('latent_var','value') %>% mutate(cell='HT29')) %>% 
  group_by(latent_var) %>%
  t_test(value ~ cell) %>% 
  adjust_pvalue(method = 'BH') %>% ungroup()

pairwise.autotransop = rbind(emb_1 %>% gather('latent_var','value') %>% mutate(cell='A375'),
                     emb_2 %>% gather('latent_var','value') %>% mutate(cell='HT29')) %>% 
  group_by(latent_var) %>%
  t_test(value ~ cell) %>% 
  adjust_pvalue(method = 'BH') %>% ungroup()
statistics_df <- rbind(pairwise.autotransop %>% mutate(model='AutoTransOp v1'),
                       pairwise.dcs %>% mutate(model='DCS')) %>% mutate(logPadj = -log10(p.adj))
create_custom_ordered_plot <- function(data, model_type) {
  data <- data %>%
    filter(model == model_type) %>%
    arrange(desc(logPadj))  # Sort by logPadj in descending order
  data$latent_var <- factor(data$latent_var, levels = data[order(data$model, -data$logPadj), ]$latent_var)
  p <- ggplot(data, #%>% mutate(statistic = ifelse(statistic>8,8,ifelse(statistic<(-8),-8,statistic))),
              aes(x = latent_var, y = logPadj)) +
    geom_point(aes(color = statistic), size = 1) +
    scale_color_gradient2(low = "blue",
                          mid = 'white',
                          high = "red",
                          midpoint = 0)+#,limits = c(-8,8)) +
    ylab('-log10(p.adjusted)') +
    xlab('latent variables') +
    geom_hline(yintercept = -log10(0.05), linetype = 'dashed', color = 'black', size = 1) +
    annotate('text',x=700,y=3,label='p.adjusted = 0.05',size=6)+
    annotate('text',x=500,y=round(max(data$logPadj),0),label=paste0('number of significant : ',nrow(data %>% filter(p.adj<0.05))),size=6)+
    theme_pubr(base_family = 'Arial', base_size = 20) +
    theme(text = element_text(size = 20, family = 'Arial'),
          axis.text.x = element_blank(),  # Rotate x-axis labels
          legend.position = 'right') +
    facet_wrap(~model, scales = "free_x")
  
  #print(p)
}
# Create a list of custom-ordered plots for each model
unique_models <- unique(statistics_df$model)
plots_list <- lapply(unique_models, function(model) {
  create_custom_ordered_plot(statistics_df, model)
})
combined_plot <- wrap_plots(plots_list) + #plot_annotation(theme = theme(legend.position = 'top')) +
  plot_layout(guides = "auto") 
print(combined_plot)
ggsave('../figures/fig1_univariately_significant_dcs_vs_model.eps',
       plot = combined_plot,
       device = cairo_ps,
       width = 12,
       height = 9,
       units = 'in',
       dpi = 600)
### Compare distance distributions of embeddings within the same cell--------
embs_proc <- process_embeddings(rbind(emb_1,emb_2),sigInfo,sigInfo %>%
                                  filter(cell_iname %in% c('A375','HT29')) %>% 
                                  select(sig_id,cell_iname,conditionId) %>%
                                  unique())
dist <- samples_separation(embs_proc,
                           compare_level='cell',
                           metric = 'cosine',
                           show_plot = T)

embs_proc_dcs <- process_embeddings(rbind(emb_dcs_1,emb_dcs_2),sigInfo,sigInfo %>%
                                  filter(cell_iname %in% c('A375','HT29')) %>% 
                                  select(sig_id,cell_iname,conditionId) %>%
                                  unique())
dist_dcs <- samples_separation(embs_proc_dcs,
                           compare_level='cell',
                           metric = 'cosine',
                           show_plot = T)
### Compare distance distributions
same_cell_dcs <- dist_dcs %>% filter(is_same=='Same cell-line') %>% mutate(model='DCS')
same_cell <- dist %>% filter(is_same=='Same cell-line')  %>% mutate(model='AutoTransOp v1')
s <- sqrt(((nrow(same_cell_dcs)-1) * sd(same_cell_dcs$value)^2 + (nrow(same_cell)-1)* sd(same_cell$value)^2)/(nrow(same_cell_dcs) + nrow(same_cell)-2))
cohen <- (mean(same_cell_dcs$value) - mean(same_cell$value))/s

d_dcs = effectsize::cohens_d(as.matrix(dist_dcs %>% filter(is_same=='Same cell-line') %>% select(value)),
                              as.matrix(dist_dcs %>% filter(is_same!='Same cell-line')%>% select(value)),
                              ci=0.95)
d = effectsize::cohens_d(as.matrix(dist %>% filter(is_same=='Same cell-line') %>% select(value)),
                             as.matrix(dist %>% filter(is_same!='Same cell-line')%>% select(value)),
                             ci=0.95)
d <- d %>% mutate(model = 'AutoTransOp v1') %>% mutate(effsize = paste0('Cohen`s D : ',round(Cohens_d,digits = 2)))
d_dcs <- d_dcs %>% mutate(model = 'DCS') %>% mutate(effsize = paste0('Cohen`s D : ',round(Cohens_d,digits = 2)))
cohen_df <- rbind(d,d_dcs)
violin_data <- rbind(same_cell_dcs,same_cell)
violin_data$model <- factor(violin_data$model, levels = c('DCS','AutoTransOp v1'))
violin <- ggplot(violin_data, aes(x=model, y=value, fill = model)) + 
  geom_violin(position = position_dodge(width = 1),width = 1)+geom_boxplot(position = position_dodge(width = 1),width = 0.05,
                                                                           outlier.shape = NA)+
  ylim(0,2)+
  ylab("Cosine Distance")+ 
  theme(axis.ticks.x=element_blank(),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(),text = element_text(family = "Arial",size = 24),legend.position = "bottom")+
  theme_minimal(base_family = "Arial",base_size = 24) +
  stat_compare_means(comparisons = list(c('DCS','AutoTransOp v1')),
                     method='t.test',
                     label.y = 1.25)+
 annotate('text',x = 1.5,y=1.5,label=paste0('Cohen`s D : ',round(cohen,digits = 2)),size=6)
print(violin)
ggsave('../figures/fig1_dcs_vs_model_samecell_cosines.eps',
       plot = violin,
       device = cairo_ps,
       width = 9,
       height = 9,
       units = 'in',
       dpi = 600)

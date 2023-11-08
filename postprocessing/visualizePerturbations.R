library(tidyverse)
library(gg3D)
library(ggpubr)
library(patchwork)
library(ggplot2)
library(colorblindr)
library(viridis)
library(RColorBrewer)
library(lsa)
library(Rtsne)
### Load data---------------------------------------------------
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

# Load embeddings and visualize drug space-------------------------------
emb1 <- distinct(data.table::fread('../results/trained_embs_all/AllEmbs_MI_pc3_withclass.csv',header = T)) %>% column_to_rownames('V1')
emb1 <- emb1 %>% mutate(cell='PC3')
emb2 <- distinct(data.table::fread('../results/trained_embs_all/AllEmbs_MI_ha1e_withclass.csv',header = T)) %>% column_to_rownames('V1')
emb2 <- emb2 %>% mutate(cell='HA1E')
all_embs <- rbind(emb1,emb2)
all_embs_proc <- left_join(all_embs %>% rownames_to_column('sig_id'),
                           sigInfo %>% select(sig_id,pert_id,cmap_name,duplIdentifier,pert_dose,pert_itime,tas,conditionId) %>% unique())

# summarised_embs <- all_embs_proc %>% group_by(conditionId) %>% summarise(samples = n())
# summarised_embs <- summarised_embs %>% unique()
# summarised_embs <- summarised_embs %>% arrange(desc(samples))
# summarised_embs <- summarised_embs[1:10,]

### visualize drugs first-----------------------------
display.brewer.all(n = 10, type = "all", select = NULL,
                   colorblindFriendly = TRUE)
colors <- brewer.pal(n = 10, name = "Paired")
colors <- c(colors,"#D3D3D3")
# colors[length(colors)-2] <- "black"

# t-SNE visualization
sample_info <- all_embs_proc %>% select(sig_id,conditionId,cmap_name,pert_id,cell,pert_dose,pert_itime,tas) %>% unique()
embs <- all_embs_proc %>% select(-cell,-cmap_name,-duplIdentifier,-pert_id,-pert_dose,-pert_itime,-tas,-conditionId) %>% unique()

rownames(embs) <- NULL
embs  <-  embs %>% column_to_rownames('sig_id')

perpl = DescTools::RoundTo(sqrt(nrow(embs)), multiple = 5, FUN = round)
emb_size = ncol(embs)
set.seed(42)
tsne_all <- Rtsne(embs, 
                  dims = 2, perplexity=perpl, 
                  verbose=T, max_iter = 1000,initial_dims = 10,check_duplicates = T,
                  normalize = F,pca_scale = F,
                  num_threads = 15)
df_tsne <- data.frame(V1 = tsne_all$Y[,1], V2 =tsne_all$Y[,2])
rownames(df_tsne) <- rownames(embs)
col_names <- paste0(rep('tSNE-',2),seq(1,2))
colnames(df_tsne) <- col_names
embs_reduced <- df_tsne %>% rownames_to_column('sig_id')
embs_reduced <- suppressWarnings(left_join(embs_reduced,
                                           sample_info %>% select(sig_id,conditionId,cmap_name,cell,pert_dose,pert_itime,tas) %>% 
                                             unique()))
embs_reduced <- embs_reduced %>% select(sig_id,conditionId,all_of(col_names)) %>% unique()
embs_reduced <- embs_reduced %>% group_by(conditionId) %>% mutate(no_sigs = ifelse(is.na(conditionId),-666,n_distinct(sig_id))) %>% ungroup()
top <- embs_reduced %>% select(conditionId,no_sigs) %>% unique() %>% arrange(-no_sigs)
top <- top[1:10,]
top <- top$conditionId
embs_reduced <- embs_reduced %>% mutate(conditionId=ifelse(conditionId %in% top,conditionId,'other'))
col_breaks <- c(top,'other')
embs_reduced <- embs_reduced %>% select(all_of(c(col_names,'conditionId')))
colnames(embs_reduced)[3] <- 'perturbation'
tsne_plot <- ggplot(embs_reduced,aes(`tSNE-1`,`tSNE-2`)) +geom_point(aes(col=perturbation,size=perturbation))+
  scale_color_manual(values = colors,breaks = col_breaks)+
  scale_size_manual(values = c(rep(2,length(colors)-1),0.25),breaks = col_breaks)+
  xlab(paste0('tSNE-1'))+ ylab(paste0('tSNE-2'))+theme_minimal()+
  theme(text = element_text(family = 'Arial',size=14),plot.title = element_blank(),
        legend.text=element_text(size=10),
        legend.title = element_text(size=11),
        legend.position = 'top')
print(tsne_plot)

ggsave('../article_supplementary_info/conditionsID_pc3_ha1e_tsne_all.png',
       plot=tsne_plot,
       width = 9,
       height = 9,
       units = 'in',
       dpi=600)
postscript('../article_supplementary_info/conditionsID_pc3_ha1e_tsne_all.eps',width = 12,height = 12)
print(tsne_plot)
dev.off()

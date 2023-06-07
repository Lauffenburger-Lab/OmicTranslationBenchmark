library(tidyverse)
library(readxl)

### Load data
human_hiv <- data.table::fread('data/Rv144_Systems_Serology.csv')
colnames(human_hiv)[1] <- 'id'
human_hiv <- human_hiv %>% mutate(trt=ifelse(trt=='PLACEBO',0,1))
human_hiv <-  human_hiv %>% mutate(infect=ifelse(infect=='Yes',0,1))
human_expr <- as.matrix(human_hiv[,12:ncol(human_hiv)])
#drop avidity data
cols <- colnames(human_expr)
inds <- grep('avidity',cols)
human_expr <- human_expr[,-inds]
cols <- colnames(human_expr)
# human_expr <- human_expr - colMeans(human_expr)
human_expr <- scale(human_expr)
hist(human_expr,100)
data.table::fwrite(human_expr,'data/human_exprs.csv',row.names = T)
data.table::fwrite(human_hiv,'data/human_metadata.csv',row.names = T)

primates_nhp <- read_excel('data/NHP_trial_1319.xlsx',sheet=1)
primates_nhp <- primates_nhp %>% mutate(Vaccine=ifelse(Vaccine=='Sham',0,1))
primates_nhp <-  primates_nhp %>% mutate(ProtectBinary=ifelse(ProtectBinary==1,0,1))
primates_expr <- read_excel('data/NHP_trial_1319.xlsx',sheet=2)
primates_expr <- as.matrix(primates_expr)
### keep only Antigen measurements
primates_expr <- primates_expr[,30:292]
# 1 to 243 columns MFIs
primates_expr <- log10(primates_expr+1)
# normalize by substracting with median of SHAM
controls <- primates_expr[which(primates_nhp$Immunization_Ag=='Sham'),]
controlsMed <- apply(controls,2,median)
primates_expr <- primates_expr - controlsMed
hist(primates_expr,100)
# primates_expr <- primates_expr - colMeans(primates_expr)
primates_expr <- scale(primates_expr)
primates_expr[is.nan(primates_expr)] <- 0. 
hist(primates_expr,100)
data.table::fwrite(primates_expr,'data/primates_exprs.csv',row.names = T)
data.table::fwrite(primates_nhp,'data/primates_metadata.csv',row.names = T)

## Perform PCA
library(factoextra)
pca_primates <- prcomp(primates_expr, scale=F)
fviz_screeplot(pca_primates)

pca_humans <- prcomp(human_expr,scale = F)
fviz_screeplot(pca_humans)

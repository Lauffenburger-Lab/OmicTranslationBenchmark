library(tidyverse)
library(readxl)

### Load data
human_hiv <- data.table::fread('data/Rv144_Systems_Serology.csv')
colnames(human_hiv)[1] <- 'id'
human_hiv <- human_hiv %>% mutate(trt=ifelse(trt=='PLACEBO',0,1))
human_hiv <-  human_hiv %>% mutate(infect=ifelse(infect=='Yes',0,1))
human_expr <- as.matrix(human_hiv[,12:ncol(human_hiv)])
cols <- colnames(human_expr)
# human_expr <- human_expr - colMeans(human_expr)
human_expr <- scale(human_expr)
hist(human_expr)
data.table::fwrite(human_expr,'data/human_exprs.csv',row.names = T)
data.table::fwrite(human_hiv,'data/human_metadata.csv',row.names = T)

mouse_nhp <- read_excel('data/NHP_trial_1319.xlsx',sheet=1)
mouse_nhp <- mouse_nhp %>% mutate(Vaccine=ifelse(Vaccine=='Sham',0,1))
mouse_nhp <-  mouse_nhp %>% mutate(ProtectBinary=ifelse(ProtectBinary==1,0,1))
mouse_expr <- read_excel('data/NHP_trial_1319.xlsx',sheet=2)
mouse_expr <- as.matrix(mouse_expr)
mouse_expr <- log10(mouse_expr+1)
# mouse_expr <- mouse_expr - colMeans(mouse_expr)
mouse_expr <- scale(mouse_expr)
mouse_expr[is.nan(mouse_expr)] <- 0. 
hist(mouse_expr)
data.table::fwrite(mouse_expr,'data/mouse_exprs.csv',row.names = T)
data.table::fwrite(mouse_nhp,'data/mouse_metadata.csv',row.names = T)


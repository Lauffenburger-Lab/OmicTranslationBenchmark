library(tidyverse)
library(reshape2)
library(ggsignif)
library(ggpubr)

### Assign labels and specific cell-types in validation data-------------
alldata <-data.table::fread('../../../Fibrosis Species Translation/human lung fibrosis/all_mouse_lung.csv')
allmeatadata <- data.table::fread('../../../Fibrosis Species Translation/human lung fibrosis/Strunz_metadata.csv')
allmeatadata <- allmeatadata %>% select(V1,c('specific_cell'='Cell_Type')) %>% unique()
alldata <- left_join(alldata,allmeatadata)
colnames(alldata)[(ncol(alldata)-3):(ncol(alldata)-1)] <- c("diagnosis","species","cell_type")
alldata <- alldata %>% select(-species)

for (i in 1:9){
  ## Load validation fold
  val <- data.table::fread(paste0('data/10foldcrossval_lung/csvFiles/val_mouse_',i,'.csv'))
  val <- val %>% select(-V1)
  val <- val %>% select(-species)
  colnames(val)[1:(ncol(val)-2)] <- colnames(alldata)[2:(ncol(alldata)-3)]
  val <- left_join(val,alldata) %>% column_to_rownames('V1')
  val <- val %>% filter(!is.na(specific_cell))
  data.table::fwrite(val,paste0('data/10foldcrossval_lung/csvFiles/labeled_val_mouse_',i,'.csv'))
  print(paste0('Finished fold ',i))
}

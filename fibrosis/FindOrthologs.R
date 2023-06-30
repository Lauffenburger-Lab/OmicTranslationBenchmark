mouse#library(AnnotationDbi)
library(tidyverse)
#library(biomaRt)

### Load lookup table
homolog_table <- read.delim('HOM_MouseHumanSequence.rpt.txt')

## Load human and mouse genes
mouse_map <- data.table::fread('data/mouse_map_genes.csv') %>% select(-V1)
human_map <- data.table::fread('data/human_map_genes.csv')%>% select(-V1)

## Begin mapping
mouse_map <- left_join(mouse_map,
                       homolog_table %>% filter(Common.Organism.Name=='mouse, laboratory') %>% 
                         select(HomoloGene.ID,Symbol) %>% unique(),
                       by = c('gene'='Symbol'))
human_map <- left_join(human_map,
                       homolog_table %>% filter(Common.Organism.Name=='human') %>% 
                         select(HomoloGene.ID,Symbol) %>% unique(),
                       by = c('gene'='Symbol'))
# keep only those that have a homologue
mouse_map <- mouse_map %>% filter(!is.na(HomoloGene.ID))
human_map <- human_map %>% filter(!is.na(HomoloGene.ID))

## Create the map which will be used during training to subset current data
map <- left_join(mouse_map,human_map %>% select(HomoloGene.ID,c('human_gene'='gene'),c('human_id'='id')) %>% unique())
map <- map %>% filter(!is.na(human_gene)) %>% filter(!is.na(id))
colnames(map)[1:2] <- c('mouse_id','mouse_gene')

## Check if there 1:many mappings
map_summary <- map %>% group_by(mouse_gene) %>% mutate(mouse2human_counts=n_distinct(human_gene)) %>% ungroup() %>%
                       group_by(human_gene) %>% mutate(human2mouse_counts=n_distinct(mouse_gene)) %>% ungroup()
# Remove multiple mappings
map_summary <- map_summary %>% filter(mouse2human_counts==1 & human2mouse_counts==1)
map_summary <- map_summary %>% select(-mouse2human_counts,-human2mouse_counts) %>% unique()

# save resulted map
data.table::fwrite(map_summary,'results/HumanMouseHomologuesMap.csv',row.names = T)

fastenrichment <- function(signature_ids,
                           gene_ids,
                           measurements,
                           enrichment_space = c("go_bp","kegg","tf_dorothea","msig_db_h",
                                                "msig_db_c1","msig_db_c2","msig_db_c3",
                                                "msig_db_c4","msig_db_c5","msig_db_c6","msig_db_c7"),
                           order_columns =T,
                           pval_adjustment=T,
                           tf_path=NULL,
                           n_permutations=1000){
  # signature_ids: the signature ids to calculate enrichment scores
  # gene_space: feature space of genes to use, one of "all","landmark" and "best_inferred"
  # gene_type: type of genes to include, one of "protein-coding","all"
  # enrichment_space: vector with types of gene sets to calculate enrichment
  #   "go_bp","kegg","tf_dorothea","msig_db_h",
  #   "msig_db_c1","msig_db_c2","msig_db_c3","msig_db_c4","msig_db_c5","msig_db_c6","msig_db_c7"
  
  # Load packages 
  library(tidyverse)
  library(fgsea)
  library(gage)
  library(EGSEAdata)
  library(AnnotationDbi)
  library(org.Hs.eg.db)
  library(cmapR)
  library(rhdf5)
  library(topGO)
  library(GO.db)
  #library(parallel)
  #library(snow)
  
  sig_ids <- as.character(signature_ids)
  
  # make sure its in correct order
  if (order_columns==T){
    measurements <- measurements[as.character(gene_ids),sig_ids]
  }else{
    measurements <- measurements[as.character(gene_ids),]
  }
  
  # load pathway data
  egsea.data(species = "human",returnInfo = TRUE)
  
  all_genesets <- list()
  # Build the list of pathways, genesets to run gsea
  print("building enrichment space")
  
  if("go_bp" %in% enrichment_space) {
    genes <- factor(x = rep(1,nrow(measurements)),levels = c(0,1))
    names(genes) <- rownames(measurements)
    
    GOobject <- new("topGOdata",ontology = "BP", allGenes = genes, annot=annFUN.org, mapping="org.Hs.eg.db", 
                    ID = "entrez", nodeSize = 10)
    term.genes <- genesInTerm(GOobject, GOobject@graph@nodes)
    names(term.genes) <- paste0("FL1000_GO_BP_",names(term.genes))
    all_genesets <- c(all_genesets,term.genes)
  }
  
  if("kegg" %in% enrichment_space) {
    kegg_list <-  kegg.pathways$human$kg.sets
    names(kegg_list) <- paste0("FL1000_KEGG_",names(kegg_list))
    all_genesets <- c(all_genesets,kegg_list)
  }
  
  if(grepl(pattern = "msig_db",x = paste0(enrichment_space,collapse = ""))) {
    
    msig_overview <- t(as.data.frame(msigdb[[1]]))
    for (i in 2:(length(msigdb)-1)) {
      msig_overview <- rbind(msig_overview,t(as.data.frame(msigdb[[i]])))
    }
    msig_overview <- as.data.frame(msig_overview)
    rownames(msig_overview) <- msig_overview$STANDARD_NAME
    
    hallmark <- list()
    c1 <- list()
    c2 <- list()
    c3 <- list()
    c4 <- list()
    c5 <- list()
    c6 <- list()
    c7 <- list()
    
    h <- 1
    c1_ind <- 1
    c2_ind <- 1
    c3_ind <- 1
    c4_ind <- 1
    c5_ind <- 1
    c6_ind <- 1
    c7_ind <- 1
    for (i in 1:(length(msigdb)-1)) {
      a <- msigdb[[i]]
      
      if (a["CATEGORY_CODE"] == "H") {
        nm <- a["STANDARD_NAME"]
        hallmark <- c(hallmark,str_split(a["MEMBERS_EZID"],pattern = ","))
        names(hallmark)[h] <- nm
        h <- h+1
      }
      if (a["CATEGORY_CODE"] == "C1") {
        nm <- a["STANDARD_NAME"]
        c1 <- c(c1,str_split(a["MEMBERS_EZID"],pattern = ","))
        names(c1)[c1_ind] <- nm
        c1_ind <- c1_ind+1
      }
      if (a["CATEGORY_CODE"] == "C2") {
        nm <- a["STANDARD_NAME"]
        c2 <- c(c2,str_split(a["MEMBERS_EZID"],pattern = ","))
        names(c2)[c2_ind] <- nm
        c2_ind <- c2_ind+1
      }
      if (a["CATEGORY_CODE"] == "C3") {
        nm <- a["STANDARD_NAME"]
        c3 <- c(c3,str_split(a["MEMBERS_EZID"],pattern = ","))
        names(c3)[c3_ind] <- nm
        c3_ind <- c3_ind+1
      }
      if (a["CATEGORY_CODE"] == "C4") {
        nm <- a["STANDARD_NAME"]
        c4 <- c(c4,str_split(a["MEMBERS_EZID"],pattern = ","))
        names(c4)[c4_ind] <- nm
        c4_ind <- c4_ind+1
      }
      if (a["CATEGORY_CODE"] == "C5") {
        nm <- a["STANDARD_NAME"]
        c5 <- c(c5,str_split(a["MEMBERS_EZID"],pattern = ","))
        names(c5)[c5_ind] <- nm
        c5_ind <- c5_ind+1
      }
      if (a["CATEGORY_CODE"] == "C6") {
        nm <- a["STANDARD_NAME"]
        c6 <- c(c6,str_split(a["MEMBERS_EZID"],pattern = ","))
        names(c6)[c6_ind] <- nm
        c6_ind <- c6_ind+1
      }
      if (a["CATEGORY_CODE"] == "C7") {
        nm <- a["STANDARD_NAME"]
        c7 <- c(c7,str_split(a["MEMBERS_EZID"],pattern = ","))
        names(c7)[c7_ind] <- nm
        c7_ind <- c7_ind+1
      }
    }
    names(hallmark) <- paste0("FL1000_MSIG_H_",names(hallmark))
    names(c1) <- paste0("FL1000_MSIG_C1_",names(c1))
    names(c2) <- paste0("FL1000_MSIG_C2_",names(c2))
    names(c3) <- paste0("FL1000_MSIG_C3_",names(c3))
    names(c4) <- paste0("FL1000_MSIG_C4_",names(c4))
    names(c5) <- paste0("FL1000_MSIG_C5_",names(c5))
    names(c6) <- paste0("FL1000_MSIG_C6_",names(c6))
    names(c7) <- paste0("FL1000_MSIG_C7_",names(c7))
    
    if("msig_db_h" %in% enrichment_space) {
      all_genesets <- c(all_genesets,hallmark)
    }
    if("msig_db_c1" %in% enrichment_space) {
      all_genesets <- c(all_genesets,c1)
    }
    if("msig_db_c2" %in% enrichment_space) {
      all_genesets <- c(all_genesets,c2)
    }
    if("msig_db_c3" %in% enrichment_space) {
      all_genesets <- c(all_genesets,c3)
    }
    if("msig_db_c4" %in% enrichment_space) {
      all_genesets <- c(all_genesets,c4)
    }
    if("msig_db_c5" %in% enrichment_space) {
      all_genesets <- c(all_genesets,c5)
    }
    if("msig_db_c6" %in% enrichment_space) {
      all_genesets <- c(all_genesets,c6)
    }
    if("msig_db_c7" %in% enrichment_space) {
      all_genesets <- c(all_genesets,c7)
    }
    
  }
  
  if("tf_dorothea" %in% enrichment_space) {
    load(tf_path)
    tf_list <- viper_regulon
    names(tf_list) <- paste0("FL1000_TF_DOROTHEA_",names(tf_list))
    all_genesets <- c(all_genesets,tf_list)
  }
  
  print("running fgsea for enrichment space")
  genesets_list <-apply(measurements,MARGIN = 2,fgsea,pathways = all_genesets,
                  minSize=10,
                  maxSize=500,
                  nperm = n_permutations)
  print("fgsea finished, preparing outputs")
  
  # Prepare output
  
  NES <- genesets_list[[1]]$NES
  if (pval_adjustment==T){
    pval <- genesets_list[[1]]$padj
  }else{
    pval <- genesets_list[[1]]$pval
  }
  
  for (i in 2:length(genesets_list)) {
    
    NES <- cbind(NES,genesets_list[[i]]$NES)
    if (pval_adjustment==T){
      pval <- cbind(pval,genesets_list[[i]]$padj)
    }else{
      pval <- cbind(pval,genesets_list[[i]]$pval)
    }
  }
  
  colnames(NES) <- names(genesets_list)
  rownames(NES) <- genesets_list[[1]]$pathway
  colnames(pval) <- names(genesets_list)
  rownames(pval) <- genesets_list[[1]]$pathway
  
  NES_go_bp <- NES[sapply(rownames(NES),FUN=grepl,pattern="FL1000_GO_BP_"),]
  NES_kegg <- NES[sapply(rownames(NES),FUN=grepl,pattern="FL1000_KEGG_"),]
  NES_tf <- NES[sapply(rownames(NES),FUN=grepl,pattern="FL1000_TF_DOROTHEA_"),]
  NES_h <- NES[sapply(rownames(NES),FUN=grepl,pattern="FL1000_MSIG_H_"),]
  NES_c1 <- NES[sapply(rownames(NES),FUN=grepl,pattern="FL1000_MSIG_C1_"),]
  NES_c2 <- NES[sapply(rownames(NES),FUN=grepl,pattern="FL1000_MSIG_C2_"),]
  NES_c3 <- NES[sapply(rownames(NES),FUN=grepl,pattern="FL1000_MSIG_C3_"),]
  NES_c4 <- NES[sapply(rownames(NES),FUN=grepl,pattern="FL1000_MSIG_C4_"),]
  NES_c5 <- NES[sapply(rownames(NES),FUN=grepl,pattern="FL1000_MSIG_C5_"),]
  NES_c6 <- NES[sapply(rownames(NES),FUN=grepl,pattern="FL1000_MSIG_C6_"),]
  NES_c7 <- NES[sapply(rownames(NES),FUN=grepl,pattern="FL1000_MSIG_C7_"),]
  
  pval_go_bp <- pval[sapply(rownames(pval),FUN=grepl,pattern="FL1000_GO_BP_"),]
  pval_kegg <- pval[sapply(rownames(pval),FUN=grepl,pattern="FL1000_KEGG_"),]
  pval_tf <- pval[sapply(rownames(pval),FUN=grepl,pattern="FL1000_TF_DOROTHEA_"),]
  pval_h <- pval[sapply(rownames(pval),FUN=grepl,pattern="FL1000_MSIG_H_"),]
  pval_c1 <- pval[sapply(rownames(pval),FUN=grepl,pattern="FL1000_MSIG_C1_"),]
  pval_c2 <- pval[sapply(rownames(pval),FUN=grepl,pattern="FL1000_MSIG_C2_"),]
  pval_c3 <- pval[sapply(rownames(pval),FUN=grepl,pattern="FL1000_MSIG_C3_"),]
  pval_c4 <- pval[sapply(rownames(pval),FUN=grepl,pattern="FL1000_MSIG_C4_"),]
  pval_c5 <- pval[sapply(rownames(pval),FUN=grepl,pattern="FL1000_MSIG_C5_"),]
  pval_c6 <- pval[sapply(rownames(pval),FUN=grepl,pattern="FL1000_MSIG_C6_"),]
  pval_c7 <- pval[sapply(rownames(pval),FUN=grepl,pattern="FL1000_MSIG_C7_"),]
  
  all_NES <- list("NES GO BP" = NES_go_bp,"NES KEGG" = NES_kegg,"NES TF" = NES_tf,
                  "NES MSIG Hallmark"=NES_h,
                  "NES MSIG C1"=NES_c1,"NES MSIG C2"=NES_c2,"NES MSIG C3"=NES_c3,
                  "NES MSIG C4"=NES_c4,"NES MSIG C5"=NES_c5,"NES MSIG C6"=NES_c6,
                  "NES MSIG C7"=NES_c7)
  all_pval <- list("Pval GO BP" = pval_go_bp,"Pval KEGG" = pval_kegg,"Pval TF"=pval_tf,
                   "Pval MSIG Hallmark"=pval_h,
                   "Pval MSIG C1"=pval_c1,"Pval MSIG C2"=pval_c2,"Pval MSIG C3"=pval_c3,
                   "Pval MSIG C4"=pval_c4,"Pval MSIG C5"=pval_c5,"Pval MSIG C6"=pval_c6,
                   "Pval MSIG C7"=pval_c7)

  result <- list(NES = all_NES, Pval = all_pval)
  
  return(result)
}

library(ggplot2)
library(lmtest)
setEPS()
for (i in 1:10){
  latent_embeddings<-read.csv(paste0("results_intermediate_encoders/embs/combined_10fold/latent_embeddings_global_",i,".csv"))
  data<-latent_embeddings[,c(1:(dim(latent_embeddings)[2]-3))]
  outcomes<-latent_embeddings[,c((dim(latent_embeddings)[2]-2):dim(latent_embeddings)[2])]
  outcomes$species[outcomes$species =="human"]<-1
  outcomes$species[outcomes$species =="primates"]<-0
  
  center_apply <- function(x) {
    apply(x, 2, function(y) y - mean(y))
  }
  centered_data <- center_apply(data)
  centered_data<-data.frame(centered_data)
  
  #https://www.statology.org/likelihood-ratio-test-in-r/
  
  results <- data.frame(matrix(ncol = 32, nrow = 2))
  
  full<-cbind(centered_data,outcomes)
  
  for (j in 1:32){
    
    full_model<-lm(full[,j]~protected+vaccinated+species,data = full)
    null_model<-lm(full[,j]~vaccinated+species,data = full)
    kp<-lrtest(full_model,null_model)
    
    results[1,j]<-kp[2,5]
    results[2,j]<-summary(full_model)[["coefficients"]][, "t value"][2]
    
  }
  
  results<-data.frame(t(results))
  colnames(results)<-c("pvalue","tvalue")
  results$logp<-(-1*log(results$pvalue))
  row.names(results)<-colnames(centered_data)
  write.csv(results,paste0("LRT_results/LRT_latent_embeddings_global",i,"_after_results.csv"))
  
  results$color <- ifelse(results$logp > (-log(0.05/32)), "selected", NA_character_)
  results$label<-colnames(centered_data)
  
  setEPS()
  postscript(paste0("LRT_latent_embeddings_global_",i,".eps"))
  ggplot(results, aes(x=tvalue, y=logp,color = color)) + geom_point()+geom_hline(yintercept= (-log(0.05/32)),linetype='dashed')+theme_bw()+  
    geom_text(data=subset(results, logp >  (-log(0.05/32))),
              aes(x=tvalue, y=logp,label=label),nudge_x=0.5, nudge_y=0.5)
  dev.off()
}
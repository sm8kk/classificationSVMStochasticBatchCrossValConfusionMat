setwd("~/Box Sync/throughput-mon/dataTransferModel/feature-analysis/ps-fdt-all-tests")
library(ggplot2)
dat=read.csv("c-gamma-rbf-diag.csv", header=T)

ggplot(dat, aes(dat$C, dat$Gamma, color=dat$DiagSum)) + 
  geom_point(size=3) +
  scale_y_log10() + scale_x_log10()  +
  labs(title = "Heat map of diagonal sum", x = "C", y = "Gamma", color = "DiagSum") +
  theme_bw() + 
  #xlab("C") + ylab("Gamma") +
  theme(legend.text=element_text(size=20), legend.title=element_text(size=20), axis.title = element_text(size=20, face="bold"), axis.text=element_text(size=20, face="bold"))

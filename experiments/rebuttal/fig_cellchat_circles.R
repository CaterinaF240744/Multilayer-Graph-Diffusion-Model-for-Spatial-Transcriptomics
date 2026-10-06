## fig_cellchat_circles_v2.R
## Figura CellChat in una sola pagina: (A) rete aggregata, (B) SPP1, (C) VEGF.
## Uso (dalla cartella python_files):
##   Rscript fig_cellchat_circles_v2.R
suppressPackageStartupMessages(library(CellChat))

rds <- "results/cellchat_full_v2/cellchat_spatial.rds"
out <- "model_validation_imgs/fig_cellchat_circles.pdf"
dir.create(dirname(out), showWarnings = FALSE, recursive = TRUE)

cc <- readRDS(rds)
group_size <- as.numeric(table(cc@idents))
stopifnot(all(c("SPP1", "VEGF") %in% cc@netP$pathways))

panels <- list(
  list(mat = cc@net$weight,             title = "A  All pathways (interaction strength)"),
  list(mat = cc@netP$prob[, , "SPP1"],  title = "B  SPP1"),
  list(mat = cc@netP$prob[, , "VEGF"],  title = "C  VEGF")
)

pdf(out, width = 13, height = 4.6)
par(mfrow = c(1, 3), xpd = TRUE, mar = c(1, 1, 2, 1))
for (p in panels) {
  netVisual_circle(p$mat, vertex.weight = group_size, weight.scale = TRUE,
                   label.edge = FALSE, title.name = p$title)
}
dev.off()
cat("Figura salvata in:", out, "\n")
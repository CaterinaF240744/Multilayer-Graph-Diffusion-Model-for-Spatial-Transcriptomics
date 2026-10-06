#!/usr/bin/env Rscript
## run_rctd_mouse.R -- RCTD (spacexr) sul V1 Mouse Kidney, Sezione 8.3.
## Modalita' "full" (raccomandata per Visium). Macro-tipo dominante = somma dei
## pesi normalizzati per macro-tipo (stesso criterio dell'analisi umana);
## l'etichetta salvata e' il tipo del riferimento con peso massimo dentro il
## macro-tipo vincente.
## Uso (cartella python_files, ambiente rctd):
##   ~/miniforge3/envs/rctd/bin/Rscript experiments/rebuttal/run_rctd_mouse.R
## Output: results/annotation/labels_rctd.csv, weights_rctd.csv
suppressPackageStartupMessages({ library(spacexr); library(Matrix) })
indir <- "data/rctd_io"; outdir <- "results/annotation"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
set.seed(0)

ref <- as(readMM(file.path(indir, "sc_counts.mtx")), "CsparseMatrix")
rownames(ref) <- make.unique(read.csv(file.path(indir, "sc_genes.csv"), header = FALSE)$V1)
bcs <- read.csv(file.path(indir, "sc_barcodes.csv"), header = FALSE)$V1
colnames(ref) <- bcs
meta <- read.csv(file.path(indir, "sc_meta.csv"), row.names = 1)
ct <- gsub("/", "_or_", as.character(meta[bcs, "cell_identity"]))
tab <- table(ct); keep <- ct %in% names(tab)[tab >= 25]
cat("Tipi esclusi (<25 cellule):", paste(names(tab)[tab < 25], collapse = ", "), "\n")
ref <- ref[, keep]; ct <- factor(ct[keep]); names(ct) <- colnames(ref)
reference <- Reference(ref, ct)

mp <- read.csv(file.path(indir, "macro_map.csv"))
macro_map <- setNames(mp$macro_type, gsub("/", "_or_", mp$cell_identity))

cnt <- as(readMM(file.path(indir, "st_counts.mtx")), "CsparseMatrix")
rownames(cnt) <- make.unique(read.csv(file.path(indir, "st_genes.csv"), header = FALSE)$V1)
colnames(cnt) <- read.csv(file.path(indir, "st_barcodes.csv"), header = FALSE)$V1
xy <- read.csv(file.path(indir, "st_coords.csv"), row.names = 1)
query <- SpatialRNA(xy[colnames(cnt), c("x", "y")], cnt)

r <- create.RCTD(query, reference, max_cores = 4)
r <- run.RCTD(r, doublet_mode = "full")
w <- as.matrix(normalize_weights(r@results$weights))
mt <- macro_map[colnames(w)]
if (any(is.na(mt))) stop("tipi senza macro-tipo: ", paste(colnames(w)[is.na(mt)], collapse = ", "))
M <- sapply(sort(unique(mt)), function(m) rowSums(w[, mt == m, drop = FALSE]))
win <- colnames(M)[max.col(M, ties.method = "first")]
lab <- vapply(seq_len(nrow(w)), function(i) {
  cols <- which(mt == win[i]); colnames(w)[cols[which.max(w[i, cols])]] }, character(1))
df <- data.frame(barcode = rownames(w), rctd_label = gsub("_or_", "/", lab), rctd_macro = win,
                 rctd_macro_weight = round(M[cbind(seq_len(nrow(M)), match(win, colnames(M)))], 4))
write.csv(df, file.path(outdir, "labels_rctd.csv"), row.names = FALSE)
write.csv(round(w, 4), file.path(outdir, "weights_rctd.csv"))
cat("RCTD completato:", nrow(df), "spot\n"); print(table(df$rctd_macro))

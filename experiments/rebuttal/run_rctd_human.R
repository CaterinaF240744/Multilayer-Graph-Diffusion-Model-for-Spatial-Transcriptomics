#!/usr/bin/env Rscript
## run_rctd_human.R -- RCTD (spacexr) sui campioni Visium umani.
## Modalita' "full" (raccomandata per Visium: piu' cellule per spot).
## Macro-tipo dominante = macro-tipo con la SOMMA dei pesi normalizzati piu' alta
## (il modello lavora sui macro-tipi; un argmax sui sottotipi penalizzerebbe i
## macro-tipi divisi in piu' sottotipi, es. PT_S1/S2/S3/iPT). L'etichetta salvata
## e' il sottotipo con peso massimo dentro il macro-tipo vincente.
## Riferimento: Susztak GSE211785 (file preparati da prepare_human_reference.py).
##
## Uso (dalla cartella python_files):
##   Rscript experiments/rebuttal/run_rctd_human.R
## Output: data/rctd_human/<dataset>/<campione>/labels_rctd.csv
##   (barcode, rctd_label, rctd_weight). I campioni gia' fatti vengono saltati.
suppressPackageStartupMessages({ library(spacexr); library(Matrix) })

ref_dir <- "data/human_ref"
mp <- read.csv(file.path(ref_dir, "human_macro_map.csv"))
macro_map <- setNames(mp$macro_type, gsub("/", "_or_", mp$cell_identity))
st_root <- "data/rctd_human"
CORES <- 4
MIN_CELLS <- 25
set.seed(0)

## ---- Riferimento (una sola volta) ----
ref <- as(readMM(file.path(ref_dir, "sc_counts.mtx")), "CsparseMatrix")   # geni x cellule
rownames(ref) <- make.unique(readLines(file.path(ref_dir, "sc_genes.txt")))
bcs <- readLines(file.path(ref_dir, "sc_barcodes.txt"))
colnames(ref) <- bcs
meta <- read.delim(file.path(ref_dir, "sc_meta.txt"))
ct <- setNames(as.character(meta$Cluster_Idents), meta$barcode)[bcs]
ct <- gsub("/", "_or_", ct)                     # RCTD non accetta "/" nei nomi
tab <- table(ct)
keep <- ct %in% names(tab)[tab >= MIN_CELLS]
cat("Tipi esclusi (<", MIN_CELLS, "cellule):", paste(names(tab)[tab < MIN_CELLS], collapse = ", "), "\n")
ref <- ref[, keep]
ct <- factor(ct[keep]); names(ct) <- colnames(ref)
reference <- Reference(ref, ct)
cat("Riferimento:", ncol(ref), "cellule,", nlevels(ct), "tipi\n")

## ---- Campioni ----
dirs <- list.dirs(st_root, recursive = TRUE)
dirs <- dirs[file.exists(file.path(dirs, "st_counts.mtx"))]
for (d in dirs) {
  out <- file.path(d, "labels_rctd.csv")
  if (file.exists(out)) { cat("gia' fatto:", d, "\n"); next }
  t0 <- Sys.time()
  cnt <- as(readMM(file.path(d, "st_counts.mtx")), "CsparseMatrix")
  rownames(cnt) <- make.unique(readLines(file.path(d, "st_genes.txt")))
  colnames(cnt) <- readLines(file.path(d, "st_barcodes.txt"))
  xy <- read.csv(file.path(d, "st_coords.csv"), row.names = 1)
  query <- SpatialRNA(xy[colnames(cnt), c("x", "y")], cnt)
  r <- create.RCTD(query, reference, max_cores = CORES)
  r <- run.RCTD(r, doublet_mode = "full")
  w <- as.matrix(normalize_weights(r@results$weights))
  mt <- macro_map[colnames(w)]
  if (any(is.na(mt))) stop("tipi senza macro-tipo: ", paste(colnames(w)[is.na(mt)], collapse = ", "))
  M <- sapply(sort(unique(mt)), function(m) rowSums(w[, mt == m, drop = FALSE]))
  if (is.null(dim(M))) M <- matrix(M, nrow = nrow(w), dimnames = list(rownames(w), sort(unique(mt))))
  win <- colnames(M)[max.col(M, ties.method = "first")]
  lab <- vapply(seq_len(nrow(w)), function(i) {
    cols <- which(mt == win[i]); colnames(w)[cols[which.max(w[i, cols])]] }, character(1))
  df <- data.frame(barcode = rownames(w),
                   rctd_label = gsub("_or_", "/", lab),
                   rctd_macro = win,
                   rctd_macro_weight = round(M[cbind(seq_len(nrow(M)), match(win, colnames(M)))], 4))
  write.csv(df, out, row.names = FALSE)
  write.csv(round(w, 4), file.path(d, "weights_rctd.csv"))
  write.csv(round(M, 4), file.path(d, "macro_weights_rctd.csv"))
  cat(d, ":", nrow(df), "spot in", round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 1), "min\n")
}
cat("RCTD completato\n")

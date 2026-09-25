#!/usr/bin/env Rscript
# run_rctd.R -- RCTD (spacexr) label transfer for the annotation-robustness
# experiment (Reviewer 2, annotation comparison: NN vs RCTD vs cell2location).
#
# Usage:
#   Rscript run_rctd.R [input_dir] [output_dir]
#
# Inputs (in input_dir, default "data/rctd_io"; produced by
#   export_data_for_cellchat.py or an equivalent exporter):
#   st_counts.mtx, st_barcodes.csv, st_genes.csv, st_coords.csv
#   sc_counts.mtx, sc_barcodes.csv, sc_genes.csv, sc_meta.csv
#     (sc_meta.csv must have a "cell_identity" column)
#
# Output (in output_dir, default "results/annotation"):
#   labels_rctd.csv  (barcode, rctd_label, rctd_score)

suppressPackageStartupMessages(library(spacexr))
suppressPackageStartupMessages(library(Matrix))

args <- commandArgs(trailingOnly = TRUE)
indir <- ifelse(length(args) >= 1, args[1], "data/rctd_io")
outdir <- ifelse(length(args) >= 2, args[2], "results/annotation")

counts <- readMM(file.path(indir, "st_counts.mtx"))          # genes x spots
st_barcodes <- read.csv(file.path(indir, "st_barcodes.csv"), header = FALSE)$V1
st_genes <- read.csv(file.path(indir, "st_genes.csv"), header = FALSE)$V1
rownames(counts) <- st_genes
colnames(counts) <- st_barcodes

ref_counts <- readMM(file.path(indir, "sc_counts.mtx"))      # genes x cells
sc_barcodes <- read.csv(file.path(indir, "sc_barcodes.csv"), header = FALSE)$V1
sc_genes <- read.csv(file.path(indir, "sc_genes.csv"), header = FALSE)$V1
rownames(ref_counts) <- sc_genes
colnames(ref_counts) <- sc_barcodes

sc_meta <- read.csv(file.path(indir, "sc_meta.csv"), row.names = 1)
cell_types <- factor(gsub("/", "_or_", sc_meta[sc_barcodes, "cell_identity"]))
names(cell_types) <- sc_barcodes

# Reference reduction (memory: RCTD densifies internally):
#   1. drop cell types with fewer than MIN_CELLS cells
#   2. subsample to at most 15,000 cells
#   3. re-apply the MIN_CELLS floor in case subsampling removed a type below it
set.seed(0)
MIN_CELLS <- 25

drop_rare_types <- function(barcodes, types, min_cells, note) {
  tab <- table(types)
  rare <- names(tab)[tab < min_cells]
  if (length(rare) == 0) return(list(barcodes = barcodes, types = types))
  drop <- barcodes %in% names(types)[types %in% rare]
  cat(note, ":", length(rare), "rare cell type(s) dropped\n")
  list(barcodes = barcodes[!drop], types = droplevels(types[barcodes[!drop]]))
}

step1 <- drop_rare_types(sc_barcodes, cell_types, MIN_CELLS, "initial filter")
sc_barcodes <- step1$barcodes
cell_types <- step1$types
ref_counts <- ref_counts[, sc_barcodes]

if (length(sc_barcodes) > 15000) {
  keep <- sample(length(sc_barcodes), 15000)
  sc_barcodes <- sc_barcodes[keep]
  ref_counts <- ref_counts[, keep]
  cell_types <- droplevels(cell_types[sc_barcodes])
  cat("reference subsampled to", length(sc_barcodes), "cells\n")
}

step2 <- drop_rare_types(sc_barcodes, cell_types, MIN_CELLS, "post-subsample filter")
sc_barcodes <- step2$barcodes
cell_types <- step2$types
ref_counts <- ref_counts[, sc_barcodes]

common_genes <- intersect(rownames(counts), rownames(ref_counts))
cat("common genes:", length(common_genes), "\n")

coords <- read.csv(file.path(indir, "st_coords.csv"), row.names = 1)
query <- SpatialRNA(coords = data.frame(x = coords[colnames(counts), "x"],
                                         y = coords[colnames(counts), "y"],
                                         row.names = colnames(counts)),
                     counts = counts[common_genes, ])

reference <- Reference(ref_counts[common_genes, ], cell_types)

rctd <- create.RCTD(query, reference, max_cores = 4)
rctd <- run.RCTD(rctd, doublet_mode = "doublet")

res <- rctd@results
df <- data.frame(
  barcode = rownames(res$results),
  rctd_label = as.character(res$results$first_type),
  rctd_score = res$results$singlet_score
)
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
write.csv(df, file.path(outdir, "labels_rctd.csv"), row.names = FALSE)
cat("RCTD done:", nrow(df), "spots\n")

#!/usr/bin/env Rscript
## cellchat_analysis.R
## --------------------
## Independent, orthogonal validation of the amplifier/sink roles reported
## in the manuscript (Table 1, Section 6.5), using CellChat v2's
## SPATIALLY-CONSTRAINED communication inference on the SAME V1 Mouse
## Kidney Visium dataset and macro-type labels used elsewhere in this work.
##
## This is the canonical version of this script: CellChat is run with
## datatype = "spatial" and distance.use = TRUE, so that inferred
## ligand-receptor communication is restricted to spatially plausible
## (paracrine/juxtacrine-range) signalling -- consistent with the spatially
## constrained diffusion model this analysis is meant to validate against.
## A non-spatial configuration (datatype = "RNA", distance.use = FALSE)
## pools signalling across the whole tissue regardless of physical distance
## between spots and should NOT be used for this comparison.
##
## Usage:
##   Rscript cellchat_analysis.R [input_dir] [output_dir]
##
## Inputs (in input_dir, default "data/cellchat_inputs"; produced by
##   export_data_for_cellchat.py):
##   v1_kidney_raw_counts.mtx   (genes x spots, RAW counts)
##   v1_kidney_meta.csv         (columns: barcode, macro_type)
##   v1_kidney_spatial_coords.csv (columns: barcode, x, y)
##   v1_kidney_genes.csv, v1_kidney_barcodes.csv
##
## Outputs (in output_dir, default "results/cellchat"):
##   cellchat_spatial.rds              full CellChat object
##   pathway_weight_<PATHWAY>.csv       per-pathway sender x receiver matrix
##   cellchat_aggregated_weight.csv     aggregated interaction strength (3x3)
##   cellchat_aggregated_count.csv      aggregated significant L-R pair count (3x3)
##   cellchat_signaling_roles.csv       outgoing/incoming strength per macro-type
##   cellchat_candidate_pathways_circle.pdf   circle plots, one page per
##                                             detected candidate pathway
##
## Next step: join cellchat_signaling_roles.csv and cellchat_aggregated_*.csv
## against the model's R_diff (Table 1) and K_dyn (Section 6.5) values with
## compare_cellchat_vs_model.py.

suppressPackageStartupMessages(library(CellChat))
suppressPackageStartupMessages(library(Matrix))
suppressPackageStartupMessages(library(dplyr))

## ---------------------------------------------------------------------
## 0. Inputs / outputs
## ---------------------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
data_dir <- ifelse(length(args) >= 1, args[1], "data/cellchat_inputs")
out_dir  <- ifelse(length(args) >= 2, args[2], "results/cellchat")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

counts_path   <- file.path(data_dir, "v1_kidney_raw_counts.mtx")
meta_path     <- file.path(data_dir, "v1_kidney_meta.csv")
coords_path   <- file.path(data_dir, "v1_kidney_spatial_coords.csv")
genes_path    <- file.path(data_dir, "v1_kidney_genes.csv")
barcodes_path <- file.path(data_dir, "v1_kidney_barcodes.csv")

## Approximate Visium center-to-center spot spacing in microns; use the SAME
## physical spacing implied by the spot-level kNN graph (Section 6.1) so the
## two methods "see" the same neighbourhood.
VISIUM_SPOT_SPACING_UM <- 100

## ---------------------------------------------------------------------
## 1. Load data
## ---------------------------------------------------------------------
counts   <- readMM(counts_path)
genes    <- read.csv(genes_path, header = FALSE)$V1
barcodes <- read.csv(barcodes_path, header = FALSE)$V1

## CellChatDB.mouse uses standard mouse gene symbol convention (first
## letter capitalised, rest lowercase, e.g. "Sulf1"), but genes are
## exported all-lowercase for SC/ST alignment consistency elsewhere in the
## pipeline. Convert here so gene symbols match the database.
capitalize_first <- function(x) paste0(toupper(substr(x, 1, 1)), substr(x, 2, nchar(x)))
genes <- capitalize_first(as.character(genes))
rownames(counts) <- genes
colnames(counts) <- as.character(barcodes)

meta   <- read.csv(meta_path, row.names = "barcode")
coords <- read.csv(coords_path, row.names = "barcode")
coords <- as.matrix(coords[rownames(meta), c("x", "y")])

stopifnot(all(rownames(meta) == colnames(counts)))

## Restrict to well-represented macro-types only. 'vascular', 'immune', and
## 'other' are excluded on this dataset: too few spots for stable CellChat
## estimates, and the immune compartment specifically could not be resolved
## because canonical macrophage markers (Adgre1, Cd68, Csf1r) are absent
## from the V1_Mouse_Kidney Visium feature matrix entirely (verified
## pre-filtering, see export_data_for_cellchat.py diagnostics) -- a
## limitation of the public demonstration dataset, not of the pipeline.
keep_types <- c("PT", "DCT", "TAL")
keep_spots <- rownames(meta)[meta$macro_type %in% keep_types]

meta   <- meta[keep_spots, , drop = FALSE]
counts <- counts[, keep_spots]
coords <- coords[keep_spots, ]
meta$macro_type <- factor(meta$macro_type, levels = keep_types)

cat("Retained", length(keep_spots), "spots across", length(keep_types), "macro-types:\n")
print(table(meta$macro_type))

## ---------------------------------------------------------------------
## 2. Build and run CellChat in SPATIAL mode
## ---------------------------------------------------------------------
## API NOTE (CellChat >= 2.2): the older `scale.factors = list(spot.diameter,
## spot)` interface was replaced by `spatial.factors = list(ratio, tol)`:
##   ratio = theoretical spot size (um) / spot_diameter_fullres (px)
##   tol   = robustness tolerance (um) when comparing center-to-center
##           distances against interaction.range; half the nominal spot size.
cellchat <- createCellChat(
  object = counts, meta = meta, group.by = "macro_type",
  datatype = "spatial", coordinates = coords,
  spatial.factors = list(ratio = 65 / 89.45675017406688, tol = 65 / 2)
)

cellchat@DB <- CellChatDB.mouse
cellchat <- subsetData(cellchat)
cat("data.signaling dim:", dim(cellchat@data.signaling), "\n")
cellchat <- identifyOverExpressedGenes(cellchat, do.fast = FALSE)
cellchat <- identifyOverExpressedInteractions(cellchat)

## interaction.range matches the ~6-nearest-neighbour physical radius used
## in the Python spatial graph (Section 6.1); this restricts inference to
## spatially plausible (paracrine/juxtacrine-range) signalling instead of
## pooling communication across the whole tissue irrespective of distance.
## contact.dependent = FALSE: all L-R pairs use the diffusion-based model
## (probability inversely proportional to spatial distance, hard cutoff at
## interaction.range), consistent with the diffusion model being validated.
cellchat <- computeCommunProb(
  cellchat, type = "truncatedMean", trim = 0.1,
  distance.use = TRUE, interaction.range = VISIUM_SPOT_SPACING_UM * 2.5,
  scale.distance = 0.011, contact.dependent = FALSE
)

cellchat <- filterCommunication(cellchat, min.cells = 10)
cellchat <- computeCommunProbPathway(cellchat)
cellchat <- aggregateNet(cellchat)

## Save the full object and per-pathway aggregated networks so manuscript
## figures (e.g. the VEGF circle plot) can be regenerated/verified without
## re-running the inference.
saveRDS(cellchat, file.path(out_dir, "cellchat_spatial.rds"))
for (pw in names(cellchat@netP$weight)) {
  write.csv(cellchat@netP$weight[[pw]],
            file.path(out_dir, paste0("pathway_weight_", pw, ".csv")))
}

## ---------------------------------------------------------------------
## 3. Aggregated network (3x3, sender x receiver)
## ---------------------------------------------------------------------
write.csv(cellchat@net$weight, file.path(out_dir, "cellchat_aggregated_weight.csv"))
write.csv(cellchat@net$count,  file.path(out_dir, "cellchat_aggregated_count.csv"))

## ---------------------------------------------------------------------
## 4. Signalling role analysis (sender / receiver / mediator / influencer)
## ---------------------------------------------------------------------
## Non-fatal tryCatch so a centrality failure is visible in the log instead
## of silently producing an empty roles table.
tryCatch(
  cellchat <- netAnalysis_computeCentrality(cellchat, slot.name = "netP"),
  error = function(e) cat("centrality failed (non-fatal):", conditionMessage(e), "\n")
)

## Outgoing (sender) and incoming (receiver) strength per macro-type,
## aggregated across all significant signalling pathways -- the quantity
## correlated against the model's R_diff / K_dyn row-sums.
centr <- slot(cellchat, "netP")$centr
roles <- lapply(names(centr), function(pw) {
  data.frame(
    pathway = pw,
    macro_type = names(centr[[pw]]$outdeg),
    outgoing = centr[[pw]]$outdeg,
    incoming = centr[[pw]]$indeg
  )
})
roles_df <- bind_rows(roles)

roles_summary <- roles_df %>%
  group_by(macro_type) %>%
  summarise(outgoing_strength = sum(outgoing, na.rm = TRUE),
            incoming_strength = sum(incoming, na.rm = TRUE)) %>%
  arrange(desc(outgoing_strength))

write.csv(roles_summary, file.path(out_dir, "cellchat_signaling_roles.csv"), row.names = FALSE)

## ---------------------------------------------------------------------
## 5. Candidate biologically-relevant pathways (circle plots)
## ---------------------------------------------------------------------
candidate_pathways <- c("COMPLEMENT", "CXCL", "CCL", "VEGF", "ANGPT", "TGFb", "TNF")
present_pathways <- intersect(candidate_pathways, unique(roles_df$pathway))
cat("Candidate biologically-relevant pathways detected by CellChat:\n")
print(present_pathways)

if (length(present_pathways) > 0) {
  pdf(file.path(out_dir, "cellchat_candidate_pathways_circle.pdf"))
  for (pw in present_pathways) {
    netVisual_aggregate(cellchat, signaling = pw, layout = "circle")
  }
  dev.off()
}

cat("Done. Outputs written to:", out_dir, "\n")
cat("Next step: run compare_cellchat_vs_model.py to join these CSVs against\n")
cat("the manuscript's R_diff (Table 1) and K_dyn (Section 6.5) values.\n")

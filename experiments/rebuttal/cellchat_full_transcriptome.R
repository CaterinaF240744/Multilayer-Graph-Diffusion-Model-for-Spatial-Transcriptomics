## cellchat_full_transcriptome.R
## -----------------------------------------------------------------------
## CellChat v2 (spatially-constrained) on the FULL-TRANSCRIPTOME V1 Mouse
## Kidney Visium matrix (32,285 genes; PT/DCT/TAL spots: 881/437/110).
## Manuscript: Section 8.6 (Table 15, Table 16). Same settings as
## cellchat_analysis_spatial.R, which ran on the 1,460-gene label-transfer
## panel (81/687 CellChatDB genes) and is kept only for the panel-vs-full
## comparison in Table 16.
##
## Usage (from repository root):
##   Rscript experiments/rebuttal/cellchat_full_transcriptome.R [input_dir] [output_dir]
## Inputs in input_dir: v1_kidney_full_counts.mtx (genes x spots, raw counts),
##   v1_kidney_full_genes.csv, v1_kidney_full_barcodes.csv,
##   v1_kidney_full_meta.csv (barcode, macro_type),
##   v1_kidney_full_coords.csv (barcode, x, y in full-resolution pixels)
## -----------------------------------------------------------------------
library(CellChat)
library(Matrix)
library(dplyr)

## ---------------------------------------------------------------------
## 0. Inputs -- paths adapted for this sandbox environment
## ---------------------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
data_dir <- if (length(args) >= 1) args[1] else "data/full_matrix_inputs"
counts_path <- file.path(data_dir, "v1_kidney_full_counts.mtx") # genes x spots, RAW counts
meta_path <- file.path(data_dir, "v1_kidney_full_meta.csv") # must contain: barcode, macro_type
coords_path <- file.path(data_dir, "v1_kidney_full_coords.csv") # must contain: barcode, x, y
genes_path <- file.path(data_dir, "v1_kidney_full_genes.csv")
barcodes_path <- file.path(data_dir, "v1_kidney_full_barcodes.csv")
out_dir <- if (length(args) >= 2) args[2] else "results/cellchat_full/"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

## Approximate Visium center-to-center spot spacing in microns; use the SAME
## physical spacing implied by your spot-level kNN graph (Section 6.1) so
## the two methods "see" the same neighbourhood.
VISIUM_SPOT_SPACING_UM <- 100

## ---------------------------------------------------------------------
## 1. Load data
## ---------------------------------------------------------------------
counts <- readMM(counts_path)
genes <- read.csv(genes_path, header = FALSE)$V1
barcodes <- read.csv(barcodes_path, header = FALSE)$V1
## CellChatDB.mouse uses standard mouse gene symbol convention
## (first letter capitalised, rest lowercase, e.g. "Sulf1"), but our
## genes were exported all-lowercase for SC/ST alignment consistency.
## Convert here so gene symbols match the database.
capitalize_first <- function(x) paste0(toupper(substr(x, 1, 1)), substr(x, 2, nchar(x)))
genes <- capitalize_first(as.character(genes))
rownames(counts) <- genes
colnames(counts) <- as.character(barcodes)


meta <- read.csv(meta_path, row.names = "barcode")
coords <- read.csv(coords_path, row.names = "barcode")
coords <- as.matrix(coords[rownames(meta), c("x", "y")])

stopifnot(all(rownames(meta) == colnames(counts)))

## Restrict to well-represented macro-types only. 'vascular' (n=8),
## 'immune' (n=0), and 'other' (n=2) are excluded: too few spots for
## stable CellChat estimates, and the immune compartment specifically
## could not be resolved because canonical macrophage markers
## (Adgre1, Cd68, Csf1r) are absent from the V1_Mouse_Kidney Visium
## feature matrix entirely (verified pre-filtering) -- a limitation of
## the public demonstration dataset, not of the analysis pipeline.
keep_types <- c("PT", "DCT", "TAL")
keep_spots <- rownames(meta)[meta$macro_type %in% keep_types]

meta <- meta[keep_spots, , drop = FALSE]
counts <- counts[, keep_spots]
coords <- coords[keep_spots, ]

meta$macro_type <- factor(meta$macro_type,
 levels = keep_types)

cat("Retained", length(keep_spots), "spots across", length(keep_types), "macro-types:\n")
print(table(meta$macro_type))

## NOTE: 'vascular' (n=8), 'immune' (n=0), and 'other' (n=2) are excluded.
## The immune compartment could not be resolved from this Visium sample:
## canonical macrophage markers (Adgre1, Cd68, Csf1r) are not detected in
## the V1_Mouse_Kidney feature matrix at all (checked pre-filtering), so
## label transfer via sc.tl.ingest has no signal to assign immune spots.
## This is a limitation of the public demonstration dataset, not of the
## CellChat/label-transfer pipeline. See export_data_for_cellchat.py
## diagnostics.


## ---------------------------------------------------------------------
## 2. Build and run CellChat in SPATIAL mode (the fix is in this block)
## ---------------------------------------------------------------------
## API NOTE (CellChat 2.2): the old `scale.factors = list(spot.diameter, spot)`
## interface was replaced by `spatial.factors = list(ratio, tol)`:
##   ratio = theoretical spot size (um) / spot_diameter_fullres (px)
##         -> 65 / 89.45675017406688 = 0.7266 um per pixel, which reproduces the
##            nominal 100 um Visium center-to-center spacing (137 px measured).
##   tol   = half the spot size in um (65/2 = 32.5), the robustness tolerance
##         used when comparing center-to-center distances against interaction.range.
cellchat <- createCellChat(object = counts, meta = meta, group.by = "macro_type",
 datatype = "spatial", coordinates = coords,
 spatial.factors = list(ratio = 65 / 89.45675017406688, tol = 65 / 2))

cellchat@DB <- CellChatDB.mouse
cellchat <- subsetData(cellchat)
cat("data.signaling dim:", dim(cellchat@data.signaling), "\n")
cellchat <- identifyOverExpressedGenes(cellchat, do.fast = FALSE)
cellchat <- identifyOverExpressedInteractions(cellchat)

## interaction.length matches the ~6-nearest-neighbour physical radius used in
## the Python spatial graph (Section 6.1); this is what restricts inference to
## spatially plausible (paracrine/juxtacrine-range) signalling instead of
## pooling communication across the whole tissue irrespective of distance.
## interaction.range (renamed from interaction.length in CellChat 2.2) matches
## the ~6-nearest-neighbour physical radius used in the Python spatial graph
## (Section 6.1); this is what restricts inference to spatially plausible
## (paracrine/juxtacrine-range) signalling instead of pooling communication
## across the whole tissue irrespective of distance.
## contact.dependent = FALSE: all L-R pairs are treated in the diffusion manner
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

## Save the full object and pathway-level aggregated networks so the
## manuscript figures (e.g. VEGF circle plot) can be regenerated/verified
## without re-running the inference.
saveRDS(cellchat, file.path(out_dir, "cellchat_spatial.rds"))
for (pw in names(cellchat@netP$weight)) {
  write.csv(cellchat@netP$weight[[pw]],
            file.path(out_dir, paste0("pathway_weight_", pw, ".csv")))
}

## ---------------------------------------------------------------------
## 3. Extract the aggregated network (3x3, weight = interaction strength)
## ---------------------------------------------------------------------
net_weight <- cellchat@net$weight
write.csv(net_weight, file.path(out_dir, "cellchat_aggregated_weight.csv"))

net_count <- cellchat@net$count # number of significant L-R pairs, for reference
write.csv(net_count, file.path(out_dir, "cellchat_aggregated_count.csv"))

## ---------------------------------------------------------------------
## 4. Signaling role analysis (sender / receiver / mediator / influencer)
## ---------------------------------------------------------------------
## Kept as a non-fatal tryCatch (safe defensive addition from the later
## session) so a centrality failure is visible in the log instead of
## silently producing an empty roles table -- but it now runs on the
## spatially-constrained network, not the non-spatial one.
tryCatch(cellchat <- netAnalysis_computeCentrality(cellchat, slot.name = "netP"),
 error = function(e) cat("centrality failed (non-fatal):", conditionMessage(e), "\n"))

## Outgoing (sender) and incoming (receiver) strength per macro-type,
## aggregated across all significant signalling pathways -- this is the
## quantity to correlate against the model's R_diff / K_dyn row-sums.
centr <- slot(cellchat, "netP")$cent
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
## 5. Pathway-level check for the specific axes discussed in the manuscript
## (PT -> immune "absorption", PT -> vascular corridor)
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
cat("the manuscript's R_diff (Table 1) and K_dyn (Table 6) values.\n")

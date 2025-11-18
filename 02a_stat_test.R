##statistical test for 
#Because each augmentation–model combination was evaluated on the identical set of 120 MCQs, per-run accuracies
#form natural paired observations; we therefore used paired non-parametric tests (Wilcoxon signed-rank) for within-model
# comparisons with Bonferroni correction was used

rm(list=ls())
# Packages
for (n in c('dplyr','tidyr', 'data.table')) {
  if(!require(n,character.only=TRUE)) { 
    install.packages(n)
  }
}
library(n,character.only=TRUE)

db_results = data.table(read.xlsx("placeholder_for_results_file", sheet=1))
#db_results = data.table(read.xlsx("data//eval_full_results_v5.xlsx", sheet=1))

# df_long structure assumed:
# df_long <- data.frame(
#   item_id      = integer,
#   run_id       = integer,
#   model        = character,
#   augmentation = character, # e.g. "None", "Text", "Multi-modal", "ColPali"
#   correct      = integer    # 0/1
# )

# ----- 1) Choose model to analyse -----

df_long<-data.table(item_id=db_results$Question_nr , run_id=db_results$timestamp ,  model=db_results$model,
                    augmentation=db_results$vd_name, correct=db_results$Cor_answer )

model_of_interests <- c("placeholder_name_the_models")  
#model_of_interests <- c("gpt-4o-2024-11-20", "gpt-4o-mini-2024-07-18", "google/gemma-3-27b-it")  

augmentations <- c("placeholder_name_RAG_frameworks")  
#augmentations <- c("no_RAG", "text_RAG", "mm_RAG", "colpali")


for (model_of_interest in model_of_interests)   {

df_model <- df_long %>%
  filter(model == model_of_interest)

# ----- 2) Compute item-wise accuracy per augmentation -----
item_acc <- df_model %>%
  group_by(item_id, augmentation) %>%
  summarise(
    acc = mean(correct),
    .groups = "drop"
  ) %>%
  # pivot to wide format: one row per item, columns = augmentations
  pivot_wider(
    names_from  = augmentation,
    values_from = acc
  )

# Inspect
head(item_acc)

# Suppose your augmentations are: "None", "Text", "Multi-modal", "ColPali"
# Check column names:
colnames(item_acc)

# ----- 4) (Optional) Run all pairwise comparisons in one go -----

#augmentations <- c("no_RAG", "text_RAG", "mm_RAG", "colpali")

pairs <- combn(augmentations, 2, simplify = FALSE)

wilcox_results <- map(pairs, function(p) {
  x <- item_acc[[p[1]]]
  y <- item_acc[[p[2]]]
  
  test <- wilcox.test(
    x, y,
    paired      = TRUE,
    alternative = "two.sided",
    exact       = FALSE
  )
  
  tibble(
    model        = model_of_interest,
    aug_1        = p[1],
    aug_2        = p[2],
    p_value      = test$p.value,
    statistic    = unname(test$statistic),
    n_items_used = sum(!is.na(x) & !is.na(y))
  )
})

wilcox_results_tbl <- bind_rows(wilcox_results)
print(wilcox_results_tbl$p_value)
print(wilcox_results_tbl$statistic)
print(wilcox_results_tbl)

}

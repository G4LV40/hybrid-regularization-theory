# ===========================================================
# Libraries
# ===========================================================
library(glmnet)
library(randomForest)
library(xgboost)
library(lightgbm)
library(catboost)
library(h2o)
library(dplyr)
library(tibble)
library(tidyr)
library(Metrics)
library(purrr)
library(doParallel)
library(foreach)

# ===========================================================
# Initialize H2O
# ===========================================================
h2o.init(nthreads = -1, max_mem_size = "30G")


source("gen_data.R")
source('func_cross_validation.R')
source('evaluate_models.R')



# ===========================================================
# Simulation wrapper
# ===========================================================


cl <- makeCluster(parallel::detectCores() - 1)  # usa todos menos 1 núcleo
registerDoParallel(cl)

simulate_multiple_parallel <- function(n_sim = 10, n_vals = n_vals, p_vals = p_vals, k = k) {
  foreach(sim = 1:n_sim, .combine = bind_rows,
          .packages = c("dplyr", "tibble", "glmnet", "randomForest",
                        "xgboost", "lightgbm", "catboost", "h2o", 
                        "Metrics", "purrr"),
          .export = c("generate_data", "evaluate_models", 
                      "eval_subset_kfold", "compute_support_recovery")) %dopar% {
                        
                        h2o::h2o.init(nthreads = -1, max_mem_size = "30G")
                        
                        df_sim <- map_dfr(n_vals, function(n) {
                          map_dfr(p_vals, function(p) {
                            data <- generate_data(n, p, seed = sim + n + p)
                            res_names <- names(evaluate_models(matrix(runif(n * p), nrow = n), rnorm(n), k = 2))
                            
                            res <- tryCatch({
                              evaluate_models(data$X, data$y, k, true_vars = data$true_vars)
                            }, error = function(e) {
                              warning(paste("Erro na simulação", sim, "- n:", n, "p:", p, ":", e$message))
                              setNames(rep(NA, length(res_names)), res_names)
                            })
                            
                            tibble(sim = sim, n = n, p = p, metric = names(res), value = unlist(res))
                          })
                        })
                        
                        # Salvamento incremental
                        write.csv(df_sim, paste0("results_sim_", sim, ".csv"), row.names = FALSE)
                        
                        df_sim
                      }
}


# ===========================================================
# simulations
# ===========================================================

df_results <- simulate_multiple_parallel(n_sim = 10,n_vals = c(200,500,1000), p_vals = c(5,10,50), k = 5)


#stopCluster(cl)

# Salving
write.csv(df_results, "results_bestsubset_rmse_corrected_1.csv", row.names = FALSE)
save(df_results, file = "results_bestsubset_rmse_corrected_1.RData")




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

# ===========================================================
# Initialize H2O
# ===========================================================
h2o.init(nthreads = -1)

# ===========================================================
# Generate Friedman data
# ===========================================================
generate_data <- function(n, p, seed = 42) {
  set.seed(seed)
  X <- matrix(runif(n * p), nrow = n)
  y <- 10 * sin(pi * X[,1] * X[,2]) +
    20 * (X[,3] - 0.5)^2 +
    10 * X[,4] + 5 * X[,5] + rnorm(n)
  list(X = X, y = y)
}

# ===========================================================
# Compute support recovery (Jaccard and recovery rate)
# ===========================================================
compute_support_recovery <- function(selected_vars, true_vars = 1:5, p) {
  selected <- rep(0, p)
  selected[selected_vars] <- 1
  true <- rep(0, p)
  true[true_vars] <- 1
  jaccard <- ifelse(sum(selected | true) == 0, 0, sum(selected & true) / sum(selected | true))
  recovery_rate <- sum(selected & true) / length(true_vars)
  list(jaccard = jaccard, recovery_rate = recovery_rate)
}

# ===========================================================
# K-fold cross-validation RMSE for a subset of features
# ===========================================================
eval_subset_kfold <- function(X, y, vars, model_func, k = 5) {
  n <- nrow(X)
  folds <- sample(rep(1:k, length.out = n))
  rmse_list <- numeric(k)
  for (i in 1:k) {
    idx_train <- which(folds != i)
    idx_test <- which(folds == i)
    X_train <- X[idx_train, vars, drop=FALSE]
    y_train <- y[idx_train]
    X_test <- X[idx_test, vars, drop=FALSE]
    y_test <- y[idx_test]
    model <- model_func(X_train, y_train)
    y_pred <- model(X_test)
    rmse_list[i] <- rmse(y_test, y_pred)
  }
  mean(rmse_list)
}

# ===========================================================
# Evaluate models: regularized + black-box + hybrid
# ===========================================================
evaluate_models <- function(X, y, k = 5) {
  p <- ncol(X)
  results <- list()
  
  # Regularized models
  reg_list <- list(
    Ridge = cv.glmnet(X, y, alpha = 0),
    Lasso = cv.glmnet(X, y, alpha = 1),
    ElasticNet = cv.glmnet(X, y, alpha = 0.5)
  )
  
  reg_subsets <- list()
  for (name in names(reg_list)) {
    model <- reg_list[[name]]
    coefs <- as.vector(coef(model, s = "lambda.min"))[-1]
    ord_vars <- order(abs(coefs), decreasing = TRUE)
    
    Nmax <- min(10, p)
    rmse_vals <- sapply(1:Nmax, function(m) {
      vars_try <- ord_vars[1:m]
      eval_subset_kfold(X, y, vars_try, function(Xt, yt) {
        fit <- lm(yt ~ ., data = as.data.frame(Xt))
        function(Xnew) predict(fit, as.data.frame(Xnew))
      }, k)
    })
    best_m <- which.min(rmse_vals)
    vars <- ord_vars[1:best_m]
    
    reg_subsets[[name]] <- vars
    
    rmse_val <- rmse_vals[best_m]
    support <- compute_support_recovery(vars, 1:5, p)
    results[[paste0(name, "_RMSE")]] <- rmse_val
    results[[paste0(name, "_Jaccard")]] <- support$jaccard
    results[[paste0(name, "_Recovery")]] <- support$recovery_rate
  }
  
  # Black-box models
  bb_models <- list(
    RF = function(X,y) randomForest(X,y, importance=TRUE),
    XGBoost = function(X,y) xgboost(data=xgb.DMatrix(X,label=y), nrounds=50, objective="reg:squarederror", verbose=0),
    LightGBM = function(X,y) lightgbm(data=lgb.Dataset(X,label=y), nrounds=50, objective="regression", verbose=-1),
    CatBoost = function(X,y) catboost.train(catboost.load_pool(X,label=y), NULL, list(loss_function="RMSE", iterations=50, verbose=0)),
    H2OGBM = function(X,y) h2o.gbm(y="y", training_frame=as.h2o(cbind(as.data.frame(X), y=y)), ntrees=50)
  )
  
  for (bb_name in names(bb_models)) {
    mod <- bb_models[[bb_name]](X,y)
    
    imp <- switch(bb_name,
                  RF = importance(mod)[,1],
                  XGBoost = {
                    imp_df <- xgb.importance(model=mod)
                    imp <- rep(0, p)
                    idx <- as.numeric(gsub("f", "", imp_df$Feature)) + 1
                    if (any(is.na(idx))) {
                      cat("XGBoost imp_df$Feature:\n")
                      print(imp_df$Feature)
                      stop("XGBoost: problema ao converter Feature em índice.")
                    }
                    imp[idx] <- imp_df$Gain
                    imp
                  },
                  LightGBM = {
                    imp_df <- lgb.importance(mod)
                    imp <- rep(0, p)
                    # Extrai o número da coluna do formato Column_X
                    idx <- as.numeric(gsub("Column_", "", imp_df$Feature)) + 1
                    if (any(is.na(idx))) {
                      cat("LightGBM imp_df$Feature:\n")
                      print(imp_df$Feature)
                      stop("LightGBM: problema ao converter Feature em índice (Column_X).")
                    }
                    imp[idx] <- imp_df$Gain
                    imp
                  }
                  ,
                  CatBoost = as.numeric(catboost.get_feature_importance(mod)),
                  H2OGBM = {
                    vi <- as.data.frame(h2o.varimp(mod))
                    imp <- rep(0, p)
                    # Extrai o número de Vx
                    idx <- as.numeric(gsub("V", "", vi$variable))
                    if (any(is.na(idx))) {
                      cat("H2OGBM vi$variable:\n")
                      print(vi$variable)
                      stop("H2OGBM: problema ao converter variable em índice (Vx).")
                    }
                    imp[idx] <- vi$relative_importance
                    imp
                  }
                  
    )
    
    
    vars <- order(imp, decreasing=TRUE)[1:min(5,p)]
    
    rmse_val <- eval_subset_kfold(X, y, vars, function(Xt, yt) {
      m <- bb_models[[bb_name]](Xt, yt)
      function(Xnew) {
        if (bb_name == "H2OGBM") {
          as.vector(h2o.predict(m, as.h2o(as.data.frame(Xnew))))
        } else if (bb_name == "CatBoost") {
          as.vector(catboost.predict(m, catboost.load_pool(Xnew)))
        } else {
          predict(m, Xnew)
        }
      }
    }, k)
    
    support <- compute_support_recovery(vars, 1:5, p)
    results[[paste0(bb_name, "_Full_RMSE")]] <- rmse_val
    results[[paste0(bb_name, "_Full_Jaccard")]] <- support$jaccard
    results[[paste0(bb_name, "_Full_Recovery")]] <- support$recovery_rate
  }
  
  # Hybrid models
  for (bb_name in names(bb_models)) {
    for (reg_name in names(reg_subsets)) {
      vars <- reg_subsets[[reg_name]]
      rmse_val <- eval_subset_kfold(X, y, vars, function(Xt, yt) {
        mod <- bb_models[[bb_name]](Xt, yt)
        function(Xnew) {
          if (bb_name == "H2OGBM") {
            as.vector(h2o.predict(mod, as.h2o(as.data.frame(Xnew))))
          } else if (bb_name == "CatBoost") {
            as.vector(catboost.predict(mod, catboost.load_pool(Xnew)))
          } else {
            predict(mod, Xnew)
          }
        }
      }, k)
      support <- compute_support_recovery(vars, 1:5, p)
      results[[paste0(bb_name, "_", reg_name, "_RMSE")]] <- rmse_val
      results[[paste0(bb_name, "_", reg_name, "_Jaccard")]] <- support$jaccard
      results[[paste0(bb_name, "_", reg_name, "_Recovery")]] <- support$recovery_rate
    }
  }
  
  return(results)
}

# ===========================================================
# Simulation wrapper
# ===========================================================
simulate_multiple <- function(n_sim=10, n_vals=c(200,500,1000), p_vals=c(5,10,50), k=5) {
  map_dfr(1:n_sim, function(sim) {
    map_dfr(n_vals, function(n) {
      map_dfr(p_vals, function(p) {
        data <- generate_data(n,p,seed=sim+n+p)
        res <- evaluate_models(data$X,data$y,k)
        tibble(sim=sim,n=n,p=p, metric=names(res), value=unlist(res))
      })
    })
  })
}

# ===========================================================
# Run simulations
# ===========================================================
df_results <- simulate_multiple(n_sim=10, k=5)
write.csv(df_results, "results_bestsubset_rmse_corrected.csv", row.names=FALSE)
save(df_results, file="results_bestsubset_rmse_corrected.RData")


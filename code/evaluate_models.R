# ===========================================================
# Compute support recovery (Jaccard and recovery rate)
# ===========================================================

evaluate_models <- function(X, y, k = 5, true_vars = 1:5) {
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
    support <- compute_support_recovery(vars, true_vars, p)
    results[[paste0(name, "_RMSE")]] <- rmse_val
    results[[paste0(name, "_Jaccard")]] <- support$jaccard
    results[[paste0(name, "_Recovery")]] <- support$recovery_rate
  }
  
  # Black-box models
  bb_models <- list(
    RF = function(X, y) randomForest(X, y, importance = TRUE),
    XGBoost = function(X, y) xgboost(data = xgb.DMatrix(X, label = y), nrounds = 50, objective = "reg:squarederror", verbose = 0),
    LightGBM = function(X, y) lightgbm(data = lgb.Dataset(X, label = y), nrounds = 50, objective = "regression", verbose = -1),
    CatBoost = function(X, y) catboost.train(catboost.load_pool(X, label = y), NULL, list(loss_function = "RMSE", iterations = 50, verbose = 0)),
    H2OGBM = function(X, y) h2o.gbm(y = "y", training_frame = as.h2o(cbind(as.data.frame(X), y = y)), ntrees = 50)
  )
  
  for (bb_name in names(bb_models)) {
    mod <- bb_models[[bb_name]](X, y)
    
    imp <- switch(bb_name,
                  RF = importance(mod)[, 1],
                  XGBoost = {
                    imp_df <- xgb.importance(model = mod)
                    imp <- rep(0, p)
                    idx <- as.numeric(gsub("f", "", imp_df$Feature)) + 1
                    imp[idx] <- imp_df$Gain
                    imp
                  },
                  LightGBM = {
                    imp_df <- lgb.importance(mod)
                    imp <- rep(0, p)
                    idx <- as.numeric(gsub("Column_", "", imp_df$Feature)) + 1
                    imp[idx] <- imp_df$Gain
                    imp
                  },
                  CatBoost = as.numeric(catboost.get_feature_importance(mod)),
                  H2OGBM = {
                    vi <- as.data.frame(h2o.varimp(mod))
                    imp <- rep(0, p)
                    idx <- as.numeric(gsub("V", "", vi$variable))
                    imp[idx] <- vi$relative_importance
                    imp
                  }
    )
    
    ord_vars <- order(imp, decreasing = TRUE)
    Nmax <- min(10, p - 1)
    rmse_vals <- sapply(1:Nmax, function(m) {
      vars_try <- ord_vars[1:m]
      eval_subset_kfold(X, y, vars_try, function(Xt, yt) {
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
    })
    
    best_m <- which.min(rmse_vals)
    vars <- ord_vars[1:best_m]
    rmse_val <- rmse_vals[best_m]
    support <- compute_support_recovery(vars, true_vars, p)
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
      support <- compute_support_recovery(vars, true_vars, p)
      results[[paste0(bb_name, "_", reg_name, "_RMSE")]] <- rmse_val
      results[[paste0(bb_name, "_", reg_name, "_Jaccard")]] <- support$jaccard
      results[[paste0(bb_name, "_", reg_name, "_Recovery")]] <- support$recovery_rate
    }
  }
  
  return(results)
}

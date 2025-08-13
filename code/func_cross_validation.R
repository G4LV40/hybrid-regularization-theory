
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

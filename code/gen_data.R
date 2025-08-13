generate_data <- function(n, p, seed = 42) {
  set.seed(seed)
  X <- matrix(runif(n * p), nrow = n)
  y <- 10 * sin(pi * X[,1] * X[,2]) +
    20 * (X[,3] - 0.5)^2 +
    10 * X[,4] + 5 * X[,5] + rnorm(n)
  
  # Embaralha colunas
  perm <- sample(p)
  X_shuffled <- X[, perm]
  
  # Mapeia as posições verdadeiras após o embaralhamento
  true_vars <- match(1:5, perm)
  
  list(X = X_shuffled, y = y, true_vars = true_vars, perm = perm)
}

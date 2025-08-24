
# ===================================================
# Script: Visualizações para Artigo 2 
# ===================================================

# Pacotes
library(ggplot2)
library(dplyr)
library(tidyr)
library(forcats)
library(ggpubr)

# Carregar o arquivo .RData
load("results_bestsubset_rmse_corrected.RData")


df<-as.data.frame(df_results)


# Separar modelo e métrica
df <- df %>%
  tidyr::separate(metric, into = c("model", "metric_type"), sep = "_(?=[^_]+$)", remove = FALSE) %>%
  filter(p == 50)

# Criar coluna de grupo
df <- df %>%
  mutate(group = case_when(
    grepl("Full$", model) ~ "BlackBox",
    grepl("Ridge|Lasso|ElasticNet", model) & grepl("CatBoost|XGBoost|LightGBM|H2OGBM|RF", model) ~ "Híbrido",
    grepl("Ridge|Lasso|ElasticNet", model) ~ "Regularizado",
    TRUE ~ "Outro"
  ))

# 1. Boxplot por grupo
ggplot(df, aes(x = group, y = value, fill = group)) +
  geom_boxplot(alpha = 0.7) +
  facet_wrap(~metric_type, scales = "free_y") +
  theme_minimal() +
  labs(title = "Distribuição das métricas por grupo de modelos",
       x = "Grupo", y = "Valor da métrica") +
  theme(legend.position = "none")

# 2. Linhas por n
df_mean <- df %>%
  group_by(model, metric_type, n) %>%
  summarise(mean_value = mean(value), .groups = "drop")

ggplot(df_mean, aes(x = factor(n), y = mean_value, color = model, group = model)) +
  geom_line() +
  geom_point() +
  facet_wrap(~metric_type, scales = "free_y") +
  theme_minimal() +
  labs(title = "Métricas por tamanho amostral", x = "n", y = "Valor médio")

# 3. Heatmap de RMSE
df %>%
  filter(metric_type == "RMSE") %>%
  group_by(model, n) %>%
  summarise(mean_rmse = mean(value), .groups = "drop") %>%
  ggplot(aes(x = factor(n), y = fct_reorder(model, mean_rmse), fill = mean_rmse)) +
  geom_tile() +
  scale_fill_viridis_c() +
  theme_minimal() +
  labs(title = "Heatmap de RMSE por modelo e n", x = "n", y = "Modelo")

# 4. Barras dos top híbridos
df %>%
  filter(metric_type == "RMSE", group == "Híbrido", n == 1000) %>%
  group_by(model) %>%
  summarise(rmse = mean(value), .groups = "drop") %>%
  slice_min(rmse, n = 10) %>%
  ggplot(aes(x = fct_reorder(model, rmse), y = rmse)) +
  geom_col(fill = "mediumpurple3") +
  coord_flip() +
  theme_minimal() +
  labs(title = "Top 10 modelos híbridos (RMSE)", x = "Modelo", y = "RMSE")



# 5. Scatter RMSE vs Jaccard com elipses por grupo - ELIPSE
jaccard <- df %>% filter(metric_type == "Jaccard", n == 1000)
rmse <- df %>% filter(metric_type == "RMSE", n == 1000)

left_join(rmse, jaccard, by = c("sim", "model", "n", "p")) %>%
  rename(RMSE = value.x, Jaccard = value.y) %>%
  mutate(group = case_when(
    grepl("Full$", model) ~ "BlackBox",
    grepl("Ridge|Lasso|ElasticNet", model) & grepl("CatBoost|XGBoost|LightGBM|H2OGBM|RF", model) ~ "Híbrido",
    grepl("Ridge|Lasso|ElasticNet", model) ~ "Regularizado",
    TRUE ~ "Outro"
  )) %>%
  ggplot(aes(x = RMSE, y = Jaccard, color = group, fill = group)) +
  geom_point(alpha = 0.6) +
  stat_ellipse(geom = "polygon", alpha = 0.15, type = "norm", level = 0.95) +
  theme_minimal() +
  labs(title = "Trade-off between RMSE e Jaccard", x = "RMSE", y = "Jaccard") +
  theme(legend.position = "bottom")


#boxplots agrupado

# Agrupamento e resumo

df_plot <- left_join(rmse, jaccard, by = c("sim", "model", "n", "p")) %>%
  rename(RMSE = value.x, Jaccard = value.y) %>%
  mutate(group = case_when(
    grepl("Full$", model) ~ "BlackBox",
    grepl("Ridge|Lasso|ElasticNet", model) & grepl("CatBoost|XGBoost|LightGBM|H2OGBM|RF", model) ~ "Hybrid",
    grepl("Ridge|Lasso|ElasticNet", model) ~ "Regularized",
    TRUE ~ "Other"
  ))

# Descriptive statistics by group
df_summary <- df_plot %>%
  group_by(group) %>%
  summarise(
    RMSE_mean = mean(RMSE, na.rm = TRUE),
    RMSE_sd = sd(RMSE, na.rm = TRUE),
    Jaccard_mean = mean(Jaccard, na.rm = TRUE),
    Jaccard_sd = sd(Jaccard, na.rm = TRUE),
    .groups = 'drop'
  )

# Plot
ggplot(df_summary, aes(x = RMSE_mean, y = Jaccard_mean, color = group)) +
  geom_point(size = 4) +
  geom_errorbar(aes(ymin = Jaccard_mean - Jaccard_sd, ymax = Jaccard_mean + Jaccard_sd), width = 0.03) +
  geom_errorbarh(aes(xmin = RMSE_mean - RMSE_sd, xmax = RMSE_mean + RMSE_sd), height = 0.03) +
  theme_minimal() +
  labs(
    title = "Trade-off between average RMSE and average Jaccard index",
    x = "RMSE (Root Mean Squared Error)",
    y = "Jaccard Index (Feature Selection)"
  ) +
  theme(legend.position = "bottom")

# 1. Transformação logit em Jaccard
# 2. Cálculo das elipses no espaço transformado
# 3. Inversão da transformação para plotagem original
# 4. Plot com stat_ellipse 


library(ellipse)
library(dplyr)
library(ggplot2)
library(tidyr)
library(MASS)  # para ellipse
library(scales)  # para inverse logit

# Dados simulados de RMSE e Jaccard
jaccard <- df %>% filter(metric_type == "Jaccard", n == 1000)
rmse <- df %>% filter(metric_type == "RMSE", n == 1000)

# Preparação dos dados
df_plot <- left_join(rmse, jaccard, by = c("sim", "model", "n", "p")) %>%
  rename(RMSE = value.x, Jaccard = value.y) %>%
  mutate(group = case_when(
    grepl("Full$", model) ~ "BlackBox",
    grepl("Ridge|Lasso|ElasticNet", model) & grepl("CatBoost|XGBoost|LightGBM|H2OGBM|RF", model) ~ "Híbrido",
    grepl("Ridge|Lasso|ElasticNet", model) ~ "Regularizado",
    TRUE ~ "Outro"
  ))

# Aplica transformação logit (com bound para evitar Inf)
eps <- 1e-5
df_plot <- df_plot %>%
  mutate(
    Jaccard_logit = log(pmin(pmax(Jaccard, eps), 1 - eps) / (1 - pmin(pmax(Jaccard, eps), 1 - eps)))
  )

# Função para gerar elipse no espaço logit e depois retransformar
make_ellipse_logit <- function(data) {
  if (nrow(data) < 3) return(NULL)
  mu <- colMeans(data[, c("RMSE", "Jaccard_logit")], na.rm = TRUE)
  Sigma <- cov(data[, c("RMSE", "Jaccard_logit")], use = "complete.obs")
  ell <- ellipse::ellipse(Sigma, centre = mu, level = 0.95, npoints = 100)
  ell <- as.data.frame(ell)
  ell$Jaccard <- plogis(ell$Jaccard_logit)  # Inversa do logit
  ell
}

# Gerar elipses por grupo
elipses <- df_plot %>%
  group_by(group) %>%
  do(make_ellipse_logit(.)) %>%
  ungroup()

# Gráfico
ggplot(df_plot, aes(x = RMSE, y = Jaccard, color = group)) +
  geom_point(alpha = 0.6) +
  geom_path(data = elipses, aes(x = RMSE, y = Jaccard, fill = group), alpha = 0.2, color = NA) +
  facet_wrap(~group, scales = "free") +
  scale_y_continuous(limits = c(0, 1), name = "Jaccard Index") +
  theme_minimal() +
  labs(
    title = "Trade-off between RMSE and Jaccard with confidence ellipses",
    x = "RMSE (Root Mean Squared Error)"
  ) +
  theme(legend.position = "bottom")





# =============================================================================
# Script 01 — Análisis de Weight of Evidence (WoE) e Information Value (IV)
# Proyecto: Predicción de Riesgo Crediticio
# Librería principal: scorecard
# =============================================================================

# -----------------------------------------------------------------------------
# 0. Instalación de dependencias
# -----------------------------------------------------------------------------
install.packages("scorecard")
install.packages("tidyverse")
install.packages("ggplot2")

# -----------------------------------------------------------------------------
# 1. Carga de librerías
# -----------------------------------------------------------------------------
library(scorecard)
library(dplyr)
library(ggplot2)

cat("=== Librerías cargadas correctamente ===\n\n")

# -----------------------------------------------------------------------------
# 2. Carga del dataset
# -----------------------------------------------------------------------------

RUTA_CSV <- "../data/raw/CreditScoring.csv"

df <- read.csv(RUTA_CSV)

# Eliminamos la columna ID — no aporta valor predictivo
df$ID <- NULL

cat(sprintf("Dataset cargado: %d filas x %d columnas\n\n", nrow(df), ncol(df)))

# -----------------------------------------------------------------------------
# 3. Limpieza previa mínima para el cálculo de WoE/IV
#
#    El cálculo de WoE requiere que la variable target sea binaria (0/1)
#    y que no haya valores que distorsionen el binning (96/98 en atrasos).
# -----------------------------------------------------------------------------

cols_atrasos <- c(
  "NumberOfTime30.59DaysPastDueNotWorse",
  "NumberOfTime60.89DaysPastDueNotWorse",
  "NumberOfTimes90DaysLate"
)

for (col in cols_atrasos) {
  if (col %in% names(df)) {
    df[[col]][df[[col]] %in% c(96, 98)] <- NA
  }
}

# Corregimos age == 0 → NA (error de datos)
df$age[df$age == 0] <- NA

cat("Limpieza de datos aplicada (96/98 → NA, age = 0 → NA)\n\n")


# -----------------------------------------------------------------------------
# 4. Definición de variables
# -----------------------------------------------------------------------------

# Variable objetivo
TARGET <- "SeriousDlqin2yrs"

# Variables predictoras (todas menos el target)
features <- setdiff(names(df), TARGET)

cat("Variables a analizar:\n")
cat(paste(" -", features, collapse = "\n"), "\n\n")

# -----------------------------------------------------------------------------
# 5. Filtrado de variables por IV (función var_filter de scorecard)
#
#    La función var_filter aplica automáticamente tres filtros:
#      - IV mínimo (por defecto 0.02): descarta variables sin poder predictivo
#      - Tasa de valores únicos (iv_limit): descarta casi-constantes
#      - Correlación entre variables (cor_limit): descarta redundantes
# -----------------------------------------------------------------------------

cat("=== Filtrando variables por IV mínimo (0.02) ===\n")

dt_filtrado <- var_filter(
  dt = df,
  y = TARGET,
  iv_limit = 0.02,
  cor_limit = 0.9,
  return_rm_reason = TRUE
)

cat(sprintf(
  "Variables que pasan el filtro: %d de %d\n\n",
  ncol(dt_filtrado$dt) - 1,
  length(features)
))

# Variables eliminadas y motivo
if (!is.null(dt_filtrado$rm)) {
  cat("=== Variables eliminadas ===\n")
  print(dt_filtrado$rm)
  cat("\n")
}

df_filtrado <- dt_filtrado$dt

# -----------------------------------------------------------------------------
# 6. Binning — discretización óptima de variables continuas
#
#    woebin() divide cada variable en tramos (bins) de forma automática
#    usando un algoritmo de árbol que maximiza la separación entre clases.
# -----------------------------------------------------------------------------

cat("=== Calculando bins óptimos (woebin) ===")

bins <- woebin(
  dt = df_filtrado,
  y = TARGET,
  print_step = 0
)

cat("Bins calculado.\n\n")


# -----------------------------------------------------------------------------
# 7. Tabla de IV por variable
#
#    El Information Value mide el poder predictivo de cada variable:
#      IV < 0.02  → sin poder predictivo (ya filtradas)
#      0.02–0.10  → poder débil
#      0.10–0.30  → poder moderado
#      0.30–0.50  → poder fuerte
#      > 0.50     → sospechoso (posible fuga de datos)
# -----------------------------------------------------------------------------

# Extraemos el IV de cada variable desde los bins
iv_tabla <- lapply(names(bins), function(var) {
  data.frame(
    variable = var,
    iv = round(sum(bins[[var]]$bin_iv, na.rm = TRUE), 4)
  )
}) |> bind_rows() |>
  arrange(desc(iv))

cat("=== Information Value por variable ===\n")
print(iv_tabla)
cat("\n")

# Clasificación del porder predictivo
iv_tabla$poder_predictivo <- cut(
  iv_tabla$iv,
  breaks = c(0, 0.02, 0.10, 0.30, 0.50, Inf),
  labels = c("Sin poder", "Débil", "Moderado", "Fuerte", "Sospechoso"),
  right = FALSE
)

cat("=== Clasificación por poder predictivo ===\n")
print(iv_tabla)
cat("\n")

# -----------------------------------------------------------------------------
# 8. Exportar tabla de IV para uso en Python
# -----------------------------------------------------------------------------

RUTA_SALIDA_IV <- "../r_analysis/outputs/iv_tabla.csv"

# Crear carpeta si no existe
dir.create(dirname(RUTA_SALIDA_IV), recursive = TRUE, showWarnings = FALSE)

write.csv(iv_tabla, RUTA_SALIDA_IV, row.names = FALSE)
cat(sprintf("Tabla de IV exportada en: %s\n\n", RUTA_SALIDA_IV))


# -----------------------------------------------------------------------------
# 9. Visualización — Gráfico de IV por variable
# -----------------------------------------------------------------------------

colores_poder <- c(
  "Débil" = "#F0A500",
  "Moderado" = "#A490D9",
  "Fuerte" = "#27AE60",
  "Sospechoso" = "#E55C5C"
)

p_iv <- ggplot(iv_tabla, aes(x = reorder(variable, iv), y = iv, fill = poder_predictivo)) +
  geom_col(width = 0.7) +
  geom_text(aes(label = sprintf("%.3f", iv)), hjust = -0.15, size = 3.5) +
  coord_flip() +
  scale_fill_manual(values = colores_poder, name = "Poder predictivo") +
  labs(
    title = "Information Value (IV) por variable",
    subtitle = "Proyecto Credit Scoring",
    x = NULL,
    y = "Information Value"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom") +
  ylim(0, max(iv_tabla$iv) * 1.2)
  
print(p_iv)

# Exportar gráfico
RUTA_SALIDA_PLOT <- "../r_analysis/outputs/woe_plots/iv_barplot.png"
dir.create(dirname(RUTA_SALIDA_PLOT), recursive = TRUE, showWarnings = FALSE)
ggsave(RUTA_SALIDA_PLOT, plot = p_iv, width = 10, height = 6, dpi = 150)
cat(sprintf("Gráfico exportado en: %s\n\n", RUTA_SALIDA_PLOT))

# -----------------------------------------------------------------------------
# 10. Visualización — Gráficos WoE por variable (bins)
#
#     woebin_plot() genera un gráfico por cada variable mostrando:
#       - Cómo se distribuyen los buenos y malos pagadores en cada tramo
#       - El WoE de cada tramo (positivo = buen pagador, negativo = moroso)
# -----------------------------------------------------------------------------

cat("=== Generando gráficos WoE por variable ===\n")

plots_woe <- woebin_plot(bins)

# Exportar cada gráfico individualmente
for (var in names(plots_woe)) {
  ruta_plot <- sprintf("../r_analysis/outputs/woe_plots/woe_%s.png", var)
  ggsave(ruta_plot, plot = plots_woe[[var]], width = 8, height = 5, dpi = 150)
}

cat(sprintf("%d gráficos WoE exportados en: ../r_analysis/outputs/woe_plots/\n\n", length(plots_woe)))

# -----------------------------------------------------------------------------
# 11. Resumen final — variables recomendadas para el modelado
# -----------------------------------------------------------------------------

vars_recomendadas <- iv_tabla |>
  filter(poder_predictivo %in% c("Moderado", "Fuerte")) |>
  pull(variable)

cat("=== Variables recomendadas para el modelado (IV moderado o fuerte) ===\n")
cat(paste(" ✓", vars_recomendadas, collapse = "\n"), "\n\n")

cat("=== Script completado exitosamente ===\n")
cat("Siguiente paso: Notebook 02 — Preprocesamiento (Python)\n")
# =============================================================================
# Script: 04_graficos_shap.R
# Sección: Resultados — Explicabilidad del Modelo (SHAP)
# Genera gráficos de importancia global y efectos de variables
# =============================================================================
# PRERREQUISITOS:
#   1. Haber ejecutado el Notebook 04 en Python
#   2. Haber ejecutado: python src/predict.py
# Archivo necesario:
#   - data/exports/predictions.csv
#
# NOTA: Los valores SHAP exactos los genera Python (librería shap).
#       Este script genera visualizaciones complementarias de explicabilidad
#       usando los datos de predicciones y features disponibles en el CSV.
#       Para los gráficos SHAP puros (beeswarm, waterfall) usar el Notebook 04.
# =============================================================================

library(ggplot2)
library(dplyr)
library(tidyr)
library(scales)
library(patchwork)

# ─────────────────────────────────────────────
# 0. Configuración global
# ─────────────────────────────────────────────

RUTA_OUTPUT <- "r_analysis/outputs/shap_plots"
dir.create(RUTA_OUTPUT, recursive = TRUE, showWarnings = FALSE)

COL_AZUL      <- "#1F4E79"
COL_NO_MOROSO <- "#4A90D9"
COL_MOROSO    <- "#E55C5C"
COL_VERDE     <- "#27AE60"
COL_AMBAR     <- "#F0A500"

tema_base <- theme_minimal(base_size = 13) +
  theme(
    plot.title    = element_text(face = "bold", color = COL_AZUL, size = 14),
    plot.subtitle = element_text(color = "gray40", size = 11),
    axis.title    = element_text(color = "gray30", size = 11),
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  )

guardar <- function(nombre, ancho = 10, alto = 6) {
  ruta <- file.path(RUTA_OUTPUT, paste0(nombre, ".png"))
  ggsave(ruta, width = ancho, height = alto, dpi = 150, bg = "white")
  cat(sprintf("  ✓ %s\n", ruta))
}


# ─────────────────────────────────────────────
# 1. Carga de datos
# ─────────────────────────────────────────────

pred <- read.csv("data/exports/predictions.csv") |>
  rename(
    prob_default  = Probabilidad.Default,
    clasificacion = Clasificación,
    riesgo        = Nivel.de.Riesgo,
    default_real  = Default.Real,
    acierto       = Acierto.del.Modelo,
    edad          = Edad,
    ingreso       = Ingreso.Mensual,
    ratio_deuda   = Ratio.de.Deuda,
    util_credito  = Utilización.Crédito,
    atrasos_90    = Atrasos..90.días,
    atrasos_3059  = Atrasos.30.59.días,
    dependientes  = Dependientes,
    lineas        = Líneas.de.Crédito,
    inmuebles     = Préstamos.Inmobiliarios
  ) |>
  mutate(riesgo = factor(riesgo, levels = c("Alto", "Medio", "Bajo")))

FEATURES <- c("util_credito", "edad", "atrasos_3059", "ratio_deuda",
              "ingreso", "lineas", "atrasos_90", "inmuebles", "dependientes")

ETIQUETAS <- c("Utilización Crédito", "Edad", "Atrasos 30-59 días",
               "Ratio de Deuda", "Ingreso Mensual", "Líneas de Crédito",
               "Atrasos +90 días", "Préstamos Inmobiliarios", "Dependientes")

cat(sprintf("Predicciones cargadas: %d clientes\n\n", nrow(pred)))


# ─────────────────────────────────────────────
# 2. Importancia de variables — correlación con probabilidad de default
#    (proxy de importancia cuando no tenemos los valores SHAP exportados)
# ─────────────────────────────────────────────
cat("Generando: importancia de variables (correlación con P(default))...\n")

importancia <- sapply(FEATURES, function(f) {
  cor(pred[[f]], pred$prob_default, use = "pairwise.complete.obs")
}) |>
  tibble::enframe(name = "feature", value = "correlacion") |>
  mutate(
    etiqueta   = ETIQUETAS[match(feature, FEATURES)],
    abs_corr   = abs(correlacion),
    direccion  = ifelse(correlacion > 0, "Aumenta riesgo", "Reduce riesgo"),
    etiqueta   = reorder(etiqueta, abs_corr)
  )

p_importancia <- ggplot(importancia,
                        aes(x = etiqueta, y = correlacion, fill = direccion)) +
  geom_col(width = 0.65) +
  geom_text(aes(label = sprintf("%.3f", correlacion),
                hjust = ifelse(correlacion > 0, -0.15, 1.15)),
            size = 3.8, fontface = "bold") +
  coord_flip() +
  geom_vline(xintercept = 0, color = "gray40", linewidth = 0.5) +
  scale_fill_manual(values = c("Aumenta riesgo" = COL_MOROSO,
                               "Reduce riesgo"  = COL_NO_MOROSO)) +
  scale_y_continuous(limits = c(-0.45, 0.65),
                     labels = function(x) sprintf("%.2f", x)) +
  labs(
    title    = "Importancia de variables — correlación con P(default)",
    subtitle = "Rojo: variable que aumenta el riesgo · Azul: variable que lo reduce",
    x        = NULL,
    y        = "Correlación de Pearson con probabilidad de default",
    fill     = NULL
  ) +
  tema_base

print(p_importancia)
guardar("01_importancia_variables", ancho = 10, alto = 6)


# ─────────────────────────────────────────────
# 3. Efecto de las 4 variables más importantes
#    sobre la probabilidad de default
# ─────────────────────────────────────────────
cat("Generando: efectos de variables clave...\n")

top4 <- importancia |>
  arrange(desc(abs_corr)) |>
  head(4) |>
  pull(feature)

top4_etiq <- ETIQUETAS[match(top4, FEATURES)]

plots_efecto <- list()

for (i in seq_along(top4)) {
  var  <- top4[i]
  etiq <- top4_etiq[i]

  datos <- pred |>
    select(x = all_of(var), prob_default, riesgo) |>
    filter(!is.na(x)) |>
    mutate(x = pmin(x, quantile(x, 0.99, na.rm = TRUE)))

  plots_efecto[[i]] <- ggplot(datos, aes(x = x, y = prob_default, color = riesgo)) +
    geom_point(alpha = 0.15, size = 0.8) +
    geom_smooth(aes(group = 1), method = "loess", se = TRUE,
                color = COL_AZUL, linewidth = 1.2, fill = "#AED6F1") +
    scale_color_manual(values = c("Alto" = COL_MOROSO,
                                  "Medio" = COL_AMBAR,
                                  "Bajo" = COL_VERDE)) +
    scale_y_continuous(labels = percent_format(accuracy = 1),
                       limits = c(0, 1)) +
    labs(
      title  = etiq,
      x      = etiq,
      y      = "P(default)",
      color  = "Riesgo"
    ) +
    tema_base +
    theme(plot.title = element_text(size = 11),
          legend.position = "none")
}

p_efectos <- wrap_plots(plots_efecto, ncol = 2) +
  plot_annotation(
    title    = "Efecto de las variables más importantes sobre P(default)",
    subtitle = "Línea azul: tendencia suavizada (LOESS) · Puntos coloreados por nivel de riesgo",
    theme    = theme(
      plot.title    = element_text(face = "bold", color = COL_AZUL, size = 14),
      plot.subtitle = element_text(color = "gray40", size = 11)
    )
  )

print(p_efectos)
guardar("02_efectos_variables_clave", ancho = 12, alto = 9)


# ─────────────────────────────────────────────
# 4. Perfil promedio por nivel de riesgo
#    (equivalente visual al SHAP global)
# ─────────────────────────────────────────────
cat("Generando: perfil promedio por nivel de riesgo...\n")

perfil <- pred |>
  group_by(riesgo) |>
  summarise(across(all_of(FEATURES), ~ mean(., na.rm = TRUE)), .groups = "drop") |>
  pivot_longer(-riesgo, names_to = "feature", values_to = "media") |>
  mutate(etiqueta = ETIQUETAS[match(feature, FEATURES)])

# Normalizar entre 0 y 1 para comparar variables en distintas escalas
perfil <- perfil |>
  group_by(feature) |>
  mutate(media_norm = (media - min(media)) / (max(media) - min(media) + 1e-9)) |>
  ungroup()

p_perfil <- ggplot(perfil,
                   aes(x = reorder(etiqueta, media_norm),
                       y = media_norm, fill = riesgo)) +
  geom_col(position = "dodge", width = 0.7) +
  coord_flip() +
  scale_fill_manual(values = c("Alto" = COL_MOROSO,
                               "Medio" = COL_AMBAR,
                               "Bajo" = COL_VERDE)) +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  labs(
    title    = "Perfil de variables por nivel de riesgo",
    subtitle = "Valores normalizados (0–100%) para comparación entre variables",
    x        = NULL,
    y        = "Valor relativo normalizado",
    fill     = "Nivel de riesgo"
  ) +
  tema_base

print(p_perfil)
guardar("03_perfil_por_nivel_riesgo", ancho = 11, alto = 7)


# ─────────────────────────────────────────────
# 5. Distribución de P(default) por nivel de riesgo
# ─────────────────────────────────────────────
cat("Generando: distribución de probabilidad por nivel de riesgo...\n")

p_violin <- ggplot(pred, aes(x = riesgo, y = prob_default, fill = riesgo)) +
  geom_violin(alpha = 0.6, trim = TRUE, show.legend = FALSE) +
  geom_boxplot(width = 0.12, fill = "white", outlier.size = 0.5,
               outlier.alpha = 0.3, show.legend = FALSE) +
  scale_fill_manual(values = c("Alto" = COL_MOROSO,
                               "Medio" = COL_AMBAR,
                               "Bajo" = COL_VERDE)) +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  labs(
    title    = "Distribución de P(default) por nivel de riesgo",
    subtitle = "Violin plot + boxplot interno",
    x        = "Nivel de riesgo",
    y        = "Probabilidad de default"
  ) +
  tema_base

print(p_violin)
guardar("04_violin_por_nivel_riesgo", ancho = 8, alto = 6)


# ─────────────────────────────────────────────
# 6. Curva Precision–Recall
# ─────────────────────────────────────────────
cat("Generando: curva Precision-Recall...\n")

umbrales <- seq(0.05, 0.95, by = 0.01)

pr_data <- lapply(umbrales, function(u) {
  pred_u <- ifelse(pred$prob_default >= u, 1, 0)
  tp <- sum(pred_u == 1 & pred$default_real == 1)
  fp <- sum(pred_u == 1 & pred$default_real == 0)
  fn <- sum(pred_u == 0 & pred$default_real == 1)
  precision <- ifelse(tp + fp == 0, 0, tp / (tp + fp))
  recall    <- ifelse(tp + fn == 0, 0, tp / (tp + fn))
  f1        <- ifelse(precision + recall == 0, 0,
                      2 * precision * recall / (precision + recall))
  data.frame(umbral = u, Precision = precision, Recall = recall, F1 = f1)
}) |> bind_rows()

umbral_optimo <- pr_data$umbral[which.max(pr_data$F1)]

p_pr <- ggplot(pr_data, aes(x = Recall, y = Precision)) +
  geom_path(color = COL_AZUL, linewidth = 1.5) +
  geom_point(data = filter(pr_data, umbral == umbral_optimo),
             aes(x = Recall, y = Precision),
             color = COL_MOROSO, size = 4) +
  annotate("text",
           x = filter(pr_data, umbral == umbral_optimo)$Recall + 0.03,
           y = filter(pr_data, umbral == umbral_optimo)$Precision + 0.03,
           label = sprintf("Umbral óptimo\n(%.2f)", umbral_optimo),
           color = COL_MOROSO, size = 3.8, fontface = "bold") +
  scale_x_continuous(labels = percent_format(accuracy = 1), limits = c(0, 1)) +
  scale_y_continuous(labels = percent_format(accuracy = 1), limits = c(0, 1)) +
  labs(
    title    = "Curva Precision–Recall — Random Forest",
    subtitle = "Punto rojo: umbral que maximiza el F1-Score",
    x        = "Recall (sensibilidad)",
    y        = "Precisión"
  ) +
  tema_base

print(p_pr)
guardar("05_curva_precision_recall", ancho = 8, alto = 6)


# ─────────────────────────────────────────────
# Resumen
# ─────────────────────────────────────────────
cat("\n=== Gráficos SHAP / explicabilidad generados ===\n")
cat(sprintf("Carpeta: %s\n", RUTA_OUTPUT))
cat("  01_importancia_variables.png\n")
cat("  02_efectos_variables_clave.png\n")
cat("  03_perfil_por_nivel_riesgo.png\n")
cat("  04_violin_por_nivel_riesgo.png\n")
cat("  05_curva_precision_recall.png\n")
cat("\n✓ Todos los scripts R completados.\n")
cat("Los gráficos SHAP puros (beeswarm, waterfall) están en el Notebook 04.\n")
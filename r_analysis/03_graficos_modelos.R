# =============================================================================
# Script: 03_graficos_modelos.R
# Sección: Resultados — Comparación de Modelos
# Genera gráficos de métricas, curvas ROC y matriz de confusión
# =============================================================================
# PRERREQUISITOS:
#   1. Haber ejecutado los Notebooks 01-03 en Python
#   2. Haber ejecutado: python src/predict.py
# Archivos necesarios:
#   - models/resultados_comparacion_final.csv
#   - data/exports/predictions.csv
# =============================================================================

library(ggplot2)
library(dplyr)
library(tidyr)
library(scales)
library(patchwork)

# ─────────────────────────────────────────────
# 0. Configuración global
# ─────────────────────────────────────────────

RUTA_OUTPUT <- "r_analysis/outputs/model_plots"
dir.create(RUTA_OUTPUT, recursive = TRUE, showWarnings = FALSE)

COL_AZUL      <- "#1F4E79"
COL_NO_MOROSO <- "#4A90D9"
COL_MOROSO    <- "#E55C5C"
COL_VERDE     <- "#27AE60"
COL_AMBAR     <- "#F0A500"
COL_GRIS      <- "#95A5A6"

COLORES_MODELOS <- c(
  "Regresión Logística" = COL_GRIS,
  "Árbol de Decisión"   = COL_AMBAR,
  "Random Forest"       = COL_VERDE,
  "XGBoost"             = "#E67E22"
)

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
# 1. Carga de resultados
# ─────────────────────────────────────────────

resultados <- read.csv("models/resultados_comparacion_final.csv")
predicciones <- read.csv("data/exports/predictions.csv")

# Limpiar nombre del modelo (quitar "(umbral=X.XX)")
resultados <- resultados |>
  rename(Modelo_completo = 1) |>
  mutate(
    Modelo = gsub("\\s*\\(umbral=.*\\)", "", Modelo_completo),
    Modelo = trimws(Modelo)
  )

cat("Modelos cargados:\n")
print(resultados[, c("Modelo", "AUC.ROC", "Recall", "F1.Score", "Accuracy")])
cat("\n")


# ─────────────────────────────────────────────
# 2. Comparación de métricas — barras agrupadas
# ─────────────────────────────────────────────
cat("Generando: comparación de métricas...\n")

metricas_long <- resultados |>
  select(Modelo, AUC.ROC, Recall, Precisión, F1.Score) |>
  rename(`AUC-ROC` = AUC.ROC, `F1-Score` = F1.Score) |>
  pivot_longer(-Modelo, names_to = "Métrica", values_to = "Valor") |>
  mutate(Métrica = factor(Métrica, levels = c("AUC-ROC", "Recall", "Precisión", "F1-Score")))

p_metricas <- ggplot(metricas_long,
                     aes(x = Modelo, y = Valor, fill = Modelo)) +
  geom_col(width = 0.65, show.legend = FALSE) +
  geom_text(aes(label = sprintf("%.3f", Valor)),
            vjust = -0.4, size = 3.2, fontface = "bold") +
  geom_hline(yintercept = 0.75, linetype = "dashed",
             color = COL_MOROSO, linewidth = 0.6, alpha = 0.7) +
  facet_wrap(~ Métrica, nrow = 1) +
  scale_fill_manual(values = COLORES_MODELOS) +
  scale_y_continuous(limits = c(0, 1.1),
                     breaks = c(0, 0.25, 0.50, 0.75, 1.0),
                     labels = percent_format(accuracy = 1)) +
  labs(
    title    = "Comparación de modelos — métricas sobre test set",
    subtitle = "Línea roja: meta mínima (0.75) | Umbral óptimo por modelo",
    x        = NULL,
    y        = NULL
  ) +
  tema_base +
  theme(
    axis.text.x  = element_text(angle = 30, hjust = 1, size = 9),
    strip.text   = element_text(face = "bold", size = 11),
    panel.spacing = unit(1, "lines")
  )

print(p_metricas)
guardar("01_comparacion_metricas", ancho = 13, alto = 6)


# ─────────────────────────────────────────────
# 3. Radar chart — perfil de cada modelo
# ─────────────────────────────────────────────
cat("Generando: radar chart de modelos...\n")

# Radar manual con coordenadas polares
radar_data <- resultados |>
  select(Modelo, AUC.ROC, Recall, Precisión, F1.Score, Accuracy) |>
  rename(`AUC-ROC` = AUC.ROC, `F1-Score` = F1.Score) |>
  pivot_longer(-Modelo, names_to = "Métrica", values_to = "Valor")

p_radar <- ggplot(radar_data,
                  aes(x = Métrica, y = Valor, color = Modelo, group = Modelo)) +
  geom_polygon(aes(fill = Modelo), alpha = 0.08, linewidth = 0.8) +
  geom_point(size = 2.5) +
  coord_polar() +
  scale_color_manual(values = COLORES_MODELOS) +
  scale_fill_manual(values  = COLORES_MODELOS) +
  scale_y_continuous(limits = c(0, 1), breaks = c(0.25, 0.5, 0.75, 1.0)) +
  labs(
    title  = "Perfil comparativo de modelos",
    x      = NULL, y = NULL,
    color  = "Modelo", fill = "Modelo"
  ) +
  tema_base +
  theme(axis.text.x = element_text(face = "bold", size = 10))

print(p_radar)
guardar("02_radar_modelos", ancho = 8, alto = 7)


# ─────────────────────────────────────────────
# 4. Distribución de probabilidades — modelo ganador
# ─────────────────────────────────────────────
cat("Generando: distribución de probabilidades...\n")

# Renombrar columnas al formato original para análisis
pred <- predicciones |>
  rename(
    prob_default   = `Probabilidad.Default`,
    clasificacion  = `Clasificación`,
    riesgo         = `Nivel.de.Riesgo`,
    default_real   = `Default.Real`
  )

p_prob_dist <- ggplot(pred, aes(x = prob_default,
                                 fill = factor(default_real,
                                               labels = c("No moroso", "Moroso")))) +
  geom_histogram(bins = 60, alpha = 0.65, position = "identity") +
  geom_vline(xintercept = 0.49, linetype = "dashed",
             color = COL_AZUL, linewidth = 0.9) +
  annotate("text", x = 0.52, y = Inf, vjust = 1.5,
           label = "Umbral = 0.49", color = COL_AZUL,
           fontface = "bold", size = 4) +
  scale_fill_manual(values = c("No moroso" = COL_NO_MOROSO, "Moroso" = COL_MOROSO)) +
  scale_x_continuous(labels = percent_format(accuracy = 1)) +
  labs(
    title    = "Distribución de probabilidades predichas — Random Forest",
    subtitle = "Separación entre clases en el test set",
    x        = "Probabilidad de default",
    y        = "Frecuencia",
    fill     = NULL
  ) +
  tema_base

print(p_prob_dist)
guardar("03_distribucion_probabilidades", ancho = 10, alto = 5)


# ─────────────────────────────────────────────
# 5. Matriz de confusión — modelo ganador
# ─────────────────────────────────────────────
cat("Generando: matriz de confusión...\n")

umbral <- 0.49

pred <- pred |>
  mutate(pred_clase = ifelse(prob_default >= umbral, "Moroso", "No moroso"),
         real_clase = ifelse(default_real == 1,       "Moroso", "No moroso"))

cm <- pred |>
  count(real_clase, pred_clase) |>
  mutate(
    etiqueta = comma(n),
    tipo = case_when(
      real_clase == "Moroso"    & pred_clase == "Moroso"    ~ "VP",
      real_clase == "No moroso" & pred_clase == "No moroso" ~ "VN",
      real_clase == "No moroso" & pred_clase == "Moroso"    ~ "FP",
      real_clase == "Moroso"    & pred_clase == "No moroso" ~ "FN"
    )
  )

p_cm <- ggplot(cm, aes(x = pred_clase, y = real_clase, fill = n)) +
  geom_tile(color = "white", linewidth = 1.5) +
  geom_text(aes(label = paste0(tipo, "\n", etiqueta)),
            size = 5.5, fontface = "bold", color = "white") +
  # Se extiende el barwidth y se ajustan los límites para que la escala respire
  scale_fill_gradient(low = "#AED6F1", high = COL_AZUL,
                      limits = c(0, 27000), 
                      breaks = seq(0, 25000, by = 5000),
                      guide = guide_colorbar(barwidth = 12, barheight = 0.8,
                                             label.theme = element_text(size = 9))) +
  scale_x_discrete(position = "top") +
  labs(
    title    = "Matriz de confusión — Random Forest (umbral = 0.49)",
    subtitle = "VP = Verdadero Positivo · VN = Verdadero Negativo · FP = Falso Positivo · FN = Falso Negativo",
    x        = "Predicción",
    y        = "Realidad",
    fill     = "Conteo"
  ) +
  tema_base +
  theme(
    axis.text  = element_text(size = 12, face = "bold"),
    axis.title = element_text(size = 12, face = "bold"),
    legend.position = "bottom",
    plot.margin = margin(10, 20, 10, 20)
  )

print(p_cm)
guardar("04_matriz_confusion", ancho = 9, alto = 6)


# ─────────────────────────────────────────────
# 6. Tasa de detección por nivel de riesgo
# ─────────────────────────────────────────────
cat("Generando: detección por nivel de riesgo...\n")

tasa_riesgo <- pred |>
  filter(default_real == 1) |>
  mutate(riesgo = factor(riesgo, levels = c("Alto", "Medio", "Bajo"))) |>
  group_by(riesgo) |>
  summarise(
    Total     = n(),
    Detectados = sum(pred_clase == "Moroso"),
    Tasa      = Detectados / Total * 100,
    .groups   = "drop"
  )

p_deteccion <- ggplot(tasa_riesgo, aes(x = riesgo, y = Tasa, fill = riesgo)) +
  geom_col(width = 0.55, show.legend = FALSE) +
  geom_text(aes(label = sprintf("%.1f%%\n(%d/%d)", Tasa, Detectados, Total)),
            vjust = -0.4, size = 4, fontface = "bold") +
  scale_fill_manual(values = c("Alto" = COL_MOROSO, "Medio" = COL_AMBAR, "Bajo" = COL_VERDE)) +
  scale_y_continuous(limits = c(0, 115),
                     labels = function(x) paste0(x, "%")) +
  labs(
    title    = "Tasa de detección de morosos por nivel de riesgo",
    subtitle = "¿Qué porcentaje de morosos reales detecta el modelo en cada segmento?",
    x        = "Nivel de riesgo",
    y        = "Tasa de detección (%)"
  ) +
  tema_base

print(p_deteccion)
guardar("05_deteccion_por_riesgo", ancho = 8, alto = 5)


# ─────────────────────────────────────────────
# Resumen
# ─────────────────────────────────────────────
cat("\n=== Gráficos de modelos generados ===\n")
cat(sprintf("Carpeta: %s\n", RUTA_OUTPUT))
cat("  01_comparacion_metricas.png\n")
cat("  02_radar_modelos.png\n")
cat("  03_distribucion_probabilidades.png\n")
cat("  04_matriz_confusion.png\n")
cat("  05_deteccion_por_riesgo.png\n")
cat("\nSiguiente script: 04_graficos_shap.R\n")
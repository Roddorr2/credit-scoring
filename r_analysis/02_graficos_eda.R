# =============================================================================
# Script: 02_graficos_eda.R
# Sección: Resultados — Análisis Exploratorio de Datos
# Genera gráficos de alta calidad listos para el informe
# =============================================================================
# Ejecutar desde la raíz del proyecto
# Outputs: r_analysis/outputs/eda_plots/
# =============================================================================

library(ggplot2)
library(dplyr)
library(tidyr)
library(scales)
library(patchwork)

# ─────────────────────────────────────────────
# 0. Configuración global
# ─────────────────────────────────────────────

RUTA_CSV    <- "data/raw/CreditScoring.csv"
RUTA_OUTPUT <- "r_analysis/outputs/eda_plots"
dir.create(RUTA_OUTPUT, recursive = TRUE, showWarnings = FALSE)

# Paleta de colores del proyecto
COL_NO_MOROSO <- "#4A90D9"
COL_MOROSO    <- "#E55C5C"
COL_AZUL      <- "#1F4E79"

# Tema base para todos los gráficos
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
# 1. Carga y limpieza mínima
# ─────────────────────────────────────────────

df <- read.csv(RUTA_CSV) |>
  select(-ID) |>
  mutate(
    Target = factor(SeriousDlqin2yrs,
                    levels = c(0, 1),
                    labels = c("No moroso", "Moroso")),
    # Reemplazar códigos especiales por NA
    across(c(NumberOfTime30.59DaysPastDueNotWorse,
             NumberOfTime60.89DaysPastDueNotWorse,
             NumberOfTimes90DaysLate),
           ~ ifelse(. %in% c(96, 98), NA, .)),
    # Corregir age == 0
    age = ifelse(age == 0, NA, age)
  )

cat(sprintf("Dataset cargado: %d filas\n\n", nrow(df)))


# ─────────────────────────────────────────────
# 2. Desbalance de clases
# ─────────────────────────────────────────────
cat("Generando: desbalance de clases...\n")

conteo_target <- df |>
  count(Target) |>
  mutate(pct = n / sum(n),
         etiqueta = paste0(comma(n), "\n(", percent(pct, accuracy = 0.1), ")"))

p_target <- ggplot(conteo_target, aes(x = Target, y = n, fill = Target)) +
  geom_col(width = 0.55, show.legend = FALSE) +
  geom_text(aes(label = etiqueta), vjust = -0.4, size = 4, fontface = "bold") +
  scale_fill_manual(values = c("No moroso" = COL_NO_MOROSO, "Moroso" = COL_MOROSO)) +
  scale_y_continuous(labels = comma, expand = expansion(mult = c(0, 0.15))) +
  labs(
    title    = "Distribución de la variable objetivo",
    subtitle = "Desbalance severo: ratio 13.9:1",
    x        = NULL,
    y        = "Número de clientes"
  ) +
  tema_base

print(p_target)
guardar("01_desbalance_clases")


# ─────────────────────────────────────────────
# 3. Valores nulos
# ─────────────────────────────────────────────
cat("Generando: valores nulos...\n")

nulos <- df |>
  select(-Target) |>
  summarise(across(everything(), ~ sum(is.na(.)))) |>
  pivot_longer(everything(), names_to = "Variable", values_to = "Nulos") |>
  filter(Nulos > 0) |>
  mutate(
    Porcentaje = Nulos / nrow(df) * 100,
    Variable   = reorder(Variable, Porcentaje)
  )

p_nulos <- ggplot(nulos, aes(x = Variable, y = Porcentaje)) +
  geom_col(fill = COL_MOROSO, width = 0.5) +
  geom_text(aes(label = sprintf("%.1f%%", Porcentaje)), hjust = -0.2, size = 4) +
  coord_flip() +
  scale_y_continuous(expand = expansion(mult = c(0, 0.2))) +
  labs(
    title    = "Porcentaje de valores nulos por variable",
    subtitle = "Solo las variables con datos faltantes",
    x        = NULL,
    y        = "% de valores nulos"
  ) +
  tema_base

print(p_nulos)
guardar("02_valores_nulos", ancho = 8, alto = 4)


# ─────────────────────────────────────────────
# 4. Histogramas por clase — variables continuas
# ─────────────────────────────────────────────
cat("Generando: histogramas por clase...\n")

vars_continuas <- c("RevolvingUtilizationOfUnsecuredLines",
                    "age", "DebtRatio", "MonthlyIncome")
etiquetas_vars <- c("Utilización Crédito", "Edad",
                    "Ratio de Deuda", "Ingreso Mensual")

plots_hist <- list()

for (i in seq_along(vars_continuas)) {
  var  <- vars_continuas[i]
  etiq <- etiquetas_vars[i]
  
  datos <- df |>
    select(valor = all_of(var), Target) |>
    filter(!is.na(valor)) |>
    mutate(valor = pmin(valor, quantile(valor, 0.99, na.rm = TRUE)))
  
  plots_hist[[i]] <- ggplot(datos, aes(x = valor, fill = Target)) +
    geom_histogram(bins = 40, alpha = 0.6, position = "identity") +
    scale_fill_manual(values = c("No moroso" = COL_NO_MOROSO, "Moroso" = COL_MOROSO)) +
    labs(title = etiq, x = NULL, y = "Frecuencia", fill = NULL) +
    tema_base +
    theme(plot.title = element_text(size = 12))
}

p_hist_combinado <- wrap_plots(plots_hist, ncol = 2) +
  plot_annotation(
    title    = "Distribución de variables continuas por clase",
    subtitle = "Cap. en percentil 99 para legibilidad",
    theme    = theme(
      plot.title    = element_text(face = "bold", color = COL_AZUL, size = 14),
      plot.subtitle = element_text(color = "gray40", size = 11)
    )
  ) +
  plot_layout(guides = "collect") &
  theme(legend.position = "bottom")

print(p_hist_combinado)
guardar("03_histogramas_por_clase", ancho = 12, alto = 8)


# ─────────────────────────────────────────────
# 5. Boxplots por clase — variables clave
# ─────────────────────────────────────────────
cat("Generando: boxplots por clase...\n")

vars_box <- c("age", "MonthlyIncome",
              "RevolvingUtilizationOfUnsecuredLines", "DebtRatio")
etiq_box <- c("Edad", "Ingreso Mensual",
              "Utilización Crédito", "Ratio de Deuda")

plots_box <- list()

for (i in seq_along(vars_box)) {
  var  <- vars_box[i]
  etiq <- etiq_box[i]
  
  datos <- df |>
    select(valor = all_of(var), Target) |>
    filter(!is.na(valor)) |>
    mutate(valor = pmin(valor, quantile(valor, 0.99, na.rm = TRUE)))
  
  plots_box[[i]] <- ggplot(datos, aes(x = Target, y = valor, fill = Target)) +
    geom_boxplot(alpha = 0.7, outlier.size = 0.8, outlier.alpha = 0.3,
                 show.legend = FALSE) +
    scale_fill_manual(values = c("No moroso" = COL_NO_MOROSO, "Moroso" = COL_MOROSO)) +
    labs(title = etiq, x = NULL, y = NULL) +
    tema_base +
    theme(plot.title = element_text(size = 12))
}

p_box_combinado <- wrap_plots(plots_box, ncol = 4) +
  plot_annotation(
    title = "Distribución por clase — variables clave",
    theme = theme(
      plot.title = element_text(face = "bold", color = COL_AZUL, size = 14)
    )
  )

print(p_box_combinado)
guardar("04_boxplots_por_clase", ancho = 14, alto = 5)


# ─────────────────────────────────────────────
# 6. Heatmap de correlación
# ─────────────────────────────────────────────
cat("Generando: heatmap de correlación...\n")

vars_corr <- df |>
  select(-Target, -SeriousDlqin2yrs) |>
  select(where(is.numeric))

# Reemplazar 96/98 antes de calcular correlación
vars_corr <- vars_corr |>
  mutate(across(everything(), ~ ifelse(. %in% c(96, 98), NA, .)))

corr_matrix <- cor(vars_corr, use = "pairwise.complete.obs") |>
  as.data.frame() |>
  rownames_to_column("Var1") |>
  pivot_longer(-Var1, names_to = "Var2", values_to = "Correlacion")

# Nombres cortos para el heatmap
nombres_cortos <- c(
  "RevolvingUtilizationOfUnsecuredLines" = "Util.Crédito",
  "age"                                  = "Edad",
  "NumberOfTime30.59DaysPastDueNotWorse" = "Atraso 30-59",
  "DebtRatio"                            = "Ratio Deuda",
  "MonthlyIncome"                        = "Ingreso",
  "NumberOfOpenCreditLinesAndLoans"      = "Líneas",
  "NumberOfTimes90DaysLate"              = "Atraso 90+",
  "NumberRealEstateLoansOrLines"         = "Inmuebles",
  "NumberOfTime60.89DaysPastDueNotWorse" = "Atraso 60-89",
  "NumberOfDependents"                   = "Dependientes"
)

corr_matrix <- corr_matrix |>
  mutate(
    Var1 = recode(Var1, !!!nombres_cortos),
    Var2 = recode(Var2, !!!nombres_cortos)
  )

p_corr <- ggplot(corr_matrix, aes(x = Var1, y = Var2, fill = Correlacion)) +
  geom_tile(color = "white", linewidth = 0.5) +
  geom_text(aes(label = sprintf("%.2f", Correlacion)),
            size = 3, color = ifelse(abs(corr_matrix$Correlacion) > 0.5, "white", "gray20")) +
  scale_fill_gradient2(
    low      = COL_NO_MOROSO,
    mid      = "white",
    high     = COL_MOROSO,
    midpoint = 0,
    limits   = c(-1, 1),
    name     = "Correlación"
  ) +
  labs(
    title    = "Matriz de correlación de Pearson",
    subtitle = "Variables predictoras del dataset",
    x        = NULL,
    y        = NULL
  ) +
  tema_base +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 10))

print(p_corr)
guardar("05_heatmap_correlacion", ancho = 10, alto = 8)


# ─────────────────────────────────────────────
# 7. Tasa de morosidad por tramo de edad
# ─────────────────────────────────────────────
cat("Generando: tasa de morosidad por edad...\n")

tasa_edad <- df |>
  filter(!is.na(age)) |>
  mutate(Tramo = cut(age,
                     breaks = c(0, 25, 35, 45, 55, 65, 110),
                     labels = c("18-25", "26-35", "36-45", "46-55", "56-65", "65+"))) |>
  group_by(Tramo) |>
  summarise(Tasa = mean(SeriousDlqin2yrs) * 100, .groups = "drop")

p_edad <- ggplot(tasa_edad, aes(x = Tramo, y = Tasa)) +
  geom_col(fill = COL_AZUL, width = 0.6) +
  geom_text(aes(label = sprintf("%.1f%%", Tasa)), vjust = -0.5, size = 4, fontface = "bold") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15)),
                     labels = function(x) paste0(x, "%")) +
  labs(
    title    = "Tasa de morosidad por tramo de edad",
    subtitle = "Los clientes más jóvenes presentan mayor riesgo",
    x        = "Tramo de edad",
    y        = "Tasa de morosidad (%)"
  ) +
  tema_base

print(p_edad)
guardar("06_morosidad_por_edad", ancho = 8, alto = 5)


# ─────────────────────────────────────────────
# 8. Tasa de morosidad por atrasos previos
# ─────────────────────────────────────────────
cat("Generando: tasa de morosidad por atrasos...\n")

tasa_atrasos <- df |>
  filter(!is.na(NumberOfTimes90DaysLate)) |>
  mutate(Atrasos = pmin(NumberOfTimes90DaysLate, 5),
         Atrasos = factor(Atrasos, labels = c("0","1","2","3","4","5+"))) |>
  group_by(Atrasos) |>
  summarise(Tasa = mean(SeriousDlqin2yrs) * 100, .groups = "drop")

p_atrasos <- ggplot(tasa_atrasos, aes(x = Atrasos, y = Tasa)) +
  geom_col(fill = COL_MOROSO, width = 0.6) +
  geom_text(aes(label = sprintf("%.1f%%", Tasa)), vjust = -0.5, size = 4, fontface = "bold") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15)),
                     labels = function(x) paste0(x, "%")) +
  labs(
    title    = "Tasa de morosidad según atrasos previos (+90 días)",
    subtitle = "Variable con mayor poder predictivo",
    x        = "Número de atrasos previos",
    y        = "Tasa de morosidad (%)"
  ) +
  tema_base

print(p_atrasos)
guardar("07_morosidad_por_atrasos", ancho = 8, alto = 5)


# ─────────────────────────────────────────────
# Resumen
# ─────────────────────────────────────────────
cat("\n=== Gráficos EDA generados ===\n")
cat(sprintf("Carpeta: %s\n", RUTA_OUTPUT))
cat("  01_desbalance_clases.png\n")
cat("  02_valores_nulos.png\n")
cat("  03_histogramas_por_clase.png\n")
cat("  04_boxplots_por_clase.png\n")
cat("  05_heatmap_correlacion.png\n")
cat("  06_morosidad_por_edad.png\n")
cat("  07_morosidad_por_atrasos.png\n")
cat("\nSiguiente script: 03_graficos_modelos.R\n")
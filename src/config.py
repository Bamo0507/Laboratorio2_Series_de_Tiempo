"""
Configuracion central del Laboratorio 2 (LSTM + catch22).
------------------------------------------------------------
Unica fuente de rutas, constantes y definicion de las series.

Las series se definen exactamente como en el Laboratorio 1: se construyen a
partir de los mismos CSV de entrenamiento / prueba (particion temporal 70/30,
Turista + Excursionista), con la misma columna de agrupacion, el mismo relleno
de meses faltantes y la misma transformacion.
"""

from pathlib import Path

# --------------------------------------------------------------------------- #
# Rutas
# --------------------------------------------------------------------------- #
RAIZ = Path(__file__).resolve().parent.parent

DIR_RAW = RAIZ / "datos" / "raw"
DIR_PROCESSED = RAIZ / "datos" / "processed"
DIR_RESULTADOS = RAIZ / "resultados"
DIR_FIGURAS = DIR_RESULTADOS / "figuras"
DIR_TABLAS = DIR_RESULTADOS / "tablas"
DIR_MODELOS = DIR_RESULTADOS / "modelos"

# Heredados del Laboratorio 1 sin modificacion (misma particion train/test).
RUTA_TRAIN = DIR_PROCESSED / "entrenamiento.csv"
RUTA_PRUEBA = DIR_PROCESSED / "prueba.csv"
RUTA_CRUDA = DIR_RAW / "Base_Migracion_2009-2026jun.xlsx"

# --------------------------------------------------------------------------- #
# Constantes del dominio
# --------------------------------------------------------------------------- #
PERIODO = 12          # estacionalidad mensual
SEMILLA = 42          # reproducibilidad
COL_VALOR = "Viajero"
COL_FECHA = "Fecha"

# Contrato de entrada: columnas que deben venir en los CSV del Laboratorio 1.
COLUMNAS_REQUERIDAS = [
    "Fecha", "Ano", "Mes cod", "pandemia", "Via", "Frontera",
    "Pais", "Region", "Region dos", "Tipo de Viajero", "Viajero",
]
# Los CSV vienen con tildes en los encabezados; se validan con los nombres reales.
COLUMNAS_REQUERIDAS_REALES = [
    "Fecha", "Año", "Mes cod", "pandemia", "Vía", "Frontera",
    "País", "Región", "Región dos", "Tipo de Viajero", "Viajero",
]

# --------------------------------------------------------------------------- #
# Definicion de las 7 series construidas en el Laboratorio 1
# --------------------------------------------------------------------------- #
# clave        -> identificador corto usado en tablas y archivos
# etiqueta     -> nombre para graficos
# columna      -> None = serie total (suma de todo); si no, columna de filtro
# valor        -> categoria a filtrar dentro de esa columna
# categoria    -> agrupacion del enunciado (para la pregunta 2.10)
# relleno_min  -> True si en el Lab 1 los meses sin registro se rellenaron con el minimo
SERIES_LAB1 = {
    "total": {
        "etiqueta": "Total mensual",
        "columna": None,
        "valor": None,
        "categoria": "Total",
        "relleno_min": False,
    },
    "reg_centro": {
        "etiqueta": "America Del Centro",
        "columna": "Región dos",
        "valor": "América Del Centro",
        "categoria": "Region",
        "relleno_min": False,
    },
    "reg_norte": {
        "etiqueta": "America Del Norte",
        "columna": "Región dos",
        "valor": "América Del Norte",
        "categoria": "Region",
        "relleno_min": True,   # abr-ago 2020 sin registro -> minimo de la serie
    },
    "reg_europa": {
        "etiqueta": "Europa",
        "columna": "Región dos",
        "valor": "Europa",
        "categoria": "Region",
        "relleno_min": True,
    },
    "fro_aurora": {
        "etiqueta": "01 La Aurora",
        "columna": "Frontera",
        "valor": "01 La Aurora",
        "categoria": "Frontera",
        "relleno_min": False,
    },
    "fro_valle_nuevo": {
        "etiqueta": "07 Valle Nuevo",
        "columna": "Frontera",
        "valor": "07 Valle Nuevo",
        "categoria": "Frontera",
        "relleno_min": False,
    },
    "fro_san_cristobal": {
        "etiqueta": "09 San Cristobal",
        "columna": "Frontera",
        "valor": "09 San Cristóbal",
        "categoria": "Frontera",
        "relleno_min": False,
    },
}

# Series elegidas para los modelos LSTM (inciso 1.1).
SERIES_LSTM = ["total", "fro_aurora"]

# --------------------------------------------------------------------------- #
# Mejor modelo del Laboratorio 1 por serie (para la comparacion del inciso 1.4)
# --------------------------------------------------------------------------- #
# Metricas fuera de muestra sobre los 63 meses de prueba, en escala original.
MEJORES_LAB1 = {
    "total": {"modelo": "Suav. exp. simple", "MAE": 158777.0, "RMSE": 173709.0},
    "reg_centro": {"modelo": "Suav. exp. simple", "MAE": 108285.0, "RMSE": 120341.0},
    "reg_norte": {"modelo": "Suav. exp. simple", "MAE": 32194.0, "RMSE": 36725.0},
    "reg_europa": {"modelo": "Suav. exp. simple", "MAE": 9292.0, "RMSE": 10695.0},
    "fro_aurora": {"modelo": "Suav. exp. simple", "MAE": 36822.0, "RMSE": 42053.0},
    "fro_valle_nuevo": {"modelo": "Prophet", "MAE": 16804.0, "RMSE": 23187.0},
    "fro_san_cristobal": {"modelo": "Suav. exp. simple", "MAE": 21103.0, "RMSE": 23740.0},
}

# --------------------------------------------------------------------------- #
# Metricas exploratorias del Laboratorio 1 (notebooks de analisisComparativo).
# NO se hardcodean: se recalculan con utils.metricas_eda_lab1(), que replica
# exactamente la funcion del Lab 1 (ACF mes 12 sobre la serie diferenciada,
# % de crecimiento 2009-2019, coeficiente de variacion y % de caida 2019->2020).
# Asi la comparacion de la pregunta 2.12 queda reproducible y no depende de que
# alguien copie bien un numero de un notebook anterior.
# --------------------------------------------------------------------------- #

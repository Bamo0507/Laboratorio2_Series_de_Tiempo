# Laboratorio 2 — Deep Learning (LSTM) y catch22

Este laboratorio continúa el análisis del ingreso de viajeros internacionales a Guatemala que se
trabajó en el [Laboratorio 1](https://github.com/Bamo0507/Laboratorio_1_Series_de_Tiempo), ahora
modelando dos de esas series con redes **LSTM** y explorando la similitud de las siete series
construidas mediante el algoritmo **catch22**.

## Estructura

```
.
├── datos/
│   ├── raw/ # Base_Migracion_2009-2026jun.xlsx (crudo, intacto)
│   └── processed/ # entrenamiento.csv y prueba.csv heredados del Lab 1
├── src/
│   ├── config.py # rutas, series, semilla y métricas del Lab 1
│   └── utils.py # construcción de series, métricas, LSTM y tuneo
├── notebooks/
│   ├── lstm/ # ejercicio 1, un notebook por serie
│   └── catch22/ # ejercicio 2
├── resultados/tablas/ # salidas regenerables (no se versionan)
└── requirements.txt
```

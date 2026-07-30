"""
Utilidades compartidas del Laboratorio 2 — Ejercicio 1 (modelos LSTM).
------------------------------------------------------------
Todo lo que los notebooks necesitan mas de una vez vive aqui, para que el codigo
de los notebooks se dedique al analisis y no a repetir plomeria:

  Series      : construccion de las series del Lab 1 (train / prueba / completa).
  Metricas    : MAE y RMSE en escala original, identicas a las del Lab 1.
  Supervisado : conversion de una serie a ventanas (X, y) para la LSTM.
  LSTM        : construccion, entrenamiento, pronostico recursivo y tuneo.
"""

from __future__ import annotations

import os
import random
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402


# --------------------------------------------------------------------------- #
# Logging minimo (mismos prefijos que el estandar del curso)
# --------------------------------------------------------------------------- #
def banner(titulo: str) -> None:
    print("=" * 70)
    print(titulo)
    print("=" * 70)


def afirmar(condicion: bool, mensaje: str) -> None:
    """Validacion fail-fast: corta la ejecucion si el supuesto no se cumple."""
    if condicion:
        print(f"[ok] {mensaje}")
    else:
        raise AssertionError(f"[FALLO] {mensaje}")


# --------------------------------------------------------------------------- #
# Construccion de las series (identica al Laboratorio 1)
# --------------------------------------------------------------------------- #
def cargar_conjuntos() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga los CSV de entrenamiento y prueba heredados del Laboratorio 1."""
    train = pd.read_csv(config.RUTA_TRAIN, parse_dates=[config.COL_FECHA])
    prueba = pd.read_csv(config.RUTA_PRUEBA, parse_dates=[config.COL_FECHA])
    return train, prueba


def _agregar_mensual(df: pd.DataFrame, columna, valor) -> pd.Series:
    """Suma mensual de viajeros, filtrando por categoria si aplica."""
    if columna is not None:
        df = df[df[columna] == valor]
    return df.groupby(config.COL_FECHA)[config.COL_VALOR].sum().asfreq("MS")


def construir_serie(clave: str, conjunto: str = "train") -> pd.Series:
    """
    Devuelve una de las series del Lab 1.

    conjunto: "train"    -> 2009-01 a 2021-03 (147 meses)
              "prueba"   -> 2021-04 a 2026-06 (63 meses)
              "completa" -> concatenacion de ambas (210 meses)

    El relleno de meses sin registro se hace con el minimo de la serie de
    ENTRENAMIENTO, igual que en el Lab 1 (no con el minimo de cada tramo, para
    que train y prueba usen el mismo valor y la serie completa sea consistente).
    """
    meta = config.SERIES_LAB1[clave]
    train_df, prueba_df = cargar_conjuntos()

    serie_train = _agregar_mensual(train_df, meta["columna"], meta["valor"])
    serie_prueba = _agregar_mensual(prueba_df, meta["columna"], meta["valor"])

    valor_relleno = serie_train.min()
    if meta["relleno_min"] or serie_train.isna().any() or serie_prueba.isna().any():
        serie_train = serie_train.fillna(valor_relleno)
        serie_prueba = serie_prueba.fillna(valor_relleno)

    if conjunto == "train":
        return serie_train
    if conjunto == "prueba":
        return serie_prueba
    if conjunto == "completa":
        return pd.concat([serie_train, serie_prueba]).asfreq("MS")
    raise ValueError(f"conjunto no valido: {conjunto}")


# --------------------------------------------------------------------------- #
# Metricas (identicas a las del Laboratorio 1)
# --------------------------------------------------------------------------- #
def metricas(y_true_log, y_pred_log) -> dict:
    """
    MAE y RMSE en la ESCALA ORIGINAL de viajeros.

    Se recibe todo en log y se deshace con exp, exactamente como en el Lab 1,
    para que los numeros sean comparables uno a uno contra SARIMA, Holt-Winters,
    SES, seasonal naive y Prophet.
    """
    y_true = np.exp(np.asarray(y_true_log, dtype=float))
    y_pred = np.exp(np.asarray(y_pred_log, dtype=float))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    return {"MAE": mae, "RMSE": rmse, "MAPE_%": mape}


# --------------------------------------------------------------------------- #
# Reproducibilidad
# --------------------------------------------------------------------------- #
def fijar_semillas(semilla: int = config.SEMILLA) -> None:
    """Fija las semillas de python, numpy y tensorflow."""
    os.environ["PYTHONHASHSEED"] = str(semilla)
    random.seed(semilla)
    np.random.seed(semilla)
    import tensorflow as tf
    tf.random.set_seed(semilla)
    tf.keras.utils.set_random_seed(semilla)


# --------------------------------------------------------------------------- #
# Serie -> aprendizaje supervisado
# --------------------------------------------------------------------------- #
def a_supervisado(valores: np.ndarray, ventana: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Convierte un vector en pares (X, y) de ventana deslizante para la LSTM.

    Con ventana=12 el modelo ve los 12 meses anteriores y predice el siguiente,
    de modo que la red tiene acceso al ciclo anual completo sin que le demos la
    estacionalidad de forma explicita.

    X queda con forma (n_muestras, ventana, 1), que es lo que espera Keras:
    (batch, pasos de tiempo, features).
    """
    X, y = [], []
    for i in range(len(valores) - ventana):
        X.append(valores[i:i + ventana])
        y.append(valores[i + ventana])
    X = np.asarray(X, dtype="float32").reshape(-1, ventana, 1)
    y = np.asarray(y, dtype="float32")
    return X, y


class EscaladorMinMax:
    """
    MinMax a [0, 1] ajustado SOLO con los datos de entrenamiento.

    Se implementa a mano (en lugar de sklearn) para dejar explicito que el
    minimo y el maximo salen unicamente del tramo de entrenamiento: si se
    ajustara sobre toda la serie, el modelo estaria viendo el rango del futuro
    y las metricas de prueba dejarian de ser honestas.
    """

    def __init__(self):
        self.minimo = None
        self.maximo = None

    def ajustar(self, x: np.ndarray) -> "EscaladorMinMax":
        self.minimo = float(np.min(x))
        self.maximo = float(np.max(x))
        return self

    def transformar(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype="float64") - self.minimo) / (self.maximo - self.minimo)

    def invertir(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype="float64") * (self.maximo - self.minimo) + self.minimo


# --------------------------------------------------------------------------- #
# LSTM
# --------------------------------------------------------------------------- #
def construir_lstm(ventana: int, unidades: int, capas: int = 1,
                   dropout: float = 0.0, tasa_aprendizaje: float = 0.001,
                   bidireccional: bool = False):
    """
    Arma una LSTM secuencial para pronostico univariado a un paso.

    ventana          : numero de meses de historia que entran a la red
    unidades         : neuronas de la capa LSTM (la segunda capa usa la mitad)
    capas            : 1 = LSTM simple, 2 = LSTM apilada
    dropout          : regularizacion entre capas (clave con 147 observaciones)
    tasa_aprendizaje : learning rate de Adam
    bidireccional    : envuelve la primera capa en Bidirectional
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    modelo = keras.Sequential(name=f"lstm_u{unidades}_c{capas}_v{ventana}")
    modelo.add(layers.Input(shape=(ventana, 1)))

    for i in range(capas):
        ultima = (i == capas - 1)
        unidades_capa = unidades if i == 0 else max(4, unidades // 2)
        capa = layers.LSTM(unidades_capa, return_sequences=not ultima)
        modelo.add(layers.Bidirectional(capa) if (bidireccional and i == 0) else capa)
        if dropout > 0:
            modelo.add(layers.Dropout(dropout))

    modelo.add(layers.Dense(1))
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=tasa_aprendizaje),
        loss="mse",
        metrics=["mae"],
    )
    return modelo


def entrenar_lstm(modelo, X, y, epocas: int = 300, lote: int = 16,
                  validacion=None, paciencia: int = 40, verbose: int = 0):
    """
    Entrena con early stopping.

    Si se pasa validacion=(X_val, y_val) el early stopping vigila val_loss; si
    no, vigila la loss de entrenamiento. Siempre restaura los mejores pesos,
    para que el modelo que se evalua no sea el de la ultima epoca sino el mejor.
    """
    from tensorflow import keras

    monitor = "val_loss" if validacion is not None else "loss"
    parada = keras.callbacks.EarlyStopping(
        monitor=monitor, patience=paciencia, restore_best_weights=True, verbose=0
    )
    historia = modelo.fit(
        X, y,
        epochs=epocas,
        batch_size=lote,
        validation_data=validacion,
        callbacks=[parada],
        shuffle=False,          # es una serie de tiempo: no se mezclan las ventanas
        verbose=verbose,
    )
    return historia


def ajustar_lstm(valores_train_log: np.ndarray, cfg: dict,
                 semilla: int = config.SEMILLA, verbose: int = 0) -> dict:
    """
    Entrena una LSTM con la configuracion `cfg` sobre la serie en log.

    Concentra las dos formas de presentarle los datos a la red, que es el eje
    `presentacion` de la grilla:

      "nivel"        -> la red recibe log(viajeros) y predice el siguiente log.
      "diferenciada" -> la red recibe los cambios mes a mes del log y predice el
                        siguiente cambio. El nivel se reconstruye despues con una
                        suma acumulada a partir del ultimo valor real conocido
                        (el ancla), ya que diferenciar y acumular son operaciones
                        inversas.

    Devuelve un diccionario con todo lo necesario para pronosticar despues, de
    modo que el tuneo y el ajuste del modelo final usan exactamente el mismo
    camino y no puede haber diferencias accidentales entre ambos.
    """
    from tensorflow import keras

    ventana = cfg["ventana"]
    diferenciada = cfg.get("presentacion", "nivel") == "diferenciada"

    if diferenciada:
        base = np.diff(valores_train_log)      # cambios mes a mes del log
        ancla = float(valores_train_log[-1])   # ultimo nivel real conocido
    else:
        base = np.asarray(valores_train_log, dtype="float64")
        ancla = None

    escalador = EscaladorMinMax().ajustar(base)
    base_esc = escalador.transformar(base)
    X, y = a_supervisado(base_esc, ventana)

    keras.backend.clear_session()
    fijar_semillas(semilla)
    modelo = construir_lstm(
        ventana=ventana, unidades=cfg["unidades"], capas=cfg["capas"],
        dropout=cfg["dropout"], tasa_aprendizaje=cfg["tasa_aprendizaje"],
        bidireccional=cfg.get("bidireccional", False),
    )
    historia = entrenar_lstm(modelo, X, y, epocas=cfg.get("epocas", 300),
                             lote=cfg.get("lote", 16),
                             paciencia=cfg.get("paciencia", 40), verbose=verbose)

    return {
        "cfg": cfg, "modelo": modelo, "escalador": escalador, "base_esc": base_esc,
        "ancla": ancla, "diferenciada": diferenciada, "ventana": ventana,
        "historia": historia, "n_muestras": len(X),
        "epocas_corridas": len(historia.history["loss"]),
    }


def pronosticar_log(ajuste: dict, pasos: int) -> np.ndarray:
    """
    Pronostico recursivo de `pasos` meses, devuelto siempre en log(viajeros).

    Si la red se entreno sobre la serie diferenciada, lo que predice son cambios,
    asi que se acumulan sobre el ancla para volver al nivel. De esta forma quien
    llama a la funcion recibe siempre lo mismo (niveles en log) sin importar como
    se le presentaron los datos a la red, y las metricas se calculan igual en
    ambos casos.
    """
    pred_esc = pronostico_recursivo(ajuste["modelo"], ajuste["base_esc"],
                                    ajuste["ventana"], pasos)
    pred = ajuste["escalador"].invertir(pred_esc)
    if ajuste["diferenciada"]:
        return ajuste["ancla"] + np.cumsum(pred)
    return pred


def pronosticar_log_un_paso(ajuste: dict, serie_completa_log: np.ndarray,
                            pasos: int) -> np.ndarray:
    """
    Pronostico a un paso con ventana deslizante, devuelto en log(viajeros).

    A diferencia de pronosticar_log, aqui la red si ve el valor real de los meses
    previos del periodo de prueba (nunca el que esta prediciendo). Es una tarea
    mas facil y NO es comparable con las metricas del Lab 1; se reporta aparte
    para poder separar dos cosas distintas: que la red no logre aprender el
    patron, o que si lo aprenda pero no aguante extrapolar 63 meses seguidos.
    """
    serie_completa_log = np.asarray(serie_completa_log, dtype="float64")
    if ajuste["diferenciada"]:
        base = np.diff(serie_completa_log)
        niveles_previos = serie_completa_log[-pasos - 1:-1]   # ancla de cada mes
    else:
        base = serie_completa_log
        niveles_previos = None

    base_esc = ajuste["escalador"].transformar(base)
    pred_esc = pronostico_un_paso(ajuste["modelo"], base_esc, ajuste["ventana"], pasos)
    pred = ajuste["escalador"].invertir(pred_esc)
    return niveles_previos + pred if ajuste["diferenciada"] else pred


def pronostico_recursivo(modelo, historia_escalada: np.ndarray,
                         ventana: int, pasos: int) -> np.ndarray:
    """
    Pronostico multi-paso realimentando las propias predicciones.

    Es el modo que corresponde para comparar contra el Lab 1: ahi los modelos
    hicieron forecast(63) desde el final del entrenamiento, sin volver a ver
    ningun dato real del periodo de prueba. Aqui la red parte de los ultimos
    `ventana` meses del entrenamiento y cada prediccion se vuelve entrada de la
    siguiente, de modo que a partir del paso `ventana+1` la red esta prediciendo
    sobre puro pronostico propio.
    """
    ventana_actual = list(np.asarray(historia_escalada, dtype="float64")[-ventana:])
    predicciones = []
    for _ in range(pasos):
        entrada = np.asarray(ventana_actual[-ventana:], dtype="float32").reshape(1, ventana, 1)
        siguiente = float(modelo.predict(entrada, verbose=0)[0, 0])
        predicciones.append(siguiente)
        ventana_actual.append(siguiente)
    return np.asarray(predicciones)


def pronostico_un_paso(modelo, serie_escalada_completa: np.ndarray,
                       ventana: int, pasos: int) -> np.ndarray:
    """
    Pronostico a un paso con ventana deslizante (walk-forward).

    Aqui la red SI ve el valor real de los meses previos del periodo de prueba
    (nunca el que esta prediciendo). Es una tarea mas facil que el pronostico
    recursivo y no es comparable con las metricas del Lab 1; se reporta aparte,
    para separar "no puede aprender el patron" de "no puede extrapolar 63 meses".
    """
    valores = np.asarray(serie_escalada_completa, dtype="float64")
    inicio = len(valores) - pasos
    predicciones = []
    for t in range(inicio, len(valores)):
        entrada = valores[t - ventana:t].astype("float32").reshape(1, ventana, 1)
        predicciones.append(float(modelo.predict(entrada, verbose=0)[0, 0]))
    return np.asarray(predicciones)


# --------------------------------------------------------------------------- #
# Tuneo de hiperparametros con validacion de origen rodante
# --------------------------------------------------------------------------- #
def cortes_origen_rodante(n: int, n_cortes: int = 4, tamano_val: int = 12,
                          minimo_train: int = 60) -> list[tuple[int, int]]:
    """
    Genera los cortes (fin_train, fin_val) de una validacion de origen rodante.

    Con 147 observaciones un solo holdout final seria muy ruidoso y, peor aun,
    caeria justo sobre el desplome de la pandemia. Rodando el origen se evalua
    cada configuracion en varios tramos distintos de la historia y se promedia,
    que es lo que hace robusta la eleccion. En ningun corte se toca el conjunto
    de prueba: todo ocurre dentro de los 147 meses de entrenamiento.
    """
    cortes = []
    for k in range(n_cortes, 0, -1):
        fin_val = n - (k - 1) * tamano_val
        fin_train = fin_val - tamano_val
        if fin_train >= minimo_train:
            cortes.append((fin_train, fin_val))
    return cortes


def evaluar_config(valores_train_log: np.ndarray, cfg: dict,
                   n_cortes: int = 4, tamano_val: int = 12,
                   semilla: int = config.SEMILLA) -> dict:
    """
    Evalua una configuracion de hiperparametros por origen rodante.

    Para cada corte: se escala con el tramo de entrenamiento del corte, se
    entrena, se pronostica de forma RECURSIVA el tramo de validacion y se mide
    el error en escala original. Se devuelve el promedio y la desviacion entre
    cortes; la desviacion importa porque una config con buen promedio pero muy
    inestable no es confiable con tan pocos datos.

    Se devuelve tambien el error de cada corte por separado y el promedio
    excluyendo el ultimo, porque el ultimo corte de estas series cae sobre el
    desplome de la pandemia: ahi ninguna configuracion puede acertar y su error
    es tan grande que por si solo domina el promedio. Tener las dos versiones
    permite verificar si la configuracion ganadora lo es de verdad o solo por
    como se comporto en ese tramo irrepetible.
    """
    rmses, maes = [], []

    for fin_train, fin_val in cortes_origen_rodante(len(valores_train_log), n_cortes, tamano_val):
        tramo_train = valores_train_log[:fin_train]
        tramo_val = valores_train_log[fin_train:fin_val]

        ajuste = ajustar_lstm(tramo_train, cfg, semilla)
        if ajuste["n_muestras"] < 20:
            continue

        pred_log = pronosticar_log(ajuste, len(tramo_val))

        m = metricas(tramo_val, pred_log)
        rmses.append(m["RMSE"])
        maes.append(m["MAE"])

    if not rmses:
        return {"RMSE_val": np.inf, "MAE_val": np.inf, "RMSE_std": np.nan,
                "RMSE_val_sin_ultimo": np.inf, "rmse_por_corte": [], "n_cortes": 0}

    return {
        "RMSE_val": float(np.mean(rmses)),
        "MAE_val": float(np.mean(maes)),
        "RMSE_std": float(np.std(rmses)),
        "RMSE_val_sin_ultimo": float(np.mean(rmses[:-1])) if len(rmses) > 1 else float(rmses[0]),
        "rmse_por_corte": [round(r, 1) for r in rmses],
        "n_cortes": len(rmses),
    }


# Tasa de aprendizaje con la que se recorre el grid. No es un eje de busqueda
# porque es justamente el parametro que distingue al Modelo 1 del Modelo 2: el
# grid encuentra la mejor configuracion con este valor fijo y despues los dos
# modelos finales se instancian con esa configuracion y solo cambian la tasa.
LR_GRID = 0.001
LR_MODELOS = {"Modelo 1": 0.001, "Modelo 2": 0.01}


def grilla_lstm() -> list[dict]:
    """
    Grilla de tuneo: 24 configuraciones sobre tres ejes.

    presentacion (2) — "nivel": la red recibe log(viajeros) y predice el nivel.
                       "diferenciada": recibe los cambios mes a mes y predice el
                       siguiente cambio, y el nivel se reconstruye acumulando.
                       Es el eje que mas puede pesar, porque no cambia que tan
                       bien la red resuelve el problema sino CUAL es el problema.

    ventana (2)      — 12 meses (un ciclo anual completo) o 24 (dos ciclos, para
                       que pueda comparar un ano contra el anterior). Por debajo
                       de 12 la red no alcanza a ver el ciclo y no podria
                       aprender la estacionalidad ni queriendo.

    arquitectura (6) — tres familias, que son las "configuraciones diferentes"
                       del enunciado:
                       A. simple       — 1 capa (16/32/64 unidades). La base.
                       B. apilada      — 2 capas + dropout 0.2 (32/64 unidades).
                          Mas capacidad, pero con 147 observaciones el riesgo de
                          sobreajuste es real y el dropout esta para contenerlo.
                       C. bidireccional— 32 unidades, recorre la ventana en los
                          dos sentidos. En pronostico es discutible, pero dentro
                          de una ventana cerrada puede caracterizar mejor el
                          ciclo, y por eso vale contrastarla.
    """
    grilla = []
    for presentacion in ("nivel", "diferenciada"):
        for ventana in (12, 24):
            comun = dict(presentacion=presentacion, ventana=ventana,
                         tasa_aprendizaje=LR_GRID)
            for unidades in (16, 32, 64):
                grilla.append(dict(familia="A_simple", unidades=unidades,
                                   capas=1, dropout=0.0, **comun))
            for unidades in (32, 64):
                grilla.append(dict(familia="B_apilada", unidades=unidades,
                                   capas=2, dropout=0.2, **comun))
            grilla.append(dict(familia="C_bidireccional", unidades=32,
                               capas=1, dropout=0.0, bidireccional=True, **comun))
    return grilla


def tunear(valores_train_log: np.ndarray, grilla: list[dict],
           n_cortes: int = 4, tamano_val: int = 12,
           semilla: int = config.SEMILLA, verbose: bool = True) -> pd.DataFrame:
    """Recorre la grilla completa y devuelve la tabla ordenada por RMSE de validacion."""
    filas = []
    for i, cfg in enumerate(grilla, 1):
        res = evaluar_config(valores_train_log, cfg, n_cortes, tamano_val, semilla)
        filas.append({**cfg, **res})
        if verbose:
            print(f"[{i:3d}/{len(grilla)}] {cfg.get('presentacion','nivel'):13s} "
                  f"{cfg.get('familia',''):16s} v={cfg['ventana']:2d} "
                  f"u={cfg['unidades']:2d}  ->  RMSE_val={res['RMSE_val']:>12,.0f}")
    return pd.DataFrame(filas).sort_values("RMSE_val").reset_index(drop=True)


def tunear_con_cache(clave_serie: str, valores_train_log: np.ndarray,
                     grilla: list[dict] | None = None, usar_cache: bool = True,
                     **kwargs) -> pd.DataFrame:
    """
    Igual que tunear(), pero guarda el resultado en resultados/tablas/.

    El tuneo cuesta varios minutos, asi que se cachea para que reejecutar el
    notebook no obligue a repetirlo. Con usar_cache=False se fuerza a correrlo
    de nuevo desde cero. La grilla y la semilla estan fijas, de modo que el
    archivo cacheado es reproducible: borrarlo y volver a correr da lo mismo.
    """
    ruta = config.DIR_TABLAS / f"tuneo_lstm_{clave_serie}.csv"
    if usar_cache and ruta.exists():
        print(f"[cargado] {ruta.name} (tuneo cacheado; usar_cache=False para repetirlo)")
        return pd.read_csv(ruta)

    tabla = tunear(valores_train_log, grilla or grilla_lstm(), **kwargs)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(ruta, index=False)
    print(f"[guardado] {ruta}")
    return tabla

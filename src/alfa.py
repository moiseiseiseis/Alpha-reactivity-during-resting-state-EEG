"""Librería compartida del proyecto: reactividad alfa en EEG (EO vs EC).

Todo el pipeline sale de aquí y de ``CONFIG``. Los notebooks solo llaman
funciones; si alguien necesita algo de otro carril, se pide como función
nueva en este archivo (por PR, con una revisión).

Dueños por función:
    A (Moisés)   cargar_raw, tabla_bloques, marcar_artefactos
    B (Adrián)   acondicionar
    C (Diego)    potencia_alfa, resumen_por_bloque
    D (Ascencio) variantes de CONFIG (no necesita funciones propias)

Si cambia la firma de una función, se avisa en el chat del equipo.
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import mne
import pandas as pd

# ---------------------------------------------------------------------------
# Parámetros del análisis
# ---------------------------------------------------------------------------

CONFIG = {
    # A justifica la lista final (distribución occipito-parietal del alfa).
    "canales": ["P3", "Pz", "P4", "PO9", "O1", "Oz", "O2", "PO10"],
    "banda_alfa": (8.0, 13.0),
    # Provisional: B decide y justifica antes del 19 nov.
    "filtro": {"l_freq": 1.0, "h_freq": 40.0, "notch": 50.0},
    "sfreq_analisis": 250,
    "ventana_s": 2.0,
    "solapamiento": 0.5,          # fracción de la ventana (0.5 = 50 %)
    "recorte_transicion_s": 5.0,  # s descartados después de cada cambio de condición
    "excluir_artefactos": True,   # usa anotaciones BAD_ con reject_by_annotation
}

ARCHIVO_VHDR = "sub-010002_task-resting_eeg.vhdr"


def variante(**cambios) -> dict:
    """Copia de CONFIG con algunos parámetros cambiados (para la Etapa 5).

    Ejemplo: ``alfa.variante(ventana_s=10.0, excluir_artefactos=False)``.
    Los diccionarios anidados (``filtro``) se actualizan, no se reemplazan.
    """
    cfg = copy.deepcopy(CONFIG)
    for clave, valor in cambios.items():
        if clave not in cfg:
            raise KeyError(f"'{clave}' no existe en CONFIG")
        if isinstance(cfg[clave], dict) and isinstance(valor, dict):
            cfg[clave].update(valor)
        else:
            cfg[clave] = valor
    return cfg


def imprimir_config(cfg: dict | None = None) -> None:
    """Imprime los parámetros en uso (va al inicio de cada notebook)."""
    cfg = CONFIG if cfg is None else cfg
    ancho = max(len(k) for k in cfg)
    for clave, valor in cfg.items():
        print(f"{clave:<{ancho}} : {valor}")


# ---------------------------------------------------------------------------
# Rutas: la misma estructura en Drive y en local
#   <DATA>/datos/       archivos del registro (.vhdr, .eeg, .vmrk, .tsv, .json)
#   <DATA>/resultados/  CSV
#   <DATA>/figuras/     PNG (fig01_senal_original.png, fig02_...)
# ---------------------------------------------------------------------------

_DATA_COLAB = "/content/drive/MyDrive/alfa-eeg-data"
_DATA_LOCAL = Path(__file__).resolve().parents[1] / "data"


def ruta_data() -> Path:
    """Carpeta raíz de datos.

    Prioridad: variable de entorno ``ALFA_DATA`` > Drive en Colab > ``data/``
    dentro del repo (ignorada por git).
    """
    if os.environ.get("ALFA_DATA"):
        return Path(os.environ["ALFA_DATA"])
    if "google.colab" in sys.modules:
        return Path(_DATA_COLAB)
    return _DATA_LOCAL


def rutas() -> dict:
    """Rutas usadas por todos los notebooks; crea resultados/ y figuras/."""
    base = ruta_data()
    r = {
        "data": base,
        "datos": base / "datos",
        "resultados": base / "resultados",
        "figuras": base / "figuras",
    }
    r["vhdr"] = r["datos"] / ARCHIVO_VHDR
    r["resultados"].mkdir(parents=True, exist_ok=True)
    r["figuras"].mkdir(parents=True, exist_ok=True)
    return r


def guardar_figura(fig, nombre: str, dpi: int = 200) -> Path:
    """Guarda una figura como PNG en <DATA>/figuras/ (p. ej. 'fig01_senal_original')."""
    destino = rutas()["figuras"] / (Path(nombre).stem + ".png")
    fig.savefig(destino, dpi=dpi, bbox_inches="tight")
    print(f"Figura guardada: {destino}")
    return destino


# ---------------------------------------------------------------------------
# Carril A · Inspección y calidad (Moisés)
# ---------------------------------------------------------------------------

def cargar_raw(ruta_vhdr=None, preload: bool = True) -> mne.io.Raw:
    """Carga el registro BrainVision con VEOG como 'eog' y unidades verificadas.

    Pendiente (A, rama a/inspeccion).
    """
    raise NotImplementedError("A: cargar_raw() pendiente")


def tabla_bloques(raw: mne.io.Raw) -> pd.DataFrame:
    """Una fila por bloque: bloque, condicion ('EO'/'EC'), inicio_s, fin_s.

    Pendiente (A, rama a/inspeccion).
    """
    raise NotImplementedError("A: tabla_bloques() pendiente")


def marcar_artefactos(raw: mne.io.Raw, cfg: dict | None = None) -> mne.io.Raw:
    """Agrega anotaciones 'BAD_...' sobre los segmentos con artefactos.

    Pendiente (A).
    """
    raise NotImplementedError("A: marcar_artefactos() pendiente")


# ---------------------------------------------------------------------------
# Carril B · Acondicionamiento (Adrián)
# ---------------------------------------------------------------------------

def acondicionar(raw: mne.io.Raw, cfg: dict | None = None) -> mne.io.Raw:
    """Devuelve una copia filtrada y remuestreada según cfg['filtro'] y cfg['sfreq_analisis'].

    Pendiente (B). No modifica ``raw``.
    """
    raise NotImplementedError("B: acondicionar() pendiente")


# ---------------------------------------------------------------------------
# Carril C · Caracterización del alfa (Diego)
# ---------------------------------------------------------------------------

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.integrate import trapezoid
from scipy.signal import welch

def potencia_alfa(raw: mne.io.Raw, bloques: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Potencia alfa por ventana deslizante y canal (PSD de Welch, un segmento por ventana).

    Ventanas de ``cfg['ventana_s']`` con traslape ``cfg['solapamiento']`` sobre todo el
    registro (también transiciones, para la curva temporal).

    Columnas:
        t_s        centro de la ventana (s)
        bloque     bloque al que pertenece el centro (NA fuera de bloques)
        condicion  'EO' / 'EC' (None fuera de bloques)
        canal      nombre del canal (cfg['canales'])
        alfa_abs   potencia en cfg['banda_alfa'] (µV², área bajo la PSD)
        alfa_rel   alfa_abs / potencia en la banda del filtro (l_freq–h_freq)
        dentro     True si la ventana cae completa en [inicio + recorte, fin] de un bloque

    Si ``cfg['excluir_artefactos']``, se descartan las ventanas que se traslapan con
    anotaciones 'BAD_...'.
    """
    cfg = CONFIG if cfg is None else cfg
    sf = raw.info["sfreq"]
    canales = cfg["canales"]
    n_win = int(round(cfg["ventana_s"] * sf))
    paso = max(1, int(round(n_win * (1 - cfg["solapamiento"]))))

    x = raw.get_data(picks=canales) * 1e6                          # V → µV
    seg = sliding_window_view(x, n_win, axis=1)[:, ::paso, :]       # (canales, ventanas, muestras)
    t_ini = np.arange(seg.shape[1]) * paso / sf
    t_fin = t_ini + n_win / sf
    t_c = (t_ini + t_fin) / 2

    f, pxx = welch(seg, fs=sf, nperseg=n_win, axis=-1)              # µV²/Hz
    a, b = cfg["banda_alfa"]
    lo = cfg["filtro"].get("l_freq") or f[1]
    hi = cfg["filtro"].get("h_freq") or sf / 2
    m_a, m_t = (f >= a) & (f <= b), (f >= lo) & (f <= hi)
    alfa_abs = trapezoid(pxx[..., m_a], f[m_a], axis=-1)
    alfa_rel = alfa_abs / trapezoid(pxx[..., m_t], f[m_t], axis=-1)

    bloque = np.full(len(t_c), np.nan)
    condicion = np.full(len(t_c), None, dtype=object)
    dentro = np.zeros(len(t_c), bool)
    for _, bl in bloques.iterrows():
        en = (t_c >= bl["inicio_s"]) & (t_c < bl["fin_s"])
        bloque[en], condicion[en] = bl["bloque"], bl["condicion"]
        dentro |= (t_ini >= bl["inicio_s"] + cfg["recorte_transicion_s"]) & (t_fin <= bl["fin_s"])

    valida = np.ones(len(t_c), bool)
    if cfg["excluir_artefactos"]:
        for on, du, de in zip(raw.annotations.onset, raw.annotations.duration,
                              raw.annotations.description):
            if de.upper().startswith("BAD"):
                valida &= ~((t_ini < on + du) & (t_fin > on))

    nc = len(canales)
    df = pd.DataFrame({
        "t_s": np.tile(t_c, nc),
        "bloque": np.tile(bloque, nc),
        "condicion": np.tile(condicion, nc),
        "canal": np.repeat(canales, len(t_c)),
        "alfa_abs": alfa_abs.ravel(),
        "alfa_rel": alfa_rel.ravel(),
        "dentro": np.tile(dentro, nc),
    })
    df = df[np.tile(valida, nc)].reset_index(drop=True)
    df["bloque"] = df["bloque"].astype("Int64")
    return df


def resumen_por_bloque(df: pd.DataFrame) -> pd.DataFrame:
    """Una fila por bloque a partir de ``potencia_alfa()``.

    Usa solo ventanas ``dentro``. Promedia entre canales en cada ventana y luego
    resume las ventanas del bloque.

    Columnas: bloque, condicion, n_ventanas, alfa_abs_media, alfa_abs_de,
    alfa_rel_media, alfa_rel_de, par (EC con el EO siguiente: 1-2, 3-4, …),
    razon_ec_eo_abs, razon_ec_eo_rel (razón de medias dentro del par).
    """
    d = df[df["dentro"]] if "dentro" in df else df.dropna(subset=["bloque"])
    por_ventana = (d.groupby(["bloque", "condicion", "t_s"], as_index=False)
                     [["alfa_abs", "alfa_rel"]].mean())
    res = (por_ventana.groupby(["bloque", "condicion"])
           .agg(n_ventanas=("t_s", "size"),
                alfa_abs_media=("alfa_abs", "mean"), alfa_abs_de=("alfa_abs", "std"),
                alfa_rel_media=("alfa_rel", "mean"), alfa_rel_de=("alfa_rel", "std"))
           .reset_index())
    res["par"] = (res["bloque"] + 1) // 2
    for m in ["abs", "rel"]:
        piv = res.pivot(index="par", columns="condicion", values=f"alfa_{m}_media")
        res[f"razon_ec_eo_{m}"] = res["par"].map(piv["EC"] / piv["EO"])
    return res

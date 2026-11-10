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

def potencia_alfa(raw: mne.io.Raw, bloques: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Potencia alfa por ventana y canal (PSD de Welch).

    Columnas: t_s, bloque, condicion, canal, alfa_abs, alfa_rel.
    Pendiente (C).
    """
    raise NotImplementedError("C: potencia_alfa() pendiente")


def resumen_por_bloque(df: pd.DataFrame) -> pd.DataFrame:
    """Media, desviación y razón EC/EO por bloque a partir de potencia_alfa().

    Pendiente (C).
    """
    raise NotImplementedError("C: resumen_por_bloque() pendiente")

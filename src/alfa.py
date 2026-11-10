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

# Marcadores del registro. MNE lee el .vmrk ("Stimulus/S  3"); el events.tsv de
# BIDS usa otros nombres para los mismos eventos ("Stimulus/S200"):
#   .vmrk   events.tsv                   significado
#   S  1    Comment/no USB Connection    inicio del archivo (muestra 1)
#   S  2    Stimulus/S  1                cambio de bloque (~10 ms antes del bloque)
#   S  3    Stimulus/S200                ojos abiertos (EO), se repite cada 2 s
#   S  4    Stimulus/S210                ojos cerrados (EC), se repite cada 2 s
# El .vmrk cuenta muestras desde 1 y el .tsv desde 0 (una muestra de diferencia).
MARCADORES_CONDICION = {
    "Stimulus/S  3": "EO", "Stimulus/S200": "EO",
    "Stimulus/S  4": "EC", "Stimulus/S210": "EC",
}
MARCADOR_CAMBIO = "Stimulus/S  2"

_UNIDADES_MICRO = {"µV", "μV", "uV"}  # µ (micro) y μ (letra griega mu)


def cargar_raw(ruta_vhdr=None, preload: bool = True) -> mne.io.Raw:
    """Carga el registro BrainVision listo para usar.

    - VEOG queda como canal 'eog' (MNE lo carga como EEG; channels.tsv dice EOG).
    - Verifica que el archivo declare µV en todos los canales. MNE convierte a
      voltios: ``raw.get_data()`` está en V; multiplicar por 1e6 para µV.
    - Agrega posiciones 10-05 estándar (para topografías y para ubicar canales).
    - ``info['line_freq'] = 50`` (PowerLineFrequency del eeg.json).

    ``preload=True`` ocupa ~1.2 GB (62 canales × 2.555 M muestras en float64).
    Con ``preload=False`` se lee del disco bajo demanda; ``raw.pick(...)`` y
    luego ``raw.load_data()`` reduce la memoria si solo se usan pocos canales.
    """
    ruta_vhdr = Path(ruta_vhdr) if ruta_vhdr is not None else rutas()["vhdr"]
    if not ruta_vhdr.exists():
        raise FileNotFoundError(
            f"No se encontró {ruta_vhdr}. ¿Está montado Drive y existe alfa-eeg-data/datos/?"
        )
    raw = mne.io.read_raw_brainvision(ruta_vhdr, preload=preload, verbose="ERROR")

    unidades = getattr(raw, "_orig_units", {}) or {}
    raras = {ch: u for ch, u in unidades.items() if u not in _UNIDADES_MICRO}
    if raras:
        raise ValueError(f"Canales con unidades distintas de µV: {raras}")

    if "VEOG" in raw.ch_names:
        raw.set_channel_types({"VEOG": "eog"})
    raw.set_montage("standard_1005", on_missing="warn", verbose="ERROR")
    raw.info["line_freq"] = 50.0
    return raw


def tabla_bloques(raw: mne.io.Raw) -> pd.DataFrame:
    """Una fila por bloque de condición, a partir de los marcadores repetidos.

    Un bloque es una racha de marcadores de condición iguales y consecutivos
    (los ``S  2`` y ``S  1`` no cortan la racha).

    Columnas:
        bloque        1..16, en orden temporal
        condicion     'EO' (ojos abiertos) o 'EC' (ojos cerrados)
        inicio_s      primer marcador de la racha
        fin_s         último marcador + separación típica entre marcadores (2 s)
        duracion_s    fin_s - inicio_s
        n_marcadores  marcadores de condición en el bloque (30 en este registro)
        cambio_s      ``S  2`` más cercano antes del bloque (NaN si no hay)

    Los huecos entre ``fin_s`` de un bloque e ``inicio_s`` del siguiente
    (1.6–2.6 s en este registro) y los ~25 s finales después del último
    bloque no pertenecen a ninguna condición.
    """
    ann = raw.annotations
    eventos = pd.DataFrame({"t": ann.onset, "desc": ann.description})
    cambios = eventos.loc[eventos["desc"] == MARCADOR_CAMBIO, "t"].to_numpy()

    cond = eventos[eventos["desc"].isin(MARCADORES_CONDICION)].copy()
    if cond.empty:
        raise ValueError("No hay marcadores de condición (S 3/S 4 o S200/S210) en raw.annotations")
    cond["condicion"] = cond["desc"].map(MARCADORES_CONDICION)
    cond = cond.sort_values("t").reset_index(drop=True)
    cond["racha"] = (cond["condicion"] != cond["condicion"].shift()).cumsum()

    paso = cond.groupby("racha")["t"].diff().median()  # separación típica (2 s)

    filas = []
    for i, (_, g) in enumerate(cond.groupby("racha"), start=1):
        inicio = float(g["t"].iloc[0])
        fin = min(float(g["t"].iloc[-1] + paso), raw.times[-1])
        previos = cambios[cambios <= inicio]
        filas.append({
            "bloque": i,
            "condicion": g["condicion"].iloc[0],
            "inicio_s": inicio,
            "fin_s": fin,
            "duracion_s": fin - inicio,
            "n_marcadores": len(g),
            "cambio_s": float(previos[-1]) if len(previos) else float("nan"),
        })
    return pd.DataFrame(filas)


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

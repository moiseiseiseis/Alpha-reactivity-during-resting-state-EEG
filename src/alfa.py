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
import numpy as np
import pandas as pd
from scipy.signal import freqz

# ---------------------------------------------------------------------------
# Parámetros del análisis
# ---------------------------------------------------------------------------

CONFIG = {
    # A (notebook 01, §4): occipitales y parieto-occipitales, simétricos. Sin PO9/PO10
    # (inferiores, sobre musculatura del cuello) ni la fila P (menos reactiva).
    "canales": ["O1", "Oz", "O2", "PO7", "PO3", "POz", "PO4", "PO8"],
    "banda_alfa": (8.0, 13.0),
    # B (notebook 02): FIR de fase cero, pasa-banda 1-40 Hz. El notch de 50 Hz es redundante
    # (el pasa-bajas ya atenua >80 dB en 50 Hz); poner 50.0 solo si la celda de decision
    # del notebook 02 (seccion 2) dice NECESARIO.
    "filtro": {"l_freq": 1.0, "h_freq": 40.0, "notch": None},
    "sfreq_analisis": 250,
    "ventana_s": 2.0,
    "solapamiento": 0.5,          # fracción de la ventana (0.5 = 50 %)
    "recorte_transicion_s": 5.0,  # s descartados después de cada cambio de condición
    "excluir_artefactos": True,   # usa anotaciones BAD_ con reject_by_annotation
    # A (notebook 01, §5): criterio de marcar_artefactos().
    "artefactos": {
        "umbral_pp_uv": 100.0,     # pico a pico por segmento en algún canal de análisis (1-40 Hz)
        "segmento_s": 1.0,         # resolución del marcado por amplitud
        "plano_pp_uv": 1.0,        # por debajo: canal plano o desconectado
        "parpadeos": True,         # marca BAD_parpadeo alrededor de cada evento del VEOG
        "margen_parpadeo_s": 0.4,  # s antes y después del pico del parpadeo
    },
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


def marcar_artefactos(raw: mne.io.Raw, cfg: dict | None = None, csv_manual="auto") -> mne.io.Raw:
    """Agrega anotaciones 'BAD_...' sobre los segmentos con artefactos y devuelve ``raw``.

    Modifica las anotaciones de ``raw`` (no la señal). Si ya tenía anotaciones
    'BAD_', se quitan antes, así la función se puede volver a correr con otro cfg.
    Criterios (parámetros en ``cfg['artefactos']``):

    - ``BAD_amplitud``: en segmentos de ``segmento_s``, algún canal de
      ``cfg['canales']`` supera ``umbral_pp_uv`` de pico a pico. Se mide sobre una
      copia filtrada con ``cfg['filtro']`` (l_freq-h_freq) para que la deriva lenta
      no dispare el umbral.
    - ``BAD_plano``: algún canal de análisis tiene menos de ``plano_pp_uv`` en el segmento.
    - ``BAD_parpadeo``: ±``margen_parpadeo_s`` alrededor de cada evento que
      ``mne.preprocessing.find_eog_events`` detecta en VEOG (si ``parpadeos``).
    - ``BAD_manual``: filas de ``csv_manual`` (columnas ``inicio_s, fin_s, motivo``).
      Con ``"auto"`` se usa ``resultados/segmentos_malos_manual.csv`` si existe.

    ``potencia_alfa`` descarta las ventanas que tocan estas anotaciones cuando
    ``cfg['excluir_artefactos']`` es True.
    """
    cfg = CONFIG if cfg is None else cfg
    pa = cfg["artefactos"]
    sf = raw.info["sfreq"]
    offset = raw.first_time if raw.annotations.orig_time is not None else 0.0

    # Copia ligera: solo canales de análisis (+ VEOG), filtrada.
    picks = [c for c in cfg["canales"] if c in raw.ch_names]
    if len(picks) != len(cfg["canales"]):
        faltan = sorted(set(cfg["canales"]) - set(picks))
        raise ValueError(f"Canales de cfg['canales'] que no están en raw: {faltan}")
    tiene_veog = "VEOG" in raw.ch_names
    tmp = raw.copy().pick(picks + (["VEOG"] if tiene_veog else [])).load_data()
    tmp.filter(cfg["filtro"].get("l_freq"), cfg["filtro"].get("h_freq"),
               picks="all", verbose=False)

    onsets, durs, descs = [], [], []

    # Amplitud y canal plano, por segmentos.
    x = tmp.get_data(picks=picks) * 1e6
    n = int(round(pa["segmento_s"] * sf))
    nseg = x.shape[1] // n
    pp = np.ptp(x[:, : nseg * n].reshape(len(picks), nseg, n), axis=2)  # (canales, segmentos)
    for desc, malo in [("BAD_amplitud", (pp > pa["umbral_pp_uv"]).any(axis=0)),
                       ("BAD_plano", (pp < pa["plano_pp_uv"]).any(axis=0))]:
        for ini, fin in _rachas(malo):
            onsets.append(ini * n / sf)
            durs.append((fin - ini) * n / sf)
            descs.append(desc)

    # Parpadeos (VEOG).
    if pa["parpadeos"] and tiene_veog:
        ev = mne.preprocessing.find_eog_events(tmp, ch_name="VEOG", verbose=False)
        for muestra in ev[:, 0]:
            t = (muestra - tmp.first_samp) / sf
            onsets.append(max(0.0, t - pa["margen_parpadeo_s"]))
            durs.append(2 * pa["margen_parpadeo_s"])
            descs.append("BAD_parpadeo")

    # Segmentos anotados a mano en el recorrido visual.
    if isinstance(csv_manual, str) and csv_manual == "auto":
        csv_manual = rutas()["resultados"] / "segmentos_malos_manual.csv"
    if csv_manual is not None and Path(csv_manual).exists():
        man = pd.read_csv(csv_manual)
        for fila in man.itertuples():
            onsets.append(float(fila.inicio_s))
            durs.append(float(fila.fin_s) - float(fila.inicio_s))
            descs.append("BAD_manual")

    viejas = raw.annotations
    conservar = [i for i, d in enumerate(viejas.description) if not d.upper().startswith("BAD")]
    nuevas = mne.Annotations(
        onset=list(viejas.onset[conservar]) + [o + offset for o in onsets],
        duration=list(viejas.duration[conservar]) + durs,
        description=list(viejas.description[conservar]) + descs,
        orig_time=viejas.orig_time,
    )
    raw.set_annotations(nuevas)
    return raw


def _rachas(mascara) -> list[tuple[int, int]]:
    """Índices [inicio, fin) de cada racha de True consecutivos."""
    m = np.concatenate([[False], np.asarray(mascara, bool), [False]])
    cambios = np.flatnonzero(m[1:] != m[:-1])
    return list(zip(cambios[::2], cambios[1::2]))


def resumen_artefactos(raw: mne.io.Raw, bloques: pd.DataFrame) -> pd.DataFrame:
    """Segundos marcados como 'BAD_' por tipo y condición.

    Para la tabla de calidad del notebook 01 y del informe. Las anotaciones de un
    mismo tipo que se traslapan se cuentan una sola vez; la fila 'cualquiera' une
    todos los tipos.
    """
    sf = raw.info["sfreq"]
    t = raw.times
    offset = raw.first_time if raw.annotations.orig_time is not None else 0.0
    cond = np.full(len(t), "fuera", dtype=object)
    for b in bloques.itertuples():
        cond[(t >= b.inicio_s) & (t < b.fin_s)] = b.condicion
    ann = raw.annotations
    tipos = sorted({d for d in ann.description if d.upper().startswith("BAD")})
    filas = []
    for tipo in tipos + ["cualquiera"]:
        malo = np.zeros(len(t), bool)
        for on, du, de in zip(ann.onset, ann.duration, ann.description):
            if de == tipo or (tipo == "cualquiera" and de.upper().startswith("BAD")):
                malo |= (t >= on - offset) & (t < on - offset + du)
        fila = {"tipo": tipo}
        for c in ["EC", "EO", "fuera"]:
            fila[f"{c}_s"] = round(malo[cond == c].sum() / sf, 1)
        filas.append(fila)
    res = pd.DataFrame(filas)
    for c in ["EC", "EO"]:
        total = (cond == c).sum() / sf
        res[f"{c}_pct"] = (100 * res[f"{c}_s"] / total).round(1)
    return res


# ---------------------------------------------------------------------------
# Carril B · Acondicionamiento (Adrián)
# ---------------------------------------------------------------------------

# Diseño FIR común a todos los filtros del carril B: fase cero (sin distorsión de
# fase ni retardo, importante para comparar tiempos de transición EO/EC),
# ventana Hamming y diseño firwin.
_FIR = dict(method="fir", phase="zero", fir_window="hamming", fir_design="firwin")


def _validar_filtro(cfg: dict, sfreq_in: float) -> None:
    """Revisa que cfg['filtro'] y cfg['sfreq_analisis'] sean coherentes con el registro."""
    f = cfg["filtro"]
    l_freq, h_freq, notch = f.get("l_freq"), f.get("h_freq"), f.get("notch")
    sfreq_out = cfg["sfreq_analisis"]
    if l_freq is None and h_freq is None:
        raise ValueError("cfg['filtro'] necesita l_freq y/o h_freq")
    if l_freq is not None and h_freq is not None and l_freq >= h_freq:
        raise ValueError(f"l_freq ({l_freq}) debe ser menor que h_freq ({h_freq})")
    if sfreq_out > sfreq_in:
        raise ValueError(f"sfreq_analisis ({sfreq_out}) > sfreq del registro ({sfreq_in}): no se sobremuestrea")
    if h_freq is not None and h_freq >= sfreq_out / 2:
        raise ValueError(f"h_freq ({h_freq}) debe ser < Nyquist de salida ({sfreq_out / 2})")
    if notch is not None and notch >= sfreq_in / 2:
        raise ValueError(f"notch ({notch}) debe ser < Nyquist del registro ({sfreq_in / 2})")


def acondicionar(raw: mne.io.Raw, cfg: dict | None = None) -> mne.io.Raw:
    """Devuelve una copia filtrada y remuestreada según cfg['filtro'] y cfg['sfreq_analisis'].

    Cadena (a la frecuencia de muestreo original, antes de remuestrear, para que el
    pasa-bajas actúe como antialiasing):

    1. Pasa-banda FIR de fase cero (``l_freq``-``h_freq``) en canales EEG y EOG.
    2. Notch FIR de fase cero en ``cfg['filtro']['notch']`` si no es None.
    3. Remuestreo a ``cfg['sfreq_analisis']`` (si difiere de la frecuencia original).

    No modifica ``raw``. Conserva las anotaciones (bloques, ``BAD_``) y todos los canales.
    """
    cfg = CONFIG if cfg is None else cfg
    f = cfg["filtro"]
    sfreq_in = float(raw.info["sfreq"])
    sfreq_out = cfg["sfreq_analisis"]
    _validar_filtro(cfg, sfreq_in)

    out = raw.copy().load_data()
    picks = mne.pick_types(out.info, eeg=True, eog=True, exclude=[])  # EEG + VEOG

    out.filter(l_freq=f.get("l_freq"), h_freq=f.get("h_freq"), picks=picks, verbose=False, **_FIR)
    if f.get("notch") is not None:
        out.notch_filter(freqs=f["notch"], picks=picks, verbose=False, **_FIR)
    if float(sfreq_out) != sfreq_in:
        out.resample(sfreq_out, verbose=False)
    return out


def respuesta_filtro(sfreq: float, cfg: dict | None = None, n_puntos: int = 2 ** 16):
    """Respuesta en frecuencia del pasa-banda de ``acondicionar`` (sin el notch).

    Devuelve ``(freqs_hz, magnitud_db, h)``, con ``h`` el kernel FIR. ``sfreq`` es la
    frecuencia a la que se filtra (la del registro original). Al ser de fase cero, el
    retardo de grupo es 0 s y la magnitud es la que se aplica a la señal.
    """
    cfg = CONFIG if cfg is None else cfg
    f = cfg["filtro"]
    h = mne.filter.create_filter(None, sfreq, f.get("l_freq"), f.get("h_freq"), verbose=False, **_FIR)
    freqs, H = freqz(h, worN=n_puntos, fs=sfreq)
    mag_db = 20.0 * np.log10(np.maximum(np.abs(H), 1e-12))
    return freqs, mag_db, h


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

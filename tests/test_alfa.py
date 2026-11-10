"""Pruebas rápidas de alfa.py con señales sintéticas (no necesitan el registro).

Correr desde la raíz del repo:  python -m pytest -q
"""
import sys
from pathlib import Path

import mne
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import alfa  # noqa: E402


def raw_sintetico(secuencia, por_bloque=30, paso=2.0, hueco=2.0, sfreq=100.0):
    """Raw de 1 canal con marcadores como los de LEMON.

    ``secuencia`` es una lista de 'EO'/'EC'. Antes de cada bloque va un S  2.
    """
    codigo = {"EO": "Stimulus/S  3", "EC": "Stimulus/S  4"}
    onsets, desc = [0.0], ["Stimulus/S  1"]
    t = 1.0
    for cond in secuencia:
        onsets.append(t - 0.01)
        desc.append(alfa.MARCADOR_CAMBIO)
        for k in range(por_bloque):
            onsets.append(t + k * paso)
            desc.append(codigo[cond])
        t += por_bloque * paso + hueco
    dur = t + 10
    info = mne.create_info(["Oz"], sfreq, "eeg")
    raw = mne.io.RawArray(np.zeros((1, int(dur * sfreq))), info, verbose="ERROR")
    raw.set_annotations(mne.Annotations(onsets, [0.0004] * len(onsets), desc))
    return raw


def test_bloques_alternados():
    seq = ["EC", "EO"] * 8
    b = alfa.tabla_bloques(raw_sintetico(seq))
    assert len(b) == 16
    assert list(b["condicion"]) == seq
    assert (b["n_marcadores"] == 30).all()
    assert np.allclose(b["duracion_s"], 60.0)
    assert np.allclose(b["inicio_s"] - b["cambio_s"], 0.01)


def test_marcadores_bids_tambien_funcionan():
    raw = raw_sintetico(["EO", "EC"])
    raw.annotations.description[:] = [
        d.replace("S  3", "S200").replace("S  4", "S210") for d in raw.annotations.description
    ]
    b = alfa.tabla_bloques(raw)
    assert list(b["condicion"]) == ["EO", "EC"]


def test_dos_s2_seguidos_no_parten_el_bloque():
    raw = raw_sintetico(["EC", "EO"])
    raw.annotations.append(0.5, 0.0004, alfa.MARCADOR_CAMBIO)  # como el S 2 extra a los 3.99 s
    b = alfa.tabla_bloques(raw)
    assert len(b) == 2
    assert b.loc[0, "cambio_s"] == pytest.approx(0.99)


def test_sin_marcadores_de_condicion():
    raw = raw_sintetico(["EC"])
    raw.set_annotations(None)
    with pytest.raises(ValueError):
        alfa.tabla_bloques(raw)


def test_variante_no_modifica_config():
    v = alfa.variante(ventana_s=10.0, filtro={"notch": None})
    assert v["ventana_s"] == 10.0 and v["filtro"]["notch"] is None
    assert alfa.CONFIG["ventana_s"] == 2.0 and alfa.CONFIG["filtro"]["notch"] == 50.0
    with pytest.raises(KeyError):
        alfa.variante(no_existe=1)


def test_cargar_raw_ruta_inexistente(tmp_path):
    with pytest.raises(FileNotFoundError):
        alfa.cargar_raw(tmp_path / "no_existe.vhdr")

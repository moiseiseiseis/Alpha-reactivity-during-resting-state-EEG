"""Pruebas rápidas de alfa.py con señales sintéticas (no necesitan el registro).

Correr desde la raíz del repo:  python -m pytest -q
"""
import copy
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
    antes = copy.deepcopy(alfa.CONFIG)
    v = alfa.variante(ventana_s=99.0, filtro={"notch": 60.0})
    assert v["ventana_s"] == 99.0 and v["filtro"]["notch"] == 60.0
    assert v["filtro"]["l_freq"] == antes["filtro"]["l_freq"]  # el resto del dict anidado se conserva
    assert alfa.CONFIG == antes
    with pytest.raises(KeyError):
        alfa.variante(no_existe=1)


def test_cargar_raw_ruta_inexistente(tmp_path):
    with pytest.raises(FileNotFoundError):
        alfa.cargar_raw(tmp_path / "no_existe.vhdr")


def raw_para_artefactos(sfreq=250.0, dur=60.0, seed=0):
    """2 canales con ruido de 10 µV (sin artefactos) y marcadores de 1 bloque EC."""
    rng = np.random.default_rng(seed)
    info = mne.create_info(["O1", "O2"], sfreq, "eeg")
    x = rng.normal(0, 10e-6, (2, int(dur * sfreq)))
    raw = mne.io.RawArray(x, info, verbose="ERROR")
    raw.set_annotations(mne.Annotations([1.0], [0.0004], ["Stimulus/S  4"]))
    return raw


def cfg_artefactos():
    return alfa.variante(canales=["O1", "O2"], artefactos={"parpadeos": False})


def test_marcar_artefactos_detecta_un_salto_de_amplitud():
    raw = raw_para_artefactos()
    sf = raw.info["sfreq"]
    raw._data[0, int(30.2 * sf):int(30.4 * sf)] += 300e-6  # escalón de 300 µV en O1
    alfa.marcar_artefactos(raw, cfg_artefactos(), csv_manual=None)
    bad = [(a["onset"], a["duration"]) for a in raw.annotations if a["description"] == "BAD_amplitud"]
    assert bad, "no marcó el salto"
    assert all(29.0 <= on and on + du <= 32.0 for on, du in bad)
    assert "Stimulus/S  4" in raw.annotations.description  # conserva los marcadores


def test_marcar_artefactos_senal_limpia_y_reejecucion():
    raw = raw_para_artefactos()
    cfg = cfg_artefactos()
    alfa.marcar_artefactos(raw, cfg, csv_manual=None)
    assert not any(d.startswith("BAD") for d in raw.annotations.description)
    raw._data[1, :] = 0.0  # O2 plano
    alfa.marcar_artefactos(raw, cfg, csv_manual=None)
    alfa.marcar_artefactos(raw, cfg, csv_manual=None)  # correr dos veces no duplica
    planos = [d for d in raw.annotations.description if d == "BAD_plano"]
    assert len(planos) == 1


def test_marcar_artefactos_csv_manual(tmp_path):
    raw = raw_para_artefactos()
    csv = tmp_path / "malos.csv"
    csv.write_text("inicio_s,fin_s,canales,motivo\n10.0,12.5,O1,movimiento\n", encoding="utf-8")
    alfa.marcar_artefactos(raw, cfg_artefactos(), csv_manual=csv)
    man = [(a["onset"], a["duration"]) for a in raw.annotations if a["description"] == "BAD_manual"]
    assert man == [(pytest.approx(10.0), pytest.approx(2.5))]


def test_marcar_artefactos_canal_inexistente():
    with pytest.raises(ValueError):
        alfa.marcar_artefactos(raw_para_artefactos(), alfa.variante(canales=["Pz"]), csv_manual=None)

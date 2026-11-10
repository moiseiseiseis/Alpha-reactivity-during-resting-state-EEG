# Reactividad alfa en EEG en reposo (ojos abiertos vs ojos cerrados)

¿Cómo cambia la actividad alfa del EEG durante la alternancia entre ojos abiertos y ojos cerrados, y qué tan consistente es este cambio a lo largo del registro? Proyecto final de *Acondicionamiento de Bioseñales y Bioimágenes*. Usa el registro `sub-010002` del dataset LEMON (Babayan et al., 2019).

## Cómo correrlo

1. En Drive, haz *Agregar acceso directo a Mi unidad* de la carpeta compartida `alfa-eeg-data/`, con `datos/`, `resultados/` y `figuras/`.
2. Abre el notebook desde GitHub en Colab: `colab.research.google.com/github/moiseiseiseis/Alpha-reactivity-during-resting-state-EEG/blob/main/notebooks/<notebook>.ipynb`.
3. Corre la primera celda: monta Drive, clona el repo e instala `mne`.
4. En local: `pip install -r requirements.txt`, pon los archivos del registro en `data/datos/` (o define `ALFA_DATA`) y abre `notebooks/` con Jupyter.
5. Pruebas de `alfa.py` (no necesitan el registro): `python -m pytest -q`.
6. Todos los parámetros están en `CONFIG` de [`src/alfa.py`](src/alfa.py). Para probar variantes usa `alfa.variante(ventana_s=10.0)`.

## Estructura

| Ruta | Qué es | Dueño |
|---|---|---|
| `src/alfa.py` | `CONFIG` y funciones compartidas | todos, solo por PR |
| `notebooks/01_inspeccion.ipynb` | Inspección, calidad, canales, artefactos | A · Moisés |
| `notebooks/02_acondicionamiento.ipynb` | Filtros y remuestreo | B · Adrián |
| `notebooks/03_alfa_temporal.ipynb` | PSD, potencia alfa, bloques y transiciones | C · Diego |
| `notebooks/04_robustez.ipynb` | Variantes de `CONFIG` y fundamento fisiológico | D · Ascencio |
| `notebooks/proyecto_final.ipynb` | Cuaderno integrado que se entrega | Moisés, fase 3 |
| `docs/plan-de-trabajo.md` | Roles, flujo, cronograma y checklist | — |

Los datos, CSV y PNG nunca se suben al repo (el `.eeg` pesa 633 MB). Viven en Drive.

## Flujo de trabajo

- Cada quien trabaja en su rama (`a/inspeccion`, `b/filtros`, `c/alfa`, `d/robustez`) y abre un PR hacia `main`.
- Moisés revisa todos los PR. Los PR del carril A los revisa Ascencio.
- Antes de guardar un notebook: *Editar → Borrar todas las salidas*.
- Entrega: 29 nov 2026. Al final, `main` se marca con el tag `v1.0-entrega`.

## Referencia del dataset

Babayan, A., Erbey, M., Kumral, D. et al. (2019). A mind-brain-body dataset of MRI, EEG, cognition, emotion, and peripheral physiology in young and old adults. *Scientific Data*, 6, 180308. https://doi.org/10.1038/sdata.2018.308

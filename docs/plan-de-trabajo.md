# Proyecto final: Reactividad alfa en EEG

Nov 10, 2026 · @Moisés

## Resumen

Cada integrante es dueño de una etapa técnica, de su propio notebook y de su sección del informe. Todo el código comparte una sola librería (`src/alfa.py`) y un diccionario `CONFIG`, así la Etapa 5 se reduce a re-correr el mismo pipeline cambiando parámetros.

Hay que entregar tres cosas: el informe escrito, al menos 5 figuras obligatorias y un Colab que reproduzca todo desde los archivos del registro. El 40 % de la calificación está en caracterización del alfa (20 %) y evaluación metodológica (20 %), así que ahí se concentra el trabajo de las semanas 2 y 3.

Entrega: Nov 29, 2026.

## Roles

Cuatro carriles, uno por persona; cada uno entrega código, figuras y texto de su parte.

| Rol | Integrante | Etapas | Entregable técnico | Secciones del informe | Peso en rúbrica |
| --- | --- | --- | --- | --- | --- |
| A · Inspección y calidad | Moisés | 1 + selección de canales | `cargar_raw()`, `tabla_bloques()`, segmentos `BAD_` marcados, revisión del VEOG, lista de canales justificada | Descripción de los datos; inspección y calidad; selección de canales y segmentos | 15 % |
| B · Acondicionamiento | Adrián | 2 | `acondicionar()`: filtros justificados (pasa-banda, notch 50 Hz si hace falta, remuestreo), comparación original vs acondicionada en tiempo y en PSD | Estrategia de acondicionamiento y justificación | 15 % |
| C · Caracterización del alfa | Diego | 3 y 4 | `potencia_alfa()`: PSD de Welch por condición, potencia absoluta y relativa 8–13 Hz por ventana, curva en el tiempo, comparación entre bloques y alrededor de transiciones | Caracterización; comparación entre condiciones; evolución temporal y consistencia | 20 % |
| D · Robustez y discusión | Ascencio | 5 y 6 | Al menos 2 variantes de `CONFIG` (p. ej. ventana 2/5/10 s y con/sin exclusión de artefactos) y tabla de robustez | Introducción y fundamento fisiológico; evaluación metodológica; discusión, limitaciones y conclusiones | 20 % + 15 % + 10 % |

Además de su carril:

- **Responsable del repo** (Moisés): crea el repo y la carpeta de Drive, revisa los PR que tocan `src/alfa.py` y arma `proyecto_final.ipynb` en la fase 3.
- **Revisión cruzada:** Moisés revisa todos los PR antes de unirlos a main. Los PR de su propio carril los revisa Ascencio, para que nadie apruebe su propio código.
- Moisés y Ascencio deben hablar desde la fase 1: la selección de canales de A necesita el fundamento fisiológico que D está escribiendo (distribución occipito-parietal del alfa).

## Flujo GitHub + Colab

El código vive en GitHub; los datos, resultados y figuras viven en una carpeta compartida de Drive. El `.eeg` pesa cientos de MB y GitHub rechaza archivos de más de 100 MB, así que nunca se sube al repo.

**Estructura del repo**

Repo: [moiseiseiseis/Alpha-reactivity-during-resting-state-EEG](https://github.com/moiseiseiseis/Alpha-reactivity-during-resting-state-EEG) (público).

```
Alpha-reactivity-during-resting-state-EEG/
├── README.md               # cómo correr todo, en 5 líneas
├── requirements.txt        # mne, ipympl, scipy, pandas
├── .gitignore              # *.eeg, *.vhdr, *.vmrk, data/, *.png, *.csv
├── docs/plan-de-trabajo.md # este documento
├── src/
│   └── alfa.py             # CONFIG + funciones compartidas
└── notebooks/
    ├── 01_inspeccion.ipynb        # A
    ├── 02_acondicionamiento.ipynb # B
    ├── 03_alfa_temporal.ipynb     # C
    ├── 04_robustez.ipynb          # D
    └── proyecto_final.ipynb       # integrado, fase 3
```

**Carpeta de Drive** `alfa-eeg-data/` con `datos/` (los archivos del registro), `resultados/` (CSV) y `figuras/` (PNG, nombrados `fig01_senal_original.png`, `fig02_...`). Cada quien hace *Agregar acceso directo a Mi unidad* con ese mismo nombre, para que la ruta sea idéntica en todos los Colab.

**Primera celda de cada notebook**

Ya viene escrita en los notebooks de `notebooks/`. Como el repo es público, no hace falta token para clonar; *Guardar una copia en GitHub* pide su propia autorización la primera vez. La misma celda funciona en Colab y en local:

```python
RAMA = "main"   # cámbiala si necesitas probar un alfa.py que aún está en un PR
# Colab: monta Drive, clona/actualiza el repo en /content/repo, instala mne
# Local: usa ../src y la carpeta data/ del repo (o la variable ALFA_DATA)
import alfa
R = alfa.rutas()   # R["vhdr"], R["resultados"], R["figuras"]
```

**Reglas**

1. Un notebook, un dueño. Nadie edita el notebook de otro; si necesitas algo de otro, se pide como función en `alfa.py`.
2. Abrir el notebook desde GitHub (`colab.research.google.com/github/...`) y guardarlo con *Archivo → Guardar una copia en GitHub* en tu rama (`a/inspeccion`, `b/filtros`, etc.), nunca directo en `main`.
3. Antes de guardar: *Editar → Borrar todas las salidas*. Las figuras que importan se guardan como PNG en Drive; así los diffs se pueden revisar y no hay conflictos por widgets.
4. `src/alfa.py` se edita en VS Code o en el editor web de GitHub, siempre por PR con una revisión. Si cambia la firma de una función, se avisa en el chat del equipo.
5. PR pequeños y frecuentes: mejor tres PR de 30 líneas que uno de 300 el día antes de entregar.

## Interfaces compartidas y CONFIG

Estas firmas se acuerdan el día 1, aunque las funciones estén vacías: así C empieza con un filtro provisional sin esperar a B, y D solo cambia `CONFIG` para sus variantes.

```python
# src/alfa.py
CONFIG = {
    "canales": ["P3", "Pz", "P4", "PO9", "O1", "Oz", "O2", "PO10"],  # A justifica la lista final
    "banda_alfa": (8, 13),
    "filtro": {"l_freq": 1.0, "h_freq": 40.0, "notch": 50.0},          # provisional, B decide
    "sfreq_analisis": 250,
    "ventana_s": 2.0,
    "solapamiento": 0.5,
    "recorte_transicion_s": 5.0,   # segundos descartados después de cada cambio de condición
    "excluir_artefactos": True,
}

def cargar_raw(ruta_vhdr) -> mne.io.Raw            # A: VEOG como 'eog', unidades verificadas
def tabla_bloques(raw) -> pd.DataFrame              # A: bloque, condicion, inicio_s, fin_s
def marcar_artefactos(raw, cfg) -> mne.io.Raw       # A: anotaciones 'BAD_...'
def acondicionar(raw, cfg) -> mne.io.Raw            # B
def potencia_alfa(raw, bloques, cfg) -> pd.DataFrame  # C: una fila por ventana y canal
                                                    #    t_s, bloque, condicion, canal, alfa_abs, alfa_rel
def resumen_por_bloque(df) -> pd.DataFrame          # C: media, desviación y razón EC/EO por bloque
```

Primera tarea de A, porque todo depende de ella: confirmar cómo se delimitan los 16 bloques. En el registro hay `S 3` y `S 4` 240 veces cada uno (marcadores repetidos dentro del bloque, no inicios), `S 2` 17 veces y `S 1` una vez. Un bloque es una racha de marcadores iguales consecutivos; hay que verificar si `S 2` coincide con los cambios.

La exclusión de artefactos usa anotaciones `BAD_` de MNE con `reject_by_annotation`, así D puede prenderla y apagarla con `excluir_artefactos` sin tocar código.

## Cronograma

**Puntos de control:** 12 nov firmas acordadas y repo corre en Colab · 19 nov canales y filtro congelados · 25 nov pipeline listo de punta a punta · 28 nov el final corre desde cero.

Nadie pasa a la siguiente fase sin cruzar su punto de control. Las secciones del informe se escriben en cuanto sale cada figura, no en la fase 3: del 26 al 29 solo se une, se corrige y se corre todo desde cero. Si al final de la semana 1 canales y filtro no están decididos, C y D se quedan con los valores provisionales de `CONFIG` y se documenta como decisión.

## Informe y figuras

Cada figura obligatoria tiene un dueño, y quien hace la figura escribe el párrafo que la describe. Todas llevan título, ejes con unidades y descripción en el informe.

| Figura obligatoria | Dueño | Contenido sugerido |
| --- | --- | --- |
| Señal original con condiciones | A | Canales elegidos + VEOG, con bloques EO/EC sombreados |
| Original vs acondicionada | B | Mismo fragmento en el tiempo + PSD antes y después |
| Contenido frecuencial por condición | C | PSD promedio EO vs EC con la banda 8–13 Hz marcada |
| Alfa a lo largo del registro | C | Potencia alfa por ventana vs tiempo, bloques sombreados |
| Comparación entre bloques | C | Boxplot o puntos por bloque, EO y EC lado a lado |
| Robustez (extra) | D | Razón EC/EO bajo cada variante de `CONFIG` |

| Sección del informe | Dueño |
| --- | --- |
| Introducción y fundamento fisiológico | D |
| Pregunta de investigación e hipótesis | D |
| Descripción de los datos | A |
| Inspección y calidad de la señal | A |
| Selección y justificación de canales y segmentos | A |
| Estrategia de acondicionamiento | B |
| Caracterización del alfa y comparación entre condiciones | C |
| Evolución temporal y consistencia | C |
| Evaluación metodológica | D, con A y B |
| Discusión, limitaciones y conclusiones | D, con aportes de todos |
| Referencias | Cada quien agrega las suyas; D unifica el formato |

El informe se escribe en un solo documento compartido. Cada quien escribe su sección en cuanto tiene su figura, no al final. La discusión debe distinguir lo que se observa en la señal de la interpretación fisiológica, y aclarar que es un solo participante.

## Checklist antes de entregar

- [ ] `proyecto_final.ipynb` corre de principio a fin en un entorno nuevo de Colab (*Entorno de ejecución → Desconectar y borrar*, luego *Ejecutar todo*)
- [ ] Todos los parámetros salen de `CONFIG` y se imprimen al inicio del notebook
- [ ] Cada decisión (canales, filtros, ventanas, segmentos excluidos) tiene su justificación en una celda de texto
- [ ] Las 5 figuras obligatorias tienen título, ejes con unidades y descripción en el informe
- [ ] La comparación EO vs EC es cuantitativa (valores por bloque, no solo espectros)
- [ ] Al menos 2 decisiones metodológicas evaluadas, con su efecto en la diferencia EO/EC
- [ ] Se citan el dataset LEMON (Babayan et al., 2019) y la literatura del alfa
- [ ] Se aclara que es un solo participante y qué no se puede generalizar
- [ ] `main` del repo tiene la versión entregada y se le pone un tag (`v1.0-entrega`)

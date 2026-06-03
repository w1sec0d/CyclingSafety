# CyclingSafety — Clasificación Multiclase de Eventos de Riesgo en Ciclismo Urbano

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Sistema de clasificación de eventos críticos de seguridad (CSE) en ciclismo urbano mediante redes neuronales, desarrollado como proyecto de término del curso de Redes Neuronales — Universidad Nacional de Colombia, Grupo 9.

> **Autores:** Juan Andrés Vallejo Rozo · Andrés Felipe Rojas Aguilar · Brayan Camilo Rodríguez Diaz · Carlos David Ramírez Muñoz

---

## Descripción general

El proyecto desarrolla dos sistemas de clasificación de CSE basados en señales IMU (acelerómetro + giroscopio) capturadas con smartphone:

| Modelo | Arquitectura | Clases | Estado |
|---|---|---|---|
| **Modelo A** | MLP heurístico | normal / bache / severo | ✅ Completo |
| **Modelo B** | MLP supervisado | normal / bache / esquivada / freno | ✅ Completo |
| **Modelo C** | 1D-CNN profunda | normal / bache / esquivada / freno | ✅ Completo  |
| **Modelo D** | 1D-CNN (AutoML) profunda | normal / bache / esquivada / freno | ✅ Completo  |

Los modelos entrenados se integran con coordenadas GPS para generar **mapas interactivos de calidad vial** sobre recorridos reales en Bogotá.

---

## Estructura del repositorio

```
CyclingSafety/
│
├── notebooks/
│   ├── MLP_Cycling_Safety.ipynb     # Pipeline principal: MLP heurístico + supervisado + mapas
│   ├── CNN_Cycling_Safety.ipynb     # Red convolucional 1D (en desarrollo)
│   └── research/
│       └── Red_ART.ipynb            # Investigación: Redes ART (Adaptive Resonance Theory)
│
├── utils/
│   └── combine_recordings.py        # Preprocesamiento de grabaciones Sensor Logger → CSV 50 Hz
│
├── data/
│   ├── raw/
│   │   ├── artificial_events/       # Grabaciones etiquetadas (bache, esquivada, freno)
│   │   └── natural_events/          # Recorridos urbanos sin etiquetar
│   └── processed/
│       ├── artificial_events/       # CSVs combinados a 50 Hz (artificial)
│       └── natural_events/          # CSVs combinados a 50 Hz (natural)
│
├── features/                        # Artefactos de features y splits (.npy, .csv)
├── models/                          # Modelos entrenados (.keras) y scalers (.pkl)
│
└── outputs/
    ├── map_mlp_events.html          # Mapa A: marcadores de eventos CSE
    ├── map_severity_heatmap.html    # Mapa B: heatmap de severidad promediada
    └── map_combined.html            # Mapa combinado con capas activables
```

---

## Dataset

Los datos propios fueron recolectados con un **Xiaomi POCO X6** montado en el cuadro de la bicicleta usando **Sensor Logger**, a 100 Hz (remuestreados a 50 Hz).

📁 **Dataset en Google Drive:** [Acceder al dataset](https://drive.google.com/drive/folders/1-24wbZkLPmLIyj7BUMyjm0R-Bd0dZiIx?usp=sharing)

---

## Instalación y uso

### Requisitos

```bash
python >= 3.10
tensorflow >= 2.21
scikit-learn
imbalanced-learn   # para SMOTE
folium             # para mapas interactivos
pandas, numpy, scipy, matplotlib, seaborn
```

### Configuración del entorno local

```bash
git clone https://github.com/w1sec0d/CyclingSafety.git
cd CyclingSafety
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

### Preprocesar datos

Antes de ejecutar el notebook, generar los CSVs combinados desde las grabaciones crudas:

```bash
# Recorridos artificiales (con procesamiento de anotaciones)
python utils/combine_recordings.py --mode artificial

# Recorridos naturales
python utils/combine_recordings.py --mode natural
```

### Ejecutar el notebook principal

```bash
jupyter notebook notebooks/MLP_Cycling_Safety.ipynb
```

O acceder directamente en **Google Colab:** [Abrir en Colab](https://drive.google.com/file/d/1XYW1Dx_ZvmpX7s5-YWOH7kgPcS5dwEK0/view?usp=sharing)

> **Nota Colab:** los datos no están incluidos en el repositorio (archivos > 100 MB). Descargar el dataset desde el enlace de Drive y montarlo en `/content/data/` antes de ejecutar.

---

## Notebooks de investigación

El directorio `notebooks/research/` contiene notebooks de estudio sobre arquitecturas de redes neuronales no directamente ligadas al problema de ciclismo, desarrollados como parte del contexto académico del curso:

| Notebook | Tema |
|---|---|
| `Red_ART.ipynb` | Redes ART (Adaptive Resonance Theory): aprendizaje no supervisado, clustering incremental, dilema plasticidad–estabilidad |

---

## Configuraciones para GPU local

Si se ejecuta localmente con GPU NVIDIA y se presentan errores de XLA, el notebook incluye las siguientes configuraciones en la primera celda:

```python
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
# jit_compile=False en model.compile(...)
```

Estas configuraciones **no se activan en Colab** (donde no generan el mismo conflicto) gracias a la detección automática del entorno de ejecución.

---

## Licencia

MIT License — ver archivo [LICENSE](LICENSE) para más detalles.

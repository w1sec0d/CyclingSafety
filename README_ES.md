# CyclingSafety

**Clasificación multiclase de eventos críticos de seguridad ciclista mediante sensores IMU de smartphone, redes neuronales y mapas de riesgo basados en GPS.**

Español · [English](README.md)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange) ![Licencia: MIT](https://img.shields.io/badge/Licencia-MIT-green)

CyclingSafety explora cómo los sensores de bajo costo de un smartphone y el aprendizaje automático pueden ayudar a identificar eventos críticos durante el ciclismo urbano. Las señales del acelerómetro y el giroscopio se procesan para clasificar eventos como impactos por baches, maniobras evasivas y frenadas de emergencia. Las predicciones se asocian con coordenadas GPS para visualizar dónde ocurrieron los eventos y dónde se concentran los indicadores de severidad vial.

## Resultados principales

| Modelo | Tarea | Conjunto de validación | Exactitud | Macro-F1 |
|---|---|---:|---:|---:|
| **Modelo A — MLP heurístico** | 3 clases: normal, bache, severo | 2.458 ventanas de recorridos naturales reservados | **98,17 %** | **0,954** |
| **Modelo B — MLP supervisado** | 4 clases: normal, bache, esquivada, frenada de emergencia | 3.235 ventanas de recorridos controlados reservados | **97,81 %** | **0,888** |

La evaluación del modelo supervisado reportó **100 % de recall en las tres clases de eventos críticos**. La precisión para la clase bache fue más moderada (0,67), por lo que el resultado debe interpretarse junto con las métricas por clase y las limitaciones del conjunto de datos. Son resultados de validación, no una afirmación de rendimiento de un sistema desplegado en toda la ciudad.

## Dataset

- **2.046.602** muestras de sensores inerciales, armonizadas a **50 Hz**.
- **35 recorridos**: 24 recorridos urbanos naturales y 11 recorridos controlados diseñados para registrar eventos etiquetados.
- Aproximadamente **11,4 horas** de grabaciones.
- **335 eventos anotados manualmente**: 152 impactos por baches, 96 maniobras evasivas y 87 frenadas de emergencia.
- Datos capturados con un Xiaomi POCO X6 y Sensor Logger; se exportaron señales de acelerómetro, giroscopio y GPS para su procesamiento.

Se evaluó el dataset público Bike&Safe, pero se excluyó del conjunto final de entrenamiento porque la escala de sus aceleraciones no era suficientemente compatible con las grabaciones propias.

## Flujo de trabajo

1. **Captura:** registrar acelerómetro, giroscopio y GPS con Sensor Logger en recorridos controlados y naturales.
2. **Preprocesamiento:** combinar grabaciones, remuestrear señales inerciales a 50 Hz, interpolar valores faltantes, propagar coordenadas GPS para asociar espacialmente las ventanas y corregir/eliminar anotaciones duplicadas.
3. **Segmentación e ingeniería de características:** dividir las series temporales en ventanas deslizantes de 128 muestras (2,56 segundos) con 50 % de solapamiento y extraer características estadísticas de las señales inerciales.
4. **Entrenamiento y evaluación:** entrenar clasificadores MLP con TensorFlow/Keras para clasificación heurística de tres clases y clasificación supervisada de cuatro clases.
5. **Visualización geoespacial:** asociar las predicciones con coordenadas GPS y generar mapas interactivos de eventos y mapas de calor de severidad vial con Folium.

## Mapas generados

El repositorio incluye mapas HTML interactivos producidos por el análisis:

- [Eventos CSE clasificados](outputs/map_mlp_events.html)
- [Mapa de calor de severidad vial](outputs/map_severity_heatmap.html)
- [Mapa combinado con capas activables](outputs/map_combined.html)

Si GitHub no muestra directamente el HTML interactivo, descarga el archivo y ábrelo en un navegador.

## Alcance de la implementación

El repositorio documenta e implementa el flujo de captura mediante smartphone, el preprocesamiento IMU, el entrenamiento/evaluación de MLP y los mapas basados en GPS. También propone una arquitectura IoT/Edge Intelligence de cuatro capas:

| Capa | Alcance |
|---|---|
| Percepción | Está implementada la captura mediante IMU/GPS de smartphone; un nodo basado en ESP32 es una extensión propuesta. |
| Comunicación | La transmisión mediante MQTT forma parte de la arquitectura propuesta. |
| Backend | La integración con ThingsBoard está propuesta y no se presenta como un despliegue finalizado. |
| Visualización | El análisis actual genera mapas y mapas de calor con Folium. |

**Importante:** la inferencia local en ESP32/TinyML y la integración en vivo con MQTT/ThingsBoard son objetivos de trabajo futuro. Los resultados reportados provienen del entrenamiento y la validación offline con las grabaciones recolectadas.

## Estructura del repositorio

```text
CyclingSafety/
├── notebooks/
│   ├── MLP_Cycling_Safety.ipynb
│   ├── CNN_Cycling_Safety.ipynb
│   └── research/
│       └── Red_ART.ipynb
├── utils/
│   └── combine_recordings.py
├── features/
├── models/
├── outputs/
└── README.md
```

El repositorio también incluye experimentos con redes 1D-CNN. Las métricas destacadas en este README corresponden a los modelos MLP resumidos en la evaluación del proyecto; aquí no se presentan métricas separadas para las CNN.

## Instalación y uso

### Requisitos

- Python 3.10 o posterior
- Paquetes incluidos en `requirements.txt`

### Instalación

```bash
git clone https://github.com/w1sec0d/CyclingSafety.git
cd CyclingSafety
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
# .venv\\Scripts\\Activate.ps1

pip install -r requirements.txt
```

### Descargar el dataset

Las grabaciones crudas no se incluyen en el repositorio por su tamaño. Descárgalas desde la [carpeta del dataset CyclingSafety en Google Drive](https://drive.google.com/drive/folders/1-24wbZkLPmLIyj7BUMyjm0R-Bd0dZiIx?usp=sharing) y ubícalas en las carpetas esperadas dentro de `data/raw/`.

### Preprocesar las grabaciones

```bash
python utils/combine_recordings.py --mode artificial
python utils/combine_recordings.py --mode natural
```

### Ejecutar el notebook principal

```bash
jupyter notebook notebooks/MLP_Cycling_Safety.ipynb
```

También puedes usar el [notebook de Google Colab](https://drive.google.com/file/d/1XYW1Dx_ZvmpX7s5-YWOH7kgPcS5dwEK0/view?usp=sharing). En Colab, descarga el dataset y móntalo o cópialo a las rutas que espera el notebook antes de ejecutar las celdas.

## Limitaciones y próximos pasos

- Ampliar las grabaciones etiquetadas en condiciones reales variadas y evaluar la generalización entre recorridos y ciudades.
- Mejorar la precisión de las clases minoritarias, especialmente la detección de baches.
- Implementar y medir el rendimiento de la inferencia en el borde con ESP32/TinyML.
- Integrar MQTT y un dashboard en ThingsBoard para disponer de monitoreo de telemetría de extremo a extremo.

## Licencia

Distribuido bajo la [licencia MIT](LICENSE).

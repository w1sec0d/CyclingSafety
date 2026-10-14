# CyclingSafety

**Clasificación multiclase de eventos críticos de seguridad ciclista mediante sensores IMU de smartphone, redes neuronales y mapas de riesgo basados en GPS.**

Español · [English](README.md)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange) ![Licencia: MIT](https://img.shields.io/badge/Licencia-MIT-green)

CyclingSafety explora cómo los sensores de bajo costo de un smartphone y el aprendizaje automático pueden ayudar a identificar eventos críticos durante el ciclismo urbano. Las señales del acelerómetro y el giroscopio se procesan para clasificar eventos como impactos por baches, maniobras evasivas y frenadas de emergencia. Las predicciones se asocian con coordenadas GPS para visualizar dónde ocurrieron los eventos y dónde se concentran los indicadores de severidad vial.

## Comparación de modelos y resultados

El notebook compara clasificadores MLP poco profundos que utilizan características estadísticas diseñadas manualmente con redes convolucionales 1D (CNN) entrenadas directamente sobre ventanas de señales inerciales. Los experimentos CNN incluyen una arquitectura definida manualmente y búsqueda bayesiana de hiperparámetros con KerasTuner (ajuste tipo AutoML).

| Arquitectura | Configuración | Exactitud en test reservado por recorrido | Macro-F1 |
|---|---|---:|---:|
| MLP — características diseñadas manualmente | Heurístico, 3 clases | 98,59 % | 0,9671 |
| MLP — características diseñadas manualmente | Supervisado, 4 clases | 97,91 % | 0,8942 |
| CNN 1D — arquitectura manual | Heurístico, 3 clases | 95,90 % | 0,9027 |
| CNN 1D — arquitectura manual | Supervisado, 4 clases | 96,13 % | 0,6936 |
| CNN 1D — búsqueda de hiperparámetros con KerasTuner | Heurístico, 3 clases | 98,30 % | 0,9542 |
| **CNN 1D — búsqueda de hiperparámetros con KerasTuner** | **Supervisado, 4 clases** | **99,14 %** | **0,9468** |

El mejor resultado supervisado reportado corresponde a la CNN 1D seleccionada con KerasTuner: 99,14 % de exactitud y 0,9468 de macro-F1 sobre 1.627 ventanas de prueba reservadas. El notebook reporta un recall del 100 % en las tres clases de eventos críticos para este modelo, con precisión por clase de 0,92 para baches, 0,94 para esquivadas y 0,76 para frenadas de emergencia. La CNN supervisada manual obtuvo un macro-F1 inferior (0,6936), lo que ilustra por qué la exactitud por sí sola no basta cuando los eventos normales superan ampliamente a los críticos.

En las configuraciones supervisadas, los umbrales de decisión por clase se ajustaron con el conjunto de validación antes de evaluar el test final. Son resultados offline sobre recorridos reservados, no evidencia de despliegue en producción ni de validación a escala de ciudad.

## Dataset

- **2.760.216** muestras de sensores inerciales tras el preprocesamiento, armonizadas a **50 Hz**.
- **48 recorridos** en la ejecución exportada de Colab: 37 recorridos urbanos naturales y 11 recorridos controlados diseñados para registrar eventos etiquetados.
- Aproximadamente **15,3 horas** de grabaciones, calculadas a partir del número de muestras remuestreadas.
- **335 eventos anotados manualmente**: 152 impactos por baches, 96 maniobras evasivas y 87 frenadas de emergencia.
- Datos capturados con un Xiaomi POCO X6 y Sensor Logger; se exportaron señales de acelerómetro, giroscopio y GPS para su procesamiento.

Se evaluó el dataset público Bike&Safe, pero se excluyó del conjunto final de entrenamiento porque la escala de sus aceleraciones no era suficientemente compatible con las grabaciones propias.

## Flujo de trabajo

1. **Captura:** registrar acelerómetro, giroscopio y GPS con Sensor Logger en recorridos controlados y naturales.
2. **Preprocesamiento:** combinar grabaciones, remuestrear señales inerciales a 50 Hz, interpolar valores faltantes, propagar coordenadas GPS para asociar espacialmente las ventanas y corregir/eliminar anotaciones duplicadas.
3. **Segmentación e ingeniería de características:** dividir las series temporales en ventanas deslizantes de 128 muestras (2,56 segundos) con 50 % de solapamiento y extraer características estadísticas de las señales inerciales.
4. **Entrenamiento y evaluación:** entrenar MLP sobre características diseñadas y CNN 1D sobre ventanas crudas. Comparar una CNN de arquitectura manual con búsqueda bayesiana de hiperparámetros mediante KerasTuner en tareas heurísticas de tres clases y supervisadas de cuatro clases.
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
│   ├── Cycling_Safety.ipynb       # Experimentos MLP y CNN 1D
│   ├── Cycling_Safety.py          # Exportación Python del notebook
│   └── research/
├── utils/
│   └── combine_recordings.py
├── features/
├── models/
├── outputs/
└── README.md
```

El mismo notebook incluye los experimentos de CNN 1D manual y con búsqueda de hiperparámetros mediante KerasTuner, además de una comparación final de seis configuraciones frente a los dos modelos MLP base.

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

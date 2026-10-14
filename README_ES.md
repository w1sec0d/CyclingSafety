# CyclingSafety

**Clasificación multiclase de eventos críticos de seguridad ciclista mediante sensores IMU de smartphone, redes neuronales y mapas de riesgo basados en GPS.**

Español · [English](README.md)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange) ![Licencia: MIT](https://img.shields.io/badge/Licencia-MIT-green)

CyclingSafety explora cómo los sensores de bajo costo de un smartphone y el aprendizaje automático pueden ayudar a identificar eventos críticos durante el ciclismo urbano. Las señales del acelerómetro y el giroscopio se procesan para clasificar eventos como impactos por baches, maniobras evasivas y frenadas de emergencia. Las predicciones se asocian con coordenadas GPS para visualizar dónde ocurrieron los eventos y dónde se concentran los indicadores de severidad vial.

## Comparación preliminar de modelos

El notebook compara modelos MLP con características estadísticas diseñadas manualmente frente a redes convolucionales 1D (CNN) entrenadas sobre ventanas crudas de señales IMU. Los experimentos CNN incluyen una arquitectura manual y búsqueda bayesiana de hiperparámetros con KerasTuner (ajuste tipo AutoML).

> **Advertencia metodológica importante:** estas cifras son salidas de la ejecución actual del notebook, pero la partición **todavía no está libre de fuga de información**. Aunque un encabezado la denomina prueba reservada por recorrido, el código aplica `train_test_split` estratificado a ventanas individuales después de segmentar las señales. Cada ventana contiene 128 muestras y el desplazamiento es de 64 (50 % de solapamiento), por lo que ventanas muy similares de una misma grabación pueden aparecer en distintas particiones. Además, las copias aumentadas se generan a partir de todas las ventanas anotadas antes de dividir los datos y después se asignan al entrenamiento; así pueden entrar al entrenamiento derivados de ventanas presentes en validación o prueba. Trata las métricas siguientes como **exploratorias y posiblemente optimistas**, no como estimaciones robustas del rendimiento en recorridos no vistos.

| Arquitectura | Tarea | Exactitud de la partición actual por ventanas | Macro-F1 |
|---|---|---:|---:|
| MLP — características diseñadas manualmente | Heurístico, 3 clases | 98,59 % | 0,9671 |
| MLP — características diseñadas manualmente | Supervisado, 4 clases | 97,91 % | 0,8942 |
| CNN 1D — arquitectura manual | Heurístico, 3 clases | 95,90 % | 0,9027 |
| CNN 1D — arquitectura manual | Supervisado, 4 clases | 96,13 % | 0,6936 |
| CNN 1D — búsqueda de hiperparámetros con KerasTuner | Heurístico, 3 clases | 98,30 % | 0,9542 |
| **CNN 1D — búsqueda de hiperparámetros con KerasTuner** | **Supervisado, 4 clases** | **99,14 %** | **0,9468** |

Para los modelos supervisados, la tabla usa umbrales de decisión ajustados con validación y luego aplicados a la partición actual de prueba. En esta ejecución, la CNN supervisada ajustada reportó un recall del 100 % en las tres clases de eventos críticos, con precisión de 0,92 para baches, 0,94 para esquivadas y 0,76 para frenadas de emergencia. Estos resultados también deben confirmarse después de corregir la partición.

**Siguiente paso recomendado:** dividir por recorrido/grabación original antes de crear las ventanas, mantener todas las ventanas de cada grabación en una única partición, generar datos aumentados solo a partir del entrenamiento, ajustar hiperparámetros y umbrales con validación y evaluar una sola vez sobre recorridos de prueba no utilizados.

## Dataset

- **2.760.216** muestras inerciales en la ejecución de Colab proporcionada, remuestreadas a **50 Hz**: 2.065.293 de grabaciones naturales y 694.923 de grabaciones controladas.
- **48 archivos CSV de grabación/registros** en esa ejecución: 37 naturales y 11 controlados.
- Aproximadamente **15,3 horas**, calculadas a partir del número de muestras remuestreadas.
- **335 marcas de eventos anotadas manualmente**: 152 impactos por baches, 96 maniobras evasivas y 87 frenadas de emergencia.
- Datos capturados con un Xiaomi POCO X6 y Sensor Logger; se exportaron señales de acelerómetro, giroscopio y GPS para su procesamiento.

Estas cifras corresponden a la exportación de notebook proporcionada; deben reconciliarse con los resúmenes anteriores del proyecto antes de utilizarlas en un informe formal o publicación.

Se evaluó el dataset público Bike&Safe, pero se excluyó del conjunto final de entrenamiento porque la escala de sus aceleraciones no era suficientemente compatible con las grabaciones propias.

## Flujo de trabajo

1. **Captura:** registrar acelerómetro, giroscopio y GPS con Sensor Logger en recorridos controlados y naturales.
2. **Preprocesamiento:** combinar grabaciones, remuestrear señales inerciales a 50 Hz, interpolar valores faltantes, propagar coordenadas GPS para asociar espacialmente las ventanas y corregir/eliminar anotaciones duplicadas.
3. **Segmentación e ingeniería de características:** dividir las series temporales en ventanas deslizantes de 128 muestras (2,56 segundos) con 50 % de solapamiento y extraer características estadísticas de las señales inerciales.
4. **Entrenamiento y comparación:** entrenar MLP sobre características diseñadas y CNN 1D sobre ventanas crudas. Comparar una CNN de arquitectura manual con búsqueda bayesiana de hiperparámetros mediante KerasTuner en tareas heurísticas de tres clases y supervisadas de cuatro clases. Consulta la advertencia metodológica antes de interpretar las métricas actuales.
5. **Visualización geoespacial:** asociar las predicciones con coordenadas GPS y generar mapas interactivos de eventos y mapas de calor de severidad vial con Folium.
6. **Inferencia en el borde:** construir un prototipo con ESP32, IMU MPU-6050, GPS NEO-6M y almacenamiento microSD. Cuantizar la CNN 1D supervisada a INT8, desplegarla con TensorFlow Lite Micro, ejecutar inferencia local y publicar eventos georreferenciados por MQTT a ThingsBoard.

## Mapas generados

El repositorio incluye mapas HTML interactivos producidos por el análisis:

- [Eventos CSE clasificados](outputs/map_mlp_events.html)
- [Mapa de calor de severidad vial](outputs/map_severity_heatmap.html)
- [Mapa combinado con capas activables](outputs/map_combined.html)

Si GitHub no muestra directamente el HTML interactivo, descarga el archivo y ábrelo en un navegador.

## Alcance de la implementación

El proyecto combina el flujo de desarrollo de modelos en Python con un prototipo IoT embebido experimental. El repositorio contiene principalmente el notebook de preprocesamiento, los experimentos MLP/CNN y los mapas Folium; el montaje de hardware y el firmware se describen en el informe complementario del proyecto.

| Capa / componente | Alcance de la implementación |
|---|---|
| Nodo sensor | Se construyó un prototipo con ESP32, IMU MPU-6050 de seis ejes, GPS NEO-6M y almacenamiento microSD. |
| Inferencia en el borde | Se aplicó cuantización post-entrenamiento INT8 a la CNN 1D supervisada y se portó a TensorFlow Lite Micro para inferencia local en el ESP32. |
| Comunicación y backend | El firmware publica predicciones georreferenciadas mediante MQTT a un dashboard de ThingsBoard. Se trata de un prototipo experimental, no de un despliegue de producción. |
| Visualización | Mapas de eventos y mapas de calor de severidad con Python/Folium, además de un dashboard de ThingsBoard para la telemetría del nodo. |

El informe registra una reducción del tamaño del modelo de aproximadamente 2,39 MB a 227 KB (cerca del 90,5 %) después de la cuantización, con una variación pequeña en las métricas reportadas para esa prueba. También señala que el rendimiento de clasificación en campo del nodo ESP32 fue inferior al obtenido con los datos del smartphone; por ello, la calibración y validación entre dispositivos siguen siendo tareas importantes. El firmware y la documentación detallada del hardware aún no están incluidos en el flujo principal de notebooks del repositorio. Las métricas preliminares del notebook siguen sujetas a la advertencia metodológica sobre la partición por ventanas.

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

El mismo notebook incluye experimentos CNN 1D manual y con búsqueda de hiperparámetros mediante KerasTuner, además de una comparación de seis configuraciones frente a los dos modelos MLP base. La secuencia actual de partición y aumentación debe corregirse antes de interpretar las métricas como una evaluación independiente.

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
jupyter notebook notebooks/Cycling_Safety.ipynb
```

También puedes usar el [notebook de Google Colab](https://drive.google.com/file/d/1XYW1Dx_ZvmpX7s5-YWOH7kgPcS5dwEK0/view?usp=sharing). En Colab, descarga el dataset y móntalo o cópialo a las rutas que espera el notebook antes de ejecutar las celdas.

## Limitaciones y próximos pasos

- Rehacer la partición a nivel de grabación/recorrido original antes de segmentar ventanas para evitar fuga por ventanas solapadas.
- Generar muestras aumentadas únicamente a partir de ventanas de entrenamiento, nunca de validación o prueba.
- Repetir las comparaciones MLP/CNN y reportar precisión, recall, macro-F1 y matrices de confusión por clase.
- Ampliar las grabaciones etiquetadas en condiciones reales variadas y evaluar la generalización entre recorridos y ciudades.
- Mejorar la precisión de las clases minoritarias, especialmente la detección de baches.
- Recalibrar y evaluar el modelo con datos capturados por la IMU MPU-6050 del ESP32, porque el informe identifica una diferencia de rendimiento frente a los datos del smartphone.
- Incorporar al repositorio el firmware del ESP32, el diagrama de conexiones y las instrucciones de compilación para reproducir el prototipo.
- Mejorar la robustez del registro en microSD y el manejo de desconexiones durante los recorridos.
- Integrar MQTT y un dashboard en ThingsBoard para disponer de monitoreo de telemetría de extremo a extremo.

## Licencia

Distribuido bajo la [licencia MIT](LICENSE).

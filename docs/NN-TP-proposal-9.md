# Sistema de Clasificación Multiclase de Eventos de Riesgo en Ciclismo Urbano mediante Redes Neuronales

Juan Andrés Vallejo Rozo, Andrés Felipe Rojas Aguilar, Brayan Camilo Rodríguez Diaz, Carlos David Ramírez Muñoz

***Abstract*****— El ciclismo en las ciudades es un medio de transporte con grandes beneficios tanto para el medio ambiente como para la salud, pero expone a sus usuarios a todo tipo de accidentes. El uso de sensores ampliamente disponibles en smartphones, como acelerómetros y giroscopios, tiene el potencial de ayudar a aliviar esta problemática. Por ello, el presente trabajo propone el desarrollo de dos sistemas basado en redes neuronales (uno basado en perceptrones multicapa combinados con feature engineering, y otro basado puramente en redes convolucionales de una dimensión) para la clasificación multiclase de Critical Safety Events (CSE) de ciclistas en entornos urbanos. Estos eventos incluyen frenados bruscos, virajes repentinos, impactos por infraestructura (como baches), y pérdidas de control o caídas. Para el desarrollo de estos modelos nos basaremos en el dataset Bike\&Safe, que contiene datos secuenciales de sensores como acelerómetros y giroscopios. Este proyecto se enmarca en el área de Human Activity Recognition (HAR), y tiene como meta la recopilación de datos de estos eventos para poder elaborar con ellos mapas de calor y así poder mejorar las condiciones de las vías y con ello la seguridad de los ciclistas.** 

***Index Terms*****— Biking Safety, Critical Safety Events, Human Activity Recognition, Multiclass Classification, Neural Networks**

1. # Introducción

La bicicleta representa actualmente un medio de transporte alternativo, ecológico, descongestionante de tráfico y mitigador de enfermedades cardiovasculares. Sin embargo, los ciclistas urbanos han tenido que enfrentar condiciones adversas de seguridad vial con preocupantes índices de víctimas de siniestros viales. Para la ciudad de Bogotá, se presentaron 69 fallecidos y 2049 lesionados en 2024, lo que los hace el tercer tipo de actor vial más vulnerable (después de motociclistas y peatones) \[1\], sumado a la percepción ciudadana general de inseguridad e incidentes leves no contempladas en estas cifras. 

Ante este panorama es fundamental la creación de mecanismos para la identificación y mitigación de eventos de riesgo de los ciclistas. Los sistemas de monitoreo actuales suelen usar GPS de baja frecuencia (\~1 Hz), mediante el uso de smartphones por ejemplo, que solo permiten análisis de trayectorias, o en modelos binarios de detección de anomalías (evento normal vs. anomalía). Sin embargo, para mejorar la seguridad vial, puede ser sumamente útil ir más allá de la detección binaria y clasificar la naturaleza específica del evento (ej. frenado de emergencia, esquiva lateral por obstáculo, o impacto por estado del pavimento, entre otros).  
El problema radica en la alta complejidad y no linealidad de las señales inerciales generadas durante la conducción. Este proyecto busca resolver este problema incluyendo en el análisis sensores inerciales (acelerómetro y giroscopio a 50 Hz) de dataset actuales en conjunción con sensores IoT para capturar datos propios. Estos datos permitirán aplicar modelos de redes neuronales para clasificar los tipos de incidentes.

2. # Objetivos

**Objetivo General:**

Diseñar, crear y evaluar un sistema de clasificación multiclase para categorizar eventos críticos, anomalías en el terreno y maniobras de riesgo en el ciclismo urbano utilizando datos de geolocalización y de telemetría inercial.

**Objetivos Específicos:**

* **Recolección y Preprocesamiento de datos:** Consolidar un conjunto de datos multivariado combinando bases de datos públicas (como el Dataset Bike\&Safe) y validaciones locales con la herramienta Sensor Logger, sincronizando señales inerciales y aplicando ventanas de tiempo. El uso de otros sensores IoT más precisos será evaluado según el avance del proyecto.  
    
* **Reconocimiento básico de eventos (Creación Red Neuronal Poco Profunda):** Desarrollar una arquitectura de red neuronal superficial basada en un perceptrón multicapa utilizando técnicas previas de Feature Engineering como extraer valores estadísticos (media, varianza, picos) de las ventanas de tiempo inerciales para clasificar de manera básica los eventos abruptos.  
    
* **Reconocimiento a avanzado de eventos (Creación Red Neuronal Profunda):**  Desarrollar una red neuronal profunda basada en capas convolucionales (1D-CNN) capaz de realizar extracción automática de características espaciotemporales directamente sobre los datos crudos de los sensores como GPS, acelerómetro y giroscopio para poder identificar y clasificar de forma precisa cada tipo de evento abrupto.  
    
* **Evaluación Comparativa:** Analizar el desempeño de ambas arquitecturas neuronales utilizando métricas de clasificación multiclase (Accuracy, Precision, Recall y F1-Score) empleando como instrumento clave las matrices de confusión para evaluar la capacidad de los modelos para distinguir los diferentes tipos de anomalías.  
    
* **Identificación de zonas de riesgo:** Integrar los resultados obtenidos de la clasificación de los eventos peligrosos junto  a las coordenadas GPS para generar mapas de calor que evidencien los tramos viales con mayor recurrencia de accidentes. Podrían incluirse datos de los tramos viales para estudiar relaciones entre la accidentalidad y el tipo de tramo, calidad del asfalto, entre otros

3. # Revisión de Literatura

   1. ## *Seguridad Vial y Maniobras de Detección Basadas en Sensores Inerciales*

El uso de sensores inerciales como el acelerómetro y el giroscopio, disponibles en smartphones, se ha convertido en una solución de bajo costo y escalable para análisis de seguridad vial. Estos sensores permiten la detección de patrones en la conducción, como el frenado, los cambios de carril, y maniobras agresivas sin necesidad de hardware vehícular especializado. En concreto la aplicación Sensor Logger permite integrar este tipo de sensores junto a la ubicación GPS de dispositivos Android e Iphone, pudiendo generar conjuntos de datos sincronizados y con preprocesamiento y calibración básicas.

2. ## *Recolección de Datos y Preprocesamiento*

El procesamiento de señales inerciales requiere una etapa de pretratamiento rigurosa para garantizar consistencia y estabilidad en los modelos de clasificación. En sistemas basados en smartphones o IMUs, es común aplicar:

* Sincronización temporal de señales multicanal  
* Filtrado para reducción de ruido  
* Eliminación de componente gravitacional  
* Normalización o estandarización  
* Segmentación en ventanas temporales deslizantes

La segmentación por ventanas es una estrategia ampliamente utilizada en reconocimiento de actividades humanas, ya que permite transformar señales continuas en muestras etiquetadas para aprendizaje supervisado \[4\]. La elección del tamaño de ventana y el porcentaje de solapamiento impacta directamente el desempeño del modelo.

3. ## *Ingeniería de Características y Modelos Superficiales*

Los enfoques tradicionales para clasificación de series temporales inerciales se basan en la extracción manual de características estadísticas a partir de cada ventana temporal. Entre las características más comunes se encuentran:

* Media  
* Varianza  
* Desviación estándar  
* Energía (RMS)  
* Amplitud pico-a-pico

Estas características reducen la dimensionalidad del problema antes de aplicar clasificadores como SVM, k-NN o Random Forest. Estudios comparativos en HAR han mostrado que estos métodos pueden alcanzar desempeños competitivos cuando las características están cuidadosamente diseñadas \[5\].  
Las redes neuronales superficiales (MLP con una o dos capas ocultas) representan una extensión de estos enfoques, permitiendo modelar relaciones no lineales entre características.

4. ## *Aprendizaje*

El aprendizaje profundo ha demostrado ventajas significativas frente a métodos tradicionales al permitir la extracción automática de representaciones jerárquicas directamente desde datos crudos.

1. ### *Convolutional Neural Networks (CNN)*

Las CNN aplicadas a señales temporales (1D-CNN) realizan convoluciones sobre el eje temporal para capturar patrones locales característicos. En HAR, CNN profundas han superado sistemáticamente métodos tradicionales basados en características manuales \[4\].

En el estudio de da Silva \[7\], una CNN aplicada al Bike\&Safe Dataset \[3\] logró detectar anomalías en comportamiento ciclista utilizando directamente las secuencias de sensores, demostrando la efectividad del enfoque profundo para análisis de seguridad vial.

### 

2. ### *Long Short-Term Memory (LSTM)*

Las redes LSTM, una variante de redes neuronales recurrentes, están diseñadas para capturar dependencias temporales de largo plazo. Estudios comparativos han demostrado que arquitecturas híbridas CNN-LSTM combinan extracción automática de características y modelado secuencial, obteniendo altos niveles de desempeño en clasificación de actividades humanas con datos IMU \[6\].

5. ## *Evaluación de Modelos de Clasificación Multiclase*

En problemas de clasificación multiclase, especialmente cuando existen clases desbalanceadas o eventos poco frecuentes, la evaluación no debe limitarse a la exactitud global. Se suele recomendar el uso de:

* Accuracy  
* Precision  
* Recall  
* F1-Score  
* Matriz de confusión

Estas métricas permiten evaluar la capacidad del modelo para distinguir entre clases similares y analizar errores sistemáticos. Estudios en HAR y clasificación de señales inerciales enfatizan la importancia de analizar métricas por clase y promedios macro para garantizar robustez del modelo \[4\], \[5\].

6. ## *Eventos de Riesgo y Anomalías*

Los eventos de tráfico que requieren maniobras evasivas rápidas (frenado, giro, o en algunos casos ambos) se denominan Critical Safety Events (CSE). Para la realización del proyecto hemos decidido clasificar el los CSE en 4 categorías principales

1. Eventos Longitudinales (Frenado Brusco): Estos representan frenados o desaceleraciones bruscas que se evidencian en picos negativos repentinos en el eje Y o X de movimiento (según el montaje). Son detenciones de emergencia ante obstáculos.  
     
2. Eventos Laterales (Viraje Repentino): Maniobras rápidas en el eje horizontal para evitar obstáculos, baches, entre otros elementos. Se ve reflejado en el giroscopio y la aceleración lateral  
     
3. Eventos Verticales (Impactos por Infraestructura): Anomalías causadas principalmente por el estado deficiente de la vía (baches, resaltos, grietas). Estos se ven reflejados en picos en el eje Z del acelerómetro.  
     
4. Pérdida de control o caída: eventos abruptos donde el giroscopio refleja un ángulo de inclinación que supera un umbral crítico, acompañado de una detención total del movimiento.

4. # Estudio de caso

Se han identificado principalmente dos investigaciones que representan soluciones paralelas y parciales hacia los objetivos propuestos.

Primeramente el estudio "*Detection of anomalies in cycling behavior with convolutional neural network and deep learning"* demuestra que es posible realizar emplear las redes convolucionales, para identificar las maniobras evasivas ciclistas y los CSE en la ciudad de Catania, Italia. En el se diseña y crea el modelo BeST-DAD (Bicyclist Safety Tracking \- Deep Anomaly Detection) que emplea una arquitectura basada en un autoencoder convolucional, logrando una clasificación binaria exitosa con un F-Score del 77% y un Recall del 100% de los casos \[2\].  Este estudio proporciona las bases para justificar a las CNN como adecuadas para procesar señales de movilidad, sin embargo, este sistema depende de forma exclusiva de los datos de geolocalización con tasas de actualización altas, como velocidad de desplazamiento y dirección, dejando de lado los sensores inerciales para la clasificación multiclase e identificación avanzada del tipo de anomalía, el cual es el objetivo principal de nuestro estudio.

Como segundo caso hemos documentado el artículo "*Enhancing Cycling Safety in Smart Cities: A Data-Driven Embedded Risk Alert System*" en este estudio, se implementa de forma exitosa un sistema de alerta de riesgo ciclista en tiempo real usando hardware de bajo costo como la tarjeta *Raspberry Pi Zero*. El proyecto emplea una recolección masiva de datos inerciales con una infraestructura IoT, pero no emplea las redes neuronales para la clasificación y análisis de dicha información \[3\]. Este estudio brinda un robusto dataset enfocado a la seguridad ciclista, el cual puede ser usado como insumo para el entrenamiento de la red neuronal, esto es explicado en mayor detalle en la siguiente sección del documento.

En síntesis, el objetivo del proyecto es superar la no inclusión de variables cinemáticas expuestas en \[2\] incluyendo en el análisis sensores clave como acelerómetro y giroscopio, mientras se superan las carencias analíticas en \[3\] mediante la construcción y uso de redes neuronales convolucionales. El resultado proyectado será un sistema capaz de reconocer los CSE en diferentes clases, siendo no solamente exhaustivos en la captura de datos sino integrando sistemas inteligentes en la interpretación de los mismos.

5. # Dataset de entrenamiento

El  Dataset Bike\&Safe\[3\], presentado y analizado en el artículo "Detection of anomalies in cycling behavior with convolutional neural network and deep learning" \[7\], constituye un conjunto de datos multivariado de acceso público diseñado específicamente para la detección de anomalías en el comportamiento de ciclistas.   
Los datos fueron recolectados mediante un teléfono Android montado en el manillar de la bicicleta, registrando señales triaxiales de acelerómetro, giroscopio, magnetómetro y GPS. Los sensores inerciales se configuraron con un retardo muestral de 20 ms en la generación de series. Cada recorrido fue repetido múltiples veces, produciendo más de 800.000 registros secuenciales multivariados almacenados en archivos CSV.  
El dataset incluye componentes marcados temporalmente de la aceleración y la velocidad en los 3 ejes, además contiene posición geográfica y mediciones de velocidad \[3\], lo que permite la sincronización precisa de estas señales multimodales.  
También se valora la posibilidad de tomar datos propios usando métodos similares al dataset, por medio de dispositivos IoT o teléfonos inteligentes.

6. # Enfoque propuesto basado en redes neuronales

   1. ## *Descripción General*

Se propone un enfoque basado en Redes Neuronales Convolucionales (CNN) para la clasificación multiclase de maniobras utilizando datos inerciales multivariados provenientes de sensores de smartphone (acelerómetro, giroscopio y GPS).  
Las CNN han demostrado un desempeño superior frente a métodos tradicionales en tareas de reconocimiento de actividad humana y clasificación de señales inerciales, al permitir la extracción automática de características directamente desde datos crudos \[4\]\[6\].   
El modelo propuesto opera directamente sobre ventanas segmentadas de señales crudas, reduciendo la dependencia de descriptores estadísticos diseñados manualmente.

2. ## *Arquitectura CNN Propuesta*

* Convolutional Layer 1  
* Max Pooling Layer 1  
* Convolutional Layer 2  
* Max Pooling Layer 2  
* Fully Connected Layer  
* Output Layer

7. # Resultados Esperados

   1. ## *Sistema esperado*

Se espera desarrollar un sistema funcional y validado de clasificación multiclase capaz de identificar y categorizar de manera precisa eventos críticos, anomalías del terreno y maniobras de riesgo en ciclismo urbano a partir de datos de geolocalización y telemetría inercial.  
El sistema deberá:

* Alcanzar una exactitud superior al 75 % en la clasificación de eventos.  
* Mantener un F1-Score balanceado entre las diferentes clases.  
* Integrar señales de acelerómetro, giroscopio y GPS de forma sincronizada.  
* Operar de manera robusta ante variaciones en los dispositivos de adquisición.

Como resultado final, se busca una herramienta capaz de transformar datos crudos de sensores en información estructurada sobre seguridad vial, permitiendo tanto el análisis automatizado de maniobras como la identificación posterior de patrones de riesgo en entornos urbanos. 

2. ## *Rendimiento frente a modelos similares*

Se espera evidenciar una mejora en el desempeño del modelo propuesto basado en 1D-CNN en comparación con enfoques que no empleen arquitecturas convolucionales profundas o que no integren información multicanal proveniente de sensores inerciales (acelerómetro y giroscopio).

En particular, se anticipa una mejora en métricas como Accuracy y F1-Score frente a modelos tradicionales basados en métodos estadísticos, enfoques lineales o arquitecturas que utilicen únicamente variables cinemáticas derivadas del GPS, como en el caso reportado en \[2\].

3. ## *Recolección de datos nuevos*

Por medio de dispositivos móviles inteligentes o dispositivos IoT, se busca tomar datos nuevos, estos deben poder ser incorporados al modelo y manteniendo un desempeño estable, adaptándose a las diferencias en muestreo de los diferentes dispositivos en escenarios más realistas.

4. ## *Identificación de Zonas de Riesgo*

Se espera poder generar mapas de calor para la identificación de tramos con alta frecuencia de eventos peligrosos, potenciales puntos críticos de accidentes.  
El resultado final deberá constituir una herramienta de apoyo para el análisis de seguridad vial, proporcionando información interpretable para estudios posteriores y toma de decisiones en infraestructura o planificación urbana.

8. # Evaluación de la solución

Para evaluar el rendimiento del modelo, se emplean las siguientes métricas:  
Accuracy=TP+TNTP+TN+FP+FN​  
Mide la proporción total de muestras correctamente clasificadas.  
Precision=TPTP+FP​  
Evalúa la proporción de predicciones correctas dentro de cada clase.  
Recall=TNTN+FN​  
Mide la capacidad del modelo para identificar correctamente las instancias de cada clase.  
**F1-Score**\=2​PrecisionRecallPrecision+Recall​

Representa el equilibrio armónico entre Precisión y Recall.  
En el contexto multiclase, se reportan promedios macro y ponderados para considerar posibles desbalances entre clases. También se usará una matriz de confusión para analizar el desempeño por clase, patrones de error, capacidad del modelo para diferenciar maniobras similares.  
Estas métricas suelen encontrarse en los resultados de herramientas similares, y serán usados para comparar y evaluar el desempeño del modelo frente a otros similares.

9. # Consideraciones Éticas

El desarrollo e implementación de este sistema de monitoreo, basado en telemetría inercial y posicionamiento global (GPS), necesita de responsabilidades éticas importantes que busquen garantizar su uso respetando la privacidad de las personas y que pueda cuidar la integridad física de quienes participan en el estudio. Por ello, el proyecto se plantean los siguientes principios a seguir:

* **Privacidad y anonimización de datos.**  
   La recolección de coordenadas GPS puede revelar información sensible, como rutinas diarias, lugares de residencia o sitios de trabajo de los ciclistas. A partir de esto se busca que todos los datos obtenidos mediante Sensor Logger o nodos IoT locales sean anonimizados de forma estricta. Antes de entrenar el modelo se eliminará cualquier información de identificación personal (PII), y los trayectos serán segmentados para evitar que pueda reconstruirse una ruta completa de origen a destino.

* **Consentimiento informado.**  
   En la fase de recolección de datos en la ciudad de Bogotá, cada ciclista que quiera participar será informado de manera clara y transparente sobre el propósito de este estudio, los sensores utilizados, el tipo de información que se almacenará y el tratamiento que se le dará. La participación será totalmente voluntaria, y cada persona podrá retirar sus datos del conjunto de entrenamiento en cualquier momento, sin ninguna consecuencia.

* **Seguridad e integridad física durante la recolección.**  
    Se pedirá a los ciclistas voluntarios que mantengan sus hábitos normales de conducción y respeten las normas de tránsito. En ningún caso se solicitarán maniobras peligrosas, caídas intencionales o situaciones forzadas para alimentar el modelo con datos “anómalos”. Este tipo de eventos se tomará únicamente de bases de datos públicas validadas (como Bike\&Safe) o de situaciones reales que ocurran de forma natural durante trayectos cotidianos.

* **Mitigación de sesgos geográficos y socioeconómicos.**  
   Bogotá presenta grandes diferencias en infraestructura ciclista y estado de la malla vial según la localidad. Para evitar que el modelo de aprendizaje automático o los mapas de calor resultantes reflejen únicamente ciertas zonas, la recolección de datos buscará cubrir distintas áreas de la ciudad (norte, centro, occidente y sur). El objetivo es que el sistema no pueda tener sesgos ante ciertos riesgos presentes en sectores periféricos o vulnerables.

* **Transparencia y uso responsable de la inteligencia artificial.**  
   El modelo entrenado y los mapas generados tienen un propósito estrictamente preventivo e investigativo. El código fuente estará disponible como software abierto en el repositorio de GitHub correspondiente, promoviendo la auditoría algorítmica y la colaboración académica. Se descarta de manera explícita cualquier uso del sistema con fines de vigilancia, perfilamiento punitivo de ciclistas o comercialización no autorizada de datos de movilidad.

10. # Enlaces a código fuente y video explicativo

El código fuente relacionado al proyecto será alojado en el siguiente repositorio de Github: [https://github.com/w1sec0d/CyclingSafety](https://github.com/w1sec0d/CyclingSafety)

El video explicativo puede ser consultado en el siguiente enlace: [https://www.loom.com/share/ece8b539e6b14aa795e881fa3e23e03d](https://www.loom.com/share/ece8b539e6b14aa795e881fa3e23e03d)

11. # Contribuciones de miembros del proyecto

| Miembro del equipo | Rol | Actividades / Contribuciones |
| ----- | ----- | ----- |
| Carlos David Ramírez Muñoz | Líder | Planteamiento inicial del proyecto, Análisis de literatura inicial, establecimiento de objetivos, estudio de caso, consideraciones éticas. Grabación de video explicativo |
| Brayan Camilo Rodriguez Diaz | Investigador | Revisión de la literatura, dataset de entrenamiento, enfoque del proyecto, resultados esperados y evaluación de la solución |
| Andrés Felipe Rojas Aguilar | Observador | Revisión de la propuesta escrita para corregir errores. Elaboración de las diapositivas para el video. |

12. # Uso de herramientas de inteligencia artificial

Gemini fue usado en su modo de investigación para mejorar la revisión inicial de la literatura, mejorando las fuentes ya conocidas previamente con la consideración de estudios previos similares. Además fue usado para considerar inicialmente las limitaciones técnicas y los requisitos generales para poder entrenar una red neuronal que pueda cumplir los objetivos propuestos exitosamente.

Las diapositivas usadas para el video explicativo fueron realizadas en Presentaciones de Google, usando su modelo generativo Nano Banana.

Referencias

\[1\] Secretaria Distrital de Movilidad, “Anuario de siniestralidad vial de Bogotá 2024,” datos.movilidadbogota.gov.co. \[Online\]. Available: [https://datos.movilidadbogota.gov.co/documents/07c7989a57954f91a2c42d792e7efd19/about](https://datos.movilidadbogota.gov.co/documents/07c7989a57954f91a2c42d792e7efd19/about)

\[2\] S. Yaqoob, S. Cafiso, G. Morabito, and G. Pappalardo, “Detection of anomalies in cycling behavior with convolutional neural network and deep learning,” European Transport Research Review, vol. 15, no. 1, pp. 9-, Mar. 2023, doi: 10.1186/s12544-023-00583-4.

\[3\] G. B. da Silva, “Bike\&Safe Dataset,” Mendeley Data. \[Online\]. Available: [https://data.mendeley.com/datasets/3j9yh8znj4/3](https://data.mendeley.com/datasets/3j9yh8znj4/3)

\[4\] F. J. Ordóñez and D. Roggen, “Deep convolutional and *LSTM* recurrent neural networks for multimodal wearable activity recognition,” *Sensors*, vol. 16, no. 1, 2016\.

\[5\] J. R. Kwapisz, G. M. Weiss, and S. A. Moore, “Activity recognition using cell phone accelerometers,” *ACM SIGKDD Explorations Newsletter*, vol. 12, no. 2, pp. 74–82, 2010\.

\[6\] M. Jaén-Vargas et al., “A deep learning approach to recognize human activity using inertial sensors,” *Frontiers in Artificial Intelligence and Applications*, 2021\.

\[7\] G. B. da Silva, “Detection of anomalies in cycling behavior with convolutional neural network and deep learning,” *European Transport Research Review*, vol. 15, no. 1, Mar. 2023, doi: 10.1186/s12544-023-00583-4.


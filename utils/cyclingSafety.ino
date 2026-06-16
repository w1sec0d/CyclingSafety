#include <WiFi.h>
#include <PubSubClient.h>
#include <Wire.h>
#include <TinyGPS++.h>
#include <SPI.h>
#include <SD.h>
#include <MPU6050_light.h>

// Librerías de TensorFlow Lite Micro (Incluidas en Chirale_TensorFLowLite)
#include "modelo_esp32.h"
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h> 
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

// ── CONFIGURACIÓN DE RED (ZONA WI-FI DEL CELULAR) ──
const char* ssid = "XXXX"; 
const char* password = "XXXXXX";

// ── CONFIGURACIÓN DE THINGSBOARD ──
const char* mqtt_server = "mqtt.thingsboard.cloud";
const char* access_token = "XXXXXX";

// ── HARDWARE ──
#define GPS_BAUDRATE 9600
const int chipSelect = 5; 
MPU6050 mpu(Wire);
TinyGPSPlus gps;
WiFiClient espClient;
PubSubClient mqttClient(espClient);
File dataFile;

// ── VARIABLES DE IA (TINYML) ──
constexpr int kTensorArenaSize = 45 * 1024; // 45 KB de RAM asignados a la red neuronal
uint8_t tensor_arena[kTensorArenaSize];
const tflite::Model* tflite_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// ── CONTROL DE VENTANA (SLIDING WINDOW) ──
const int WINDOW_SIZE = 120; // 2.56 segundos a 50Hz
const int NUM_FEATURES = 6;  // ax, ay, az, gx, gy, gz
int muestras_recolectadas = 0;

// ── CONTROL DE TIEMPO A 50 HZ ──
unsigned long startTimeMicros = 0;
unsigned long hwPreviousMicros = 0;
const unsigned long hwIntervalMicros = 20000; // 50 Hz exactos

void setup_wifi() {
  delay(10);
  Serial.print("[WIFI] Conectando a ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\n[WIFI] Conectado exitosamente al Hotspot del celular.");
}

void reconnect_mqtt() {
  while (!mqttClient.connected()) {
    Serial.print("[MQTT] Intentando conectar a ThingsBoard...");
    // El username es el Access Token, el password va en blanco
    if (mqttClient.connect("ESP32_Ciclista", access_token, NULL)) {
      Serial.println(" ¡Conectado!");
    } else {
      Serial.print(" Falló, rc=");
      Serial.print(mqttClient.state());
      Serial.println(". Reintentando en 3 segundos...");
      delay(3000);
    }
  }
}

void setup() {
  Serial.begin(115200);
  Wire.begin();
  Serial2.begin(GPS_BAUDRATE, SERIAL_8N1, 16, 17);

  // 1. Iniciar Conectividad
  setup_wifi();
  mqttClient.setServer(mqtt_server, 1883);

  // 2. Iniciar Sensores y SD
  byte status = mpu.begin();
  if(status != 0){
    Serial.println("[ERROR] MPU6050 no detectado.");
    while(1) delay(10);
  }
  if (!SD.begin(chipSelect)) {
    Serial.println("[ERROR] Falla en la MicroSD.");
    while (1) delay(10);
  }
  
  dataFile = SD.open("/rutas_ia.csv", FILE_APPEND);
  if (!dataFile) {
    Serial.println("[ERROR] No se pudo crear archivo en SD.");
    while (1) delay(10);
  }
  if (dataFile.size() == 0) {
    dataFile.println("timestamp,ax,ay,az,gx,gy,gz,prediccion,lat,lng");
    dataFile.flush();
  }

  Serial.println("[IMU] Calibrando offsets en plano horizontal... ¡NO MUEVAS LA PLACA!");
  delay(1500); 
  mpu.calcOffsets();

  // 3. Iniciar Motor TensorFlow Lite Micro
  // Cargar el arreglo hexadecimal del modelo int8
  tflite_model = tflite::GetModel(modelo_tflite);
  
  // Reemplazo del AllOpsResolver por el MutableOpResolver
  // Se asignan exactamente 4 espacios en memoria para las operaciones de la CNN
  static tflite::MicroMutableOpResolver<10> resolver;
  resolver.AddConv2D();         // TFLite procesa tu Conv1D mapeándola internamente como Conv2D
  resolver.AddRelu();           // Activación de las capas ocultas
  resolver.AddFullyConnected(); // Capas densas finales
  resolver.AddSoftmax();        // Salida probabilística para las 4 clases
  resolver.AddExpandDims();
  resolver.AddMul();
  resolver.AddAdd();
  resolver.AddReshape();
  resolver.AddMaxPool2D();
  resolver.AddMean();

  // Construir el intérprete en silencio (pasando NULL en lugar del ErrorReporter)
  static tflite::MicroInterpreter static_interpreter(
      tflite_model, resolver, tensor_arena, kTensorArenaSize, NULL);
  interpreter = &static_interpreter;

  // Asignar memoria a los tensores
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("[ERROR TFLITE] Falla al asignar memoria Tensor_Arena.");
    while (1);
  }

  input = interpreter->input(0);
  Serial.print("[DEBUG IA] Input Scale: "); Serial.println(input->params.scale, 6);
  Serial.print("[DEBUG IA] Input Zero Point: "); Serial.println(input->params.zero_point);
  output = interpreter->output(0);
  
  Serial.println("[OK] Modelo IA cargado en RAM. Iniciando inferencia en tiempo real...");
  
  startTimeMicros = micros();
  hwPreviousMicros = micros();
}

void loop() {
  if (WiFi.status() == WL_CONNECTED && !mqttClient.connected()) {
    reconnect_mqtt();
  }
  mqttClient.loop();

  // Mantener actualizado el GPS en cada ciclo de CPU
  while (Serial2.available() > 0) {
    gps.encode(Serial2.read());
  }

  unsigned long currentMicros = micros();
  if (currentMicros - hwPreviousMicros >= hwIntervalMicros) {
    hwPreviousMicros += hwIntervalMicros;

    mpu.update();

    // ── 1. OBTENCIÓN DE DATOS Y CONVERSIÓN DE UNIDADES ──
    float roll  = mpu.getAngleX() * (PI / 180.0);
    float pitch = mpu.getAngleY() * (PI / 180.0);

    float ax_raw = mpu.getAccX();
    float ay_raw = mpu.getAccY();
    float az_raw = mpu.getAccZ();

    float gx_grav = -sin(pitch);
    float gy_grav = sin(roll) * cos(pitch);
    float gz_grav = cos(roll) * cos(pitch);

    float ax = (ax_raw - gx_grav) * 9.81;
    float ay = (ay_raw - gy_grav) * 9.81;
    float az = (az_raw - gz_grav) * 9.81;

    float gx = mpu.getGyroX() * (PI / 180.0);
    float gy = mpu.getGyroY() * (PI / 180.0);
    float gz = mpu.getGyroZ() * (PI / 180.0);

    // ── 2. ESTANDARIZACIÓN Z-SCORE ──
    const float feature_means[6] = { 0.021366, -0.090056, 0.090380, 0.010523, -0.003118, -0.002767 };
    const float feature_stds[6] = { 1.322040, 2.846342, 4.455335, 0.154088, 0.203197, 0.180061 };

    float ax_scaled = (ax - feature_means[0]) / feature_stds[0];
    float ay_scaled = (ay - feature_means[1]) / feature_stds[1];
    float az_scaled = (az - feature_means[2]) / feature_stds[2];
    float gx_scaled = (gx - feature_means[3]) / feature_stds[3];
    float gy_scaled = (gy - feature_means[4]) / feature_stds[4];
    float gz_scaled = (gz - feature_means[5]) / feature_stds[5];

    // ── 3. CUANTIZACIÓN AL VUELO ──
    float scale = input->params.scale;
    int zero_point = input->params.zero_point;
    
    int base_idx = muestras_recolectadas * NUM_FEATURES;
    
    auto quantize = [scale, zero_point](float val) -> int8_t {
      int quantized = round(val / scale) + zero_point;
      return (int8_t) max(-128, min(127, quantized));
    };

    input->data.int8[base_idx + 0] = quantize(ax_scaled);
    input->data.int8[base_idx + 1] = quantize(ay_scaled);
    input->data.int8[base_idx + 2] = quantize(az_scaled);
    input->data.int8[base_idx + 3] = quantize(gx_scaled);
    input->data.int8[base_idx + 4] = quantize(gy_scaled);
    input->data.int8[base_idx + 5] = quantize(gz_scaled);

    muestras_recolectadas++;

    // Variable temporal para atrapar la predicción si la hay en este ciclo
    String evento_detectado = "";

    // ── 4. MOTOR DE INFERENCIA ──
    if (muestras_recolectadas == WINDOW_SIZE) {
      Serial.println("\n--- [DEBUG] CONTENIDO DE LA VENTANA PASADA A LA IA ---");
      Serial.println("Muestra |  ax_int8  |  ay_int8  |  az_int8  |  gx_int8  |  gy_int8  |  gz_int8");
      for(int m = 0; m < 5; m++) { // Imprimimos las primeras 5 muestras de la ventana de 100
        int idx = m * 6;
        Serial.printf("  #%d    |    %d    |    %d    |    %d    |    %d    |    %d    |    %d\n", 
                      m, 
                      input->data.int8[idx+0], input->data.int8[idx+1], input->data.int8[idx+2],
                      input->data.int8[idx+3], input->data.int8[idx+4], input->data.int8[idx+5]);
      }
      Serial.println("-----------------------------------------------------");

      if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("[ERROR] Falla al invocar al intérprete.");
        return;
      }

      int max_idx = 0;
      int8_t max_val = output->data.int8[0];
      for (int i = 1; i < 4; i++) {
        if (output->data.int8[i] > max_val) {
          max_val = output->data.int8[i];
          max_idx = i;
        }
      }

      if (max_idx != 0) {
        if (max_idx == 1) evento_detectado = "bache";
        else if (max_idx == 2) evento_detectado = "esquivada";
        else if (max_idx == 3) evento_detectado = "frenada";

        // Obtener GPS para MQTT (también se usará para la SD abajo)
        float lat = gps.location.isValid() ? gps.location.lat() : 4.6582; 
        float lng = gps.location.isValid() ? gps.location.lng() : -74.0931;

        String payload = "{";
        payload += "\"evento\": \"" + evento_detectado + "\", ";
        payload += "\"confidence\": " + String(max_val) + ", ";
        payload += "\"location_latitude\": " + String(lat, 6) + ", ";
        payload += "\"location_longitude\": " + String(lng, 6);
        payload += "}";

        if (mqttClient.connected()) {
          mqttClient.publish("v1/devices/me/telemetry", payload.c_str());
          Serial.println("⚠️ [ALERTA] " + evento_detectado + " enviado a MQTT.");
        }
      }
      muestras_recolectadas = 0;
    }

    // ── 5. GUARDEADO HISTÓRICO EN LA SD (Posición Final) ──
    // Se ejecuta cada 16.6ms. Si hubo inferencia y se detectó anomalía, 'evento_detectado' tendrá texto.
    float seconds_elapsed = (float)(currentMicros - startTimeMicros) / 1000000.0;
    
    // Extraemos el GPS general para la SD, validando si hay satélites
    float current_lat = gps.location.isValid() ? gps.location.lat() : 4.6582;
    float current_lng = gps.location.isValid() ? gps.location.lng() : -74.0931;

    // Ensamblaje dinámico de la fila, reemplazando el ",,," hardcodeado
    String fila = String(seconds_elapsed, 4) + "," + 
                  String(ax, 4) + "," + String(ay, 4) + "," + String(az, 4) + "," + 
                  String(gx, 4) + "," + String(gy, 4) + "," + String(gz, 4) + "," + 
                  evento_detectado + "," + 
                  String(current_lat, 6) + "," + String(current_lng, 6);
                  
    if (dataFile) {
      dataFile.println(fila);
      // Forzar escritura física en la SD cada vez que se cierra la ventana o hay evento
      if (evento_detectado != "" || muestras_recolectadas == 0) {
        dataFile.flush(); 
      }
    }
  }
}
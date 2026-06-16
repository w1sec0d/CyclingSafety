import time
import json
import random
import paho.mqtt.client as mqtt
import os
from dotenv import load_dotenv
load_dotenv()


# ── CONFIGURACIÓN DEL BACKEND ─────────────────────────────────────────────────
THINGSBOARD_HOST = os.getenv("THINGSBOARD_HOST")   
MQTT_PORT = int(os.getenv("MQTT_PORT")) 
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")  
# El tópico oficial de ThingsBoard para publicar telemetría
MQTT_TOPIC = os.getenv("MQTT_TOPIC")
# ──────────────────────────────────────────────────────────────────────────────

print(MQTT_PORT)
def simular_viaje():
    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2, client_id="Simulador_ESP32_Kubuntu")
    events = ["Emergency Brake", "Pothole", "Dodge"] 
    
    client.username_pw_set(ACCESS_TOKEN, password=None)
    
    print(f"🔗 Conectando al broker MQTT en {THINGSBOARD_HOST}:{MQTT_PORT}...")
    try:
        client.connect(THINGSBOARD_HOST, MQTT_PORT, keepalive=60)
    except Exception as e:
        print(f"❌ Error de conexión: {e}")
        return

    print("🚀 Simulador iniciado. Enviando datos a 50 Hz durante un evento de bache...")
    
    # Coordenadas base en Bogotá (cerca de la zona de estudio)
    lat_base = 4.6582
    lng_base = -74.0931
    
    start_time = time.time()
    
    # Simularemos 100 muestras (~2 segundos de datos a 50 Hz)
    for i in range(100):
        current_time = time.time()
        seconds_elapsed = current_time - start_time
        
        # Por defecto, el camino es plano (ruido cercano a 0 en aceleración lineal)
        ax = random.uniform(-0.2, 0.2)
        ay = random.uniform(-0.2, 0.2)
        az = random.uniform(-0.2, 0.2)
        annotation = random.choice(events) # Equivalente a null en JSON
        
        # Simular el impacto del "hueco" en la ventana central (muestras 45 a 55)
        if 45 <= i <= 55:
            ax = random.uniform(-2.5, 2.5)
            ay = random.uniform(-1.5, 1.5)
            az = random.uniform(-8.0, 11.0)  # Fuerte oscilación vertical en bache
            annotation = "hueco"              # El modelo detecta y etiqueta el evento
            
        # El giroscopio registra el bamboleo en rad/s
        gx = random.uniform(-0.1, 0.1)
        gy = random.uniform(-0.1, 0.1)
        gz = random.uniform(-0.1, 0.1)
        
        # Variación milimétrica del GPS simulando movimiento en la bici
        lat_base += random.uniform(-0.00001, 0.00001)
        lng_base += random.uniform(-0.00001, 0.00001)
        
        # Construcción de la carga útil JSON con la estructura exacta de tu modelo
        payload = {
            "timestamp": round(seconds_elapsed, 4),
            "ax": round(ax, 4),
            "ay": round(ay, 4),
            "az": round(az, 4),
            "gx": round(gx, 4),
            "gy": round(gy, 4),
            "gz": round(gz, 4),
            "annotation": annotation,
            "location_latitude": round(lat_base, 6),
            "location_longitude": round(lng_base, 6)
        }
        
        # Publicar el paquete JSON en el broker
        payload_str = json.dumps(payload)
        client.publish(MQTT_TOPIC, payload_str, qos=1)
        
        if annotation:
            print(f"⚠️ [ENVIADO - EVENTO DETECTADO]: {payload_str}")
        else:
            print(f"📊 [Enviado]: ts={payload['timestamp']:.2f}s | az={payload['az']:.2f}")
            
        # Forzar la frecuencia a 50 Hz (1 segundo / 50 muestras = 0.02 segundos de espera)
        time.sleep(0.02)
        
    client.disconnect()
    print("🏁 Simulación terminada. Revisa el dashboard de ThingsBoard.")

if __name__ == "__main__":
    simular_viaje()
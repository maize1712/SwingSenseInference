/* ESP32-S3 Port for TensorFlow Lite Micro
 * Provides the missing InitializeTarget() and DebugLog() functions
 * for ESP32-S3 boards (like Seeed XIAO ESP32-S3)
 */

#include <Arduino.h>

// DebugLog implementation for ESP32-S3
extern "C" void DebugLog(const char* s) {
  Serial.print(s);
}

// InitializeTarget implementation for ESP32-S3
namespace tflite {

constexpr unsigned long kSerialMaxInitWait = 4000;  // milliseconds

void InitializeTarget() {
  // Serial is already initialized by Arduino framework on ESP32-S3
  // But we'll set a reasonable baud rate if needed
  if (!Serial) {
    Serial.begin(115200);
    unsigned long start_time = millis();
    while (!Serial) {
      // allow for serial port synchronization
      if (millis() - start_time > kSerialMaxInitWait) {
        break;
      }
    }
  }
}

}  // namespace tflite

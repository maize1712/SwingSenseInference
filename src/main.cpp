#include <Arduino.h>
#include <Wire.h>

// OLED: U8g2 HW I2C, SSD1306 128x64 (the one on the XIAO expansion board)
#include <U8x8lib.h>

//// debounce


// Flag to keep classification result on screen until next recording
bool g_showingResult = false;

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Your TFLM model data (change header + symbol name to match your export)
#include "SwingSenseModelData.h"   // TODO: create like magic_wand_model_data.h
// extern const unsigned char g_tennis_swing_model_data[];
// extern const int g_tennis_swing_model_data_len;

// tflm_esp32_port.cpp should be compiled as a separate source file in the project
// so we can call tflite::InitializeTarget();

// ===================== Hardware config =====================

// IMU I2C pins & config (same as your data collection code)
#define I2C_SDA 5
#define I2C_SCL 6
#define ICM20600_ADDR 0x69

#define REG_PWR_MGMT_1   0x6B
#define REG_ACCEL_CONFIG 0x1C
#define REG_GYRO_CONFIG  0x1B
#define REG_ACCEL_XOUT_H 0x3B
#define REG_WHO_AM_I     0x75

// conversions for ±2g accel, ±250 dps gyro
static const float ACC_LSB_PER_G   = 16384.0f;
static const float GYR_LSB_PER_DPS = 131.0f;

// Button (same as data collection)
// D7 = GPIO 7 on Seeed XIAO ESP32-S3
#define RECORDING_PIN D7

// LEDs (pick two free pins on the header)
// D8 = GPIO 47, D9 = GPIO 48 on Seeed XIAO ESP32-S3
#define LED_GOOD_PIN 47
#define LED_BAD_PIN  48

// Sampling rate
const uint32_t PERIOD_MS = 25;   // ~40 Hz

// ===================== OLED =====================

// Uses hardware I2C on default Wire instance and address 0x3C
U8X8_SSD1306_128X64_NONAME_HW_I2C u8x8(U8X8_PIN_NONE);

// ===================== TFLM globals =====================

constexpr int kTensorArenaSize = 25 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

const tflite::Model*      g_model       = nullptr;
tflite::MicroInterpreter* g_interpreter = nullptr;

// We have 4 classes: 0 = GOOD, 1 = NO_FOLLOW, 2 = ENDS_LOW, 3 = STARTS_HIGH
constexpr int kLabelCount = 4;
const char* kLabels[kLabelCount] = {"GOOD", "NO_FOLLOW", "ENDS_LOW", "STARTS_HIGH"};

// You can either let the model handle normalization via a Normalization layer,
// or replicate the mean/std we used in Colab. Fill these from Colab if needed.
constexpr int kNumChannels = 6;
float kChannelMean[kNumChannels] = {
  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f   // TODO: replace with real means or leave as 0
};
float kChannelStd[kNumChannels] = {
  1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f   // TODO: replace with real stds or leave as 1
};

// ===================== IMU helpers =====================

static bool i2cWrite(uint8_t reg, uint8_t val) {
  Wire.beginTransmission((uint8_t)ICM20600_ADDR);
  Wire.write(reg);
  Wire.write(val);
  return Wire.endTransmission() == 0;
}

static bool i2cReadBytes(uint8_t reg, uint8_t* buf, size_t len) {
  Wire.beginTransmission((uint8_t)ICM20600_ADDR);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;

  if (Wire.requestFrom((uint8_t)ICM20600_ADDR, (uint8_t)len) != (int)len) return false;
  for (size_t i = 0; i < len; ++i) {
    buf[i] = Wire.read();
  }
  return true;
}

static bool icmInit(uint8_t& who) {
  who = 0;
  if (!i2cWrite(REG_PWR_MGMT_1, 0x01)) return false;   // wake, PLL
  delay(50);
  if (!i2cWrite(REG_ACCEL_CONFIG, 0x00)) return false; // ±2g
  if (!i2cWrite(REG_GYRO_CONFIG,  0x00)) return false; // ±250 dps
  delay(10);
  return i2cReadBytes(REG_WHO_AM_I, &who, 1);
}

static bool icmRead(float& ax,float& ay,float& az,
                    float& gx,float& gy,float& gz) {
  uint8_t raw[14];
  if (!i2cReadBytes(REG_ACCEL_XOUT_H, raw, sizeof(raw))) return false;

  auto s16 = [&](int i)->int16_t {
    return (int16_t)((raw[i] << 8) | raw[i+1]);
  };

  ax = s16(0)  / ACC_LSB_PER_G;
  ay = s16(2)  / ACC_LSB_PER_G;
  az = s16(4)  / ACC_LSB_PER_G;

  gx = s16(8)  / GYR_LSB_PER_DPS;
  gy = s16(10) / GYR_LSB_PER_DPS;
  gz = s16(12) / GYR_LSB_PER_DPS;

  return true;
}

// ===================== Recording buffer =====================

constexpr int kMaxSamples = 150;

// [time][channel] float buffer
float g_samples[kMaxSamples][kNumChannels];
int   g_sample_count = 0;

volatile bool g_isRecording   = false;
bool          g_lastRecState  = false;

// debounce
const unsigned long DEBOUNCE_MS = 100;
int                last_switch_state     = HIGH; // pullup
unsigned long      last_debounce_time_ms = 0;

// Flag to say “we finished a swing and need to classify it”
bool g_swingReady = false;

// ===================== Helpers =====================

void OledClear() {
  u8x8.clearDisplay();
  u8x8.setCursor(0,0);
}

void OledPrintCentered(const char* line1, const char* line2 = nullptr) {
  u8x8.clearDisplay();
  int len1 = strlen(line1);
  int col1 = max(0, (16 - len1) / 2);
  u8x8.setCursor(col1, 2);
  u8x8.print(line1);

  if (line2) {
    int len2 = strlen(line2);
    int col2 = max(0, (16 - len2) / 2);
    u8x8.setCursor(col2, 4);
    u8x8.print(line2);
  }
}

// Normalize one value (if you used mean/std in Colab)
float NormalizeChannel(float x, int ch) {
  float stdv = kChannelStd[ch];
  if (stdv == 0.0f) stdv = 1.0f;
  return (x - kChannelMean[ch]) / stdv;
}

// ===================== TFLM setup =====================

void SetupTflm() {
  tflite::InitializeTarget();

  Serial.println("Setting up TFLM model...");

  g_model = tflite::GetModel(g_swing_sense_model_data);
  if (g_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version mismatch!");
    while (1) delay(1000);
  }

  // Increase resolver capacity to accommodate all ops used by the model
  static tflite::MicroMutableOpResolver<16> micro_op_resolver;
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddMean();
  micro_op_resolver.AddMaxPool2D();
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddSoftmax();
  micro_op_resolver.AddMul();
  micro_op_resolver.AddAdd();
  // Some exported models use ExpandDims; ensure it's registered
  micro_op_resolver.AddExpandDims();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddQuantize();
  micro_op_resolver.AddDequantize();
  micro_op_resolver.AddLogistic();

  static tflite::MicroInterpreter static_interpreter(
      g_model, micro_op_resolver, tensor_arena, kTensorArenaSize);
  g_interpreter = &static_interpreter;

  if (g_interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("Tensor allocation failed!");
    while (1) delay(1000);
  }

  TfLiteTensor* input = g_interpreter->input(0);
  Serial.print("Input tensor type: ");
  Serial.println(input->type);
  Serial.print("Input dims: ");
  for (int i = 0; i < input->dims->size; ++i) {
    Serial.print(input->dims->data[i]);
    Serial.print(" ");
  }
  Serial.println();

  Serial.println("TFLM setup done.");
}

// ===================== Classification =====================

void RunClassification() {
  if (g_sample_count <= 0) {
    Serial.println("No samples captured; skipping classification.");
    return;
  }

  Serial.print("Raw sample_count = ");
  Serial.println(g_sample_count);

  // Prepare exactly 150 samples:
  float tmp[150][kNumChannels];

  if (g_sample_count >= kMaxSamples) {
    // Take the last 150 samples (swing usually ends with impact / follow-through)
    int start = g_sample_count - kMaxSamples;
    for (int i = 0; i < kMaxSamples; ++i) {
      for (int c = 0; c < kNumChannels; ++c) {
        tmp[i][c] = g_samples[start + i][c];
      }
    }
  } else {
    // Copy what we have and pad with zeros
    int i = 0;
    for (; i < g_sample_count; ++i) {
      for (int c = 0; c < kNumChannels; ++c) {
        tmp[i][c] = g_samples[i][c];
      }
    }
    for (; i < kMaxSamples; ++i) {
      for (int c = 0; c < kNumChannels; ++c) {
        tmp[i][c] = 0.0f;
      }
    }
  }

  // Copy into TFLM input (quantized int8)
  TfLiteTensor* input = g_interpreter->input(0);

  const float input_scale = input->params.scale;
  const int   input_zp    = input->params.zero_point;

  // Expected layout is [1, 150, 6] in row-major order
  int idx = 0;
  for (int t = 0; t < kMaxSamples; ++t) {
    for (int c = 0; c < kNumChannels; ++c) {
      float x = tmp[t][c];
      // Optional: comment this out if the model already has a Normalization layer
      x = NormalizeChannel(x, c);

      // quantize float -> int8
      int32_t q = static_cast<int32_t>(roundf(x / input_scale)) + input_zp;
      if (q < -128) q = -128;
      if (q > 127)  q = 127;
      input->data.int8[idx++] = static_cast<int8_t>(q);
    }
  }

  if (g_interpreter->Invoke() != kTfLiteOk) {
    Serial.println("ERROR: Inference failed!");
    return;
  }

  TfLiteTensor* output = g_interpreter->output(0);
  const float out_scale = output->params.scale;
  const int   out_zp    = output->params.zero_point;

  // Find argmax
  float best_score = -1e9;
  int   best_idx   = -1;

  for (int i = 0; i < kLabelCount; ++i) {
    int8_t q = output->data.int8[i];
    float  f = (static_cast<int>(q) - out_zp) * out_scale;  // dequantize
    if (f > best_score) {
      best_score = f;
      best_idx   = i;
    }
    Serial.print("Class ");
    Serial.print(kLabels[i]);
    Serial.print(": ");
    Serial.println(f, 4);
  }

  if (best_idx < 0) {
    Serial.println("No valid class index!");
    return;
  }
  
  const char* label = kLabels[best_idx];
  bool isGood = (best_idx == 0);   // index 0 = GOOD

  Serial.print("PREDICTION: ");
  Serial.print(label);
  Serial.print(" (confidence: ");
  Serial.print(best_score, 4);
  Serial.println(")");

  // LEDs (commented out until you have resistors)
  /*
  digitalWrite(LED_GOOD_PIN, isGood ? HIGH : LOW);
  digitalWrite(LED_BAD_PIN,  isGood ? LOW  : HIGH);
  */

  if (isGood) {
    OledPrintCentered("GOOD SWING", "Nice form!");
  } else {
    OledPrintCentered(label, "Needs work");
  }

  g_showingResult = true;
}

// ===================== Setup & Loop =====================

void setup() {
  // Serial FIRST - needed for all debug output
  Serial.begin(115200);
  delay(500);  // Give serial monitor time to connect
  Serial.println("\n\n========== SwingSense Starting ==========");

  // Button + LEDs
  pinMode(RECORDING_PIN, INPUT_PULLUP);
  // pinMode(LED_GOOD_PIN, OUTPUT);
  // pinMode(LED_BAD_PIN,  OUTPUT);
  // digitalWrite(LED_GOOD_PIN, LOW);
  // digitalWrite(LED_BAD_PIN,  LOW);
  Serial.println("[GPIO] Button and LED pins configured");

  // I2C + IMU
  Wire.begin(I2C_SDA, I2C_SCL, 400000);
  delay(100);

  uint8_t who = 0;
  if (icmInit(who)) {
    Serial.printf("[I2C] ICM20600 OK, WHO_AM_I=0x%02X\n", who);
  } else {
    Serial.println("[I2C] ICM init FAILED");
  }

  // OLED
  u8x8.begin();
  u8x8.setPowerSave(0);
  u8x8.setFont(u8x8_font_chroma48medium8_r);
  OledPrintCentered("SwingSense", "Press button");

  // TFLM - TEMPORARILY DISABLED to debug button
  SetupTflm();

  g_sample_count = 0;
  g_isRecording  = false;
  g_lastRecState = false;

  Serial.println("Ready. Press button to record a swing.");
  Serial.println("Button pin is GPIO 7 (D7)");
  // Serial.println("LED Good pin is GPIO 47 (D8)");
  // Serial.println("LED Bad pin is GPIO 48 (D9)");
}

void loop() {
  // -------- Debounce + toggle recording --------
  int pin_read = digitalRead(RECORDING_PIN);

  // Debug: Print pin state CONSTANTLY
  //Serial.print("Button = ");
  //Serial.println(pin_read);

  if (pin_read != last_switch_state) {
    Serial.print("[BTN] Pin changed to ");
    Serial.println(pin_read);
    last_debounce_time_ms = millis();
    last_switch_state     = pin_read;
  }

  bool toggled = false;

  if ((millis() - last_debounce_time_ms) > DEBOUNCE_MS) {
    bool newRecording = (pin_read == LOW); // active low
    if (newRecording != g_isRecording) {
      g_isRecording = newRecording;
      toggled = true;

      Serial.print("[REC] Recording toggled: ");
      Serial.println(g_isRecording ? "STARTED" : "STOPPED");
    }
  }

  // Handle state transitions
  if (toggled) {
    if (g_isRecording) {
      // Start of swing
      g_sample_count = 0;
      g_swingReady   = false;
      g_showingResult = false;  // Clear old result when starting new recording
      OledPrintCentered("Recording", "Swing now");
    } else {
      // End of swing
      g_swingReady = true;
      OledPrintCentered("Processing...", nullptr);
    }
  }

  // -------- Sampling while recording --------
  static uint32_t last_sample_ms = 0;
  uint32_t now = millis();

  if (g_isRecording) {
    if (now - last_sample_ms >= PERIOD_MS) {
      last_sample_ms = now;

      if (g_sample_count < kMaxSamples * 3) {
        float ax, ay, az, gx, gy, gz;
        if (icmRead(ax, ay, az, gx, gy, gz)) {
          if (g_sample_count < kMaxSamples) {
            g_samples[g_sample_count][0] = ax;
            g_samples[g_sample_count][1] = ay;
            g_samples[g_sample_count][2] = az;
            g_samples[g_sample_count][3] = gx;
            g_samples[g_sample_count][4] = gy;
            g_samples[g_sample_count][5] = gz;
            g_sample_count++;

            Serial.print("Sample ");
            Serial.print(g_sample_count);
            Serial.print(": ");
            Serial.print(ax, 3); Serial.print(", ");
            Serial.print(ay, 3); Serial.print(", ");
            Serial.print(az, 3); Serial.print(", ");
            Serial.print(gx, 3); Serial.print(", ");
            Serial.print(gy, 3); Serial.print(", ");
            Serial.println(gz, 3);
          }
        }
      }
    }
  }

  // -------- Run classification when a swing is ready --------
  if (g_swingReady && !g_isRecording) {
    g_swingReady = false;
    RunClassification();

  }
}
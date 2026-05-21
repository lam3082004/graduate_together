# Tài liệu ôn thi vấn đáp IOT — Bùi Tùng Lâm & Nguyễn Thế Vinh

---

# PHẦN 1: BÙI TÙNG LÂM — Cảm biến bụi mịn Sharp GP2Y1010AU0F (19%)

## 1.1. Phần cứng
- **Cảm biến:** Sharp GP2Y1010AU0F đo bụi mịn PM2.5
- **Chân điều khiển LED hồng ngoại:** GPIO 32 (`SHARP_LED_PIN`) — dùng chế độ `OUTPUT_OPEN_DRAIN`
- **Chân đọc điện áp đầu ra:** GPIO 35 (`SHARP_VO_PIN`) — chỉ đọc vào (ADC1)

> **Tại sao dùng OUTPUT_OPEN_DRAIN?**
> ESP32 chạy 3.3V nhưng cảm biến Sharp cần 5V. Chế độ `OUTPUT_OPEN_DRAIN`:
> - Khi xuất `HIGH` → chân thả nổi (ngắt hoàn toàn dòng) = LED TẮT
> - Khi xuất `LOW` → chân nối đất = LED BẬT
> Nhờ đó dùng được transistor/mạch 5V bên ngoài mà không phá ESP32.

---

## 1.2. Chu kỳ đọc LED hồng ngoại (quan trọng nhất)

Mỗi lần lấy 1 mẫu gồm 5 bước, tổng ~10ms:

| Bước | Hành động | Thời gian |
|------|-----------|-----------|
| 1 | Bật LED (`LOW`) | — |
| 2 | Chờ | 280 µs |
| 3 | Đọc ADC (`analogRead`) | — |
| 4 | Chờ | 40 µs |
| 5 | Tắt LED (`HIGH`) | — |
| 6 | Chờ hết chu kỳ | 9680 µs |
| **Tổng** | | **~10 ms/mẫu** |

> **Tại sao phải chờ 280µs rồi mới đọc?**
> Đây là thời gian LED ổn định và hạt bụi tán xạ đủ ánh sáng để cảm biến quang (phototransistor) cho ra điện áp chính xác. Đọc quá sớm sẽ bị nhiễu.

---

## 1.3. Lấy trung bình 20 mẫu (Moving Average)

```cpp
for (int i = 0; i < 20; i++) {
    digitalWrite(SHARP_LED_PIN, LOW);
    delayMicroseconds(280);
    int voMeasured = analogRead(SHARP_VO_PIN);  // ADC 12-bit: 0–4095
    delayMicroseconds(40);
    digitalWrite(SHARP_LED_PIN, HIGH);
    delayMicroseconds(9680);

    float calcVoltage = voMeasured * (3.3 / 4095.0);  // Đổi ADC → Volt
    sumVoltage += calcVoltage;
}
float avgVoltage = sumVoltage / 20;
```

> **Tại sao lấy 20 mẫu?** Để lọc nhiễu ADC và dao động ngẫu nhiên. Kết quả ổn định hơn nhiều so với đọc 1 lần.

---

## 1.4. Công thức chuyển đổi Voltage → PM2.5

```
Dust Density (mg/m³) = 0.17 × avgVoltage - 0.1
PM2.5 (µg/m³)       = dustDensity × 1000
```

- Công thức tuyến tính của **Chris Nafis**, phổ biến cho Sharp trên Arduino/ESP32
- Nếu `dustDensity < 0` → gán bằng 0 (không có bụi âm)
- **Đổi đơn vị:** `mg/m³ × 1000 = µg/m³` (đơn vị chuẩn PM2.5)

**Ví dụ tính tay:**
- `avgVoltage = 0.8V` → `dustDensity = 0.17×0.8 - 0.1 = 0.036 mg/m³` → `PM2.5 = 36 µg/m³`

---

## 1.5. Tính chỉ số AQI từ PM2.5 (chuẩn EPA Mỹ)

```
PM2.5 ≤ 12.0   → AQI = (50/12) × PM2.5                  → Tốt (xanh)
PM2.5 ≤ 35.4   → AQI = (49/23.3) × (PM2.5 - 12.1) + 51  → Trung bình (vàng)
PM2.5 ≤ 55.4   → AQI = ...+101                            → Kém (cam)
PM2.5 > 55.4   → ...                                      → Xấu/Nguy hại (đỏ)
```

---

## 1.6. Câu hỏi vấn đáp cho phần này

**Q: Tại sao GPIO 35 không dùng được OUTPUT?**
A: GPIO 34, 35 của ESP32 là **input-only**, không có driver output bên trong. Chỉ dùng để đọc ADC.

**Q: ADC của ESP32 có vấn đề gì?**
A: ADC 12-bit (0–4095), điện áp tham chiếu 3.3V. Nhưng **phi tuyến** ở đầu dải (gần 0V và gần 3.3V) nên cần calibration hoặc lấy trung bình nhiều mẫu.

**Q: Tại sao không dùng `delay()` mà dùng `delayMicroseconds()`?**
A: Chu kỳ LED tính bằng **micro giây** (µs). `delay()` chỉ đến mili giây (ms) — độ phân giải thô hơn 1000 lần, không đủ chính xác.

**Q: Nếu không bật/tắt LED theo đúng chu kỳ thì sao?**
A: Không đo được ánh sáng tán xạ từ hạt bụi, điện áp đầu ra sẽ bão hòa hoặc nhiễu loạn, kết quả PM2.5 sai hoàn toàn.

---

# PHẦN 2: NGUYỄN THẾ VINH — LCD, Nút nhấn, Buzzer, LED (19%)

## 2.1. Màn hình LCD 16x2 qua I2C

- **Thư viện:** `LiquidCrystal_I2C`
- **Địa chỉ I2C:** `0x27` (hoặc `0x3F` tùy module)
- **Cấu hình:** 16 cột × 2 hàng (`LCD_COLS=16`, `LCD_ROWS=2`)
- **Giao tiếp:** I2C — chỉ cần 2 dây (SDA, SCL) thay vì 6–8 dây của LCD song song

```cpp
LiquidCrystal_I2C lcd(0x27, 16, 2);
lcd.init();
lcd.backlight();
```

---

## 2.2. Ba màn hình (3 chế độ bách biến)

Biến `currentLcdMode` điều khiển: `0 → 1 → 2 → 0 → ...`

### Chế độ 0: Chất lượng không khí (`displayAirQuality`)
```
Dòng 1: PM:36.5   T:28C
Dòng 2: Gas:0.95  H:65%
```

### Chế độ 1: Ngày giờ (`displayTimeScreen`)
```
Dòng 1: Date: 21/05/2026
Dòng 2: Time: 10:30:45
```
- Lấy giờ từ **NTP Server** (`pool.ntp.org`) múi giờ GMT+7

### Chế độ 2: Thời tiết (`displayWeatherScreen`)
```
Dòng 1: Weather: Ha Noi
Dòng 2: 32.0C Clouds
```
- Dữ liệu từ **OpenWeatherMap API**, cập nhật mỗi 30 phút
- Nếu chưa có data → hiển thị `"Loading data..."`

---

## 2.3. Nút nhấn chuyển chế độ — Thuật toán Debounce phi tuần tự

**Vấn đề:** Nút cơ học khi nhấn/nhả tạo ra nhiều xung giả (bounce) trong ~50ms đầu.

**Giải pháp: Debounce dùng `millis()` (không dùng `delay()`)**

```cpp
void checkModeButton() {
  static unsigned long lastDebounce = 0;
  static int lastButtonState = HIGH;
  int buttonState = digitalRead(MODE_BUTTON_PIN); // GPIO 5, INPUT_PULLUP

  if (buttonState != lastButtonState) {
    lastDebounce = millis(); // Reset timer khi trạng thái thay đổi
  }

  if ((millis() - lastDebounce) > 50) { // Ổn định sau 50ms
    static int actualState = HIGH;
    if (buttonState != actualState) {
      actualState = buttonState;
      if (actualState == LOW) { // LOW = nhấn (vì INPUT_PULLUP)
        currentLcdMode = (currentLcdMode + 1) % 3; // Xoay vòng 0→1→2→0
        refreshLCD();
      }
    }
  }
  lastButtonState = buttonState;
}
```

> **Tại sao "phi tuần tự"?** Dùng `millis()` thay `delay(50)` nên hàm trả về ngay, không chặn `loop()`. Blynk và Timer vẫn chạy bình thường trong lúc chờ debounce.

> **Tại sao `INPUT_PULLUP`?** Không cần điện trở ngoài. Khi nút nhả → HIGH (3.3V qua trở kéo nội bộ). Khi nút nhấn → kéo xuống GND = LOW.

---

## 2.4. Buzzer cảnh báo ngắt quãng

Timer gọi `handleWarningBuzzer()` mỗi **500ms**:

```cpp
void handleWarningBuzzer() {
  static bool buzzerState = false;
  if (warningLevel == 0) {
    digitalWrite(BUZZER_PIN, LOW);       // Tắt hoàn toàn
  } else if (warningLevel == 1) {
    buzzerState = !buzzerState;
    digitalWrite(BUZZER_PIN, buzzerState ? HIGH : LOW); // Bíp ngắt quãng 500ms
  } else if (warningLevel == 2) {
    digitalWrite(BUZZER_PIN, HIGH);      // Kêu liên tục
  }
}
```

| Mức cảnh báo | LED sáng | Buzzer | Điều kiện |
|---|---|---|---|
| 0 — An toàn | Xanh lá | Tắt | PM2.5 ≤ 35 và GasRatio ≥ 0.8 |
| 1 — Cảnh báo | Vàng | Bíp 500ms | PM2.5 > 35 hoặc GasRatio < 0.8 |
| 2 — Nguy hiểm | Đỏ | Kêu liên tục | PM2.5 > 75 hoặc GasRatio < 0.5 |

---

## 2.5. Hệ thống 3 LED (GPIO 25, 26, 27)

```cpp
// Chỉ một LED sáng tại một thời điểm
if (warningLevel == 2) {
    digitalWrite(LED_RED_PIN, HIGH);    // GPIO 27
    digitalWrite(LED_YELLOW_PIN, LOW);
    digitalWrite(LED_GREEN_PIN, LOW);
} else if (warningLevel == 1) {
    digitalWrite(LED_RED_PIN, LOW);
    digitalWrite(LED_YELLOW_PIN, HIGH); // GPIO 26
    digitalWrite(LED_GREEN_PIN, LOW);
} else {
    digitalWrite(LED_RED_PIN, LOW);
    digitalWrite(LED_YELLOW_PIN, LOW);
    digitalWrite(LED_GREEN_PIN, HIGH);  // GPIO 25
}
```

---

## 2.6. Bộ lọc chống nhiễu mức cảnh báo (Hysteresis + Tích lũy thời gian)

Để tránh LED/Buzzer nhấp nháy khi giá trị dao động quanh ngưỡng:

1. **Hysteresis (khoảng đệm):** Ngưỡng giảm cấp = Ngưỡng tăng cấp − Margin
   - PM2.5 margin: 3 µg/m³
   - GasRatio margin: 0.05
2. **Tích lũy thời gian:** Phải ổn định **3 chu kỳ liên tiếp (~6 giây)** mới đổi mức

---

## 2.7. Câu hỏi vấn đáp cho phần này

**Q: Tại sao dùng `millis()` thay `delay()` cho debounce?**
A: `delay()` chặn toàn bộ `loop()` — Blynk mất kết nối, timer không chạy, buzzer không bíp được. `millis()` cho phép code tiếp tục chạy trong khi chờ.

**Q: I2C LCD khác LCD song song thế nào?**
A: LCD song song cần 6–8 chân GPIO. I2C chỉ cần 2 dây (SDA + SCL) nhờ chip chuyển đổi PCF8574 trên module. ESP32 dùng GPIO 21 (SDA) và GPIO 22 (SCL) mặc định.

**Q: Làm sao biết địa chỉ I2C của LCD là 0x27?**
A: Chạy I2C Scanner (quét từ 0x00 đến 0x7F, thử `Wire.beginTransmission()`) hoặc xem jumper A0/A1/A2 trên chip PCF8574. Mặc định tất cả jumper = 0 → địa chỉ 0x27.

**Q: Tại sao buzzer dùng timer 500ms thay vì đặt trong loop()?**
A: Nếu đặt trong `loop()` không có delay sẽ toggle hàng nghìn lần/giây — tai người không nghe được và buzzer có thể hỏng. Timer 500ms cho tiếng bíp rõ ràng, không chặn code khác.

**Q: `static` trong hàm có nghĩa gì?**
A: Biến `static` bên trong hàm chỉ khởi tạo 1 lần, giữ giá trị giữa các lần gọi — tương đương biến global nhưng chỉ truy cập được trong hàm đó.

---

# PHẦN 3: TỔNG QUAN HỆ THỐNG (cho câu hỏi tích hợp)

## Sơ đồ luồng dữ liệu

```
[DHT11]──────────────┐
[MQ135]──────────────┤
[Sharp GP2Y1010]─────┼──→ ESP32 xử lý ──→ Tính warningLevel
[OpenWeatherMap API]─┘         │
                               ├──→ LCD 16x2 (3 chế độ)
                               ├──→ LED Xanh/Vàng/Đỏ
                               ├──→ Buzzer (ngắt quãng / liên tục)
                               └──→ Blynk Cloud → Push Notification điện thoại
```

## Bảng chân GPIO tổng hợp

| GPIO | Tên | Chức năng |
|------|-----|-----------|
| 32 | SHARP_LED_PIN | Điều khiển LED hồng ngoại cảm biến bụi |
| 35 | SHARP_VO_PIN | Đọc ADC điện áp cảm biến bụi (input-only) |
| 34 | MQ135_PIN | Đọc ADC tín hiệu khí độc (input-only) |
| 4 | DHT_PIN | Data DHT11 |
| 5 | MODE_BUTTON_PIN | Nút chuyển chế độ LCD |
| 0 | RESET_BUTTON_PIN | Nút reset WiFi/Blynk (giữ 5 giây) |
| 25 | LED_GREEN_PIN | LED xanh — An toàn |
| 26 | LED_YELLOW_PIN | LED vàng — Cảnh báo |
| 27 | LED_RED_PIN | LED đỏ — Nguy hiểm |
| 14 | BUZZER_PIN | Còi báo |
| 21 | SDA (mặc định) | I2C Data — LCD |
| 22 | SCL (mặc định) | I2C Clock — LCD |

## Timer (BlynkTimer)

| Hàm | Chu kỳ | Mục đích |
|-----|--------|----------|
| `sendSensorData()` | 2000ms | Đọc tất cả cảm biến, gửi Blynk, cập nhật LCD, tính warningLevel |
| `handleWarningBuzzer()` | 500ms | Toggle buzzer ngắt quãng ở mức 1 |
| `fetchWeatherData()` | 1800000ms (30 phút) | Gọi OpenWeatherMap API |

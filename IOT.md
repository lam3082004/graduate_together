# Tài liệu ôn thi vấn đáp IOT — Bùi Tùng Lâm & Nguyễn Thế Vinh

---

# PHẦN 1: BÙI TÙNG LÂM — Cảm biến bụi mịn Sharp GP2Y1010AU0F (19%)

## 1.1. Phần cứng & cấu hình chân

| Chân ESP32 | Define | Hướng | Ghi chú |
|---|---|---|---|
| GPIO 32 | `SHARP_LED_PIN` | OUTPUT_OPEN_DRAIN | Điều khiển LED hồng ngoại |
| GPIO 35 | `SHARP_VO_PIN` | INPUT (ADC1) | Đọc điện áp đầu ra, input-only |

### Tại sao dùng `OUTPUT_OPEN_DRAIN`?
ESP32 chạy 3.3V nhưng cảm biến Sharp thiết kế cho 5V:
- `HIGH` → chân **thả nổi** (ngắt dòng) → LED **TẮT**
- `LOW` → chân nối GND → LED **BẬT**

Nhờ vậy mạch 5V bên ngoài điều khiển được mà không phá GPIO ESP32.

---

## 1.2. Code khởi tạo cảm biến (`sensors.h` — hàm `initSensor`)

```cpp
void initSensor() {
  // OUTPUT_OPEN_DRAIN: HIGH = thả nổi (TẮT LED), LOW = kéo GND (BẬT LED)
  // Cần thiết vì mạch LED Sharp chạy 5V, ESP32 chỉ xuất 3.3V
  pinMode(SHARP_LED_PIN, OUTPUT_OPEN_DRAIN);

  // Mặc định TẮT LED khi mới khởi động, tránh đọc nhiễu
  digitalWrite(SHARP_LED_PIN, HIGH);

  // GPIO 35 là input-only trên ESP32 — không có OUTPUT driver
  pinMode(SHARP_VO_PIN, INPUT);

  dht.begin(); // Khởi động DHT11 trên GPIO 4
}
```

---

## 1.3. Chu kỳ đọc 10ms — trái tim của cảm biến bụi

Nguyên lý: LED hồng ngoại chiếu vào buồng đo, hạt bụi tán xạ ánh sáng sang phototransistor, điện áp đầu ra tỉ lệ với nồng độ bụi. **Phải đọc ADC đúng thời điểm LED đang bật ổn định.**

```
|←── 280µs ──→|← 40µs →|←────── 9680µs ──────→|
  LED BẬT      ĐỌC ADC   LED TẮT, chờ hết 10ms
```

### Code đọc 20 mẫu lấy trung bình (`sensors.h` — hàm `readSensorData`)

```cpp
void readSensorData(float &pm25, int &aqi) {
  float sumVoltage = 0;
  int sampleCount = 20; // Lấy 20 mẫu để triệt nhiễu ADC
  int validSamples = 0;

  for (int i = 0; i < sampleCount; i++) {

    // ① Bật LED hồng ngoại (kéo xuống GND)
    digitalWrite(SHARP_LED_PIN, LOW);

    // ② Chờ 280µs để LED ổn định và phototransistor phản ứng
    //    Đọc sớm hơn → điện áp chưa đạt đỉnh → kết quả thấp hơn thực tế
    delayMicroseconds(280);

    // ③ Đọc ADC 12-bit (0–4095), tương ứng 0–3.3V
    //    GPIO 35 thuộc ADC1 — dùng ADC1 khi WiFi hoạt động (ADC2 bị vô hiệu hóa)
    int voMeasured = analogRead(SHARP_VO_PIN);

    // ④ Chờ thêm 40µs trước khi tắt (theo datasheet Sharp)
    delayMicroseconds(40);

    // ⑤ Tắt LED (thả nổi — HIGH trong chế độ open-drain)
    digitalWrite(SHARP_LED_PIN, HIGH);

    // ⑥ Chờ hết phần còn lại của chu kỳ 10ms (280+40+9680 = 10000µs)
    delayMicroseconds(9680);

    // Đổi giá trị ADC sang Volt: V = ADC × (3.3V / 4095)
    float calcVoltage = voMeasured * (3.3 / 4095.0);
    sumVoltage += calcVoltage;
    validSamples++;
  }

  // Tránh chia cho 0 nếu vòng lặp bị lỗi
  if (validSamples == 0) validSamples = 1;

  float avgVoltage = sumVoltage / validSamples; // Điện áp trung bình 20 mẫu

  // Công thức Chris Nafis (tuyến tính, phổ biến cho Sharp trên ESP/Arduino)
  // dustDensity (mg/m³) = 0.17 × V - 0.1
  float dustDensity = 0.17 * avgVoltage - 0.1;

  // Kẹp về 0 nếu tính ra âm (không khí rất sạch, điện áp thấp)
  if (dustDensity < 0) dustDensity = 0.0;

  // Đổi mg/m³ → µg/m³ (nhân 1000) — đơn vị chuẩn quốc tế cho PM2.5
  pm25 = dustDensity * 1000.0;
  aqi  = calculateAQI(pm25);
}
```

---

## 1.4. Công thức tính AQI từ PM2.5 (chuẩn EPA Mỹ)

```cpp
int calculateAQI(float pm25) {
  // Mỗi dải PM2.5 ánh xạ tuyến tính sang dải AQI tương ứng
  if (pm25 <= 12.0)  return round((50.0 / 12.0)  * pm25);                        // Tốt
  if (pm25 <= 35.4)  return round((49.0 / 23.3)  * (pm25 - 12.1) + 51);         // Trung bình
  if (pm25 <= 55.4)  return round((49.0 / 19.9)  * (pm25 - 35.5) + 101);        // Kém
  if (pm25 <= 150.4) return round((49.0 / 94.9)  * (pm25 - 55.5)  + 151);       // Xấu
  if (pm25 <= 250.4) return round((99.0 / 99.9)  * (pm25 - 150.5) + 201);       // Rất xấu
  if (pm25 <= 350.4) return round((99.0 / 99.9)  * (pm25 - 250.5) + 301);       // Nguy hại
  return 500; // Vượt thang đo — cực kỳ nguy hại
}
```

**Ví dụ tính tay:**
- `avgVoltage = 0.8V` → `dustDensity = 0.17×0.8 − 0.1 = 0.036 mg/m³` → `PM2.5 = 36 µg/m³` → AQI ≈ 102

---

## 1.5. Câu hỏi vấn đáp — Phần cảm biến bụi

**Q1: Tại sao phải chờ 280µs rồi mới đọc ADC?**
A: Đây là thời gian LED hồng ngoại phát sáng ổn định và phototransistor trong cảm biến hội tụ dòng đủ để cho ra điện áp chính xác. Theo datasheet Sharp, đọc sớm hơn sẽ lấy điện áp chưa đạt đỉnh — kết quả bụi thấp hơn thực tế.

**Q2: Tại sao GPIO 35 không thể OUTPUT?**
A: GPIO 34 và 35 của ESP32 là **input-only pins** — không có transistor driver output bên trong chip. Chúng chỉ kết nối vào ADC1, dùng để đọc tín hiệu analog.

**Q3: Tại sao phải dùng ADC1 (GPIO 32–39) mà không dùng ADC2?**
A: Khi WiFi đang hoạt động, **ADC2 bị vô hiệu hóa** bởi RF module của ESP32. ADC1 không bị ảnh hưởng, nên tất cả cảm biến analog (Sharp, MQ135) đều dùng ADC1.

**Q4: ADC ESP32 có vấn đề gì cần lưu ý?**
A: ADC 12-bit (0–4095) nhưng **phi tuyến** ở đầu dải (dưới ~0.1V và trên ~3.1V) — độ chính xác kém. Giải pháp: lấy trung bình nhiều mẫu, tránh đo ở đầu dải, hoặc dùng hàm `analogReadMilliVolts()` đã có calibration sẵn.

**Q5: Tại sao lấy 20 mẫu thay vì 1 mẫu?**
A: ADC của ESP32 có nhiễu nội bộ và hệ thống còn WiFi, Blynk gây nhiễu điện. Trung bình 20 mẫu giảm nhiễu ngẫu nhiên xuống ~4.5 lần (theo √N), giúp PM2.5 ổn định không nhảy liên tục.

**Q6: Nếu cảm biến đo được điện áp thấp hơn điện áp không bụi thì sao?**
A: `dustDensity` tính ra âm → code kẹp về 0. Nghĩa là không khí sạch hơn điểm hiệu chuẩn, PM2.5 = 0 µg/m³.

**Q7: Công thức `0.17 × V − 0.1` lấy từ đâu?**
A: Từ nghiên cứu thực nghiệm của **Chris Nafis** — đo nhiều mức bụi thực tế rồi fit đường thẳng vào đồ thị Voltage vs Dust Density của Sharp GP2Y1010. Không phải công thức trong datasheet gốc (datasheet Sharp dùng đồ thị, không cho công thức tường minh).

**Q8: Đơn vị mg/m³ và µg/m³ khác nhau thế nào? Chuẩn nào dùng µg/m³?**
A: 1 mg/m³ = 1000 µg/m³. Chuẩn WHO và EPA Mỹ dùng **µg/m³** cho PM2.5. Ngưỡng WHO 24h: 15 µg/m³. Ngưỡng Việt Nam QCVN 05:2023: 50 µg/m³.

---

# PHẦN 2: NGUYỄN THẾ VINH — LCD, Nút nhấn, Buzzer, LED (19%)

## 2.1. LCD 16x2 qua I2C — code khởi tạo (`display.h` — hàm `initDisplay`)

```cpp
// Khai báo đối tượng LCD: địa chỉ I2C 0x27, 16 cột, 2 hàng
// Module LCD I2C dùng chip PCF8574 để chuyển I2C → 8-bit parallel cho LCD HD44780
LiquidCrystal_I2C lcd(LCD_I2C_ADDR, LCD_COLS, LCD_ROWS); // 0x27, 16, 2

void initDisplay() {
  lcd.init();      // Khởi động LCD qua I2C (gửi lệnh khởi tạo HD44780)
  lcd.backlight(); // Bật đèn nền (điều khiển transistor trên module PCF8574)

  // Màn hình chào mừng giữ 2 giây để người dùng thấy hệ thống đang boot
  lcd.setCursor(0, 0);
  lcd.print("Air Quality");
  lcd.setCursor(0, 1);
  lcd.print("Monitor v1.0");
  delay(2000);
  lcd.clear(); // Xóa màn hình trước khi vào vòng lặp chính
}
```

---

## 2.2. Ba màn hình hiển thị

### Chế độ 0: Chất lượng không khí (`displayAirQuality`)

```cpp
void displayAirQuality(float pm25, float gasRatio, float temp, float hum) {
  lcd.clear();

  // Dòng 0: "PM:36.5   T:28C"
  lcd.setCursor(0, 0);
  lcd.print("PM:");
  lcd.print(pm25, 1);    // 1 chữ số thập phân
  lcd.setCursor(9, 0);   // Căn cột 9 để T: luôn ở vị trí cố định
  lcd.print(" T:");
  if (!isnan(temp)) lcd.print(temp, 0); // 0 chữ số thập phân cho nhiệt độ
  else lcd.print("--");                 // DHT11 lỗi → hiển thị "--"
  lcd.print("C");

  // Dòng 1: "Gas:0.95  H:65%"
  lcd.setCursor(0, 1);
  lcd.print("Gas:");
  lcd.print(gasRatio, 2); // 2 chữ số thập phân
  lcd.setCursor(9, 1);
  lcd.print(" H:");
  if (!isnan(hum)) lcd.print(hum, 0);
  else lcd.print("--");
  lcd.print("%");
}
```

### Chế độ 1: Đồng hồ NTP (`displayTimeScreen`)

```cpp
void displayTimeScreen() {
  struct tm timeinfo;
  // getLocalTime() lấy giờ từ bộ nhớ NTP đã sync — không gọi mạng lại
  if (!getLocalTime(&timeinfo)) {
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("Syncing Time..."); // Chưa sync được NTP lần nào
    return;
  }
  lcd.clear();
  // tm_mon bắt đầu từ 0 (tháng 1 = 0) nên phải +1
  // tm_year tính từ 1900 nên phải +1900
  lcd.setCursor(0, 0);
  lcd.printf("Date: %02d/%02d/%04d", timeinfo.tm_mday, timeinfo.tm_mon + 1, timeinfo.tm_year + 1900);
  lcd.setCursor(0, 1);
  lcd.printf("Time: %02d:%02d:%02d", timeinfo.tm_hour, timeinfo.tm_min, timeinfo.tm_sec);
}
```

### Chế độ 2: Thời tiết OpenWeatherMap (`displayWeatherScreen`)

```cpp
void displayWeatherScreen(float wTemp, String wDesc) {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Weather: Ha Noi");

  // wDesc rỗng = chưa gọi API lần nào (mới boot hoặc mạng lỗi)
  if (wDesc == "") {
    lcd.setCursor(0, 1);
    lcd.print("Loading data...");
    return;
  }

  // Ví dụ: "32.0C Clouds"
  lcd.setCursor(0, 1);
  lcd.print(wTemp, 1);
  lcd.print("C ");
  lcd.print(wDesc); // "Clear", "Clouds", "Rain"...
}
```

---

## 2.3. Nút nhấn Debounce phi tuần tự (`main.cpp` — hàm `checkModeButton`)

**Vấn đề bounce:** Tiếp điểm cơ học rung ~50ms khi nhấn/nhả → ESP32 đọc hàng chục lần nhấn giả.

```cpp
void checkModeButton() {
  // static: biến giữ giá trị giữa các lần gọi hàm (không reset mỗi lần)
  static unsigned long lastDebounce = 0;  // Thời điểm lần cuối trạng thái thay đổi
  static int lastButtonState = HIGH;       // Trạng thái vật lý lần trước

  // GPIO 5, INPUT_PULLUP: nhả = HIGH (3.3V qua trở nội bộ), nhấn = LOW (kéo GND)
  int buttonState = digitalRead(MODE_BUTTON_PIN);

  // Phát hiện cạnh thay đổi → reset timer debounce
  if (buttonState != lastButtonState) {
    lastDebounce = millis();
  }

  // Chỉ xử lý khi trạng thái đã ổn định 50ms (hết bounce)
  if ((millis() - lastDebounce) > 50) {
    static int actualState = HIGH; // Trạng thái đã được xác nhận (qua debounce)

    if (buttonState != actualState) {
      actualState = buttonState;

      if (actualState == LOW) { // Cạnh xuống = người dùng vừa nhấn
        // Xoay vòng 0 → 1 → 2 → 0 bằng modulo
        currentLcdMode = (currentLcdMode + 1) % 3;

        // Chuyển sang chế độ thời tiết lần đầu → tải data ngay
        if (currentLcdMode == 2 && g_weatherDesc == "") {
          fetchWeatherData();
        }

        refreshLCD(); // Cập nhật LCD ngay lập tức, không chờ timer 2s
      }
    }
  }
  lastButtonState = buttonState; // Lưu lại để so sánh lần sau
}
```

> **"Phi tuần tự"** = không dùng `delay(50)` để chờ — hàm luôn trả về ngay. `loop()` tiếp tục gọi `Blynk.run()`, `timer.run()` bình thường. Thời gian chờ được tracking bằng `millis()`.

---

## 2.4. Buzzer cảnh báo ngắt quãng (`main.cpp` — hàm `handleWarningBuzzer`)

```cpp
// Được gọi bởi BlynkTimer mỗi 500ms — KHÔNG chặn code khác
void handleWarningBuzzer() {
  static bool buzzerState = false; // Nhớ trạng thái bật/tắt hiện tại

  if (warningLevel == 0) {
    // An toàn: tắt hoàn toàn
    digitalWrite(BUZZER_PIN, LOW);

  } else if (warningLevel == 1) {
    // Cảnh báo: toggle mỗi 500ms → tiếng bíp bíp
    // Mỗi lần hàm được gọi (500ms/lần) thì đổi trạng thái một lần
    buzzerState = !buzzerState;
    digitalWrite(BUZZER_PIN, buzzerState ? HIGH : LOW);

  } else if (warningLevel == 2) {
    // Nguy hiểm: kêu liên tục (không toggle)
    digitalWrite(BUZZER_PIN, HIGH);
  }
}
```

---

## 2.5. Hệ thống 3 LED + logic cảnh báo (`main.cpp` — trong `sendSensorData`)

```cpp
// Chỉ một LED sáng tại một thời điểm
if (warningLevel == 2) {
  digitalWrite(LED_RED_PIN,    HIGH); // GPIO 27 — Nguy hiểm
  digitalWrite(LED_YELLOW_PIN, LOW);
  digitalWrite(LED_GREEN_PIN,  LOW);
} else if (warningLevel == 1) {
  digitalWrite(LED_RED_PIN,    LOW);
  digitalWrite(LED_YELLOW_PIN, HIGH); // GPIO 26 — Cảnh báo
  digitalWrite(LED_GREEN_PIN,  LOW);
} else {
  digitalWrite(LED_RED_PIN,    LOW);
  digitalWrite(LED_YELLOW_PIN, LOW);
  digitalWrite(LED_GREEN_PIN,  HIGH); // GPIO 25 — An toàn
}
```

### Bảng ngưỡng cảnh báo

| Mức | LED | Buzzer | Điều kiện tăng | Điều kiện giảm |
|---|---|---|---|---|
| 0 — An toàn | Xanh lá | Tắt | PM2.5 > 35 hoặc Gas < 0.8 | — |
| 1 — Cảnh báo | Vàng | Bíp 500ms | PM2.5 > 75 hoặc Gas < 0.5 | PM2.5 ≤ 32 và Gas ≥ 0.85 |
| 2 — Nguy hiểm | Đỏ | Liên tục | — | PM2.5 ≤ 72 và Gas ≥ 0.55 |

---

## 2.6. Bộ lọc chống nhiễu mức cảnh báo (`main.cpp` — trong `sendSensorData`)

### Hysteresis (khoảng đệm)
Ngưỡng để **xuống** cấp thấp hơn ngưỡng để **lên** một khoảng margin:
- PM2.5 margin: ±3 µg/m³
- GasRatio margin: ±0.05

```cpp
float pmMargin = 3.0;   // µg/m³
float gasMargin = 0.05;

// Ví dụ khi đang ở mức 2 (Nguy hiểm):
// Chỉ xuống mức 1 nếu PM2.5 ≤ 75-3=72 VÀ gasRatio ≥ 0.5+0.05=0.55
// (không phải đúng ngưỡng 75 — tránh dao động lên xuống liên tục)
```

### Tích lũy thời gian (3 chu kỳ ≈ 6 giây)

```cpp
static int pendingLevel    = warningLevel; // Mức đang chờ xác nhận
static int levelStableCount = 0;           // Số chu kỳ liên tiếp ổn định

if (targetLevel != warningLevel) {
  if (targetLevel == pendingLevel) {
    levelStableCount++;
    if (levelStableCount >= 3) {   // Phải ổn định 3 chu kỳ × 2s = 6 giây
      warningLevel = targetLevel;  // Lúc này mới thực sự đổi mức
      levelStableCount = 0;
    }
  } else {
    pendingLevel = targetLevel;
    levelStableCount = 1; // Bắt đầu đếm lại
  }
} else {
  pendingLevel = warningLevel;
  levelStableCount = 0; // Về mức cũ → reset bộ đếm
}
```

---

## 2.7. Câu hỏi vấn đáp — Phần LCD, Buzzer, LED

**Q1: Tại sao dùng `millis()` thay `delay()` cho debounce?**
A: `delay(50)` chặn toàn bộ `loop()` — trong 50ms đó Blynk mất kết nối, timer buzzer không chạy, không đọc được cảm biến. `millis()` chỉ ghi thời điểm rồi trả về ngay, code tiếp tục chạy bình thường.

**Q2: I2C LCD tiết kiệm chân GPIO như thế nào?**
A: LCD song song HD44780 cần **6 chân** (RS, EN, D4–D7). Module I2C dùng chip **PCF8574** chuyển đổi → chỉ cần **2 chân** SDA (GPIO 21) và SCL (GPIO 22). Tiết kiệm 4 chân GPIO cho các ngoại vi khác.

**Q3: Làm sao biết địa chỉ I2C của LCD là 0x27?**
A: Chip PCF8574 có 3 chân địa chỉ A0, A1, A2. Khi tất cả nối GND (mặc định) → địa chỉ = 0x20 + 0b111 = **0x27**. Nếu có jumper thay đổi có thể là 0x3F. Xác nhận bằng I2C Scanner (`Wire.beginTransmission()` quét 0x00–0x7F).

**Q4: `static` bên trong hàm có nghĩa gì?**
A: Biến `static` trong hàm chỉ khởi tạo **1 lần duy nhất** (lần đầu gọi hàm), giữ giá trị giữa các lần gọi. Ví dụ: `static bool buzzerState = false` — biến này nhớ trạng thái bật/tắt từ lần gọi trước mà không cần biến global.

**Q5: Tại sao buzzer dùng Timer 500ms thay vì toggle trong `loop()`?**
A: `loop()` chạy hàng nghìn lần/giây — toggle ở đó tạo ra tần số âm thanh hàng kHz (tiếng the thé, không phải bíp). Timer 500ms cho tiếng **bíp… bíp…** nghe rõ ràng. Ngoài ra timer không chặn `loop()`.

**Q6: Hysteresis là gì và tại sao cần nó?**
A: Hysteresis là kỹ thuật dùng **hai ngưỡng khác nhau** cho chiều tăng và chiều giảm. Ví dụ: PM2.5 > 35 → tăng mức; nhưng phải PM2.5 < 32 mới giảm mức. Nếu không có hysteresis, khi giá trị dao động quanh 35 thì LED/Buzzer nhấp nháy liên tục (flip-flop).

**Q7: Tại sao cần thêm bộ lọc tích lũy 6 giây trên đỉnh Hysteresis?**
A: Hysteresis chỉ lọc dao động tại một thời điểm. Nhưng nếu cảm biến bị nhiễu 1 chu kỳ vượt ngưỡng rồi về ngay, không nên cảnh báo. Bộ đếm 3 chu kỳ đảm bảo ô nhiễm phải **thực sự kéo dài 6 giây** mới kích hoạt — tránh báo động giả.

**Q8: Tại sao chuyển sang chế độ thời tiết cần gọi `fetchWeatherData()` ngay?**
A: Timer cập nhật thời tiết chạy **mỗi 30 phút**. Khi mới boot, `g_weatherDesc = ""` nên LCD sẽ hiển thị "Loading data..." mãi. Khi nhấn nút chuyển sang chế độ thời tiết lần đầu, code tải data ngay lập tức để người dùng thấy kết quả ngay.

**Q9: `tm_mon + 1` và `tm_year + 1900` trong code đồng hồ là sao?**
A: Struct `tm` trong C có quy ước lạ: `tm_mon` từ 0–11 (tháng 1 = 0), `tm_year` tính từ năm 1900. Phải cộng chỉnh khi hiển thị để ra ngày tháng năm đúng.

**Q10: `INPUT_PULLUP` là gì? Tại sao không dùng điện trở ngoài?**
A: `INPUT_PULLUP` kích hoạt trở kéo lên nội bộ của ESP32 (~45kΩ). Khi nút nhả → chân đọc được 3.3V (HIGH). Khi nút nhấn → kéo xuống GND (LOW). Không cần điện trở ngoài → đơn giản mạch, tiết kiệm linh kiện.

---

# PHẦN 3: TỔNG QUAN HỆ THỐNG

## Sơ đồ luồng dữ liệu

```
[Sharp GP2Y1010]──ADC──┐
[MQ-135]──────────ADC──┤
[DHT11]────────────────┼──→ ESP32 (xử lý + lọc) ──→ tính warningLevel
[OpenWeatherMap API]───┘              │
                                      ├──→ LCD 16x2 I2C (3 chế độ)
                                      ├──→ LED Xanh / Vàng / Đỏ
                                      ├──→ Buzzer (tắt / bíp 500ms / liên tục)
                                      └──→ Blynk Cloud → Push Notification điện thoại
```

## Bảng chân GPIO đầy đủ

| GPIO | Define | Chức năng |
|------|--------|-----------|
| 32 | SHARP_LED_PIN | OUTPUT_OPEN_DRAIN — Điều khiển LED hồng ngoại Sharp |
| 35 | SHARP_VO_PIN | INPUT / ADC1 — Đọc điện áp Sharp (input-only) |
| 34 | MQ135_PIN | INPUT / ADC1 — Đọc tín hiệu MQ-135 (input-only) |
| 4 | DHT_PIN | Data DHT11 |
| 5 | MODE_BUTTON_PIN | INPUT_PULLUP — Nút chuyển chế độ LCD |
| 0 | RESET_BUTTON_PIN | INPUT_PULLUP — Giữ 5s để reset WiFi + Blynk |
| 25 | LED_GREEN_PIN | OUTPUT — LED xanh (An toàn) |
| 26 | LED_YELLOW_PIN | OUTPUT — LED vàng (Cảnh báo) |
| 27 | LED_RED_PIN | OUTPUT — LED đỏ (Nguy hiểm) |
| 14 | BUZZER_PIN | OUTPUT — Còi báo |
| 21 | SDA (mặc định) | I2C Data — LCD |
| 22 | SCL (mặc định) | I2C Clock — LCD |

## Bảng Timer (BlynkTimer)

| Hàm | Chu kỳ | Nhiệm vụ |
|-----|--------|----------|
| `sendSensorData()` | 2000ms | Đọc Sharp + MQ135 + DHT11, gửi Blynk, cập nhật LCD, tính warningLevel, bật LED |
| `handleWarningBuzzer()` | 500ms | Toggle buzzer ngắt quãng ở mức cảnh báo 1 |
| `fetchWeatherData()` | 1800000ms | Gọi OpenWeatherMap API, cập nhật `g_weatherTemp` và `g_weatherDesc` |

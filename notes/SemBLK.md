Ok, mình chuyển sang dạng **mũi tên ngang (linear pipeline)** cho dễ nhìn + dễ đưa vào paper.

---

# 🧭 1. Semantic Communication (baseline)

```
X → Encoder (E) → Z → Modulation → X_s → Channel → Y_s → Demodulation → Ẑ → Decoder (D) → X̂
```

👉 Ý nghĩa:

* **X**: dữ liệu gốc
* **Z**: semantic latent
* **X_s**: tín hiệu truyền (OFDM symbols)
* **X̂**: kết quả nhận

---

# ⚔️ 2. SemBLK (attack vào X_s — physical layer)

```
X → E → Z → Mod → X_s → (+ ε) → X_s^adv → Channel → Y_s → Demod → Ẑ → D → X̂
```

---

## 🧠 Cách nó hoạt động (bổ sung):

```
X → Surrogate Encoder → Z' → Semantic Loss → optimize ε → apply to X_s
```

---

## 📌 Bản chất:

* Attack tại: **X_s**
* Nhưng optimize dựa trên: **semantic loss (latent-level idea)**

👉 => **semantic-guided signal attack**

---

# 🧠 3. Latent Attack (proposal của bạn)

```
X → E → Z → (+ δ) → Z^adv → Mod → X_s → Channel → Y_s → Demod → Ẑ → D → X̂
```

---

## 📌 Bản chất:

* Attack tại: **Z**
* Tác động trực tiếp semantic

👉 => **semantic-level attack**

---

# ⚫ 4. Black-box query loop (áp dụng cho cả hai)

```
Initialize δ / ε
→ Generate candidate (Z+δ hoặc X_s+ε)
→ Feed into system
→ Observe output (X̂ / score)
→ Compute loss
→ Update δ / ε
→ Repeat
```

---

# ⚖️ 5. So sánh cực ngắn (1 dòng mỗi loại)

```
Input attack:      X → (+δ) → E → ...
SemBLK:            X → E → Z → Mod → X_s → (+ε) → ...
Latent attack:     X → E → Z → (+δ) → Mod → ...
```

---

# 🔥 6. Insight chốt (rất quan trọng)

```
SemBLK:  attack ở signal (X_s) nhưng “hiểu semantic”
Bạn:     attack trực tiếp semantic (Z)
```

---

# 🎯 1 câu gọn cho paper

> SemBLK perturbs transmitted signals (X_s) with semantic guidance, whereas our method directly manipulates the latent semantic representation (Z).

---

Nếu bạn muốn, mình có thể làm thêm bản:

👉 **“notation-consistent system model” (chuẩn IEEE, ký hiệu đẹp, đưa thẳng vào paper)**

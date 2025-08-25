# DRS–Hyperspherical: Kế hoạch triển khai hoàn chỉnh

**Mục tiêu:** Kết hợp **LoRA⁻ (subtraction)** ở tham số phẳng với **hình học cầu** ở không gian đặc trưng để giảm drift, tăng ổn định liên nhiệm vụ, **không cần lưu exemplar**. Bản này hợp nhất blueprint trước đó + các bổ sung: warm-up/anneal, EMA prototypes, quy trình ước lượng ổn định, lịch huấn luyện 2-phase, ma trận ablation, chẩn đoán và YAML cấu hình.

---

## 1) Kiến trúc tổng quan

- **Backbone (frozen):** ViT/ResNet tiền huấn luyện (W₀).
- **LoRA per-layer:** mỗi nhiệm vụ có nhánh LoRA riêng; trước task t, áp **LoRA⁻**: $\tilde W_t = W_0 - \sum_{j=1}^{t-1} \Delta W_j$.
- **Embedding chuẩn hoá cầu:** $\hat{x} = x/\|x\|\in S^{d-1}$.
- **DRS–cầu:** chọn prototype $\mu$ (per-class/global), log-map lên tiếp tuyến $T_\mu S^{d-1}$, PCA lấy cơ sở ổn định $U_t$, chiếu gradient theo hình học Riemann.
- **Loss:** ArcFace (cosine margin softmax) + Angular Triplet (với negatives gồm prototypes cũ).
- **Tuỳ chọn:** spCauchy-VAE head để mã hoá latent cầu.

---

## 2) Hình học & phép chiếu

- **Log-map:** $\log_\mu(x) = \tfrac{\theta}{\sin\theta}\,(x - \cos\theta\,\mu)$, $\theta=\arccos(\mu^\top x)$.
- **Exp-map/Retraction:** $\exp_\mu(v)=\cos\|v\|\,\mu+\sin\|v\|\,\tfrac{v}{\|v\|}$ (thực tế có thể retraction bằng chuẩn hoá).
- **Chiếu Riemann:** $g_T=(I-\mu\mu^\top)g$; nếu dùng không gian con ổn định $U_t$: $g_{proj}=U_t U_t^\top g_T$.
- **Khoảng cách địa tuyến:** $d_g(u,v)=\arccos(u^\top v)$.

---

## 3) Prototypes: tránh shift distribution do LoRA⁻

**Rủi ro:** LoRA⁻ có thể dịch phân phối khiến prototypes cũ không đại diện.
**Khắc phục (bắt buộc):**

1. **Tính prototypes ngay sau LoRA⁻** ($\tilde W_t$) **và trước khi train** LoRA mới.
2. **Cập nhật μ bằng EMA** trong suốt huấn luyện (momentum 0.95–0.99).
3. **Tái ước lượng cuối task** bằng 1 pass nhanh trên tập đại diện (hoặc full).
   **Bổ trợ:** warm-up 1–2 epoch chỉ CE, temperature annealing cho classifier.

**Per-class vs global μ:**

- Đủ mẫu → per-class μ (mean-direction).
- Few-shot/imbalance → global μ_t hoặc **multi-anchors** (K-means trên cầu). Chọn anchor gần nhất per-batch.

---

## 4) Quy trình DRS–cầu (per task)

**Bước A: Xây DRS sau LoRA⁻**

1. Forward tập đại diện Dt → thu $\hat{x}$.
2. Ước lượng μ (per-class/global), chuẩn hoá.
3. Log-map vào tiếp tuyến tại μ: $v_i=\log_\mu(\hat{x}_i)$.
4. PCA trên {v_i} → lấy $U_t$ (top-k, giữ ≥90% năng lượng, k≤128).

**Bước B: Huấn luyện LoRA_t (2-phase)**

- **Warm-up (Ew):** CE/cosine, chỉ chiếu Riemann (chưa dùng $U_t$). Anneal s↑, m↑. EMA(μ) online.
- **Main (Em):** CE + Angular Triplet, chiếu Riemann + $U_t$. EMA tiếp tục. Cuối task: re-estimate μ.

**Hook thực dụng (khuyến nghị):** Gắn `register_hook` tại tensor embedding trước classifier để chiếu gradient (đảm bảo ‖proj‖≤‖grad‖ và ⟂ μ).

---

## 5) Loss & Head

- **ArcFace head:** trọng số W được chuẩn hoá; logits = s·cos(θ+m) cho class đúng, cos(θ) cho class khác. Anneal **s: 10→30**, **m: 0→0.2**.
- **Angular Triplet:** $\max(0, d_g(a,p)-d_g(a,n)+m)$; negatives lấy từ batch & prototypes cũ. Trọng số λ_triplet ≈ 0.5 (tune 0.1–1.0).

---

## 6) Hyper-parameters (khởi điểm)

- LoRA rank r: 16 (8–32).
- PCA: năng lượng 0.90, k_max=128.
- Warm-up Ew: 3–5; Main Em: 15–50.
- Optimizer: Adam, LR 1e−3, WD 1e−5, batch 128.
- EMA momentum: 0.95–0.99.
- Label smoothing 0.05 (3–5 epoch đầu, tùy chọn).

---

## 7) Đo lường & biểu đồ

- **ACC vs task**; **A_old/A_new** theo task.
- **BWT:** ((1/(T-1)) \sum\_{t=1}^{T-1} (R\_{T,t}-R\_{t,t})).
- **Geodesic drift:** trung bình arccos(p_old·p_new).
- **Stability gap:** ACC_old − ACC_new.
- **Runtime/Memory.**

---

## 8) Chẩn đoán nhanh (failure modes → fix)

- **Quên mạnh:** tăng Ew; giảm m_end; bật smoothing; giảm LR.
- **Học mới yếu:** tăng Em; tăng s_end; tăng k_max; tăng λ_triplet; dùng multi-anchors.
- **NaN/Instability:** clamp cosθ∈\[−1+1e−7,1−1e−7]; small-angle approx; kiểm tra θ≈π; seed cố định.
- **Shift do LoRA⁻:** đảm bảo quy trình 3).

---

## 9) Chi phí & bộ nhớ (ước lượng)

- Cov (d×d) với d=768 ≈ 2.3MB; U_t (768×64) ≈ 0.19MB/task (fp32).
- Prototypes: \~3KB/lớp (fp32).
- Tăng tốc: **eigh** trên covariance; incremental PCA; sample 30–50k vector.

---

## 10) Reproducibility & cấu hình mẫu (YAML)

```yaml
seed: 1337
model: { backbone: vit_b16_in21k, lora_rank: 16, lora_alpha: 16 }
train:
  {
    epochs_warm: 4,
    epochs_main: 26,
    batch_size: 128,
    lr: 1e-3,
    wd: 1e-5,
    ema_momentum: 0.97,
  }
projection: { use_U: true, pca_energy: 0.90, k_max: 128 }
loss:
  {
    s_start: 10.0,
    s_end: 30.0,
    m_start: 0.0,
    m_end: 0.2,
    triplet_lambda: 0.5,
    label_smoothing: 0.05,
  }
logging: { eval_interval: 1, save_prototypes: true, save_U: true }
```

---

## 11) Ma trận ablation tối thiểu

1. Linear DRS (gốc).
2. Riemannian-only (không U_t).
3. - U_t.
4. - U_t + Angular Triplet.
5. (4) + EMA.
6. (5) + Warm-up & Anneal.
7. (6) + spCauchy-VAE head.

---

## 12) Pseudo-code (tóm tắt thực thi)

```python
# Sau khi áp LoRA⁻ → W_tilde
embs = collect_embeddings(model=W_tilde, data=Dt)
mu_init = estimate_prototypes(embs, per_class=True, multi_anchor=False)
V = log_map_batch(mu_init, embs)  # vào tiếp tuyến
U_t = tangent_pca(V, energy=0.90, k_max=128)

for epoch in range(Ew + Em):
    s = anneal(epoch, 0, Ew+Em, 10, 30)
    m = anneal(epoch, 0, Ew+Em, 0.0, 0.2)
    phase_main = epoch >= Ew
    for x,y in loader(Dt):
        z = normalize(backbone_with_lora(x))
        mu = ema_update(mu, z.detach(), y, momentum=0.97)
        z = z.detach().requires_grad_(True)
        attach_projection_hook(z, mu_batch(y, mu), U_t if phase_main else None)
        logits = arcface_head(z, y, s=s, m=m)
        ce = cross_entropy(logits, y, label_smoothing=0.05 if epoch<Ew else 0.0)
        loss = ce if not phase_main else ce + 0.5*angular_triplet(z, y, prototypes=mu)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
# Cuối task: re-estimate prototypes bằng 1 pass
```

---

## 13) Lộ trình thực hiện

- Ngày 0–2: baseline + LoRA⁻ + utilities log/exp/normalize + unit tests.
- Ngày 3–6: xây DRS (μ, EMA, PCA) + test.
- Ngày 7–10: loop huấn luyện 2-phase + hooks + losses.
- Ngày 11–14: chạy CIFAR-100 (50 tasks), tinh chỉnh.
- Ngày 15–21: ablation + ImageNet-R / CUB / DomainNet.

---

## 14) Kết luận

DRS–Hyperspherical hợp lý khi embedding mang tính hướng. Kết hợp **LoRA⁻** (reset drift) với **projection Riemann + U_t** và **loss góc** giúp cân bằng stability–plasticity, vẫn giữ ưu điểm **không lưu exemplar**. Bản hoàn chỉnh này sẵn sàng để code hoá và chạy ablation theo ma trận đã nêu.

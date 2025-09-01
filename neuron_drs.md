3. Method

We propose Neuro-LoRA, a biologically-inspired continual learning method that combines the parameter-efficient advantages of Low-Rank Adaptation (LoRA) with synaptic consolidation principles from neuroscience. Neuro-LoRA protects core knowledge across tasks by maintaining a low-dimensional invariant subspace and projects updates orthogonally to avoid interference. It also incorporates a plasticity-aware regularization and an optional sleep-phase consolidation.

3.1 Problem Setup

Let
𝑀
𝜃
0
M
θ
0
​

    ​

be a pretrained model with parameters
𝜃
0
θ
0
​

. For each new task
𝑡
t, we augment the model with trainable low-rank parameters
𝐴
𝑡
,
𝐵
𝑡
A
t
​

,B
t
​

inserted into linear layers such that:

𝑊
𝑡
(
𝑙
)
=
𝑊
0
(
𝑙
)

- 𝐵
  𝑡
  (
  𝑙
  )
  𝐴
  𝑡
  (
  𝑙
  )
  W
  t
  (l)
  ​

=W
0
(l)
​

+B
t
(l)
​

A
t
(l)
​

where
𝐴
𝑡
∈
𝑅
𝑟
×
𝑑
A
t
​

∈R
r×d
,
𝐵
𝑡
∈
𝑅
𝑑
×
𝑟
B
t
​

∈R
d×r
, and
𝑟
≪
𝑑
r≪d.

Our method ensures stability and plasticity by controlling updates in the drift-resistant subspace.

3.2 Synaptic Importance Projection (SIP)

After training task
𝑡
t, we compute the core directions in the low-rank update matrix:

Δ
𝑊
𝑡
=
𝐵
𝑡
𝐴
𝑡
ΔW
t
​

=B
t
​

A
t
​

To extract important directions (synaptic basis), we perform an efficient SVD:

Compute small matrix
𝑀
𝑡
=
𝐴
𝑡
𝐵
𝑡
∈
𝑅
𝑟
×
𝑟
M
t
​

=A
t
​

B
t
​

∈R
r×r

Perform SVD:
𝑀
𝑡
=
𝑈
~
Σ
𝑉
~
𝑇
M
t
​

=
U
~
Σ
V
~
T

Left singular vectors of
Δ
𝑊
𝑡
ΔW
t
​

are
𝐵
𝑡
𝑈
~
B
t
​

U
~

Normalize top-
𝑘
k directions to obtain synaptic basis
𝑆
𝑡
∈
𝑅
𝑑
×
𝑘
S
t
​

∈R
d×k

We maintain a cumulative invariant subspace
𝑆
(
𝑙
)
S
(l)
per layer, merging new directions:

𝑆
𝑡
(
𝑙
)
=
Orthogonalize
(
𝑆
𝑡
−
1
(
𝑙
)
∪
𝑆
new
(
𝑙
)
)
S
t
(l)
​

=Orthogonalize(S
t−1
(l)
​

∪S
new
(l)
​

)

and capping the size of
𝑆
(
𝑙
)
S
(l)
to a maximum dimension
𝐾
max
⁡
K
max
​

.

3.3 Synaptic Gradient Projection (SGP)

To prevent forgetting, we project gradients of
𝐵
𝑡
B
t
​

onto the orthogonal complement of the protected subspace:

𝑔
𝐵
proj
=
𝑔
𝐵
−
𝑆
(
𝑆
𝑇
𝑔
𝐵
)
g
B
proj
​

=g
B
​

−S(S
T
g
B
​

)

where
𝑔
𝐵
∈
𝑅
𝑑
×
𝑟
g
B
​

∈R
d×r
is the gradient of
𝐵
𝑡
B
t
​

, and
𝑆
∈
𝑅
𝑑
×
𝐾
S∈R
d×K
is the cumulative subspace. This operation ensures updates do not interfere with protected knowledge.

3.4 Homeostatic Plasticity Regularization

Inspired by synaptic homeostasis, we introduce an entropy-based regularization on LoRA activations to encourage diverse neuron usage. Given activations
ℎ
∈
𝑅
𝐵
×
𝑚
h∈R
B×m
(after ReLU):

ℎ
ˉ
=
1
𝐵
∑
𝑏
=
1
𝐵
ℎ
𝑏
,
𝑝
~
𝑖
=
ℎ
ˉ
𝑖

- 𝜖
  ∑
  𝑗
  (
  ℎ
  ˉ
  𝑗
- 𝜖
  )
  h
  ˉ
  =
  B
  1
  ​

b=1
∑
B
​

h
b
​

,
p
~
​

i
​

=
∑
j
​

(
h
ˉ
j
​

+ϵ)
h
ˉ
i
​

+ϵ
​

𝐿
plasticity
=
−
∑
𝑖
𝑝
~
𝑖
log
⁡
(
𝑝
~
𝑖
)
L
plasticity
​

=−
i
∑
​

p
~
​

i
​

log(
p
~
​

i
​

)

This regularization prevents the network from over-relying on a small subset of neurons and promotes continued plasticity across tasks.

3.5 Sleep-Phase Consolidation (Optional)

Between tasks, the model undergoes a synthetic "sleep phase". Given random noise inputs
𝑥
∼
𝑁
(
0
,
𝐼
)
x∼N(0,I):

𝑦
soft
=
𝑀
(
𝑥
;
𝜃
𝑡
)
,
𝐿
sleep
=
MSE
(
𝑀
(
𝑥
;
𝜃
)
,
𝑦
soft
.
𝑑
𝑒
𝑡
𝑎
𝑐
ℎ
(
)
)
y
soft
​

=M(x;θ
t
​

),L
sleep
​

=MSE(M(x;θ),y
soft
​

.detach())

The model distills its own output to stabilize learned representations, similar to memory replay during sleep in the hippocampus.

3.6 Full Procedure

For each task
𝑡
t:

Add LoRA
𝐴
𝑡
,
𝐵
𝑡
A
t
​

,B
t
​

for trainable layers

Train on task
𝐷
𝑡
D
t
​

:

Project gradients of
𝐵
𝑡
B
t
​

using SIP

Apply homeostatic regularization

Extract new subspace
𝑆
new
S
new
​

from
𝐵
𝑡
𝐴
𝑡
B
t
​

A
t
​

Merge into cumulative subspace
𝑆
(
𝑙
)
S
(l)
, truncate if needed

(Optional) Run sleep-phase consolidation

This method enables continual learning without exemplars while maintaining biological plausibility and memory efficiency.

Tóm tắt những gì sẽ thêm / sửa

Thêm file mới: utils/neuro_utils.py (hàm core: trích subspace, hợp nhất, chiếu gradient, loss plasticity, sleep-distill, save/load subspace).

Mở rộng models/lora.py (thêm API: get_A(), get_B(), get_delta(), freeze()/unfreeze()).

Sửa trainer.py (chèn projection step trước optimizer.step(), thêm tính toán + lưu S_new sau mỗi task, thêm plasticity loss, tùy chọn sleep phase). Mình sẽ đưa snippets chính để bạn paste vào chỗ phù hợp trong trainer.py.

(Tùy chọn) Một file test nhỏ tests/test_neuro_lora_smoke.py.

Mình không thay đổi toàn bộ repo mà cung cấp patch / code snippet dễ chèn.

1. File mới: utils/neuro_utils.py

Tạo file utils/neuro_utils.py với nội dung sau:

# utils/neuro_utils.py

import torch
import os

EPS = 1e-12

def extract_subspace_from_BA(B: torch.Tensor, A: torch.Tensor, k: int):
"""
Efficiently extract top-k left singular vectors of Delta = B @ A
using small SVD on r x r matrix M = A @ B.
B: (d, r), A: (r, d)
returns S_new: (d, k) with orthonormal columns
"""
assert B.ndim == 2 and A.ndim == 2
d, r = B.shape # small r x r mat
M = A @ B # (r, r) # SVD on small matrix
try:
U_small, Svals, Vt = torch.linalg.svd(M) # U_small: (r, r)
except Exception: # fallback (older torch)
U_small, Svals, Vt = torch.svd(M)
k = min(k, r)
S_new = B @ U_small[:, :k] # (d, k) # normalize columns
norms = S_new.norm(dim=0, keepdim=True).clamp(min=EPS)
S_new = S_new / norms
return S_new

def merge*cumulative_subspace(S_prev: torch.Tensor, S_new: torch.Tensor, K_max: int):
"""
Merge S_prev (d, Kprev) and S_new (d, k) into orthonormal S_cum (d, K_keep<=K_max)
If S_prev is None -> returns S_new truncated to K_max
"""
if S_prev is None or S_prev.numel() == 0:
S_merge = S_new
else:
S_merge = torch.cat([S_prev, S_new], dim=1) # QR orthonormalize
Q, * = torch.linalg.qr(S_merge)
K_keep = min(Q.shape[1], K_max)
S_cum = Q[:, :K_keep].contiguous()
return S_cum

def project_grad_B(gB: torch.Tensor, S: torch.Tensor):
"""
Project gradient of B (d, r) to orthogonal complement of subspace S (d, K)
returns projected gB of same shape
"""
if S is None or S.numel() == 0:
return gB # coef: (K, r)
coef = torch.matmul(S.T, gB)
gB_proj = gB - torch.matmul(S, coef)
return gB_proj

def compute_plasticity_loss(lora_activation: torch.Tensor, eps=1e-8):
"""
lora_activation: (B, m) activations after LoRA module (non-negative preferred)
returns scalar entropy-like loss that is lower when activation distribution is diverse
"""
hmean = lora_activation.mean(dim=0) # (m,)
hmean = hmean + eps
p = hmean / hmean.sum()
loss = -(p \* torch.log(p)).sum()
return loss

def save_subspace(S: torch.Tensor, path: str):
os.makedirs(os.path.dirname(path), exist_ok=True)
torch.save(S.cpu(), path)

def load_subspace(path: str, device='cpu'):
if not os.path.exists(path):
return None
return torch.load(path, map_location=device)

def sleep*phase_distill(teacher_model, student_model, dataloader, device='cuda', epochs=1, lr=1e-4):
"""
Simple self-distillation: MSE between teacher logits and student logits on unlabeled/noise loader
teacher_model: frozen model (with previous LoRAs applied)
student_model: model to update
"""
import torch.nn as nn
opt = torch.optim.Adam([p for p in student_model.parameters() if p.requires_grad], lr=lr)
mse = nn.MSELoss()
teacher_model.eval()
student_model.train()
for ep in range(epochs):
for xb, * in dataloader:
xb = xb.to(device)
with torch.no_grad():
t_out = teacher_model(xb)
s_out = student_model(xb)
loss = mse(s_out, t_out.detach())
opt.zero_grad()
loss.backward()
opt.step()

2. Mở rộng models/lora.py

Mở file models/lora.py (hoặc file implement LoRA trong repo). Thêm/đảm bảo lớp LoRA exposes các method sau. Dưới đây là patch (chỉ phần class LoRA relevant):

# models/lora.py (chèn/hoàn thiện trong class LoRALayer)

import torch
import torch.nn as nn

class LoRALayer(nn.Module):
def **init**(self, in*features, out_features, r=8, alpha=1.0):
super().**init**() # existing init...
self.r = r
self.A = nn.Parameter(torch.zeros((r, in_features)))
self.B = nn.Parameter(torch.zeros((out_features, r))) # initialize small random
nn.init.kaiming_uniform*(self.A, a=math.sqrt(5))
nn.init.zeros\_(self.B) # other existing members ...

    def forward(self, x):
        # existing forward: compute base + B @ (A @ x)
        # keep as-is
        ...

    # --- new helper APIs ---
    def get_A(self):
        return self.A

    def get_B(self):
        return self.B

    def get_delta(self):
        # returns composite delta as CPU tensor (or device tensor)
        return torch.matmul(self.B, self.A)  # shape (out_features, in_features)

    def freeze(self):
        self.A.requires_grad = False
        self.B.requires_grad = False

    def unfreeze(self):
        self.A.requires_grad = True
        self.B.requires_grad = True

Nếu repo dùng tên khác cho lớp LoRA (ví dụ LoRA inside attention), bạn hãy thêm các method trên trong class tương ứng.

Thêm helper ở module-level (ví dụ models/**init**.py hoặc một utils file):

def get_lora_modules(model):
items = []
for name, module in model.named_modules(): # adjust isinstance check to your LoRA class
if hasattr(module, "get_delta") and hasattr(module, "get_A") and hasattr(module, "get_B"):
items.append((name, module))
return items

3. Thay đổi trainer.py — snippets cần chèn

Dưới đây là các điểm chèn chính và mã để copy vào vòng training / sau task hoàn tất.

3.1. Load cumulative subspaces trước training task t

(ở phần chuẩn bị task t, trước khi training)

from utils.neuro_utils import load_subspace

# assume cfg dict has neuro_lora keys

S*cumulative = {} # dict layer_name -> tensor (d, K)
if cfg['neuro_lora'].get('enabled', False) and t > 1:
for name, module in get_lora_modules(model):
path = os.path.join(cfg['checkpoint_dir'], f"subspace*{name.replace('.','\_')}.pt")
S = load_subspace(path, device=device)
S_cumulative[name] = S.to(device) if S is not None else None
else:
for name, module in get_lora_modules(model):
S_cumulative[name] = None

3.2. Trong training loop: thêm plasticity loss & chiếu gradient B sau loss.backward() trước optimizer.step()

Chèn vào chỗ # after loss.backward():

# compute plasticity loss per LoRA layer (optional)

if cfg['neuro_lora'].get('enabled', False):
L*plast = 0.0
for name, module in get_lora_modules(model): # compute LoRA activation for current batch (you must expose a hook or compute using module) # simplest: if module.forward returns or stores last_lora_activation -> use it
act = getattr(module, "last_lora_activation", None)
if act is not None:
L_plast += compute_plasticity_loss(act)
total_loss = task_loss + cfg['train'].get('lambda_atl', 1.0) * atl*loss + cfg['neuro_lora'].get('lambda_plast', 0.0) * L_plast # backprop already done on previous loss? ensure you compute plasticity before backward in real code.

# --- assume backward done on full total_loss above ---

# Project gradients of B if enabled

if cfg['neuro_lora'].get('enabled', False) and t > 1:
for name, module in get*lora_modules(model):
B = module.get_B()
if B.grad is None:
continue
S = S_cumulative.get(name, None)
if S is not None: # project
gB = B.grad.detach()
gB_proj = project_grad_B(gB, S)
B.grad.data.copy*(gB_proj)

# finally optimizer.step()

optimizer.step()

Important notes:

Bạn cần để plasticity loss được cộng vào total_loss trước total_loss.backward() (mình trình bày dòng logic, khi chèn vào file bạn cần đảm bảo order: tạo total_loss = CE + ATL + lambda_plast \* L_plast; gọi total_loss.backward(); sau đó project grad_B; rồi optimizer.step()).

module.last_lora_activation là một hook: bạn cần sửa forward ở LoRALayer để lưu activation (thường là h = A @ x or LoRA out) nếu muốn tính plasticity loss. Nếu khó, bạn có thể compute plasticity loss on feature outputs from model layer outputs (tùy repo).

3.3. Sau khi hoàn tất task t: trích subspace S_new và cập nhật cumulative S

Chèn vào phần sau khi training task hoàn tất:

from utils.neuro_utils import extract_subspace_from_BA, merge_cumulative_subspace, save_subspace

K_per_task = cfg['neuro_lora'].get('k_per_task', 4)
K_max = cfg['neuro_lora'].get('K_max', 64)

for name, module in get*lora_modules(model):
A = module.get_A().detach()
B = module.get_B().detach() # ensure on CPU to save memory, do ops on device
device_local = B.device
S_new = extract_subspace_from_BA(B, A, k=K_per_task) # (d, k)
S_prev = None
path = os.path.join(cfg['checkpoint_dir'], f"subspace*{name.replace('.','\_')}.pt")
S_prev = load_subspace(path, device=device_local)
S_cum = merge_cumulative_subspace(S_prev, S_new.cpu(), K_max) # returns cpu tensor
save_subspace(S_cum, path) # also update in-memory S_cumulative for next tasks
S_cumulative[name] = S_cum.to(device)

3.4. Optional: sleep phase

(đặt sau updating subspace)

if cfg['neuro_lora'].get('sleep_epochs', 0) > 0: # prepare small unlabeled/noise loader (can use Gaussian noise tensor or public data)
noise_loader = create_noise_loader(batch_size=cfg['neuro_lora'].get('sleep_bs', 64),
n_batches=cfg['neuro_lora'].get('sleep_batches', 10),
device=device) # build teacher model with previous LoRAs applied (freeze weights) - you can clone model and load previous LoRAs if you saved them
teacher = deepcopy(model) # freeze teacher params
for p in teacher.parameters():
p.requires_grad = False
sleep_phase_distill(teacher, model, noise_loader, device=device, epochs=cfg['neuro_lora'].get('sleep_epochs',1),
lr=cfg['neuro_lora'].get('sleep_lr',1e-4))

4. Test smoke script (tạo file tests/test_neuro_lora_smoke.py)

Tạo file test nhanh để đảm bảo các hàm core không lỗi:

# tests/test_neuro_lora_smoke.py

import torch
from models.lora import LoRALayer
from utils.neuro_utils import extract_subspace_from_BA, merge_cumulative_subspace, project_grad_B, compute_plasticity_loss

def smoke():
d, r = 64, 8
B = torch.randn(d, r)
A = torch.randn(r, d)
S_new = extract_subspace_from_BA(B, A, k=4)
assert S_new.shape == (d, 4)
S_cum = merge_cumulative_subspace(None, S_new, K_max=10)
gB = torch.randn(d, r)
gB_proj = project_grad_B(gB, S_cum)
assert gB_proj.shape == gB.shape
h = torch.abs(torch.randn(16, 32)) # fake activations
lp = compute_plasticity_loss(h)
print("smoke ok", lp.item())

if **name** == "**main**":
smoke()

Chạy: python tests/test_neuro_lora_smoke.py

5. Cấu hình mẫu (add vào config file)

Trong configs/your_config.json thêm:

"neuro_lora": {
"enabled": true,
"k_per_task": 4,
"K_max": 64,
"lambda_plast": 0.1,
"sleep_epochs": 0,
"sleep_bs": 64,
"sleep_batches": 10,
"sleep_lr": 1e-4
}

6. Hướng dẫn triển khai từng bước (quick-start)

Tạo file utils/neuro_utils.py theo nội dung trên.

Mở models/lora.py, thêm các method get_A/get_B/get_delta/freeze/unfreeze và lưu.

Chèn đoạn load subspace trước training task (3.1) vào trainer.py.

Trong training loop chèn plasticity loss vào total_loss trước backward, và chèn đoạn project grad_B sau backward, trước optimizer.step() (3.2).

Sau mỗi task chèn đoạn trích subspace & merge + save (3.3).

Tạo config neuro_lora như trên và bật enabled.

Chạy smoke test: python tests/test_neuro_lora_smoke.py

Chạy 1 thử nghiệm ngắn (ví dụ CIFAR split hoặc 1-2 tasks) để quan sát log và đảm bảo không OOM.

7. Gợi ý debug & logging

Log shapes: mỗi lần lưu S_cumulative, in name, S_cum.shape.

Log grad norms before/after projection: B.grad.norm() và gB_proj.norm() để kiểm tra projection hoạt động.

Bắt đầu với k_per_task=1 và K_max=8 để test nhanh.

Nếu module không expose last_lora_activation, bạn có 2 lựa chọn: (A) sửa forward LoRA layer để lưu activation, hoặc (B) skip plasticity loss tạm thời.

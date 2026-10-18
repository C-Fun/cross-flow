# CrossFlow × pixelFAR 集成分析

日期：2026-10-18 ｜ 对象：`FAR_pMF/pixel_far/`（pixel-space FAR + MeanFlow，UCF101 128²，`pixelFAR_L`）

## 0. 你的提案（我的理解）

- **编码端**：预训练 VAE（建议 VA-VAE，它本身是 DINOv2 对齐的）产生每帧 latent；DINOv2/v3 用作感知损失（和/或 latent 对齐）。
- **生成端**：当前帧从 latent 噪声出发，**一步直接生成像素**（CrossFlow 目标），上下文帧通过 FAR 的帧因果注意力提供。
- 期望收益：latent 轨迹更低维、更语义化 → 一步映射更容易；无独立解码器；像素级感知/对抗损失可直接作用于输出。

下面先给两种可行架构，再逐项评估实现难度和风险。

## 1. 两种架构

### 方案 A（推荐先做）：单模型，"latent 进、像素出、像素做上下文"

```
训练:   context = GT 像素帧 (patch-16 嵌入, 现有 x_embedder)
        current = 噪化 latent  z_t = (1−t)·E(x_F) + t·ε   (1×1 patch 嵌入, 32ch)
        输出    = 当前帧像素 x̂_F  (现有 768-dim head = 16·16·3, 不变)
        损失    = CrossFlow eq.(11') L1 + DINO 感知 on x̂ (+ 以后 GAN)，只在当前帧

推理:   每帧:  ε ~ N(0,I) in latent space → F(ε, r=0, context) → 像素帧
        像素帧直接 append 为下一步的 context（无需再编码）
```

关键观察：**VA-VAE 是 f16，在 128² 上 latent 网格是 8×8，恰好等于 pixelFAR 当前 patch-16 的 8×8 token 网格**。所以 Transformer 主干、RoPE、因果掩码、KV cache、`token_per_frame` 全部不变；只换当前帧的嵌入器（3ch×16×16 → 32ch×1×1）。子代理列出的 5 个"输入输出空间耦合"阻塞点里，#2（unpatchify 网格）、#3（H==W 断言）、#4（RoPE/token 数）因此自动消解。

优点：保留 pixelFAR 的核心卖点（推理循环里没有编码器/解码器，context 就是输出本身，不经额外模块放大漂移）；训练时 VAE 只用来算 GT 当前帧的 latent，可离线预计算。
缺点：同一序列里混两种 token（像素嵌入的 context + latent 嵌入的 current），模型要学会对齐两种空间；这是论文没做过的设定。

### 方案 B：双模型，latent-FAR 自回归 + CrossFlow 做帧解码器

上游 FAR 已有 latent 空间的 FAR 训练器（DC-AE）；CrossFlow 只当解码器（论文 §4.4 证明它作为解码器比 VAE 自带解码器 FID 更好，且对"生成的、非干净的 latent"更鲁棒）。

优点：解码器就是我们 ImageNet 那份代码按帧跑，**几乎零改动**；AR 模型复用现有 latent FAR。
缺点：退回"两阶段"，失去 pixelFAR 的无解码器叙事；AR 漂移发生在 latent 空间，解码器只能缓解不能消除；两个模型的训练预算叠加。

**建议**：方案 B 作为一周内可出数的基线（验证 CrossFlow 解码器在视频上的收益），方案 A 作为主线。

## 2. 方案 A 的实现清单（难度按工作量）

| 模块 | 改动 | 工作量 | 依据 |
|---|---|---|---|
| **数据** | UCF101 每帧 VA-VAE latent：在线编码（VA-VAE encoder 128² 很便宜）或离线预计算（4 KB/帧，全量 ~10 GB） | 0.5 天 | 现有 `use_latent` 分支是 `NotImplementedError`，需补 |
| **模型** `crossflowFAR_model.py` | 复制 `pixelFAR`：(1) 新增 latent 嵌入器 `Conv2d(32→inner, k=1)`，按帧类型选择嵌入器；(2) 去掉 u/v 双头，单头直接输出 x 预测，**删除** `u=(z−net)/t` 变换（它把输入空间和输出空间硬绑定，是阻塞点 #1）；(3) 时间条件改 **r-only**（context 帧 r=−1 哨兵），**删除** omega/t_min/t_max 嵌入器 | 1 天 | 当前 `(z_sel−net_u)/t_sel` 在 latent 输入下形状不匹配 |
| **训练器** `trainer_crossflow_far.py` | 损失整体从 `crossflow.py::compute_loss` 移植：shifted-time 采样 + 50% 对角线、JVP **只对当前帧 latent 取切向 ε−z**（context 切向置零，`u_fn` 只需 1 个 primal，比现在的 3 个简单）、eq.(11') L1、x̂ = F + t(t−r)/r·sg(dF/dt)、DINO-Huber 感知（移植 `utils/perceptual.py`，复用现有 `_paired_random_resized_crop` 到 224）、context/clean-context 掩码 | 1.5 天 | `SDPBackend.MATH` 和 autograd-JVP 路径现成 |
| **优化器** | `train.py` 断言 AdamW；需加 Muon（矩阵参数）+ AdamW（其余） | 0.5 天 | 见 §3.3，这不是可选项 |
| **采样器** | 每帧一步：`ε → F(ε, r=0)`，像素 append；KV cache 写入时对新 context 帧走像素嵌入器 | 0.5 天 | 现有 `is_cache_step` 逻辑可复用 |
| **评估** | FVD 不变；加每帧 FID 漂移曲线（已有 `fid_vs_frame.py`） | 0 | |

合计约 **4–5 天到第一次可训练**，不含调参。CFG 和 GAN 另计（§3.4、§3.6）。

## 3. 可能出问题的地方（按严重程度）

### 3.1 VA-VAE 的分辨率失配 —— 先做 10 分钟验证
VA-VAE 在 256² 上训练；UCF 是 128²。f16 在 128² 只剩 8×8×32 = 2048 维 latent，比论文 ImageNet 的 8192 维少 4 倍。**先验证**：对 UCF 帧做 VA-VAE encode→decode，看 PSNR（ImageNet 256 上是 25.8 dB）。若 128² 上明显掉，两条路：(a) 帧上采到 256² 再编码（token 变 16×16，注意力代价 ×16，不可接受）；(b) 换 f8 VAE（SD-VAE 4ch 或 16ch），latent 16×16，用 patch 2 → 仍是 8×8 token。我们 ImageNet 的经验是 4ch SD-VAE 明显差于 VA-VAE，但那是 256²；128² 上未知。

### 3.2 一步生成的"静态/拷贝 context"坍缩 —— 最大的算法风险
pixelFAR 已知的失败模式是 1-step 运动坍缩（`self-forcing.md`）。CrossFlow 会放大这个风险：训练时 50% 样本是对角线 r=t，小 t 时当前帧的 latent **就是答案**，模型可以学成"看 z_t 解码、不看 context"；而推理时 z_t 是纯噪声，内容全靠 context。论文 Fig 6 正是这种低 t 坍缩（忽略条件和噪声）。缓解：
- 时间分布向高 t 偏（论文的 shift 公式在 2048 维 latent 下 α=0.71，反而偏低 t——要手动改）；
- 感知损失必须覆盖全部 (t,r)（这是我们 v4 的教训），并考虑按 t 加权；
- 监控指标：生成视频的帧间 LPIPS（太低 = 静态）、FVD 与逐帧 FID 的背离。

### 3.3 优化器：必须带 Muon（或退回自适应归一化）
我们 ImageNet 上的硬数据：AdamW + L1 + r/t² 权重 = v4，FID 99 停滞（解码样本梯度大 50 倍，裁剪后垄断更新）；换 Muon + 去裁剪 + time_eps 0.05 = v5，FID 23。pixelFAR 的 `train.py` 硬编码 AdamW 且 `max_grad_norm=1.0`，直接套 CrossFlow 损失会复现 v4 的失败。两种选择：移植 Muon（推荐），或沿用 pixelFAR 现有的 `_adp_wt_fn` 自适应归一化（等价我们的 v2/v3，上限差一截）。

### 3.4 CFG：pixelFAR 的引导蒸馏整套失效
pixelFAR 的 CFG 是**速度空间**的蒸馏（omega/t_min/t_max token + 3 次额外无梯度前向算引导目标）。CrossFlow 消去了速度，这套不能用。而我们实测过**像素空间推理时外推 CFG 是有害的**（CIFAR：FID 12.9 → 17.8）。可选：
- 先不做 CFG（论文默认无 CFG、无 label dropout；UCF 的 101 类条件可能够用）；
- x 空间蒸馏：目标 x̂_g = x̂_u + w(x̂_c − x̂_u)，用无梯度前向算，未验证；
- 推理时 1.2–1.5 的温和 w。
建议第一版不做 CFG，这也省掉 pixelFAR 每步 3 次额外前向，**训练反而更快**。

### 3.5 推理时的 context 一致性与 KV cache
训练时 context 是 GT 像素，推理时是模型自己生成的像素——和 pixelFAR 相同的 exposure bias，不新增。新增的一点：生成帧要走**像素嵌入器**进 cache，而它生成时走的是 latent 嵌入器；两种嵌入必须在训练里见过同一帧的两种身份（当前帧→latent，下一步→像素 context），现有训练把所有帧都同时当 current 和 context，天然覆盖。

### 3.6 感知/对抗损失的开销
DINO 在 b×16 帧 = 64 张/GPU/步、224² 上跑，叠在 VGG + ConvNeXt 之上会明显吃显存和时间。建议只保留 DINO（论文配方），去掉 VGG/ConvNeXt，并对每个 clip 随机抽 4–8 帧算感知损失。现有 `_paired_random_resized_crop` 是 Python 循环，批量大时要改成向量化。GAN 以后加 dinoGAN 头（复用 DINO 特征，+10–15%），时间权重 (1−t)² 必需，否则低 t 坍缩。

### 3.7 x̂ 无界系数
t(t−r)/r 在 r→0 处无界，ImageNet 上靠 time_eps 0.05 压到 ≤17 才稳住；视频里感知损失按帧算，垃圾 x̂ 的比例同样约 25%。沿用 0.05 + clamp，或对感知项单独设 r ≥ 0.1。

### 3.8 工程细节（会咬人的）
- `autograd_jvp(create_graph=True)` 走双反向，加进 `u_fn` 的任何算子都要支持 double-backward；VA-VAE 编码器必须在 `u_fn` 之外算（闭包捕获 latent），否则会被当成 primal 求导。
- `functorch` 和 `autograd` 两条 JVP 路径对 `u_fn` 返回值要求不同，改返回结构要同时改两处。
- `pipeline` 用 `config.in_channels × sample_size` 造初始噪声，latent 输入后要改成 latent 形状。
- `EMAModel` 的 `copy_to` 在当前上游版本里疑似不存在，子代理读到的是 `step/store/restore`——集成前确认。
- 感知损失现在对 context 帧（t=−1）也算了一遍再被 mask 掉，白跑 10%；顺手修。

## 4. 最终建议的推进顺序
1. **10 分钟**：VA-VAE 在 UCF 128² 帧上的重建 PSNR（决定 §3.1 走哪条路）。
2. **1 周**：方案 B 基线——用现有 latent-FAR + 我们的 CrossFlow 代码按帧当解码器，得到 FVD 对照（同时验证 VA-VAE latent 在视频上好不好 AR）。
3. **1–2 周**：方案 A 第一版：无 CFG、无 GAN、Muon、DINO 感知全 (t,r)、time_eps 0.05；重点看帧间 LPIPS 和逐帧 FID 漂移。
4. 之后：dinoGAN、x 空间 CFG 蒸馏。

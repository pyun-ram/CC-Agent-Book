# Diffusion Models ($ 版本)

Reference: https://lilianweng.github.io/posts/2021-07-11-diffusion-models/

## 1. Diffusion process
给定一个从真实数据分布采样的数据点 $\mathbf{x}_0 \sim q(\mathbf{x})$，我们定义一个前向扩散过程：在 $T$ 步中逐步向样本中加入高斯噪声，得到一系列噪声样本 $\mathbf{x}_1, \ldots, \mathbf{x}_T$。每一步的噪声大小由方差调度 $\{\beta_t \mid 0 < \beta_t < 1\}$ 控制。

扩散过程的转移概率为：
$$
q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}\left(\mathbf{x}_t; \sqrt{1-\beta_t}\,\mathbf{x}_{t-1},\, \beta_t \mathbf{I}\right)
$$

该过程的一个优良性质是：我们可以利用重参数化技巧，在任意时间步 $t$ 以闭式形式采样 $\mathbf{x}_t$。

$$
\begin{aligned}
\mathbf{x}_t &= \sqrt{1 - \beta_t}\, \mathbf{x}_{t-1} + \sqrt{\beta_t}\, \boldsymbol{\epsilon}_{t-1}, \quad \boldsymbol{\epsilon}_{t-1} \sim \mathcal{N}(0, \mathbf{I}) \\
             &= \sqrt{\alpha_t}\, \mathbf{x}_{t-1} + \sqrt{1 - \alpha_t}\, \boldsymbol{\epsilon}_{t-1} \\
             &= \sqrt{\alpha_t} \left( \sqrt{\alpha_{t-1}}\, \mathbf{x}_{t-2} + \sqrt{1 - \alpha_{t-1}}\, \boldsymbol{\epsilon}_{t-2} \right) + \sqrt{1 - \alpha_t}\, \boldsymbol{\epsilon}_{t-1} \\
             &= \sqrt{\alpha_t \alpha_{t-1}}\, \mathbf{x}_{t-2} + \sqrt{\alpha_t (1 - \alpha_{t-1})}\, \boldsymbol{\epsilon}_{t-2} + \sqrt{1 - \alpha_t}\, \boldsymbol{\epsilon}_{t-1} \\
             &\quad \text{其中 } \boldsymbol{\epsilon}_{t-1}, \boldsymbol{\epsilon}_{t-2} \sim \mathcal{N}(0, \mathbf{I}) \\
             &= \cdots \\
             &= \sqrt{\prod_{s=1}^t \alpha_s}\, \mathbf{x}_0 + \sum_{k=1}^t \left( \sqrt{ \left( \prod_{s=k+1}^t \alpha_s \right) (1 - \alpha_k) }\, \boldsymbol{\epsilon}_{k-1} \right) \\
             &= \sqrt{\bar{\alpha}_t}\, \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\, \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})
\end{aligned}
$$

其中，$\alpha_t = 1 - \beta_t$，$\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$。

> Recall that when we merge two Gaussians with different variances $\mathcal{N}(0, \sigma_1^2 \mathbf{I})$ and $\mathcal{N}(0, \sigma_2^2 \mathbf{I})$, the new distribution is $\mathcal{N}(0, (\sigma_1^2 + \sigma_2^2)\mathbf{I})$.

## 2. 反向扩散过程（Reverse Diffusion Process）

如果我们能够反转上述前向扩散过程，并从 $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$ 采样，就可以从高斯噪声输入 $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 逐步还原出真实样本。注意，当 $\beta_t$ 足够小时，$q(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$ 也是高斯分布。然而，$q(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$ 依赖于整个数据分布，难以直接计算，因此我们需要学习一个模型 $p_\theta$ 来近似这些条件概率，从而实现反向扩散采样。

我们有：
$$
p_\theta(\mathbf{x}_{0:T}) = p(\mathbf{x}_T) \prod_{t=1}^T p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)
$$
其中
$$
p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t) = \mathcal{N}\left(\mathbf{x}_{t-1};\, \boldsymbol{\mu}_\theta(\mathbf{x}_t, t),\, \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t)\right)
$$

值得注意的是，若已知 $\mathbf{x}_0$，则反向条件概率 $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)$ 具有解析形式：
$$
q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}\left(\mathbf{x}_{t-1};\, \tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0),\, \tilde{\beta}_t \mathbf{I}\right)
$$

利用贝叶斯公式，有：
$$
q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) = \frac{q(\mathbf{x}_t \mid \mathbf{x}_{t-1})\, q(\mathbf{x}_{t-1} \mid \mathbf{x}_0)}{q(\mathbf{x}_t \mid \mathbf{x}_0)}
$$

其中各项分布为：
$$
\begin{aligned}
q(\mathbf{x}_{t-1} \mid \mathbf{x}_0) &= \mathcal{N}\left(\mathbf{x}_{t-1};\, \sqrt{\bar{\alpha}_{t-1}}\, \mathbf{x}_0,\, (1 - \bar{\alpha}_{t-1})\mathbf{I}\right) \\
q(\mathbf{x}_t \mid \mathbf{x}_0) &= \mathcal{N}\left(\mathbf{x}_t;\, \sqrt{\bar{\alpha}_t}\, \mathbf{x}_0,\, (1 - \bar{\alpha}_t)\mathbf{I}\right) \\
q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) &= \mathcal{N}\left(\mathbf{x}_t;\, \sqrt{\alpha_t}\, \mathbf{x}_{t-1},\, (1-\alpha_t)\mathbf{I}\right)
\end{aligned}
$$

将上述高斯分布代入，$q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)$ 仍为高斯分布，其均值和方差为：
$$
\begin{aligned}
\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0) &= \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}\, \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\, \beta_t}{1 - \bar{\alpha}_t}\, \mathbf{x}_0 \\
\tilde{\beta}_t &= \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t}\, \beta_t
\end{aligned}
$$

由于
$$
\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\, \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\, \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$
可解得
$$
\mathbf{x}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}} \left( \mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t}\, \boldsymbol{\epsilon} \right)
$$

因此，$\tilde{\boldsymbol{\mu}}_t$ 也可以用 $\mathbf{x}_t$ 和噪声 $\boldsymbol{\epsilon}_t$ 表示：
$$
\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0) = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}}\, \boldsymbol{\epsilon}_t \right)
$$

注意，此时 $\boldsymbol{\epsilon}_t$ 并不是一个随机变量，而是一个确定的样本。只要我们知道 $\boldsymbol{\epsilon}_t$，就可以反推出 $\mathbf{x}_0$，从而逆转采样过程。直观地说，我们可以在训练时用神经网络 $f_\theta(\mathbf{x}_t, t)$ 来回归噪声 $\boldsymbol{\epsilon}_t$，即
$$
f_\theta(\mathbf{x}_t, t) \approx \boldsymbol{\epsilon}_t
$$
这样，模型学会了给定 $\mathbf{x}_t$ 和 $t$ 预测出对应的噪声，从而实现对去噪过程的建模.

---

## 3. Score Function 定义

定义时间步 $t$ 的边缘分布 $q(\mathbf{x}_t)$ 的 score function：
$$
s_t(\mathbf{x}_t) := \nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t).
$$

定义条件（给定 $\mathbf{x}_0$）的 score：
$$
s_{\text{cond}}(\mathbf{x}_t \mid \mathbf{x}_0) := \nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t \mid \mathbf{x}_0).
$$

由前向扩散的闭式：
$$
q(\mathbf{x}_t \mid \mathbf{x}_0)=\mathcal N\bigl(\mathbf{x}_t;\sqrt{\bar\alpha_t}\mathbf{x}_0,(1-\bar\alpha_t)\mathbf{I}\bigr),
$$
直接求导可得
$$
s_{\text{cond}}(\mathbf{x}_t \mid \mathbf{x}_0) = -\frac{1}{1-\bar\alpha_t}\bigl(\mathbf{x}_t - \sqrt{\bar\alpha_t}\mathbf{x}_0\bigr).
$$

使用重参数化
$$
\mathbf{x}_t = \sqrt{\bar\alpha_t}\mathbf{x}_0 + \sqrt{1-\bar\alpha_t}\boldsymbol{\epsilon},\quad \boldsymbol{\epsilon}\sim\mathcal N(\mathbf{0},\mathbf{I}),
$$
得
$$
s_{\text{cond}}(\mathbf{x}_t \mid \mathbf{x}_0)= -\frac{1}{\sqrt{1-\bar\alpha_t}}\boldsymbol{\epsilon},
$$
并且在给定 $(\mathbf{x}_0,\mathbf{x}_t)$ 时
$$
\boldsymbol{\epsilon} = \frac{\mathbf{x}_t - \sqrt{\bar\alpha_t}\mathbf{x}_0}{\sqrt{1-\bar\alpha_t}}
$$
为确定值。

## 4. Score Function 是条件 Score 的条件期望

由于
$$
q(\mathbf{x}_t)=\int q(\mathbf{x}_t\mid \mathbf{x}_0)\,q(\mathbf{x}_0)\,d\mathbf{x}_0,
$$
对 $\mathbf{x}_t$ 求梯度：
$$
\nabla_{\mathbf{x}_t} q(\mathbf{x}_t) = \int q(\mathbf{x}_t\mid \mathbf{x}_0)\, s_{\text{cond}}(\mathbf{x}_t\mid \mathbf{x}_0)\, q(\mathbf{x}_0)\, d\mathbf{x}_0.
$$
故
$$
s_t(\mathbf{x}_t)=\nabla_{\mathbf{x}_t}\log q(\mathbf{x}_t)
=\frac{1}{q(\mathbf{x}_t)}\int q(\mathbf{x}_t\mid \mathbf{x}_0) q(\mathbf{x}_0) s_{\text{cond}}(\mathbf{x}_t\mid \mathbf{x}_0)\, d\mathbf{x}_0
=\mathbb E_{q(\mathbf{x}_0\mid \mathbf{x}_t)}\bigl[s_{\text{cond}}(\mathbf{x}_t\mid \mathbf{x}_0)\bigr].
$$

即
$$
s_t(\mathbf{x}_t)=\mathbb E_{\mathbf{x}_0\mid \mathbf{x}_t}[s_{\text{cond}}(\mathbf{x}_t\mid \mathbf{x}_0)].
$$

## 5. 模型：噪声预测形式与隐式 score

DDPM 采用噪声预测网络（Ho et al.）：
$$
\hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t,t)\approx \boldsymbol{\epsilon}.
$$
利用上式关系，可定义模型隐式 score：
$$
s_\theta(\mathbf{x}_t,t):= -\frac{1}{\sqrt{1-\bar\alpha_t}}\,\hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t,t).
$$
由于这是关于 $\hat{\boldsymbol{\epsilon}}_\theta$ 的线性可逆变换，只要噪声预测最优，则对应的 $s_\theta$ 也最优。

## 6. DDPM 的 ELBO（严格推导与参数化，参照引用博客）

我们的目标是最大化数据似然 $\log q(\mathbf{x}_0)$ 的下界（ELBO）。构造生成模型：
$$
p_\theta(\mathbf{x}_{0:T}) = p(\mathbf{x}_T)\prod_{t=1}^T p_\theta(\mathbf{x}_{t-1}\mid \mathbf{x}_t), \quad p(\mathbf{x}_T)=\mathcal N(\mathbf{0},\mathbf{I}).
$$
前向（变分）分布：
$$
q(\mathbf{x}_{0:T}) = q(\mathbf{x}_0)\prod_{t=1}^T q(\mathbf{x}_t\mid \mathbf{x}_{t-1}),
\quad q(\mathbf{x}_t\mid \mathbf{x}_{t-1})=\mathcal N\bigl(\sqrt{\alpha_t}\mathbf{x}_{t-1},\beta_t \mathbf{I}\bigr).
$$

对 $\log q(\mathbf{x}_0)$ 写出下界：
$$
\log q(\mathbf{x}_0)=\log \int q(\mathbf{x}_{1:T}\mid \mathbf{x}_0)\frac{p_\theta(\mathbf{x}_{0:T})}{p_\theta(\mathbf{x}_{0:T})} d\mathbf{x}_{1:T}
\ge \mathbb E_{q(\mathbf{x}_{1:T}\mid \mathbf{x}_0)}\bigl[\log p_\theta(\mathbf{x}_{0:T}) - \log q(\mathbf{x}_{1:T}\mid \mathbf{x}_0)\bigr]
= -\mathcal L_{\text{ELBO}}.
$$

将 $\mathcal L_{\text{ELBO}}$ 分块（与 Ho et al. / 引用博客一致）：
$$
\mathcal L_{\text{ELBO}} = \mathcal L_T + \sum_{t=2}^T \mathcal L_{t-1} + \mathcal L_0,
$$
其中
$$
\mathcal L_T = \mathrm{KL}\bigl(q(\mathbf{x}_T\mid \mathbf{x}_0)\,\|\,p(\mathbf{x}_T)\bigr),
$$
$$
\mathcal L_{t-1} = \mathbb E_{q(\mathbf{x}_0,\mathbf{x}_t)}\Bigl[\mathrm{KL}\bigl(q(\mathbf{x}_{t-1}\mid \mathbf{x}_t,\mathbf{x}_0)\,\|\,p_\theta(\mathbf{x}_{t-1}\mid \mathbf{x}_t)\bigr)\Bigr],\quad 2\le t\le T,
$$
$$
\mathcal L_0 = -\log p_\theta(\mathbf{x}_0\mid \mathbf{x}_1).
$$

其中：
$$
q(\mathbf{x}_{t-1}\mid \mathbf{x}_t,\mathbf{x}_0)=\mathcal N\bigl(\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t,\mathbf{x}_0),\tilde{\beta}_t\mathbf{I}\bigr),
\quad
\tilde{\boldsymbol{\mu}}_t=\frac{1}{\sqrt{\alpha_t}}\Bigl(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\boldsymbol{\epsilon}\Bigr),
\quad
\tilde{\beta}_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t.
$$

设生成后验近似：
$$
p_\theta(\mathbf{x}_{t-1}\mid \mathbf{x}_t)=\mathcal N\bigl(\boldsymbol{\mu}_\theta(\mathbf{x}_t,t),\sigma_t^2 \mathbf{I}\bigr).
$$
常见策略：
1) 固定 $\sigma_t^2 = \tilde{\beta}_t$；
2) 固定为 $\beta_t$；
3) 或学习插值（此处取 $\sigma_t^2=\tilde{\beta}_t$）。

在固定协方差时：
$$
\mathcal L_{t-1} = \mathbb E_{q(\mathbf{x}_0,\mathbf{x}_t)}
\left[\frac{1}{2\sigma_t^2}\|\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t,\mathbf{x}_0)-\boldsymbol{\mu}_\theta(\mathbf{x}_t,t)\|^2\right] + C_t.
$$

采用噪声预测参数化：
$$
\boldsymbol{\mu}_\theta(\mathbf{x}_t,t)=\frac{1}{\sqrt{\alpha_t}}\Bigl(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t,t)\Bigr).
$$

则
$$
\tilde{\boldsymbol{\mu}}_t - \boldsymbol{\mu}_\theta
= \frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1-\bar\alpha_t}}\bigl(\hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t,t)-\boldsymbol{\epsilon}\bigr),
$$
所以
$$
\mathcal L_{t-1}
= \mathbb E_{q(\mathbf{x}_0,\mathbf{x}_t)}\left[
   \frac{1}{2\sigma_t^2}
   \frac{\beta_t^2}{\alpha_t(1-\bar\alpha_t)}
   \|\hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t,t)-\boldsymbol{\epsilon}\|^2
\right] + C_t.
$$

定义
$$
w_t := \frac{\beta_t^2}{2\sigma_t^2 \alpha_t(1-\bar\alpha_t)},
$$
得
$$
\mathcal L_{t-1} = w_t\,\mathbb E\|\hat{\boldsymbol{\epsilon}}_\theta - \boldsymbol{\epsilon}\|^2 + C_t.
$$

在 $\sigma_t^2=\tilde{\beta}_t$ 时，
$$
w_t = \frac{\beta_t^2}{2\tilde{\beta}_t \alpha_t(1-\bar\alpha_t)}.
$$

“简化损失”（舍弃 $w_t$）：
$$
\mathcal L_{\text{simple}} = \mathbb E_{t,\mathbf{x}_0,\boldsymbol{\epsilon}}\|\boldsymbol{\epsilon}-\hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t,t)\|^2,
\quad \mathbf{x}_t=\sqrt{\bar\alpha_t}\mathbf{x}_0+\sqrt{1-\bar\alpha_t}\boldsymbol{\epsilon}.
$$

## 7. 噪声损失与条件 score 代换（含权重形式）

固定 $t$，加权损失：
$$
\mathcal J_t := w_t\,\mathbb E_{\mathbf{x}_0,\boldsymbol{\epsilon}}\|\boldsymbol{\epsilon}-\hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t,t)\|^2.
$$
乘以 $(1-\bar\alpha_t)$：
$$
(1-\bar\alpha_t)\mathcal J_t = w_t\,\mathbb E\|\sqrt{1-\bar\alpha_t}\boldsymbol{\epsilon}-\sqrt{1-\bar\alpha_t}\hat{\boldsymbol{\epsilon}}_\theta\|^2.
$$

用
$$
\sqrt{1-\bar\alpha_t}\,\boldsymbol{\epsilon}=-(1-\bar\alpha_t)s_{\text{cond}}(\mathbf{x}_t\mid \mathbf{x}_0),\quad
\sqrt{1-\bar\alpha_t}\,\hat{\boldsymbol{\epsilon}}_\theta=-(1-\bar\alpha_t)s_\theta(\mathbf{x}_t,t),
$$
得
$$
(1-\bar\alpha_t)\mathcal J_t = w_t(1-\bar\alpha_t)^2
\mathbb E\|s_{\text{cond}}(\mathbf{x}_t\mid \mathbf{x}_0)-s_\theta(\mathbf{x}_t,t)\|^2.
$$

去掉与 $\theta$ 无关的正因子：
$$
\mathcal J_t \propto
\mathbb E_{\mathbf{x}_t}\mathbb E_{\mathbf{x}_0\mid \mathbf{x}_t}
\|s_{\text{cond}} - s_\theta\|^2.
$$

条件方差-偏差分解（固定 $\mathbf{x}_t$）：
$$
\mathbb E_{\mathbf{x}_0\mid \mathbf{x}_t}\|s_{\text{cond}} - s_\theta\|^2
= \mathbb E_{\mathbf{x}_0\mid \mathbf{x}_t}\|s_{\text{cond}} - s_t\|^2
  + \|s_t(\mathbf{x}_t)-s_\theta(\mathbf{x}_t,t)\|^2,
$$
因为
$$
\mathbb E_{\mathbf{x}_0\mid \mathbf{x}_t}[s_{\text{cond}}]=s_t.
$$

故
$$
\mathcal J_t \propto
\mathbb E_{\mathbf{x}_t}\Bigl[
  \underbrace{\mathbb E_{\mathbf{x}_0\mid \mathbf{x}_t}\|s_{\text{cond}} - s_t\|^2}_{\text{常数}}
  + \|s_t - s_\theta\|^2
\Bigr],
$$
其与 $\theta$ 相关部分为
$$
\mathbb E_{\mathbf{x}_t}\|s_\theta(\mathbf{x}_t,t)-s_t(\mathbf{x}_t)\|^2.
$$

因此无论是否保留原始权重 $w_t$，最小化噪声预测 MSE（或其加权形式）都等价于进行分层的 score matching，进而与最大似然（ELBO）优化一致。

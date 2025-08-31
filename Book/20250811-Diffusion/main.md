# Diffusion Models

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

定义任意随机变量分布 $q(\mathbf{x}_t)$，其 score function 定义为：
$$
s_t(\mathbf{x}_t) := \nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t).
$$

我们额外定义条件 score（给定 $\mathbf{x}_0$）：
$$
s_{\text{cond}}(\mathbf{x}_t \mid \mathbf{x}_0) := \nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t \mid \mathbf{x}_0).
$$

由前向扩散我们有：
$$
q(\mathbf{x}_t \mid \mathbf{x}_0)=\mathcal N\bigl(\mathbf{x}_t;\sqrt{\bar\alpha_t}\mathbf{x}_0,(1-\bar\alpha_t)\mathbf{I}\bigr),
$$
我们可以直接求导，得到
$$
s_{\text{cond}}(\mathbf{x}_t \mid \mathbf{x}_0) = -\frac{1}{1-\bar\alpha_t}\bigl(\mathbf{x}_t - \sqrt{\bar\alpha_t}\mathbf{x}_0\bigr).
$$
把前向扩散使用重参数化，得到
$$
\mathbf{x}_t = \sqrt{\bar\alpha_t}\mathbf{x}_0 + \sqrt{1-\bar\alpha_t}\boldsymbol{\epsilon},\quad \boldsymbol{\epsilon}\sim\mathcal N(\mathbf{0},\mathbf{I}),
$$
带入条件 score，得到
$$
s_{\text{cond}}(\mathbf{x}_t \mid \mathbf{x}_0)= -\frac{1}{\sqrt{1-\bar\alpha_t}}\boldsymbol{\epsilon},
$$
注意，此时我们已经固定了 $\mathbf{x_0}$ 和 $\mathbf{x_t}$, 此时 $\mathbf{\epsilon}$ 被唯一确定，因此  $\mathbf{\epsilon}$ 不再是随机变量，而是确定采样值：
$$
\boldsymbol{\epsilon} = \frac{\mathbf{x}_t - \sqrt{\bar\alpha_t}\mathbf{x}_0}{\sqrt{1-\bar\alpha_t}}
$$

## 4. Score Function 是条件 Score 的条件期望

由于
$$
q(\mathbf{x}_t)=\int q(\mathbf{x}_t\mid \mathbf{x}_0)\,q(\mathbf{x}_0)\,d\mathbf{x}_0,
$$
我们对 $\mathbf{x}_t$ 求梯度：
$$
\nabla_{\mathbf{x}_t} q(\mathbf{x}_t) = \int q(\mathbf{x}_t\mid \mathbf{x}_0)\, s_{\text{cond}}(\mathbf{x}_t\mid \mathbf{x}_0)\, q(\mathbf{x}_0)\, d\mathbf{x}_0.
$$
故
$$
\begin{aligned}
s_t(\mathbf{x}_t) &= \nabla_{\mathbf{x}_t}\log q(\mathbf{x}_t) \\
                  &= \frac{1}{q(\mathbf{x}_t)}\int q(\mathbf{x}_t\mid \mathbf{x}_0) q(\mathbf{x}_0) s_{\text{cond}}(\mathbf{x}_t\mid \mathbf{x}_0)\, d\mathbf{x}_0 \\
                  &= \mathbb{E}_{q(\mathbf{x}_0\mid \mathbf{x}_t)}\bigl[s_{\text{cond}}(\mathbf{x}_t\mid \mathbf{x}_0)\bigr]
\end{aligned}
$$

即
$$
s_t(\mathbf{x}_t)=\mathbb E_{\mathbf{x}_0\mid \mathbf{x}_t}[s_{\text{cond}}(\mathbf{x}_t\mid \mathbf{x}_0)].
$$

## 5. 模型：噪声预测形式与隐式 score

DDPM 采用噪声预测网络（Ho et al.）：
$
\hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t,t)\approx \boldsymbol{\epsilon}.
$
利用上式关系，可定义模型隐式 score：
$$
s_\theta(\mathbf{x}_t,t):= -\frac{1}{\sqrt{1-\bar\alpha_t}}\,\hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t,t).
$$
由于这是关于 $\hat{\boldsymbol{\epsilon}}_\theta$ 的线性可逆变换，只要噪声预测最优，则对应的 $s_\theta$ 也最优。

## 6. ELBO: Loss Function 推导
我们的目标是最大化  $\log p_{\mathbf{\theta}}(\mathbf{x}_0)$, 我们可以最大化其下界。

根据 Diffusion 反向扩散过程，我们可以得到联合概率：
$$
p_\theta(\mathbf{x}_{0:T}) = p(\mathbf{x}_T)\prod_{t=1}^T p_\theta(\mathbf{x}_{t-1}\mid \mathbf{x}_t), \quad p(\mathbf{x}_T)=\mathcal N(\mathbf{0},\mathbf{I}).
$$

我们构造 ELBO：
$$
\begin{aligned}
- \log p_\theta(\mathbf{x}_0) 
&\leq -\log p_\theta(\mathbf{x}_0) + \mathrm{KL}\big(q(\mathbf{x}_{1:T}|\mathbf{x}_0)\,\|\,p_\theta(\mathbf{x}_{1:T}|\mathbf{x}_0)\big) \\
&= -\log p_\theta(\mathbf{x}_0) + \mathbb{E}_{\mathbf{x}_{1:T} \sim q(\mathbf{x}_{1:T}|\mathbf{x}_0)}\left[\log \frac{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}{p_\theta(\mathbf{x}_{1:T}|\mathbf{x}_0)}\right]
\end{aligned}
$$
注意到
$$
p_\theta(\mathbf{x}_{1:T}|\mathbf{x}_0) = \frac{p_\theta(\mathbf{x}_{0:T})}{p_\theta(\mathbf{x}_0)}
$$
因此

$$
\begin{aligned}
&- \log p_\theta(\mathbf{x}_0) + \mathbb{E}_{\mathbf{x}_{1:T} \sim q(\mathbf{x}_{1:T}|\mathbf{x}_0)}\left[\log \frac{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})/p_\theta(\mathbf{x}_0)}\right] \\
&= - \log p_\theta(\mathbf{x}_0) + \mathbb{E}_{\mathbf{x}_{1:T} \sim q(\mathbf{x}_{1:T}|\mathbf{x}_0)}\left[\log \frac{q(\mathbf{x}_{1:T}|\mathbf{x}_0)\,p_\theta(\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})}\right] \\
&= - \log p_\theta(\mathbf{x}_0) + \mathbb{E}_{\mathbf{x}_{1:T} \sim q(\mathbf{x}_{1:T}|\mathbf{x}_0)}\left[\log \frac{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})}\right] + \log p_\theta(\mathbf{x}_0) \\
&= \mathbb{E}_{\mathbf{x}_{1:T} \sim q(\mathbf{x}_{1:T}|\mathbf{x}_0)}\left[\log \frac{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})}\right]
\end{aligned}
$$

对不等式两边对 $q(\mathbf{x}_0)$ 求期望，有
$$
\begin{aligned}
    -\mathbb{E}_{q(\mathbf{x}_0)}[\log p_\theta(\mathbf{x}_0)] 
    &\leq \mathbb{E}_{\mathbf{x}_0 \sim q(\mathbf{x}_0)}\left[
        \mathbb{E}_{\mathbf{x}_{1:T} \sim q(\mathbf{x}_{1:T}|\mathbf{x}_0)}
        \left[
            \log \frac{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})}
        \right]
    \right] \\
    &= \mathbb{E}_{\mathbf{x}_{0:T} \sim q(\mathbf{x}_{0:T})}
    \left[
        \log \frac{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})}
    \right]
\end{aligned}
$$
我们定义
$$
\mathcal{L}_{\mathrm{ELBO}} = \mathbb{E}_{\mathbf{x}_{0:T} \sim q(\mathbf{x}_{0:T})}
\left[
    \log \frac{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})}
\right]
$$
通过最小化 $\mathcal{L}_{\mathrm{ELBO}}$，即可最大化 $\log p_\theta(\mathbf{x}_0)$。

进一步推导（详见 [Lilian Weng 博客](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#reverse-diffusion-process)），我们可以将 $\mathcal{L}_{\mathrm{ELBO}}$ 拆解为如下形式：

$$
\begin{aligned}
\mathcal{L}_{\mathrm{ELBO}} =\; &\mathbb{E}_{q(\mathbf{x}_{0:T})} \Bigg[
\mathrm{KL}\big(q(\mathbf{x}_T|\mathbf{x}_0)\;\|\;p(\mathbf{x}_T)\big)
+ \sum_{t=2}^T \mathrm{KL}\big(q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)\;\|\;p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)\big) - \log p_\theta(\mathbf{x}_0|\mathbf{x}_1) \Bigg]
\end{aligned}
$$
其中第一项是常数。

根据前述推导，反向真实后验分布 $q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)$ 具有如下高斯形式（与前文符号保持一致）：
$$
q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}\left(\mathbf{x}_{t-1};\, \tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0),\, \tilde{\beta}_t \mathbf{I}\right)
$$
其中
$$
\begin{aligned}
\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0) &= \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\boldsymbol{\epsilon}\right) \\
\tilde{\beta}_t &= \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t
\end{aligned}
$$
$\boldsymbol{\epsilon}$ 为前向过程中的噪声，$\alpha_t = 1-\beta_t$，$\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$。

在训练时，$\mathbf{x}_t$ 和 $\mathbf{x}_0$ 已知，$\boldsymbol{\epsilon}$ 也可由重参数化得到。

我们希望模型 $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$ 能尽可能拟合上述真实后验，因此构造：
$$
p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t) = \mathcal{N}\left(\mathbf{x}_{t-1};\, \boldsymbol{\mu}_\theta(\mathbf{x}_t, t),\, \Sigma_\theta(\mathbf{x}_t, t)\right)
$$
其中，$\boldsymbol{\mu}_\theta$ 采用噪声预测参数化（noise prediction parameterization）：
$$
\boldsymbol{\mu}_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t)\right)
$$
$\hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t)$ 为模型预测的噪声。

因此，模型的反向分布为
$$
p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t) = \mathcal{N}\left(\mathbf{x}_{t-1};\, \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t)\right),\, \Sigma_\theta(\mathbf{x}_t, t)\right)
$$

带入上述推导，$L_{\mathrm{VLB}}$ 的损失函数可写为：

$$
\begin{aligned}
L_t &= \mathbb{E}_{\mathbf{x}_0,\, \boldsymbol{\epsilon}} \left[ 
    \frac{1}{2 \|\Sigma_\theta(\mathbf{x}_t, t)\|_2^2} 
    \left\| \tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0) - \boldsymbol{\mu}_\theta(\mathbf{x}_t, t) \right\|^2 
\right] \\
&= \mathbb{E}_{\mathbf{x}_0,\, \boldsymbol{\epsilon}} \left[
    \frac{(1-\alpha_t)^2}{2\alpha_t(1-\bar{\alpha}_t)\|\Sigma_\theta(\mathbf{x}_t, t)\|_2^2}
    \left\| \boldsymbol{\epsilon} - \hat{\boldsymbol{\epsilon}}_\theta\left(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon},\, t\right) \right\|^2
\right]
\end{aligned}
$$
Ho et al. 2020 针对这个公式提出了简化方案，首先不估计方差，而是用 alpha 相关的表达式代替，其次舍弃了每一项前的权重系数。实验结果证明收敛性能更好了。简化后的 Loss function:
$L_t^{simple} = E_{t\in [1,T], x0, \epsilon_t} [|| \boldsymbol{\epsilon} - \hat{\boldsymbol{\epsilon}}_\theta\left(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon},\, t\right) ||^2]$

## 7. 噪声损失与条件 score 代换（含权重形式）

我们将 $\boldsymbol{\epsilon}$ 与条件 score 的关系带入上式，得到：

$$
\begin{aligned}
L_t^{\mathrm{simple}} 
&= \mathbb{E}_{t \sim [1,T],\, \mathbf{x}_0,\, \boldsymbol{\epsilon}} \left[
    \left\| -\sqrt{1-\bar{\alpha}_t}\, s_{\mathrm{cond}}(\mathbf{x}_t|\mathbf{x}_0) + \sqrt{1-\bar{\alpha}_t}\, s_\theta(\mathbf{x}_t, t) \right\|^2
\right] \\
&= (1-\bar{\alpha}_t)\, \mathbb{E}_{t,\, \mathbf{x}_0,\, \boldsymbol{\epsilon}} \left[
    \left\| s_{\mathrm{cond}}(\mathbf{x}_t|\mathbf{x}_0) - s_\theta(\mathbf{x}_t, t) \right\|^2
\right]
\end{aligned}
$$

因此，有
$$
L_t^{\mathrm{simple}} \propto \mathbb{E}_{\mathbf{x}_0,\, \boldsymbol{\epsilon}} \left[
    \left\| s_{\mathrm{cond}}(\mathbf{x}_t|\mathbf{x}_0) - s_\theta(\mathbf{x}_t, t) \right\|^2
\right]
$$

进一步，可以写为
$$
\propto \mathbb{E}_{\mathbf{x}_t} \left[
    \mathbb{E}_{\mathbf{x}_0|\mathbf{x}_t} \left\| s_{\mathrm{cond}}(\mathbf{x}_t|\mathbf{x}_0) - s_\theta(\mathbf{x}_t, t) \right\|^2
\right]
$$

利用条件方差-偏差分解，有
$$
\mathbb{E}_{\mathbf{x}_0|\mathbf{x}_t} \left\| s_{\mathrm{cond}} - s_\theta \right\|^2
= \mathbb{E}_{\mathbf{x}_0|\mathbf{x}_t} \left\| s_{\mathrm{cond}} - s_t \right\|^2 + \left\| s_t - s_\theta \right\|^2
$$
其中 $s_t = \mathbb{E}_{\mathbf{x}_0|\mathbf{x}_t} [s_{\mathrm{cond}}]$，第一项与 $\theta$ 无关。

因此，最终优化目标为
$$
L_t^{\mathrm{simple}} \propto \mathbb{E}_{q(\mathbf{x}_t, t)} \left[
    \left\| s_t - s_\theta(\mathbf{x}_t, t) \right\|^2
\right]
$$
即模型 $s_\theta$ 拟合条件 score 的均值 $s_t$。因此，优化 $\boldsymbol{\epsilon}$ 等价于优化 score function。用公式表达为：
$$
\begin{aligned}
L_t^{\mathrm{simple}} &\propto \mathbb{E}_{q(\mathbf{x}_t, t)} \left[
    \left\| s_t - s_\theta(\mathbf{x}_t, t) \right\|^2
\right] \\
&= \mathbb{E}_{q(\mathbf{x}_t, t)} \left[
    \left\| \mathbb{E}_{\mathbf{x}_0|\mathbf{x}_t} [s_{\mathrm{cond}}(\mathbf{x}_t|\mathbf{x}_0)] - s_\theta(\mathbf{x}_t, t) \right\|^2
\right]
\end{aligned}
$$
因此，最小化噪声预测损失 $L_t^{\mathrm{simple}}$，本质上就是让 $s_\theta$ 拟合条件 score 的均值 $s_t$，即优化 $\boldsymbol{\epsilon}$ 等价于优化 score function。
# Diffusion Models from SDE Perspective

## Step 0. 离散前向马尔可夫链定义

给定数据 $\mathbf{x}_0 \sim q(\mathbf{x})$，一步扩散：
$$
q(\mathbf{x}_t|\mathbf{x}_{t-1}) = \mathcal{N}\big(\sqrt{1-\beta_t}\,\mathbf{x}_{t-1},\; \beta_t \mathbf{I}\big),\quad t=1,\dots,T.
$$
等价写成显式更新：
$$
\mathbf{x}_t = \sqrt{1-\beta_t}\,\mathbf{x}_{t-1} + \sqrt{\beta_t}\,\boldsymbol{\epsilon}_{t-1},\quad \boldsymbol{\epsilon}_{t-1}\sim\mathcal{N}(\mathbf{0},\mathbf{I}).
$$

## Step 1. 目标：让步长变小以形成连续极限

我们希望把"很多很小的高斯加噪"极限化为一个时间连续扩散过程。为此让总时间区间固定为 $[0,1]$，把离散索引映射：
$$
\tau_t = t\Delta\tau,\quad \Delta\tau = \frac{1}{T}.
$$

设存在一个光滑、正、界内的函数 $\beta(\tau)$，定义：
$$
\beta_t = \beta(\tau_t)\Delta\tau.
$$
理由：
1. 保证 $\beta_t = O(\Delta\tau)$（小步）以便泰勒展开。
2. 累积噪声方差：$\sum \beta_t \approx \int_0^1 \beta(\tau)d\tau$ 有一个有限极限。

（若你原始调度是一个数组，可以用插值定义 $\beta(\tau)$；理论推导反过来假设它本就存在。）

## Step 2. 写成"增量"形式

从更新式减去 $\mathbf{x}_{t-1}$：
$$
\mathbf{x}_t - \mathbf{x}_{t-1} = \big(\sqrt{1-\beta_t}-1\big)\mathbf{x}_{t-1} + \sqrt{\beta_t}\,\boldsymbol{\epsilon}_{t-1}.
$$

## Step 3. 泰勒展开（保留一阶）

用 $\sqrt{1-u} = 1 - \frac{1}{2} u - \frac{1}{8} u^2 + O(u^3)$，令 $u=\beta_t$：
$$
\sqrt{1-\beta_t}-1 = -\frac{1}{2} \beta_t + O(\beta_t^2).
$$
因为 $\beta_t = O(\Delta\tau)$，故 $\beta_t^2 = O((\Delta\tau)^2)$ 是高阶，可忽略（构造一阶弱精度 SDE 足够）。
得到：
$$
\mathbf{x}_t - \mathbf{x}_{t-1} = -\frac{1}{2} \beta_t \mathbf{x}_{t-1} + \sqrt{\beta_t}\,\boldsymbol{\epsilon}_{t-1} + O((\Delta\tau)^2)\mathbf{x}_{t-1}.
$$

代入尺度 $\beta_t = \beta(\tau_{t-1})\Delta\tau$

（在一阶精度里 $\tau_t$ 或 $\tau_{t-1}$ 均可，差别是高阶）
$$
\begin{aligned}
\mathbf{x}_t - \mathbf{x}_{t-1}
&= -\frac{1}{2} \beta(\tau_{t-1}) \mathbf{x}_{t-1}\Delta\tau \\
&\quad + \sqrt{\beta(\tau_{t-1})}\sqrt{\Delta\tau}\,\boldsymbol{\epsilon}_{t-1} + o(\Delta\tau).
\end{aligned}
$$

**识别 Euler–Maruyama 模板**

标准 Euler–Maruyama 离散化：
$$
\Delta \mathbf{x} = f(\mathbf{x},\tau)\Delta\tau + g(\tau)\sqrt{\Delta\tau}\,\boldsymbol{\epsilon}.
$$
对比得：
$$
f(\mathbf{x},\tau) = -\frac{1}{2} \beta(\tau)\mathbf{x},\qquad g(\tau)=\sqrt{\beta(\tau)}.
$$

## Step 4. 把高斯噪声刻画成 Wiener 增量

独立标准正态 $\boldsymbol{\epsilon}_{t-1}$ 乘 $\sqrt{\Delta\tau}$ 满足：
$$
\sqrt{\Delta\tau}\,\boldsymbol{\epsilon}_{t-1} \stackrel{d}{=} \mathbf{W}_{\tau_t} - \mathbf{W}_{\tau_{t-1}},
$$
其中 $\mathbf{W}_\tau$ 是 $d$ 维标准 Wiener 过程（布朗运动）：增量独立、$\mathcal{N}(\mathbf{0}, h\mathbf{I})$。  
因此：
$$
\sqrt{\beta(\tau_{t-1})}\sqrt{\Delta\tau}\,\boldsymbol{\epsilon}_{t-1}
= \sqrt{\beta(\tau_{t-1})}\big(\mathbf{W}_{\tau_t}-\mathbf{W}_{\tau_{t-1}}\big).
$$

**写出极限 SDE（前向扩散 SDE）**

令 $\Delta\tau \to 0$：
$$
d\mathbf{x}_\tau = -\frac{1}{2} \beta(\tau)\mathbf{x}_\tau d\tau + \sqrt{\beta(\tau)} d\mathbf{W}_\tau,\quad \mathbf{x}_0 \sim q(\mathbf{x}).
$$
这就是（VP）前向扩散 SDE。  

## Step 5. 反向扩散 SDE

有了前向扩散 SDE，可通过 Anderson 1982 得到反向扩散 SDE（时间变量反向）：

若正向：
$$
d\mathbf{x}_\tau = f(\mathbf{x}_\tau,\tau)d\tau + g(\tau)d\mathbf{W}_\tau
$$

则逆向（把时间变量反向）过程的 SDE：
$$
d\mathbf{x}_\tau = \left(f(\mathbf{x}_\tau,\tau) - g(\tau)^2 \nabla_{\mathbf{x}} \log q_\tau(\mathbf{x}_\tau)\right)d\tau + g(\tau)d\bar{\mathbf{W}}_\tau,
$$

其中此处的 $d\tau$ 是向"减小方向"走（实现时常用离散循环从 $t=T$ 到 $1$），$d\bar{\mathbf{W}}_\tau$ 是新的 Wiener 过程（与前向独立）。

对于我们的 VP 扩散，代入 $f(\mathbf{x},\tau) = -\frac{1}{2} \beta(\tau)\mathbf{x}$ 和 $g(\tau)=\sqrt{\beta(\tau)}$：
$$
\begin{aligned}
d\mathbf{x}_\tau &= \left(-\frac{1}{2} \beta(\tau)\mathbf{x}_\tau - \beta(\tau) \nabla_{\mathbf{x}} \log q_\tau(\mathbf{x}_\tau)\right)d\tau \\
&\quad + \sqrt{\beta(\tau)} d\bar{\mathbf{W}}_\tau
\end{aligned}
$$

这就是反向扩散 SDE，其中 $\nabla_{\mathbf{x}} \log q_\tau(\mathbf{x}_\tau)$ 是数据分布的对数概率密度梯度，需要通过神经网络学习估计。


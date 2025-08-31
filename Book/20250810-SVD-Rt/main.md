# 1. 问题定义


已知有 $N$ 对对应点 $(\mathbf{p}_i, \mathbf{q}_i),\ i=1, \ldots, N,\ \mathbf{p}_i, \mathbf{q}_i \in \mathbb{R}^3$，目标：找到旋转矩阵 ${\mathbf{R}} \in SO(3)\ (\mathbf{R}^\mathrm{T}\mathbf{R} = \mathbf{I},\ \det\mathbf{R} = 1)$ 和平移向量 ${\mathbf{t}} \in \mathbb{R}^3$ 使得

$$
\mathbf{R},\ \mathbf{t} = \arg\min_{\mathbf{R},\ \mathbf{t}} F(\mathbf{R}, \mathbf{t}) = \sum_{i=1}^N \left\| \mathbf{R} \mathbf{p}_i + \mathbf{t} - \mathbf{q}_i \right\|^2
$$

# 2. Solution

## 2.1. 分离平移: 对 $\mathbf{t}$ 求最优解

目标函数是 $F(\mathbf{R}, \mathbf{t}) = \sum_{i=1}^N \left\| \mathbf{R} \mathbf{p}_i + \mathbf{t} - \mathbf{q}_i \right\|^2$，首先我们对 $\mathbf{t}$ 求导并令其为零，得到最优的平移向量：

$$
\begin{aligned}
\frac{\partial F(\mathbf{R}, \mathbf{t})}{\partial \mathbf{t}} &= 0 \\
2 \sum_{i=1}^N \left( \mathbf{R} \mathbf{p}_i + \mathbf{t} - \mathbf{q}_i \right) &= 0
\end{aligned}
$$

我们得到最优平移向量
$\mathbf{t}^* = \frac{1}{N} \sum_{i=1}^N \mathbf{q}_i - \mathbf{R} \left( \frac{1}{N} \sum_{i=1}^N \mathbf{p}_i \right)$，
令
$\bar{\mathbf{p}} = \frac{1}{N} \sum_{i=1}^N \mathbf{p}_i, \quad \bar{\mathbf{q}} = \frac{1}{N} \sum_{i=1}^N \mathbf{q}_i$，
则有
$\mathbf{t}^* = \bar{\mathbf{q}} - \mathbf{R} \bar{\mathbf{p}}$。
将其代入目标函数，得到
$F(\mathbf{R}, \mathbf{t}^*) = \sum_{i=1}^N \left\| \mathbf{R} (\mathbf{p}_i - \bar{\mathbf{p}}) - (\mathbf{q}_i - \bar{\mathbf{q}}) \right\|^2$，
令
$\tilde{\mathbf{p}}_i = \mathbf{p}_i - \bar{\mathbf{p}}, \quad \tilde{\mathbf{q}}_i = \mathbf{q}_i - \bar{\mathbf{q}}$，
则有

$$F(\mathbf{R}, \mathbf{t}^*) = \sum_{i=1}^N \left\| \mathbf{R} \tilde{\mathbf{p}}_i - \tilde{\mathbf{q}}_i \right\|^2$$

## 2.2. 优化目标转换
对于其中的一项，$\left\| \mathbf{R} \tilde{\mathbf{p}}_i - \tilde{\mathbf{q}}_i \right\|^2$，我们将其展开，有：

$$
\begin{aligned}
\left\| \mathbf{R} \tilde{\mathbf{p}}_i - \tilde{\mathbf{q}}_i \right\|^2 
&= \left( \mathbf{R} \tilde{\mathbf{p}}_i - \tilde{\mathbf{q}}_i \right)^\mathrm{T} \left( \mathbf{R} \tilde{\mathbf{p}}_i - \tilde{\mathbf{q}}_i \right) \\
&= \left\| \mathbf{R} \tilde{\mathbf{p}}_i \right\|^2 + \left\| \tilde{\mathbf{q}}_i \right\|^2 - 2 \tilde{\mathbf{q}}_i^\mathrm{T} \mathbf{R} \tilde{\mathbf{p}}_i
\end{aligned}
$$

由于旋转保持长度，即 $\left\| \mathbf{R} \tilde{\mathbf{p}}_i \right\|^2 = \left\| \tilde{\mathbf{p}}_i \right\|^2$，因此有：

$$
\left\| \mathbf{R} \tilde{\mathbf{p}}_i - \tilde{\mathbf{q}}_i \right\|^2 
= \left\| \tilde{\mathbf{p}}_i \right\|^2 + \left\| \tilde{\mathbf{q}}_i \right\|^2 - 2 \tilde{\mathbf{q}}_i^\mathrm{T} \mathbf{R} \tilde{\mathbf{p}}_i
$$

其中前两项与 $\mathbf{R}$ 无关，因此我们的目标从最小化 $F(\mathbf{R}, \mathbf{t})$ 等价于最大化 $\sum_{i=1}^N \tilde{\mathbf{q}}_i^\mathrm{T} \mathbf{R} \tilde{\mathbf{p}}_i$。

对于每一项 $\tilde{\mathbf{q}}_i^\mathrm{T} \mathbf{R} \tilde{\mathbf{p}}_i$，我们有
$$
\tilde{\mathbf{q}}_i^\mathrm{T} \mathbf{R} \tilde{\mathbf{p}}_i = \operatorname{Tr}\left( \tilde{\mathbf{q}}_i^\mathrm{T} \mathbf{R} \tilde{\mathbf{p}}_i \right ) = \operatorname{Tr}\left( \mathbf{R} \tilde{\mathbf{p}}_i \tilde{\mathbf{q}}_i^\mathrm{T} \right )
$$
这一步是由于标量等于标量的迹以及 Trace 的交换律: 

> $s_{1\times1} = \operatorname{Tr}(s_{1\times1})$
>
> $\operatorname{Tr}(\mathbf{A}\mathbf{B}\mathbf{C}) = \operatorname{Tr}(\mathbf{C}\mathbf{A}\mathbf{B}) = \operatorname{Tr}(\mathbf{B}\mathbf{C}\mathbf{A})$

因此，此时我们的目标可以写为
$$
\mathbf{R}^* = \arg\max_{\mathbf{R} \in \mathrm{SO}(3)} \sum_{i=1}^N \tilde{\mathbf{q}}_i^\mathrm{T} \mathbf{R} \tilde{\mathbf{p}}_i
$$
进一步利用上式，有
$$
\mathbf{R}^* = \arg\max_{\mathbf{R} \in \mathrm{SO}(3)} \operatorname{Tr} \left( \mathbf{R} \sum_{i=1}^N \tilde{\mathbf{p}}_i \tilde{\mathbf{q}}_i^\mathrm{T} \right )
$$

定义协方交叉矩阵
$$
\mathbf{H} = \sum_{i=1}^N \tilde{\mathbf{p}}_i \tilde{\mathbf{q}}_i^\mathrm{T}
$$
因此我们的优化目标为
$$
\mathbf{R}^* = \arg\max_{\mathbf{R} \in \mathrm{SO}(3)} \operatorname{Tr}(\mathbf{R} \mathbf{H})
$$

对 $\mathbf{H}$ 进行 SVD 分解，得到
$$
\mathbf{H} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^\mathrm{T}
$$
其中
$$
\boldsymbol{\Sigma} = \operatorname{diag}(\sigma_1, \sigma_2, \sigma_3), \quad \sigma_1 > \sigma_2 > \sigma_3
$$

目标函数
$$
\operatorname{Tr}(\mathbf{R} \mathbf{H}) = \operatorname{Tr}(\mathbf{R} \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^\mathrm{T}) = \operatorname{Tr}(\mathbf{V}^\mathrm{T} \mathbf{R} \mathbf{U} \boldsymbol{\Sigma})
$$

令
$$
\mathbf{S} = \mathbf{V}^\mathrm{T} \mathbf{R} \mathbf{U}
$$
则
$$
\mathbf{R} = \mathbf{V} \mathbf{S} \mathbf{U}^\mathrm{T}
$$
因为 $\mathbf{R}$、$\mathbf{U}$、$\mathbf{V}$ 均为正交矩阵，$\mathbf{S}$ 也为正交矩阵（$\mathbf{S} \in \mathrm{O}(3)$）。


我们有
$$
\det(\mathbf{R}) = \det(\mathbf{V}^\mathrm{T}) \det(\mathbf{S}) \det(\mathbf{U}) = 1,
$$
以及
$$
\det(\mathbf{S}) = \det(\mathbf{V}^\mathrm{T}) \det(\mathbf{R}) \det(\mathbf{U}) = \det(\mathbf{V}^\mathrm{T}) \det(\mathbf{U}) = \det(\mathbf{V}\mathbf{U}^\mathrm{T}),
$$


此时我们的目标变为
$$
\mathbf{S}^* = \arg\max_{\mathbf{S} \in \mathrm{O}(3),\ \det(\mathbf{S}) = \det{(\mathbf{VU^T})}} \operatorname{Tr}(\mathbf{S} \boldsymbol{\Sigma})
$$

## 2.3. 求解 $\mathbf{R}^*$

**首先我们忽略 $\det(\mathbf{S})$ 的约束**

此时
$$
\operatorname{Tr}(\mathbf{S} \boldsymbol{\Sigma}) = \sum_{i=1}^3 \sigma_i S_{ii}
$$
由于 $\mathbf{S}$ 是正交矩阵，$\mathbf{S}$ 中各个元素满足 $|S_{ij}| \leq 1$（参考正交矩阵定义），注意“单位向量”。因此最理想的情况是每个 $S_{ii} = 1$，其余元素均为 $0$，此时 $\mathbf{S} = \mathbf{I}$，$\operatorname{Tr}(\mathbf{S} \boldsymbol{\Sigma}) = \sum_{i=1}^3 \sigma_i$。

> 正交矩阵定义：$\mathbf{S}^\mathrm{T} \mathbf{S} = \mathbf{I}$，意味着每一行及每一列都是单位向量，且彼此正交。

**但同时要满足 $\det(\mathbf{S}) = \det(\mathbf{V}\mathbf{U}^\mathrm{T})$**

$\det(\mathbf{V}\mathbf{U}^\mathrm{T})$ 可能为 $+1$ 或 $-1$。

- 若 $\det(\mathbf{V}\mathbf{U}^\mathrm{T}) = 1$，则 $\det(\mathbf{S}) = \det(\mathbf{V}\mathbf{U}^\mathrm{T}) = 1$，此时为最优解。
- 若 $\det(\mathbf{V}\mathbf{U}^\mathrm{T}) = -1$，则 $\det(\mathbf{S}) \neq \det(\mathbf{V}\mathbf{U}^\mathrm{T})$，需要在 $\mathbf{S} = \mathbf{I}$ 的基础上翻转（取负）一个正交轴。

为了使 $\operatorname{Tr}(\mathbf{S} \boldsymbol{\Sigma}) = \sum_{i=1}^3 \sigma_i S_{ii}$ 最大，我们选择反转 $\sigma_3$（即最小特征值）对应的轴。

因此，
$$
\mathbf{S}_\text{opt} = \operatorname{diag}(1,\, 1,\, \det(\mathbf{V}\mathbf{U}^\mathrm{T}))
$$
此时
$$
\mathbf{R}^* = \mathbf{V} \mathbf{S}_\text{opt} \mathbf{U}^\mathrm{T}
$$
带回可得
$$
\mathbf{t}^* = \bar{\mathbf{q}} - \mathbf{R}^* \bar{\mathbf{p}}
$$

# 3. 延伸

- 若 $\sigma_3 \approx 0$（即点集共面或近线性），翻转最小轴几乎不影响目标函数，最优 $\mathbf{R}$ 可能在该轴的转角上不唯一。

- 数值实现时，可以对 $\det(\mathbf{V}\mathbf{U}^\mathrm{T})$ 做阈值判断（例如仅当其接近 $-1$ 时才翻转），以抑制浮点噪声。

- 若有权重 $w_i$，则：
质心为：
$
\bar{\mathbf{p}} = \frac{\sum_i w_i \mathbf{p}_i}{\sum_i w_i}, \qquad
\bar{\mathbf{q}} = \frac{\sum_i w_i \mathbf{q}_i}{\sum_i w_i}
$
协方差矩阵为：
$
\mathbf{H} = \sum_i w_i\, \tilde{\mathbf{p}}_i\, \tilde{\mathbf{q}}_i^\mathrm{T}
$
其中 $\tilde{\mathbf{p}}_i = \mathbf{p}_i - \bar{\mathbf{p}}$，$\tilde{\mathbf{q}}_i = \mathbf{q}_i - \bar{\mathbf{q}}$。其余步骤相同。

- 若加入全局尺度 $s$（即相似变换），目标为：
$
\min_{s,\,\mathbf{R},\,\mathbf{t}} \sum_i \left\| s\,\mathbf{R}\mathbf{p}_i + \mathbf{t} - \mathbf{q}_i \right\|^2
$
先如前求 $\mathbf{R},\,\mathbf{t}$，然后
$
s^* = \frac{\sum_i \tilde{\mathbf{q}}_i^\mathrm{T} \mathbf{R}^* \tilde{\mathbf{p}}_i}{\sum_i \|\tilde{\mathbf{p}}_i\|^2} = \frac{\operatorname{trace}(\mathbf{R}^* \mathbf{H})}{\sum_i \|\tilde{\mathbf{p}}_i\|^2}
$



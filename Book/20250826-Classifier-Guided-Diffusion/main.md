# 1.Classifier Guided Diffusion

在基础扩散模型中，我们想要得到的是数据分布 $p(\mathbf{x})$, 我们通过反向过程从高斯噪声中去噪还原出样本。
Classifier Guided Diffusion 的目标是生成满足特定条件 (例如标签 $y$)的样本，即近似从条件分布 $p(\mathbf{x}|y)$ 采样。
其核心思想是：在反向扩散每一步更新中，利用一个对噪声污染样本仍然可微的分类器的梯度信息，来推动采样轨迹朝着标签一致的区域。

所以我们想要建模 $p(\mathbf{x}_0|y)$ ，和 Diffusion model 类似，我们需要找到反向扩散过程 $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t,y)$
我们可以利用 score 分解, 由 Bayes 公式我们得到：

$$\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t|y) = \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t) + \nabla_{\mathbf{x}_t} \log p(y|\mathbf{x}_t),$$

右侧第一项为无条件 Diffusion Model，右侧第二项我们可以通过分类器梯度给出。

## 1.1. 连续时间视角 (SDE 形式)

Diffusion Model 的前向&反向扩散可以用 SDE 表述：
前向 SDE $d\mathbf{x} = \mathbf{f}(\mathbf{x}, t) dt + g(t) d\mathbf{w}$, 逆向（无条件) SDE 为：
$$d\mathbf{x} = [\mathbf{f}(\mathbf{x}, t) - g(t)^2 \nabla_{\mathbf{x}} \log p_t(\mathbf{x})] dt + g(t) d\tilde{\mathbf{w}}$$

若要条件化，我们将 $\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t|y) = \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t) + \nabla_{\mathbf{x}_t} \log p(y|\mathbf{x}_t)$, 带入上式的 $\nabla_{\mathbf{x}} \log p_t(\mathbf{x})$ 中

$$d\mathbf{x} = [\mathbf{f}(\mathbf{x}, t) - g(t)^2 (\nabla_{\mathbf{x}} \log p_t(\mathbf{x}) + \nabla_{\mathbf{x}} \log p_t(y|\mathbf{x}))] dt + g(t) d\tilde{\mathbf{w}}$$

实际实现中， $\nabla_{\mathbf{x}_t} \log p(y|\mathbf{x}_t)$ 通过分类器梯度近似

为了调节条件的强度， 引入 guidance scale $s \geq 0$

$$d\mathbf{x} = [\mathbf{f}(\mathbf{x}, t) - g(t)^2 (\nabla_{\mathbf{x}} \log p_t(\mathbf{x}) + s\nabla_{\mathbf{x}} \log p_t(y|\mathbf{x}))] dt + g(t) d\tilde{\mathbf{w}}$$

## 1.2. 如何训练

训练分类器 $p_\phi(y|\mathbf{x}_t, t)$ 需让其在不同噪声水平仍能识别类别：

步骤：

1. 采样真实样本 $(\mathbf{x}_0, y)$
2. 采样时间步 $t \sim \text{Uniform}\{1,\ldots,T\}$
3. 采样噪声 $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$，构造 $\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t} \boldsymbol{\epsilon}$
4. 输入分类器：$\text{Classifier}(\mathbf{x}_t, t)$（$t$ 可嵌入为时间编码）
5. 交叉熵损失最小化 $-\log p_\phi(y|\mathbf{x}_t, t)$

推断时需对 $\mathbf{x}_t$ 计算 $\nabla_{\mathbf{x}_t} \log p_\phi(y|\mathbf{x}_t, t)$。注意：

- 分类器必须对噪声分布有鲁棒性；
- 可对输入标准化或使用额外归一层提升梯度稳定；
- 计算梯度是整个 guidance 额外推断开销的主要来源。

# 2. Classifier Free Diffusion

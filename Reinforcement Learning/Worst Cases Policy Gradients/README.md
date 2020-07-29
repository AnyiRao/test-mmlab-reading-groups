# Worst Cases Policy Gradients

📔 Peng Zhenghao

🖇️ [https://arxiv.org/pdf/1911.03618.pdf](https://arxiv.org/pdf/1911.03618.pdf)

🖋️ Tang, Yichuan Charlie, Jian Zhang, and Ruslan Salakhutdinov. "Worst cases policy gradients." arXiv preprint arXiv:1911.03618 (2019).

🏫 Apple

![Worst%20Cases%20Policy%20Gradients%20a6790e03d8194ee197326a0fb81c67ff/Untitled.png](Worst%20Cases%20Policy%20Gradients%20a6790e03d8194ee197326a0fb81c67ff/Untitled.png)


## Highlights


- 利用分布式RL的视角，提出一种能够动态控制策略风险厌恶程度的方法
- 基于DDPG算法，其Critic不仅输出Q值，而且输出Q值的方差，利用Wasserstein距离来更新Critic输出的方差项
- Return分布的百分比alpha作为衡量risk的指标，在训练时随机采样，在测试时可以手动条件策略的风险厌恶程度。

## Formulation

- 在RL中如何引入风险这个概念？如何最小化风险？
- 我们能否得到一种办法以动态的调整策略对风险的偏好？
- 设回报Return是一个概率分布，给定一个0、1之间的数alpha，现希望最大化Return的alpha-百分数。当alpha趋于0的时候，则我们希望最大化在”非常少见的、回报很低的“情况下的回报。此时即是Worst Cases。

## Method


### 训练模型来拟合回报的分布

- 假设return(s,a)的分布是一个高斯分布，其均值为传统的Q(s,a)函数给出的值，而其方差表示为 $\Upsilon(s,a) = E[R^2] - Q^2(s,a)$
- 使用一个神经网络来拟合方差：

![Worst%20Cases%20Policy%20Gradients%20a6790e03d8194ee197326a0fb81c67ff/Untitled%201.png](Worst%20Cases%20Policy%20Gradients%20a6790e03d8194ee197326a0fb81c67ff/Untitled%201.png)

- 使用TD learning来学习方差：

    用Wasserstein Metric做loss。

    ![Worst%20Cases%20Policy%20Gradients%20a6790e03d8194ee197326a0fb81c67ff/Untitled%202.png](Worst%20Cases%20Policy%20Gradients%20a6790e03d8194ee197326a0fb81c67ff/Untitled%202.png)

    F-1是inverse cumulative distribution function。设一个老的高斯分布u和一个新的高斯分布v分别服从：u~N(μ1, C1), v~N(μ2, C2), 则二者的Wasserstein距离为

    ![Worst%20Cases%20Policy%20Gradients%20a6790e03d8194ee197326a0fb81c67ff/Untitled%203.png](Worst%20Cases%20Policy%20Gradients%20a6790e03d8194ee197326a0fb81c67ff/Untitled%203.png)

### CVaR

- 因为我们已经假设了回报的分布是一个高斯函数，可以由Q和Υ分别表示均值和方差，所以这个分布的alpha-percentile的可以直接写出来

![Worst%20Cases%20Policy%20Gradients%20a6790e03d8194ee197326a0fb81c67ff/Untitled%204.png](Worst%20Cases%20Policy%20Gradients%20a6790e03d8194ee197326a0fb81c67ff/Untitled%204.png)

- 现在原始的policy gradient就可以写成（其实就是用上式替换了Q）下式。从而可以把 $\Gamma$ 展开来写从而得到最终的J。

![Worst%20Cases%20Policy%20Gradients%20a6790e03d8194ee197326a0fb81c67ff/Untitled%205.png](Worst%20Cases%20Policy%20Gradients%20a6790e03d8194ee197326a0fb81c67ff/Untitled%205.png)

- 一个新问题产生了，新的J与alpha有关，该选哪个alpha呢？直观的做法是将alpha离散化到0~1直接的N个区间，并训N个不同的策略，但这样参数太多了。本文的做法是将alpha作为输入传入策略网络中。在每个episode开始前，均匀随机采样一个alpha，并在这个episode中固定住它。

## Experiment


### Experimental Setting

- 两个连续动作的驾驶环境：1. 转弯和2. 汇入

![Worst%20Cases%20Policy%20Gradients%20a6790e03d8194ee197326a0fb81c67ff/Untitled%206.png](Worst%20Cases%20Policy%20Gradients%20a6790e03d8194ee197326a0fb81c67ff/Untitled%206.png)

- 环境的具体设计
    1. 本车
        1. 运动学模型为离散时间的kinematics bicycle model
        2. 转向由Stanley controller控制（一种非线性闭环转向控制器）
        3. 初始速度为5~20m/s随机采样
    2. 环境
        1. 大概200x200米
        2. 碰撞reward -50
        3. 成功reward $50\times e^{-t/50} + 10$，t表示目前的step
        4. 没有操作成功则是 0
    3. 他车
        1. rule-based行为
        2. 速度随机选取
        3. 可以执行adaptive curise control：基于前车的情况自动加减速
        4. 可以安全的变道：动态规划出一条平滑的变道轨迹
        5. 既然小车有这些能力了，他们在出生的时候随机选取三种性格：yield（礼让的），ignore（分心的），accelerate（路怒症）
- 网络的具体设计：
    1. alpha作为actor和critic的输入
    2. critic不仅输出均值Q，而且输出方差。方差使用softplus函数保证恒大于0。
    3. 输出的是距离本车最近的一些车的state如速度方向位置等。输出为小车的加速度。

![Worst%20Cases%20Policy%20Gradients%20a6790e03d8194ee197326a0fb81c67ff/Untitled%207.png](Worst%20Cases%20Policy%20Gradients%20a6790e03d8194ee197326a0fb81c67ff/Untitled%207.png)

### Result

- 下图左：预测的R的方差随着alpha的增大而增大。中：随着alpha的增加episode length变小，说明车开的越来越快。右图：输出的方差随着alpha的增大而增大。

![Worst%20Cases%20Policy%20Gradients%20a6790e03d8194ee197326a0fb81c67ff/Untitled%208.png](Worst%20Cases%20Policy%20Gradients%20a6790e03d8194ee197326a0fb81c67ff/Untitled%208.png)

- 下图展示数值结果。每一行表示一种setting。括号左边的百分比表示碰撞率，括号内的表示成功率。因为alpha在训练的时候作为输入传进网络，所以可以在测试的时候调整alpha的值从而画出不同的列，这点很好。

![Worst%20Cases%20Policy%20Gradients%20a6790e03d8194ee197326a0fb81c67ff/Untitled%209.png](Worst%20Cases%20Policy%20Gradients%20a6790e03d8194ee197326a0fb81c67ff/Untitled%209.png)

- 作者也在Carla上进行了测试，手工提取Carla中车的信息并传给策略网络。
- 笔者注：作者没有与其他Risk-averse的算法进行对比。

## Related Works

### Risk-sensitive, Safe RL, Robust MDP

- [ ]  L. Pinto, J. Davidson, R. Sukthankar, and A. Gupta. Robust adversarial reinforcement learning. *ICML*,
2017. URL https://arxiv.org/abs/1703.02702.

分成两类

- 修改探索过程
    - 利用外部知识
    - risk-directed exploration
- 修改训练时的最优条件。本篇工作就是这类。

### Distributional RL

- [ ]  M. G. Bellemare, W. Dabney, and R. Munos. A distributional perspective on reinforcement learning.
*CoRR*, abs/1707.06887, 2017. URL http://arxiv.org/abs/1707.06887.
- [ ]  W. Dabney, M. Rowland, M. G. Bellemare, and R. Munos. Distributional reinforcement learning with
quantile regression. *arXiv preprint arXiv:1710.10044*, 2017.
- [ ]  G. Barth-Maron, M. W. Hoffman, D. Budden, W. Dabney, D. Horgan, A. Muldal, N. Heess, and T. Lilli- crap. Distributed distributional deterministic policy gradients. arXiv preprint arXiv:1804.08617, 2018.

## Remained Questions


### Distributional RL

定义，内涵，用处，未来。

### Wasserstein Metric

- 为什么用它不用KL？
- 他的表达式是怎么计算的？

![Worst%20Cases%20Policy%20Gradients%20a6790e03d8194ee197326a0fb81c67ff/Untitled%2010.png](Worst%20Cases%20Policy%20Gradients%20a6790e03d8194ee197326a0fb81c67ff/Untitled%2010.png)

- 我估计这个东西肯定是distributional RL的重点，可以了解一下
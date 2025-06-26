# flow-matching 小demo

这个小lab的是从lab2改编的，用于模拟真实的生成场景

## 流程：

1. 得到data （图像数据在data文件夹里。你可以clone整个文件夹，或者下载运行sata_generator）

    关于图像处理，如何变成tensor，看flow_matching.ipynb的part1

2. 构建向量成和simulator。这部分和lab2几乎一摸一样，只是在lab2的基础上删除了不必要的内容

3. 构建 alpha, beta, 还有 Gaussian Conditional Vector Fiedl。也跟lab2的核心思路一摸一样，删减不需要的内容

4. 建立nn以及封装training process。跟lab2的思路一摸一样，删减不需要部分

5. 实际训练，看loss，看生成效果

## 原来的 failed_proj 的问题：

1. 最主要的问题：算 loss 的时候 xt 应该从 p_t (x) 采样，但是我忘记了，我错误的从 p_0 (x) 采样。这样自然得不到想要的结果。自然，step是2的时候生成效果最好，因为实际就只学习了一步。昨天基本在白干活

2. 原来的生成效果差可能是因为我的data是纯黑白的点图，就是说只有0和255两个值，跳度太大，而且没有归一化。重构之后，将散点图变成了概率密度图，让他变连续了。同时，在操作的时候把0，1，..，255归一化为[0, 1]。

3. 发现原来`一遍生成一遍训练`的办法**效率太低**。GPU只有40%，但是CPU单核满了（生成训练数据似乎是CPU的活）。重构的版本里，采用`data_generator`这个文件（AI写的，感谢GPT）来生成data，储存到data这个文件夹中。standard-gaussian/stretched-gaussian/moons/circles/checkerboards每一类1000张1 * 64 * 64的png图片（实际上standard-gaussian是多余的）

## 这次碰到的问题

1. tensor，mlp要在同一个device，疯狂踩坑debug

2. 在 simulator 里，应该是 x = step(x, t, h) 而不是 xt = step(x, t, h)，这样simulate当然是错的。我今天前面都是构建对了，这里构建错了，导致我生成出来还是一坨。后来发现了这个问题。现在应该能正常生成了。

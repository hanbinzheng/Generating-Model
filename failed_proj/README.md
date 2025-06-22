## 目的：从二维的高斯噪声出发，生成一些toy data，具体见下：

标准二维高斯噪声图片：

![unable to show the standard gaussian noise](https://github.com/hanbinzheng/generating-model/blob/main/failed_proj/images/std-gaussian.png)

希望生成的图片

moons

![unable to show the moons](https://github.com/hanbinzheng/generating-model/blob/main/failed_proj/images/moons.png)

stretched-gaussian

![unable to show the stretched gaussian](https://github.com/hanbinzheng/generating-model/blob/main/failed_proj/images/stretched-gaussian.png)

circles:

![unable to show the circles](https://github.com/hanbinzheng/generating-model/blob/main/failed_proj/images/circles.png)

checkerboard:

![unable to show the checkerboard](https://github.com/hanbinzheng/generating-model/blob/main/failed_proj/images/checker-board.png)

由于我不会画图，也不会生成数据，这些数据点是由gemini根据mit lab里的sampleable class改编的。





## 设计：

1. 与mit lab的不同：

   （1）简化了设计，比如删掉了一些为了讲解概念的内容，比如density的函数，还有$p_t(x)$的分布到底长什么样这类的。这里只保留了跟vecror field紧密相关的内容。

   （2） 学习的图像的“表达性质”不同。mit的lab学的是（x,y) 这个坐标对的分布。他的一个shape(num_samples, 2) tensor是一张图，代表由“num_samples个（x，y）坐标点”构成，他把num_samples当作“batch_size"，学“点”的分布。这里面我是让Gemini改编mit lab里面生成Sampleable的办法。我让Gemini把一张图片变成shape(1, 64, 64)的tensor，然后flattened的形式给出，即shape(1 * 64 * 64)。我的一个batch tensor是shape(batch_size, dims), 其中batch_size默认128，dims默认1 * 64 * 64





## 问题：

- 模型不收敛： 该模型在训练过程中损失函数持续震荡且有增大的趋势。
- 随着训练次数的增加，模型生成的图像会逐渐退化为二维高斯噪声。
  - 对于 `stretched_gaussian` 目标，模型在训练50次时表现尚可?随后趋于高斯噪声。1000次能辨认，再多没有试过。
  - 对于 `moons`, `circles`, `checker board` 等目标，模型在训练100-1000次时已基本变为高斯噪声，在5-15次训练时偶尔能有勉强可接受的表现。


经过毫无章法的到处试参数，最终得到稍微能看一点的结果如下：

总而言之，这个模型根本不具备稳定，正确训练-生成的能力，能生成图像纯粹是瞎猫碰上死耗子

moon:

![moons](https://github.com/hanbinzheng/generating-model/blob/main/failed_proj/images/Moon.png)

circle:

![circle](https://github.com/hanbinzheng/generating-model/blob/main/failed_proj/images/Circle.png)

checkerboard:

![checkerboard](https://github.com/hanbinzheng/generating-model/blob/main/failed_proj/images/Checkerboard.png)

stretched-gaussian

![stretched-gaussian](https://github.com/hanbinzheng/generating-model/blob/main/failed_proj/images/StretchedGaussian.png)


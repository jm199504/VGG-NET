## VGG-NET

背景：使用VGG做图像风格迁移

内容：对内容损失和风格损失组合，训练模型输出带有风格特征的内容图像

准备：TensorFlow、numpy、scipy、(vgg.mat)等

Neural Style Transfer 神经风格迁移，是计算机视觉流行的一种算法，最先论文来源《 A Neural Algorithm of Artistic Style》

所谓的图像风格迁移，是指利用算法学系著名画作的风格，将这个风格应用到我们自定义的图片上，其中著名的图像处理应用Prisma是用利用风格迁移技术，将普通人的照片自动转换为具有艺术气息风格的图片。

将用到ImageNet VGG模型来图像风格迁移，其实VGGNet的本意并不是为了风格迁移，而是输入图像，提取图像中的特征，输出图像的类别，即图像识别，而图像风格迁移恰好与其相反，输入的是特征，输出的具有这种特征的图片。

风格迁移使用卷积层的中间特征还原对应的特征原始图像，个人理解：你使用一系列的卷积层对底层图（称content_img）进行特征学习，现在你又将某一层/几层的特征返过来试着生成原始图，显然你没有使用全部卷积层是几乎不能生成与原图完全一样的图形（可以理解为“反卷积”操作），那么你生成的图（称Generated_img）与原图就存在差异，这种差异可以称为内容损失（content loss），另外浅层卷积层还原效果会比深层卷积层好（深层卷积层往往只保留图像中物体形状和位置），另外是使用梯度下降法还原图像，以内容损失作为损失/优化函数。

除了使用图像还原，还需要对内容加入“风格”，这里就要引入我们的风格图(style_img)，可以使用风格图的卷积层特征的Gram矩阵去表示“风格”，Gram矩阵即格拉姆矩阵，是关于一组向量的内积的对称矩阵，是在n维欧式空间中任意k(k≤n)个向量α1,α2,⋯,αk的内积所组成的矩阵，这里就不过多描述。

总体而言，实现图像风格迁移即将内容损失(内容/底层图与内容还原图)和风格损失(风格图与风格还原图)的组合，是由两个神经网络组合而成，利用内容损坏和风格损失训练图像生成网络。

**算法步骤大致总结一下：**

①构建content图像损失函数loss(C,Gc)

②构建style图像损失函数loss(S,Gs)

③生成合并：rs_img = αloss(C,Gc)+βloss(S,Gs)

**效果图：**

<img src="https://github.com/jm199504/VGG-NST/blob/master/images/1.png">

补充知识点（参考来源：https://zhuanlan.zhihu.com/p/41423739）：

VGG是Oxford的Visual Geometry Group提出而得名为VGG，其中VGG16相比AlexNet的一个改进是采用连续的几个3x3的卷积核代替AlexNet中的较大卷积核（11x11，7x7，5x5），对于给定的感受野（与输出有关的输入图片的局部大小），采用堆积的小卷积核是优于采用大的卷积核，因为多层非线性层可以增加网络深度来保证学习更复杂的模式，而且代价还比较小（参数更少）。

**总结：**

1×1卷积核在保证具有相同感知野的条件下，提升了网络的深度，在一定程度上提升了神经网络的效果。

3个步长为1的3x3卷积核的一层层叠加作用可看成一个大小为7的感受野（其实就表示3个3x3连续卷积相当于一个7x7卷积），其参数总量为 3x((3×3)xC^2) ，如果直接使用7x7卷积核，其参数总量为 (7×7)xC^2 ，这里 C 指的是输入和输出的通道数。很明显，27xC^2 < 49xC^2，即减少了参数；而且3x3卷积核有利于更好地保持图像性质。

使用2个3x3卷积核可以来代替5×5卷积核图解：

<img src="https://github.com/jm199504/VGG-NST/blob/master/images/2.jpg" width="400">

VGG网络结构：

<img src="https://github.com/jm199504/VGG-NST/blob/master/images/3.jpg" width="400">

VGG16包含了16个隐藏层（13个卷积层和3个全连接层），如上图中的D列所示

VGG19包含了19个隐藏层（16个卷积层和3个全连接层），如上图中的E列所示

VGG网络的结构非常一致，从头到尾全部使用的是3x3的卷积和2x2的max pooling。

**VGG优点**

VGGNet的结构非常简洁，整个网络都使用了同样大小的卷积核尺寸（3x3）和最大池化尺寸（2x2）。

几个小滤波器（3x3）卷积层的组合比一个大滤波器（5x5或7x7）卷积层好：

验证了通过不断加深网络结构可以提升性能。

**VGG缺点**

VGG耗费更多计算资源，并且使用了更多的参数（这里不是3x3卷积的锅），导致更多的内存占用（140M）。其中绝大多数的参数都是来自于第一个全连接层。VGG可是有3个全连接层！

训练请参考：tensorflow-vgg：<https://link.zhihu.com/?target=https%3A//github.com/machrisaa/tensorflow-vgg>

快速测试请参考：VGG-in TensorFlow：<https://link.zhihu.com/?target=https%3A//www.cs.toronto.edu/~frossard/post/vgg16/>

VGG-Tensorflow测试代码：<https://link.zhihu.com/?target=https%3A//www.cs.toronto.edu/~frossard/post/vgg16/>

其他参考来源：

论文：https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1409.1556

机器学习进阶笔记之五 | 深入理解 VGG Residual Network：https://zhuanlan.zhihu.com/p/23518167

深度学习经典卷积神经网络之VGGNet：https://link.zhihu.com/?target=https%3A//blog.csdn.net/marsjhao/article/details/72955935

---
title: Tensorflow Dataset（数据集）
mathjax: false
date: 2023-12-03 11:27:26
tags:
  - [Deep Learning, 深度学习, Tensorflow 数据集, AI]
categories:
  - [深度学习, Tensorflow]
---

TensorFlow Dataset 是 TensorFlow 中用于处理数据的模块，它提供了一种高效的数据输入管道，用于加载和预处理数据，以供模型训练和评估使用。

TensorFlow Dataset 的数据结构主要包括以下几个核心组件：

1. Dataset：代表一系列元素的集合，可以是张量、numpy 数组、Python 生成器或其他数据源。

2. Data Transformation：用于对 Dataset 进行转换和处理的方法，例如 map、filter、batch 等，用于对数据进行预处理、筛选和批处理等操作。

3. Iterator：用于遍历 Dataset 中元素的迭代器，例如 iter、next 等。

通过这些组件，TensorFlow Dataset 提供了一种灵活且高效的数据处理方式，能够方便地构建数据输入管道，加速模型训练过程。

#### 最简单的 Dataset 使用方式 Dataset.from_tensor_slices

```python
import tensorflow as tf

# 假设有一些数据行
datarows = [[0.1, 0.2, 0.3], 
            [0.4, 0.5, 0.6]]
#以及对应的标签
labels = [0, 
          1]

# 创建包含元组的 Dataset
dataset = tf.data.Dataset.from_tensor_slices((datarows, labels))

# 打印 Dataset 中的数据
for data, label in dataset:
    print(data)
    print(label)  
    break #打印一行后退出

```

可见，本质上 Tensorflow Dataset 是一个**可迭代**的数据结构

没有人能够阻止我们 用 *take(n)* 拿出指定行数，因此，上面的代码可以更优雅的写为：

```python
for data, label in dataset.take(1):
    print(data)
    print(label)  
```

#### 迭代 Dataset 的方式

可以使用 Dataset 的迭代器来遍历数据集中的元素。以下是一个简单的示例代码，演示了如何创建一个迭代器并使用它来遍历数据集中的元素：

```python
import tensorflow as tf

# 创建一个简单的数据集
data = [1, 2, 3, 4, 5]
dataset = tf.data.Dataset.from_tensor_slices(data)

# 创建一个迭代器
iterator = iter(dataset)

# 使用迭代器遍历数据集
while True:
    try:
        element = next(iterator)
        print(element)
    except StopIteration:
        break
```
在这个示例中，我们首先创建了一个包含整数的数据集。然后，我们使用 iter 函数创建了一个迭代器，并使用 next 函数来逐个获取数据集中的元素。当数据集中的所有元素都被遍历完毕后，迭代器会抛出 StopIteration 异常，我们利用这一点来结束遍历过程。

当然，在这里，对 Dataset 使用 *for* 语句显得更直接。

#### Dataset 一定存储 (data, label) 形状的元组吗？

不一定，就像上面由 range 一维数组生成的 Dataset，我们并没有给他 label（标签）。

没有标签的数据集也是有用的。在机器学习和深度学习中通常用于预处理、数据增强、特征提取等操作。例如，在训练卷积神经网络时，可以使用没有标签的数据集进行数据增强，如随机裁剪、翻转、旋转等操作，以扩充训练数据集。此外，没有标签的数据集也可以用于特征提取，例如在使用预训练模型进行迁移学习时，可以使用没有标签的数据集来提取特征。

同样，我们可以给一个数据集构造多个标签列，甚至多个数据列，这也没有问题。

```python
import tensorflow as tf

# 创建一个包含0到9的张量的 Dataset
dataset = tf.data.Dataset.range(10)

# 对 Dataset 进行转换和处理
dataset = dataset.filter(lambda x: x % 2 == 0)  # 筛选出偶数
dataset = dataset.map(lambda x: (x, 0, 1))  # 将每个元素与标签 0 组合

# 迭代数据
for data, label, c in dataset:
    print(data, label, c)
```

#### Dataset每次迭代出的数据结构：tf.Tensor

我们观察 print(data) 和 print(label) 代码返回的结果

```bash
tf.Tensor([0.1 0.2 0.3], shape=(3,), dtype=float32) 
tf.Tensor(0, shape=(), dtype=int32)
```
可以看到返回的类型是 tf.Tensor，这是因为为了与 TensorFlow 的计算图和自动微分机制兼容

如果我们要拿到 numpy 数组，可以使用 *.numpy()* 方法将 Tensor 化简

```python
print(data.numpy(), label.numpy())
```

#### 如何构造一个自定义的 Dataset ？

我们可以用类继承的方式构造一个自己的 Dataset

```python
import tensorflow as tf

class CustomDataset(tf.data.Dataset):
    def _generator(data, labels):
        for i in range(len(data)):
            yield data[i], labels[i]

    def __new__(cls, data, labels):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_signature=(
                tf.TensorSpec(shape=(2,), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32)
            ),
            args=(data, labels)
        )

# Example usage
input_data = [[1, 2], [3, 4], [5, 6], [7, 8]]
labels = [0, 1, 0, 1]

custom_dataset = CustomDataset(input_data, labels)

for data, label in custom_dataset:
    print(data, label)
```

关注上面的代码，派生类的构造函数 *\_\_new\_\_* 的实现可以看到迭代出 tf.Tensor 实例的逻辑。

那 *_generator* 函数重载里的 *yeild* 有什么意义呢？

yield 是 Python 中用于定义生成器函数的关键字。生成器函数是一种特殊的函数，它可以在迭代过程中暂停并恢复执行，从而可以逐个产生值而不需要一次性将所有值存储在内存中。

当生成器函数中包含 yield 语句时，调用该函数并不会立即执行函数体内的代码，而是返回一个生成器对象。每次调用生成器对象的 \_\_next\_\_ 方法（或使用 for 循环迭代生成器对象时），生成器函数会从上一次暂停的地方恢复执行，直到遇到下一个 yield 语句，然后再次暂停并将 yield 后面的值返回给调用者。

因此，yield 的作用是定义生成器函数的暂停点，并 **返回一个值给调用者，同时保持函数的状态** ，以便下次调用时可以从暂停的地方继续执行。这种特性使得生成器函数非常适合用于处理大量数据或需要逐个产生值的场景，因为它可以节省内存并提高效率。

#### map、filter方法，把 Dataset 里的原始数据转化为训练程序可用的数据

很多时候，网路上的 Dataset，或者系统提供的 Dataset，离我们要在 Tensorflow 代码中要直接用的数据表达方式是有偏差的。

例如，

 - 原始的 Dataset 可能存储的是文件名（或 URL ）作为一条记录，而我们需要的是这个文件的数据本身（或者压缩采样后的数据）作为一条记录
 - 原始的 Dataset 的标签列为字符串类型，我们需要训练器可以理解的一个在 1 ~ 10 范围的数值编码

当然，我们可以自己迭代一遍原始的 Dataset，在其间书写自己的转化逻辑。然后构造生成满足需求的，可以传递给 Keras 的数据结构。但这不够优雅。

使用 map 是一种更有效率的方式，以下是一个构造简单图片分类的 Dataset 的示例代码：

```python
import tensorflow as tf
import os

# 定义图片路径和标签
image_paths = ["path_to_image1.jpg", "path_to_image2.jpg", "path_to_image3.jpg"]
labels = [0, 1, 0]  # 假设有两个类别，0 和 1

# 创建 Dataset
dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

# 定义图片加载函数
def load_and_preprocess_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [228, 228])  # 假设图片大小为 228x228
    image = tf.cast(image, tf.float32) / 255.0  # 归一化
    return image, label

# 对 Dataset 进行转换和处理
dataset = dataset.map(load_and_preprocess_image)



# 打印 Dataset 中的数据
for images, labels in dataset:
    print(images.shape, labels)
```

我们还可以使用 *filter* 将 Dataset 中不需要的行过滤掉。

```python
import tensorflow as tf

# 创建一个包含整数的数据集
dataset = tf.data.Dataset.range(10)

# 定义过滤函数，过滤偶数
def filter_even_numbers(x):
    return x % 2 == 0

# 应用过滤函数到数据集上
filtered_dataset = dataset.filter(filter_even_numbers)

# 打印过滤后的数据集中的元素
for element in filtered_dataset:
    print(element.numpy())
```

#### 洗牌（shuffle）和 批处理（batch）

```python
# 设置 Batch 大小并打乱数据
batch_size = 32
dataset = dataset.shuffle(buffer_size=len(image_paths)).batch(batch_size)

```
*shuffle* 操作的意义是不言自明的。

而 *batch* 操作会在数据集的最前面增加一个维度，通常用于表示批次（batch）的维度。假设原始数据集中的每个元素的形状是 (a, b, c)，经过 batch 操作后，每个元素的形状将变为 (batch_size, a, b, c)，其中 batch_size 表示批次的大小。 batch 操作通常用于将数据集划分为批次，以便进行批处理训练。

#### 总结，通常这样使用 Dataset：

1. 创建 Dataset：使用 tf.data.Dataset 类的构造函数来创建一个 Dataset 对象，可以从张量、List、numpy 数组、文本文件、TFRecord 文件等数据源中创建 Dataset。或者从 Tensorflow 的库中装载现成的 Dataset

2. 数据转换：通过调用 Dataset 对象的方法，如 map、filter、batch 等，对数据进行转换和处理，以便用于模型训练和评估。

3. 迭代数据：使用迭代器（Iterator）来遍历 Dataset 中的元素，可以使用 for...in 循环或者 iter、next 方法来逐个获取数据样本。


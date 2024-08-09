import numpy as np 
np.version.full_version

a = np.arange(20)


a = a.reshape(4,5)

a = a.reshape(2,2,5)


# ndim 维度
# shape 维度大小
# size 元素个数 = 维度大小的乘积
# dtype 元素类型
# dsize 元素占位



#数组的创建可通过转换列表实现，高维数组可通过转换嵌套列表实现
raw = [0,1,2,3,4]
a = np.array(raw)


raw = [[0,1,2,3,4], [5,6,7,8,9]]
b = np.array(raw)

# 4*5的全零矩阵：
d = (4, 5)



#默认生成的类型是浮点型，可以通过指定类型改为整型：

d = (4, 5)
#print(np.ones(d, dtype=int))

#[0, 1)区间的随机数数组：
#print(np.random.rand(5))

#简单的四则运算已经重载过了，全部的+，-，*，/运算都是基于全部的数组元素的，以加法为例：

a = np.array([[1.0, 2], [2, 4]])

b = np.array([[3.2, 1.5], [2.5, 4]])

print (a+b)
#类似C++，+=、-=、*=、/=操作符在NumPy中同样支持：
a /= 2
print (a)

#开根号求指数也很容易：

print (np.exp(a))



a = np.arange(20).reshape(4,5)
print ("a:")
print (a)
print ("sum of all elements in a: " + str(a.sum()))
print ("maximum element in a: " + str(a.max()))
print ("minimum element in a: " + str(a.min()))
print ("maximum element in each row of a: " + str(a.max(axis=1)))
print ("minimum element in each column of a: " + str(a.min(axis=0)))



#科学计算中大量使用到矩阵运算，除了数组，NumPy同时提供了矩阵对象（matrix）。矩阵对象和数组的主要有两点差别：一是矩阵是二维的，而数组的可以是任意正整数维；二是矩阵的*操作符进行的是矩阵乘法，乘号左侧的矩阵列和乘号右侧的矩阵行要相等，而在数组中*操作符进行的是每一元素的对应相乘，乘号两侧的数组每一维大小需要一致。数组可以通过asmatrix或者mat转换为矩阵，或者直接生成也可以：
a = np.arange(20).reshape(4, 5)
a = np.asmatrix(a)
print (type(a))

b = np.matrix('1.0 2.0; 3.0 4.0')
print (type(b))


#矩阵的乘法，这使用arange生成另一个矩阵b，arange函数还可以通过arange(起始，终止，步长)的方式调用生成等差数列，注意含头不含尾。

b = np.arange(2, 45, 3).reshape(5, 3)
b = np.mat(b)
print (b)

#arange指定的是步长，如果想指定生成的一维数组的长度怎么办？好办，linspace就可以做到：

print(np.linspace(0, 2, 9))
#数组和矩阵元素的访问可通过下标进行，以下均以二维数组（或矩阵）为例

a = np.array([[3.2, 1.5], [2.5, 4]])
print (a[0][1])
print (a[0, 1])


b = a
a[0][1] = 2.0
#明明改的是a[0][1]，怎么连b[0][1]也跟着变了？这个陷阱在Python编程中很容易碰上，其原因在于Python不是真正将a复制一份给b，而是将b指到了a对应数据的内存地址上。想要真正的复制一份a给b，可以使用copy：

a = np.array([[3.2, 1.5], [2.5, 4]])
b = a.copy()



a = np.arange(20).reshape(4, 5)
#可以访问到某一维的全部数据，例如取矩阵中的指定列：
print (a[:,[1,3]])
#我们尝试取出满足某些条件的元素，这在数据的处理中十分常见，通常用在单行单列上。下面这个例子是将第一列大于5的元素（10和15）对应的第三列元素（12和17）取出来
print(a[:, 2][a[:, 0] > 5])


# 矩阵转置

# 如果a 是个array，用transpose 
a = np.random.rand(2,4)
a = np.transpose(a)


# 如果b是个矩阵，b.T
b = np.random.rand(2,4)
b = np.mat(b)
b.T

import numpy.linalg as nlg
a = np.random.rand(2,2)
a = np.mat(a)

ia = nlg.inv(a)

print (ia)
print (a * ia)


# 求特征值和特征向量
a = np.random.rand(3,3)
eig_value, eig_vector = nlg.eig(a)


# 向量拼接
a = np.array((1,2,3))
b = np.array((2,3,4))
print (np.column_stack((a,b)))



a = np.random.rand(2,2)
b = np.random.rand(2,2)

# h [A B]  v[A/B]
c = np.hstack([a,b])
d = np.vstack([a,b])


#NumPy提供nan作为缺失值的记录，通过isnan判定。
a = np.random.rand(2,2)
a[0, 1] = np.nan
print (np.isnan(a))
# nan_to_num可用来将nan替换成0
print (np.nan_to_num(a))
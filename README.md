# TopPush Python Code

本项目是对[《Top Rank Optimization in Linear Time》](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/nips14.pdf) 论文的python代码实现。实现过程参考了官方matlab和c代码，原论文以及代码见[official](official)目录。


## 纯python实现
主体`toppush`函数以及`epne`函数都用python实现。

### 代码文件
* python/toppush.py

### 使用方式
直接调用`toppush.py`中的`topPush`函数即可。
例如：
~~~ python
w = topPush(X, y)
~~~

相关参数以全局变量形式定义：
~~~ python
lambdaa = 1  # radius of l2 ball
maxIter = 10000  # maximal number of iterations
tol = 1e-4  # the relative gap
debug = False  # the flag whether it is for debugging
delta = 1e-6
~~~

## python+C实现
用python实现`toppush`函数，用C实现`epne`函数。

### 代码文件
* python_with_c/epne.c
* python_with_c/toppush.py

### 使用方式
1.编译`epne.c`生成`epne.so`动态链接库文件。
~~~ shell
gcc -shared -o epne.so epne.c
~~~
2.直接调用`toppushWithC.py`中的`topPush`函数即可。
例如：
~~~ python
w = topPush(X, y)
~~~

相关参数以全局变量形式定义：
~~~ python
# load C func epne
mylib = cdll.LoadLibrary('python_with_c/epne.so')
epne = mylib.epne

# params
lambdaa = 1  # radius of l2 ball
maxIter = 10000  # maximal number of iterations
tol = 1e-4  # the relative gap
debug = False  # the flag whether it is for debugging
delta = 1e-6
~~~

## 实验对比
主要包括`功能性实验`和`对比实验`。
* **功能性实验**：验证计算结果的正确性；
* **对比实验**：对比两种实现方式的速度差别；

### 代码文件
* main.py

### 实验数据
* data/spambase.mat
实验数据中的`ans`字段是利用matlab使用官方提供的代码计算保留的结果，用于验证本项目代码的正确性。

### 实验结果
~~~ text
--------------------------------------  start functional test --------------------------------------
----------------------------------- start python functional test -----------------------------------
the max diff is: 7.580741590018647e-16 less than 1e-08
---------------------------------- python functional test passed -----------------------------------
------------------------------- start python-with-c functional test --------------------------------
the max diff is: 7.580741590018647e-16 less than 1e-08
------------------------------- python-with-c functional test passed -------------------------------
----------------------------------------- start speed test -----------------------------------------
------------------------------------- start python speed test --------------------------------------
topPush time: 17.7223345 s
------------------------------- start python-with-c functional test --------------------------------
topPushWithC time: 0.7064349 s

Process finished with exit code 0

~~~
通过实验可以发现：
* **功能性实验**通过，计算结果在计算误差范围内；
* **对比实验**：python-with-c速度远快于python，建议使用python-with-c

### 经验教训
numpy.ndarray运算后会生成新的numpy.ndarray，不是在原数据上进行修改。例如（data = -data，会生成新的数据）
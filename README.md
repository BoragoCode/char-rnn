# char-rnn
字符级别的RNN模型，[karpathy](https://github.com/karpathy)公布了一个用python实现的100来行的[char-rnn模型](https://gist.github.com/karpathy/d4dee566867f8291f086)。模型核心是一个RNN，输入单元为一个字符，训练语料为若干篇英文文本。模型通过RNN对连续序列有很好的学习能力，对文本进行建模，学习得到字符级别上的连贯性，从而可以利用训练好的模型来进行文本生成。

========================================================
参照karpathy公布的代码，我用theano重写了该模型，使模型能够支持中文文本，同时将源程序中的RNN模型替换成了[GRU(Cho et al., 2014b)](http://arxiv.org/abs/1406.1078)。RNN模型在长序列上存在梯度消失或梯度爆炸的问题([Bengio et al., 1994](http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=279181&url=http%3A%2F%2Fieeexplore.ieee.org%2Fxpls%2Fabs_all.jsp%3Farnumber%3D279181))，而GRU和LSTM一样能够解决这一问题，但是GRU和LSTM的效果差不多却又比LSTM更简单([Greff et al., 2015](http://arxiv.org/abs/1503.04069))，因而选用GRU替换源程序中的RNN。该模型训练中文语料有一个很好的优势是不用对中文进行分词，模型输入单元为一个一个的字，这样可以避免分词带来的错误。利用在大规模中文文本上训练好的模型，可以生成一篇短文。具体可以参考karpathy写的一篇[blog](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)。

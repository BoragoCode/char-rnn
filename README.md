# char-rnn
字符级别的RNN模型，[karpathy](https://github.com/karpathy)公布了一个用python实现的100来行的轻量级[char-rnn模型](https://gist.github.com/karpathy/d4dee566867f8291f086)。模型核心是一个RNN，输入单元为一个字符，训练语料为若干篇英文文本。模型通过RNN对连续序列有很好的学习能力，对文本进行建模，学习得到字符级别上的连贯性，从而可以利用训练好的模型来进行文本生成。这里还有一个用tensorflow实现的版本[char-rnn-tf](https://github.com/hit-computer/char-rnn-tf)，这个tensorflow版本的程序和karpathy的完整版char-rnn一致，效果也会比该程序好，并且tensorflow版本程序还支持多种生成策略（max，sample，以及beam-search）。

------------------------------------------------------------
参照karpathy公布的代码，我用theano重写了该模型，使模型能够支持中文文本，同时将源程序中的RNN模型替换成了[GRU(Cho et al., 2014b)](http://arxiv.org/abs/1406.1078)。GRU，和LSTM一样解决了基本RNN模型在长序列上的梯度消失或梯度爆炸问题([Bengio et al., 1994](http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=279181&url=http%3A%2F%2Fieeexplore.ieee.org%2Fxpls%2Fabs_all.jsp%3Farnumber%3D279181))，然而GRU和LSTM的效果差不多却比LSTM更简单([Greff et al., 2015](http://arxiv.org/abs/1503.04069))。陆陆续续有研究者开始在英文上使用char-rnn，有论文证实char-rnn比以词为输入的rnn模型要好，它有一个优势就是可以解决未登录词的问题([Dhingra et al., 2016](http://arxiv.org/abs/1605.03481))。然而，char-rnn训练中文语料还有另外一个优势是不用对中文进行分词，模型输入单元为一个一个的字，这样可以避免分词带来的一些错误，所以采用char-rnn在中文语料上就两个好处。利用在大规模中文文本上训练好的模型，可以生成一篇短文(生成的时候也是一个字一个字的产生)。具体可以参考karpathy写的一篇[blog](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)。

========================================================
###运行说明
在命令行中输入：

    THEANO_FLAGS='mode=FAST_RUN,floatX=float32' python char-rnn.py [训练语料]

若机器上有GPU，可以使用GPU进行训练，速度比CPU能快很多，输入命令改为：

    THEANO_FLAGS='mode=FAST_RUN,device=gpu,floatX=float32' python char-rnn.py [训练语料]

注意：训练语料为文本文件，请采用utf-8编码，可以考虑在每一个语义段落前加上起始符‘^’。


char-rnn.py文件里面有以下参数可以设定：
- hidden_size：神经网络隐含层的维度
- seq_length：RNN展开的步骤数（每次训练多少个字符）
- learning_rate：学习率
- iter：迭代次数
- save_freq：每迭代多少次保存一次模型，同时进行一次生成
- idx_of_begin：生成语段的起始字符
- len_of_sample：生成语段的字符数目

========================================================
###实验结果
本实验选取了大量和“选择”相关的作文作为训练语料，在生成的时候起始符设定为“选”字，生成字符设定为100个字符，以下是部分生成结果（训练语料规模：1.12M）：

运行环境：CentOS Linux release 7.2.1511，一块显卡Tesla K40m，Theano 0.7.0。全部数据迭代一轮大致需要45分钟左右。

迭代50次：

>选择了，我们的选择是一种美丽的人生，就是一个人的人生，就是一个人的人生，就是一个人的人生，就是一个人的人生，就是一个人的人生，就是一个人的人生，就是一个人的人生，就是一个人的人生，就是一个人的人生，就是

迭代100次：
 >选择了，我们不能选择自己的人生，就是一个人的人生，不是因为我们的选择，不是一个人的人生，就是一个人的人生，不是因为我们的选择，不是一个人的人生，就是一个人的人生，不是因为我们的选择，不是一个人的人生，就

迭代200次：
>选择了，我们不能选择自己的人生，不是因为我们的选择，不要放弃，就是一个人生的价值，我们的选择是一种选择，而是一个人的人生，不是因为我们的选择，不是一个人的人生，不是因为我们的选择，不是一个人的人生，不是

从实验结果看出，模型生成每个短句（由逗号隔开的）还算通顺，但整段连起来看就不知所云了。当然，这和选用的语料以及参数设定有很大关系，但我们依旧可以看出一些问题，就是后半段生成的内容会出现重复，这表明序列太长即使是GRU仍然会有信息丢失。所以，目前在自然语言生成方面的研究大多都是在做短句的生成，例如生成诗歌或者生成歌词这些，并且不是整段一并生成的。

对于出现文本重复现象的一点经验性总结（同时可以参考[char-rnn-tf](https://github.com/hit-computer/char-rnn-tf)的实验结果以及实验结果分析）：其实在做文本生成的时候有两种策略，一种是max还有一种是sample（在karpathy的程序中有体现），本程序用的是max策略，这种策略会导致重复的现象，而sample策略不会但句子连贯性方面会比max策略稍差一些，随着语料和迭代次数的增加sample策略所生产的文本连贯性会越好（karpathy的程序默认采用的是sample策略）。我最近用tensorflow重写这个模型（[char-rnn-tf](https://github.com/hit-computer/char-rnn-tf)）后发现增加训练语料以及采用多层RNN能使重复现象出现时序列长度更长（采用max策略时）。

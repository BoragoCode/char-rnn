# char-rnn
    字符级别的RNN模型，[karpathy](https://github.com/karpathy)公布了一个用python实现的100来行的[char-rnn模型](https://gist.github.com/karpathy/d4dee566867f8291f086)。模型核心是一个RNN，输入单元为一个字符，训练语料为若干篇文本。模型通过RNN对连续序列有很好的学习能力，对文本进行建模，学习得到字符级别上的连贯性，从而可以利用训练好的文本来进行文本生成。
    我用theano重构了该模型，使模型能够支持中文文本，该模型训练中文语料有一个很好的优势是不用对中文进行分词，模型输入单元为一个一个的字，这样可以避免分词带来的错误。利用在大规模中文文本上训练好的模型，可以生成一篇短文。具体可以参考karpathy写的一篇[blog](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)。

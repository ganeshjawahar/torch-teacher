## A Thorough Examination of the CNN/Daily Mail Reading Comprehension Task

[Torch](http://torch.ch) implementation of the model 'Neural Net (Relabeling)' proposed in [Chen et al.](https://arxiv.org/abs/1606.02858)'s work.

### Features

* Train the model with the benchmarked corpus, [CNN News Test](https://github.com/deepmind/rc-data) (CNN) out of the box.
* Easy to try out all the variants of activations functions of RNNs such as vanilla RNN, GRU and LSTM.
* Support for tuning other hyper-parameters of the model reported in the paper.

### Quick Start

Download and extract the `cnn.tgz` file from [DeepMind Q&A Dataset](http://cs.nyu.edu/~kcho/DMQA/)'s page.

To generate and save the data tensors (objects readable for Torch),

```
th preprocess.lua -data cnn/questions/
```

where the data value `cnn/questions/` points to the extracted directory containing the training, validation and testing files for NE prediction task

To kick-start the training,

```
th main.lua
```

To know the hyper-parameters relevant for both the steps,

```
th preprocess.lua --help
th main.lua --help
```

### Training options

#### Data Preprocessing (`preprocess.lua`)

* `word_type`: class type of the prediction word. Specify `NE` for Named Entity, `CN` for Common Noun, `P` for Preposition and `V` for Verb
* `data`: path to the data folder containing train, validation and test records
* `out`: output file name for the tensors to be saved
* `batch_size`: sgd mini-batch size
* `vocab_size`: size of the word vocabulary. this is constructed by taking the top <int> most frequent words. rest are replaced with <unk> tokens.'
* `question_pad`: which side to pad the question to make sequences in a batch to be of same size? `left` or `right`?
* `passage_pad`: which side to pad the passage to make sequences in a batch to be of same size? `left` or `right`?

#### Training Options (`main.lua`)

* `input`: input file name for the saved tensors
* `seed`: seed value for the random generator
* `glove_file`: file containing the pre-trained glove word embeddings
* `dim`: dimensionality of word embeddings
* `hid_size`: RNN's hidden layer size
* `num_epochs`: number of full passes through the training data
* `lr`: adam's learning rate
* `grad_clip`: clip gradients at this value
* `dropout`: dropout for regularization, used before the prediction layer. 0 = no dropout

#### Torch Dependencies
* nn
* cunn
* cutorch
* rnn
* optim
* xlua
* tds
* nngraph

#### Author
[Ganesh J](https://researchweb.iiit.ac.in/~ganesh.j/)

#### Licence
MIT



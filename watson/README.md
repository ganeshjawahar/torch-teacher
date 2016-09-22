## Text Understanding with the Attention Sum Reader Network

[Torch](http://torch.ch) implementation of the model 'AS Reader (Single Model)' proposed in [Kadlec et al.](https://arxiv.org/abs/1603.01547)'s work.

### Features

* Train the model with the benchmarked corpus, [Children's Book Test](https://research.facebook.com/research/babi/) (CBT) out of the box.
* Easy to try out all the variants of activations functions of RNNs such as vanilla RNN, GRU and LSTM.
* Support for tuning other hyper-parameters of the model reported in the paper.

### Quick Start

Download and extract the `CBTest.tgz` file from [FB Research](https://research.facebook.com/research/babi/)'s page.

To generate and save the data tensors (objects readable for Torch),

```
th preprocess.lua -data CBTest/data/
```

where the data value `CBTest/data/` points to the extracted directory containing the training, validation and testing files for all the 4 prediction tasks, viz., `Named Entity` (NE), `Common Noun` (CN), `Preposition` (P) and `Verb` (V).

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

#### `preprocess.lua`

* `word_type`: class type of the prediction word. Specify `NE` for Named Entity, `CN` for Common Noun, `P` for Preposition and `V` for Verb
* `data`: path to the data folder containing train, validation and test records
* `out`: output file name for the tensors to be saved
* `question_pad`: which side to pad the question to make sequences in a batch to be of same size? `left` or `right`?
* `passage_pad`: which side to pad the passage to make sequences in a batch to be of same size? `left` or `right`?

#### `main.lua`

* `input`: input file name for the saved tensors
* `seed`: seed value for the random generator
* `dim`: dimensionality of word embeddings
* `hid_size`: RNN's hidden layer size
* `num_epochs`: number of full passes through the training data
* `lr`: adam's learning rate
* `bsize`: mini-batch size for adam
* `grad_clip`: clip gradients at this value

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


## The Goldilocks Principle: Reading Children's Books with Explicit Memory Representations (ICLR 2016)

[Torch](http://torch.ch) implementation of the model 'MemNNs (Window Memory + Self-Supervision)' proposed in [Hill et al.](http://arxiv.org/abs/1511.02301)'s work.

### Features

* Train the model with the benchmarked corpus, [Children's Book Test](https://research.facebook.com/research/babi/) (CBT) out of the box.
* Support for the selection of window composition, viz., Summation or Concatentation of the word vectors within the window
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
* `b`: size of the window memory (same symbol used in the paper)
* `out`: output file name for the tensors to be saved

#### `main.lua`

* `input`: input file name for the saved tensors
* `seed`: seed value for the random generator
* `p`: dimensionality of word embeddings (same symbol used in the paper)
* `num_epochs`: number of full passes through the training data
* `lr`: learning rate (note: currently waiting for the first author to reply the optimal learning rate decay used in the experiments.)
* `window_compo`: how to compose the window representations from the word vectors? sum or concatenation?

#### Torch Dependencies
* nn
* cunn
* cutorch
* xlua
* tds

#### Author
[Ganesh J](https://researchweb.iiit.ac.in/~ganesh.j/)

#### Licence
MIT

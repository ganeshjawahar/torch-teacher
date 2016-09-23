# Teaching Machines to Read and Comprehend CNN News and Children Books using Torch

This software repository hosts the self-contained implementation of the state-of-the-art models used in Machine Reading and Comprehension Task.

| Folder | Reference |
|---|---|
| [watson/](https://github.com/ganeshjawahar/torch-teacher/tree/master/watson)| **Text Understanding with the Attention Sum Reader Network**, *Kadlec et al.*, *ACL 2016*. |
| [stanford/](https://github.com/ganeshjawahar/torch-teacher/tree/master/stanford) | **A Thorough Examination of the CNN/Daily Mail Reading Comprehension Task**, *Chen et al.*, *ACL 2016*. |
| [fair/](https://github.com/ganeshjawahar/torch-teacher/tree/master/fair) | **The Goldilocks Principle: Reading Children's Books with Explicit Memory Representations**, *Hill et al.*, *ICLR 2016*. |

### Benchmarking Training Time

| mins per batch (mins per epoch) | [watson/](https://github.com/ganeshjawahar/torch-teacher/tree/master/watson) | [stanford/](https://github.com/ganeshjawahar/torch-teacher/tree/master/stanford) | [fair/](https://github.com/ganeshjawahar/torch-teacher/tree/master/fair) |
|---|:---:|:---:|:---:|
|GPU\Batch Size|32|32|1|
|[K40](http://www.nvidia.com/object/tesla-servers.html)|`806 ms` (`46m 16s`)|`800 ms` (`2h 40m`)|`18ms` (`34m 8s`)|
|[Titan X](http://www.geforce.com/hardware/desktop-gpus/geforce-gtx-titan-x/specifications)|`746 ms` (`42m 38s`)|-|`13ms` (`24m 45s`)|
|[1080](http://www.geforce.com/hardware/10series/geforce-gtx-1080) |`889 ms` (`51m 8s`)|-|`13ms` (`25m 29s`)|

### Acknowledgements
This repository would not be possible without the efforts of the maintainers of the following libraries:
* [Element-Research/rnn](https://github.com/Element-Research/rnn)
* [MemNN](https://github.com/facebook/MemNN)
* [Torch](https://github.com/torch) (Ofcourse!)

### Author
[Ganesh J](https://researchweb.iiit.ac.in/~ganesh.j/)

### Licence
MIT

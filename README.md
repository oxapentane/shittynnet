# ShittyNNet

Weekend of messing with NNs instead of writing.

Written on top of numpy for education of a single person, aim was to make micrograd-ish thing but operating on matricies instead of scalars. Heavily inspired by [micrograd][mg], with also having a look at [tinygrad][tg]. Both forward and backward pass work, though optimizer is just free-range, hand-written for loop for now.

The `Tensor` class inside of [tensor.py](./shittynnet/tensor.py) is the workhorse, that can backpropagate gradients to the parameters. For usage example see [2binnet](./2binnet.ipynb), which successfuly leanrs to classify points.

## TODO
- [ ] more activation functions
- [ ] double check backpropogation derivatives
- [ ] optimizer machinery
- [ ] training machinery

## Bugs
Yes.

<!-- links go here -->
[mg]:https://github.com/karpathy/micrograd
[tg]:https://github.com/tinygrad/tinygrad
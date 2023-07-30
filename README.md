# Shittynet

Basically me learing about neural nets, aim is to make micrograd-ish thing, but operating on tensors instead of scalars. Heavily inspired by [micrograd][mg], with also having a look at [tinygrad][tg].

Built on top of numpy. For simplicity sake any value has 2 dimensions, numpy broadcasting is implicit pretty much everywhere where there is a shape mismatch. Both forward and backward pass work, though optimizer is written out by hand for now in examples.

## TODO
- [ ] more activation functions
- [ ] double check backpropogation derivatives
- [ ] optimizer

## Nice-to-have
- [ ] mypy type checking

## Bugs
Yes.

Very yes around free-range, hand-calculated `_backward()` derivatives.

<!-- links go here -->
[mg]:https://github.com/karpathy/micrograd
[tg]:https://github.com/tinygrad/tinygrad
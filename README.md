# Shittynet

Basically me learing about neural nets, aim is to make micrograd-ish thing, but operating on tensors instead of scalars.

Heavily inspired by [micrograd][mg], with having a look at [tinygrad][tg].

Built on top of numpy. For simplicity sake scalars are length-1 vectors, numpy broadcasting is implicit pretty much everywhere where there is a shape mismatcn

## TODO
- [ ] dot: backward
- [ ] div
- [ ] div: backward
- [ ] pow
- [ ] pow: backward
- [ ] exp
- [ ] exp: backward

## Nice-to-have
- [ ] mypy type checking

## Bugs
Yes

<!-- links go here -->
[mg]:https://github.com/karpathy/micrograd
[tg]:https://github.com/tinygrad/tinygrad
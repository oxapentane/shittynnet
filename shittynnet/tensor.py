import numpy as np

# quite some bits are copypasted^W were heavily inspired by https://github.com/karpathy/micrograd
class Tensor:
  def __init__(self, data, _prev = (), _op='', _type=np.float16):
    if not hasattr(data, "__len__"): data = [data] # for simplicity storing everything as matrices
    self.data = data if isinstance(data, np.ndarray) and len(data.shape) == 2 else np.atleast_2d(np.asarray(data)).astype(_type)
    self._shape = np.shape(self.data)
    assert len(self._shape) == 2, f"shape doesn't seem to be 2D, got {self._shape}"
    self._type = _type
    # autograd bits
    self._zero_grad()
    self._backward = lambda: None
    self._prev = set(_prev)
    self._op = _op

  # Gradient operations

  # the backpropogator
  def backward(self):
    assert self._shape == (1,1), f"backward() can only be called on a scalar tensor, got shape: {self._shape}"
    self._one_grad()
    for v in self._build_rtopo():
      v._backward()

  # Vector/matrix ops
  def dot(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    out = Tensor(np.matmul(self.data, other.data), (self, other), 'matmul')
    def _backward():
      self.grad += np.matmul(out.grad, other.data.T)
      other.grad += np.matmul(self.data.T, out.grad)
    out._backward = _backward
    return out

  def T(self):
    out = Tensor(self.data.T, (self,), 'T')
    def _backward():
      self.grad += out.grad.T
    out._backward = _backward
    return out

  # element-wise binary ops (broadcasted for arrays)
  def __add__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    out = Tensor(np.add(self.data, other.data), (self, other), '+')

    def _backward():
      self.grad += out._val_shape(out.grad)
      other.grad += out._val_shape(out.grad)
    out._backward = _backward

    return out

  def __mul__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    out = Tensor(self.data * other.data, (self, other), '*')

    def _backward():
      self.grad += self._val_shape(out.grad) * other.data
      other.grad += self._val_shape(out.grad) * self.data
    out._backward = _backward

    return out

  # element-wise unary ops
  def __pow__(self, exponent):
    assert isinstance(exponent, (float, int)), "power implemented only for float and int exponents"
    out = Tensor(self.data ** exponent, (self,), f'**{exponent}')

    def _backward():
      self.grad += (exponent * self.data**(exponent - 1)) * out.grad
    out._backward = _backward

    return out
  
  def exp(self):
    out = Tensor(np.exp(self.data), (self,), 'exp')
    def _backward():
      self.grad += np.exp(self.data) * out.grad
    out._backward = _backward
    return out
    
  # Activation functions
  def relu(self):
    out = Tensor(self.data * (self.data > 0), (self,), 'ReLU')

    def _backward():
      self.grad += out.grad * (out.data > 0)
    out._backward = _backward

    return out

  def sigm(self): return 1 / (1 + (-self).exp())
  def softmax(self): raise NotImplementedError

  # generation helpers
  def uniform(shape, low=-1, high=1, _seed=None): return Tensor((high - low) * np.random.default_rng(_seed).random(shape) + low)

  # free real estate
  def __rmul__(self, other): return  self * other
  def __sub__(self, other): return  self + (-other)
  def __rsub__(self, other): return  (-self) + other
  def __radd__(self, other): return self + other
  def __neg__(self): return self * -1
  def __truediv__(self, other): return self * other**-1
  def __rtruediv__(self, other): return other * self**-1
  # internal grad helpers
  def _zero_grad(self): self.grad = np.zeros(self._shape)
  def _one_grad(self): self.grad =  np.ones(self._shape)
  def _val_shape(self, val): return np.ones(self._shape) * val
  def _build_topo(self):
    topo = []
    visited = set()
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)
    build_topo(self)
    return topo
  def _build_rtopo(self): return reversed(self._build_topo())

  # formatter for pretty printing
  def __repr__(self): return f'Tensor(shape={self._shape}, grad={self.grad}, _op={self._op},\n{self.data}\n)\n'
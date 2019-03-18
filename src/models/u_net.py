"""U-Net model class."""

import torch

###############################################################
# Create a tensor and set ``requires_grad=True`` to track computation with it
x = torch.ones(2, 2, requires_grad=True)
x = torch.tensor([[2., 2.], [2., 2.]], requires_grad=True)
print(x)

###############################################################
# Do a tensor operation:
y = x + 2
print(y)

###############################################################
# ``y`` was created as a result of an operation, so it has a ``grad_fn``.
print(y.grad_fn)

###############################################################
# Do more operations on ``y``
z = y * y * 3
out = z.mean()

print(z, out)

###############################################################
# Gradients
# ---------
# Let's backprop now.
# Because ``out`` contains a single scalar, ``out.backward()`` is
# equivalent to ``out.backward(torch.tensor(1.))``.

out.backward(torch.tensor(2.))
out.backward()

###############################################################
# Print gradients d(out)/dx
#

print(x.grad)

###############################################################
# You should have got a matrix of ``4.5``. Let’s call the ``out``
# *Tensor* “:math:`o`”.
# We have that :math:`o = \frac{1}{4}\sum_i z_i`,
# :math:`z_i = 3(x_i+2)^2` and :math:`z_i\bigr\rvert_{x_i=1} = 27`.
# Therefore,
# :math:`\frac{\partial o}{\partial x_i} = \frac{3}{2}(x_i+2)`, hence
# :math:`\frac{\partial o}{\partial x_i}\bigr\rvert_{x_i=1} = \frac{9}{2} = 4.5`.

###############################################################
# Mathematically, if you have a vector valued function :math:`\vec{y}=f(\vec{x})`,
# then the gradient of :math:`\vec{y}` with respect to :math:`\vec{x}`
# is a Jacobian matrix:
#
# .. math::
#   J=\left(\begin{array}{ccc}
#    \frac{\partial y_{1}}{\partial x_{1}} & \cdots & \frac{\partial y_{1}}{\partial x_{n}}\\
#    \vdots & \ddots & \vdots\\
#    \frac{\partial y_{m}}{\partial x_{1}} & \cdots & \frac{\partial y_{m}}{\partial x_{n}}
#    \end{array}\right)
#
# Generally speaking, ``torch.autograd`` is an engine for computing
# vector-Jacobian product. That is, given any vector
# :math:`v=\left(\begin{array}{cccc} v_{1} & v_{2} & \cdots & v_{m}\end{array}\right)^{T}`,
# compute the product :math:`v^{T}\cdot J`. If :math:`v` happens to be
# the gradient of a scalar function :math:`l=g\left(\vec{y}\right)`,
# that is,
# :math:`v=\left(\begin{array}{ccc}\frac{\partial l}{\partial y_{1}} & \cdots & \frac{\partial l}{\partial y_{m}}\end{array}\right)^{T}`,
# then by the chain rule, the vector-Jacobian product would be the
# gradient of :math:`l` with respect to :math:`\vec{x}`:
#
# .. math::
#   J^{T}\cdot v=\left(\begin{array}{ccc}
#    \frac{\partial y_{1}}{\partial x_{1}} & \cdots & \frac{\partial y_{m}}{\partial x_{1}}\\
#    \vdots & \ddots & \vdots\\
#    \frac{\partial y_{1}}{\partial x_{n}} & \cdots & \frac{\partial y_{m}}{\partial x_{n}}
#    \end{array}\right)\left(\begin{array}{c}
#    \frac{\partial l}{\partial y_{1}}\\
#    \vdots\\
#    \frac{\partial l}{\partial y_{m}}
#    \end{array}\right)=\left(\begin{array}{c}
#    \frac{\partial l}{\partial x_{1}}\\
#    \vdots\\
#    \frac{\partial l}{\partial x_{n}}
#    \end{array}\right)
#
# (Note that :math:`v^{T}\cdot J` gives a row vector which can be
# treated as a column vector by taking :math:`J^{T}\cdot v`.)
#
# This characteristic of vector-Jacobian product makes it very
# convenient to feed external gradients into a model that has
# non-scalar output.

###############################################################
# Now let's take a look at an example of vector-Jacobian product:

x = torch.randn(3, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)

###############################################################
# Now in this case ``y`` is no longer a scalar. ``torch.autograd``
# could not compute the full Jacobian directly, but if we just
# want the vector-Jacobian product, simply pass the vector to
# ``backward`` as argument:
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)

print(x.grad)

###############################################################
# You can also stop autograd from tracking history on Tensors
# with ``.requires_grad=True`` by wrapping the code block in
# ``with torch.no_grad():``
print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
	print((x ** 2).requires_grad)

###############################################################
# **Read Later:**
#
# Documentation of ``autograd`` and ``Function`` is at
# https://pytorch.org/docs/autograd

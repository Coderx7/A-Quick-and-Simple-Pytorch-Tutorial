#%%
# in the name of God the most compassionate the most merciful
# lets create a value class that does stuff
import math
from typing import Any
from graphviz import Digraph
import numpy as np
import random
from collections.abc import Iterable
from typing import Union


class Value:
    def __init__(self, data, children=(), op="", label="") -> None:
        self.data = float(data)
        self.grad = 0.0
        self.children = set(children)
        self.backward = lambda: None
        self.op = op
        self.label = str(id(self))[-4:] if not label else label

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, children=(self, other), op="+")
        out.label = f"add{str(id(out))[-4:]}"

        def backward():
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad

        out.backward = backward
        return out

    __radd__ = __add__

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + -other

    __rsub__ = __sub__

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), op="*")
        out.label = f"mul{str(id(out))[-4:]}"

        def backward():
            other.grad += self.data * out.grad
            self.grad += other.data * out.grad

        out.backward = backward
        return out

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = self * other**-1
        # def backward():
        #     other.grad = out.grad*self.data
        #     self.grad = out.grad * other.data
        # out.backward = backward
        return out

    __rtruediv__ = __truediv__


    def exp(self):
        out = Value(math.exp(self.data),(self,), op='exp', label='exp')
        def backward():
            self.grad += out.data*out.grad
        out.backward = backward
        return out
    
    def tanh(self):
        out = Value((math.exp(2*self.data)-1)/(math.exp(2*self.data)+1), (self,), op='tanh')
        def backward():
            self.grad += 1-(out.data**2)*out.grad
        out.backward = backward
        return out
    
    def relu(self):
        out = Value(0 if self.data<0 else self.data, children=(self,), op='ReLU')
        def backward():
            self.grad += out.grad if self.data>0 else 0
        out.backward = backward
        return out
    
    def __pow__(self, other):
        out = Value(self.data**other, (self,), op="pow")

        def backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out.backward = backward
        return out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.data:.4f})[{self.label}]"

    def backwardit(self):
        # create a tree of children and then run backward on them all
        topo=[]
        def dobackward(root:Value):
            if root not in topo:
                # print(f'{root.label} added')
                for child in root.children:
                    dobackward(child)
                topo.append(root)
            return topo
        dobackward(self)
        # print(topo)
        self.grad = 1
        for t in reversed(topo):
            t.backward()


# displays this in ipython
def show_graph(graph_objc:Digraph):
    import sys
    if 'ipykernel' in sys.modules:
        from IPython.display import display
        display(graph_objc)
    else:
        graph_objc.view()

# draw a graph
def draw(root):
    def build_graph(root: Value):
        nodes = set()
        edges = set()

        def build(node: Value):
            if node not in nodes:
                nodes.add(node)
                for child in node.children:
                    edges.add((child, node))
                    build(child)

        build(root)
        return nodes, edges

    nodes, edges = build_graph(root)
    # print(f'nodes: {nodes}')
    # print(f'edges: {edges}')
    graph = Digraph(format="svg", graph_attr={"rankdir": "LR"})  # left to right graph
    for n in nodes:
        uid = str(id(n))
        graph.node(name=uid, label=f"{{{n.label}| data {n.data:.4f}|grad {n.grad:.4f} }}", shape="record")
        if n.op:
            graph.node(name=uid + n.op, label=n.op)
            graph.edge(uid + n.op, uid)

    for n1, n2 in edges:
        graph.edge((str(id(n1))), str(id(n2)) + n2.op)

    show_graph(graph)
    return graph

#%%
class __Value:
    """stores a single scalar value and its gradient"""

    def __init__(self, data, children=(), op="", label=""):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self.children = set(children)
        self.op = op  # the op that produced this node, for graphviz / debugging / etc
        self.label = label

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward

        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), "ReLU")

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward

        return out

    def backwardit(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def tanh(self):
        out = Value((math.exp(2*self.data)-1)/(math.exp(2*self.data)+1), (self,), op='tanh')
        def _backward():
            self.grad += 1-(out.data**2)*out.grad
        out._backward = _backward
        return out

    def __neg__(self):  # -self
        return self * -1

    def __radd__(self, other):  # other + self
        return self + other

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __rmul__(self, other):  # other * self
        return self * other

    def __truediv__(self, other):  # self / other
        return self * other**-1

    def __rtruediv__(self, other):  # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
# a = Value(2, label="a")
# b = Value(3, label="b")
# c = Value(4, label="c")
# print(a, b, c)
# e = a * b;      e.label = "e"
# f = e + 10;     f.label = "f"
# d = f / c;      d.label = "d"
# h = 0.0001
# c += h
# d2 = f / c;     d2.label = "d2"
# print((d2.data - d.data / h))
# draw(d)
#%%

a = Value(2, label="a")
b = Value(-3, label="b")
c = Value(10, label="c")
f = Value(-2, label="f")
e = a * b;      
d = e + c;      
l = d / f;      
# l = l.tanh()
e.label = "e"
d.label = "d"
l.label = "l"
print(a, b, c, d, e, f)

def gradient_check(var="a", h=0.0001, l=l):

    a = Value(2, label="a")
    b = Value(-3, label="b")
    c = Value(10, label="c")
    f = Value(-2, label="f")
    a.data = a.data+h if var == "a" else a.data
    b.data = b.data+h if var == "b" else b.data
    c.data = c.data+h if var == "c" else c.data
        
    # e -6
    e = a * b;      e.label = "e"; e.data=e.data+h if var == "e" else e.data
    # d 4
    d = e + c;      d.label = "d"; d.data=d.data+h if var=='d' else d.data; f.data =f.data+h if var=='f' else f.data
    # f -2
    l2 = d / f;     l2.label = "l"
    # l2 = l2.tanh()
    l2.backwardit()
    grad = (l2.data-l.data)/h
    print(f"loss:{l2.data:.1f}, {var}.grad: {grad:.1f} ")
    return draw(l2)
gradient_check("a")

#%%
draw(l)
#%%
l.grad = 1
# print(d.op)
# print(d.children)
# a.grad = b.grad = c.grad = 0
l.backwardit()
print(f.grad, e.grad, c.grad, b.grad, a.grad)
draw(l)
#%%
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
b = Value(6.88137335870195432,label='b')
x1w1 = x1*w1 ; x1w1.label='x1w1'
# x1w2 = x1*w2 ; x1w2.label='x1w2'
# x2w1 = x2*w1 ; x2w1.label='x2w1'
x2w2 = x2*w2 ; x2w2.label='x2w2'
xwsum = x1w1 + x2w2 ;xwsum.label='xwsum'
n = xwsum+b ;n.label='n'
# o = n.tanh(); 
# we can implenet tanh using builtin exp and division which uses mul and pow
e = (2*n).exp();  e.label='e'
o = (e-1)/(e+1)  ;  o.label='o'
# o.grad=1
o.backwardit()
draw(o)
#%%
a = Value(3)
b = a+a 
# b.grad=1
b.backwardit()
draw(b)

# good, now lets create higher abstractions such as neurons, layers and network!
#%%
class Neuron:
    def __init__(self, num_input,nonlin=True) -> None:
        self.num_input = num_input
        self.weights = list(Value(random.uniform(-1,1)) for _  in range(num_input))
        self.bias = Value(random.uniform(-1,1))
        self.nonlin = nonlin
        
    def __repr__(self) -> str:
        return  f'Neuron({self.num_input})\n weights: {self.weights}\n'
    
    def __call__(self, inputs:Iterable) -> Value:
        # out = wx +b
        # return out.act()
        output = sum((x*w for x, w in zip(inputs, self.weights)), self.bias)
        # with tanh, it just doesnt converge easily if the bias is set as 0!
        # second the learning rate 0.001 seems to be somewhat ok with tanh always applied
        # but after some iterations it will diverge. if we replace tanh with relu, 
        # it acts the same way, they somewhat work if bias is random(-1,1), if set to 0
        # they will flat out diverge nearly immediately.
        # larger learing rates will diverge as well. 
        #  
        # so using activation functions such as relu/tanh on all layers just kills the training!
        # it keeps outputing 4.0 for relu and just diverge for tanh (loss increases) and 
        # learning just doesnt happen!
        # to fix this herefore we only apply activation function on the last output for this to work 
        # consistently and ok, otherwise it fails. 
        # relu is just masively more stable than tanh in our case!! lr =0.01 seems to be good for relu
        # but for tanh lr=0.001 seems to be better (but it mayvery well diverge!! quickly anyway)
        return output.relu() if self.nonlin else output
    
    def parameters(self):
        return self.weights + [self.bias]
  
n = Neuron(2)
print(f'neuron params: {n.parameters()}')

# now lets create a layer it has some input and output!
class Layer:
    def __init__(self, num_in, num_out, **kwargs) -> None:
        self.neurons = list(Neuron(num_in,**kwargs) for _ in range(num_out))
        # print(self.neurons)
    def __call__(self, inputs: Iterable) -> Iterable[Value]:
        out = [n(inputs) for n in self.neurons]
        return out[0] if len(out) == 1 else out
    
    def parameters(self):
        return [p for n in self.neurons 
                  for p in n.parameters()]
            
l = Layer(5,3)
l([1,2,3,4,5])
print(f'layer params: {l.parameters()}')

# now lets create an MLP!
class MLP:
    def __init__(self, num_input, layers_output:Iterable[int|float]) -> None:
        # lets create a list with the first item be the number of input for our first layer
        # and the rest be each layers output number. 
        # every layers input after the first layer will be previous layers output
        # so by doing this we can easily use a i and i+1 index to initialize all layers
        # becasue we are getting any iterable, we convert it to list so all iterables work well
        size_info = [num_input]+list(layers_output)
        # note we are using the len of layers_output so we dont face index out of range at the last item
        self.layers = [Layer(size_info[i], size_info[i+1], nonlin=i!=len(layers_output)-1) for i in range(len(layers_output))]
    
    def __call__(self, x) -> Value:
        for l in self.layers:
            x = l(x)
        return x

    def parameters(self):
        return [p for layer in self.layers 
                for p in layer.parameters()]
        
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0
                
mlp = MLP(3,[4,4,1])
# x = [1,2,3]
# o=mlp(x)
# o.grad=1
# o.backwardit()
# draw(o)

#%%
mlp = MLP(3,[4,4,1])
xs = [[2.0,3.0,-1.0],
      [3.0,-1.0,0.5],
      [0.5,1.0,1.0],
      [1.0,1.0,-1.0]]

ys =[1.0,-1.0,-1.0,1.0]
output = [mlp(x) for x in xs]
loss=sum(((o-y)**2 for y,o in zip(ys,output)))
loss
# draw(loss)
#%%
for i in range(10):
    output = [mlp(x) for x in xs]
    loss=sum(((o-y)**2 for y,o in zip(ys,output)))
    # print(*output,sep='\n')
    # print('loss:',loss.data)

    # for p in mlp.parameters():
    #     p.grad =0
    mlp.zero_grad()
    loss.backwardit()
    # mlp.layers[0].neurons[0].weights[0].grad
    # mlp.parameters()

    # so now lets optimize it 
    # we know that gradient here refers to the direction by which 
    # the fucntion increases (here our loss) so each parameters gradient
    # shows the direction to a higher loss (the direction which increases the functions output)
    # so what we need to do is to take a step in that direction to increase the function output
    # but since our function is basically our loss, we need to decrease it not increase it so we
    # go the other way, that is we use the nagative of the gradient! 
    for p in mlp.parameters():
        # print(f'p.data before: {p.data}')
        p.data += (-0.01* p.grad)
        # print(f'p.data after: {p.data}')

    print(i, loss.data)#, *output,sep='\n')
    
[mlp(x) for x in xs]
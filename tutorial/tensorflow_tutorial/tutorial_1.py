import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
my_tensor = torch.tensor([[1,2,3],[4,5,6]], dtype=torch.float32, requires_grad=True)
print(my_tensor)

print(my_tensor.dtype)
print(my_tensor.device)
print(my_tensor.shape)
print(my_tensor.requires_grad)

x = torch.empty(size=(3,3))
y = torch.zeros((3,3))
z = torch.eye(5,5)
a = torch.rand((3,3))
e = torch.empty(size=(1,5)).normal_(mean=0,std=1)
f = torch.empty(size=(1,5)).uniform_(0, 1)


# convert tensor to different type
tensor = torch.arange(4)
print(tensor.bool())
print(tensor.short())
print(tensor.long())
print(tensor.half())
print(tensor.float())


import numpy as np
np_array = np.zeros((5,5))
tensor = torch.from_numpy(np_array)
array_back = tensor.numpy()


# math comp

x = torch.tensor([1,2,3])
y = torch.tensor([4,5,6])

z1 = torch.empty(3)
torch.add(x,y,out=z1)

z2 = torch.add(x,y)
z3 = x+y

# subtraction
s1 = x-y
s2 = torch.true_divide(x,y)

# inplace
t = torch.zeros(3)
t.add_(x)
t+=x


# exponential
e1 = x.pow(2)
e2 = x**2

# matrix multiplication
x1 = torch.rand((3,3))
x3 = torch.mm(x1,x1)

# matrix exp
matrix_exp = torch.rand(5,5)
print(matrix_exp)
print(matrix_exp.matrix_power(3))

# element wise mult
z = x*y
print(z)

# dot product
z = torch.dot(x,y)
print(z)


# batch matrix multiplication
############


# board casting
x1 = torch.rand((5,5))
x2 = torch.rand((1,5))

z = x1-x2
z = x1**x2

# other useful tensor operation
sum_x = torch.sum(x,dim=0)
values,indices = torch.max(x, dim=0)
values, indices = torch.min(x, dim=0)
abs_x = torch.abs(x)
argmax = torch.argmax(x, dim=0)
print(argmax)

mean_x = torch.mean(x.float(), dim=0)
equal = torch.eq(x,y)
print(equal)
A,B = torch.sort(y, dim=0, descending=False)

clamp = torch.clamp(x,min=0, max=10) # less than 0 set to 0, same for max

x = torch.tensor([1,1,1,0,1], dtype=torch.bool)
a = torch.any(x)
b = torch.all(x)


###################
x = torch.arange(10)
indices = [2,5,8]
print(x[indices])

#advance indexing
x = torch.arange(10)
print(x[(x<2)|(x>8)])
print(x[x.remainder(2)==0])

print(torch.where(x>5, x, x+2))
print(torch.tensor([0,0,1,1,2,2,]).unique())
print(x.ndimension())
print(x.numel())


x = torch.arange(9)
x_33 = x.view(3,3)
x_33_ = x.reshape(3,3)

x1 = torch.rand((2,5))
x2 = torch.rand((2,5))
print(torch.cat((x1,x2), dim=0).shape)

z = x1.view(-1)
print(z.shape)

# un row
unrow = x1.view(-1)
print(unrow.shape)
batch = 64
x = torch.rand((batch,2,5))
print(x.view(batch,-1).shape)

z = x.permute(0, 2, 1) # switch dimension

x = torch.arange(10)
print(x.unsqueeze(-1).shape)


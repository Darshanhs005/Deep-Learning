import torch
import torch.nn as nn

#  ==== Put your solutions here ====
# Each function receives an nn.Linear.
# Hint #1: Make sure you print out the shape of the weights
# Hint #2: You can fill multiple weights at once by assigning
#  the weights to a tensor. e.g.
# lin.weight.data[:] = torch.tensor([
#     [1, 0],
#     [0, 1],
# ], dtype=lin.weight.dtype, device=lin.weight.device)

def modify_lin_1(lin):
    ''' lin = nn.Linear(1, 1) '''
    # TODO    
    pass
    

def modify_lin_2(lin):
    ''' lin = nn.Linear(1, 1) '''
    # TODO
    pass

def modify_lin_3(lin):
    ''' lin = nn.Linear(4, 1) '''
    # TODO
    pass

def modify_lin_4(lin):
    ''' lin = nn.Linear(4, 2) '''
    # TODO
    pass

def modify_lin_5(lin):
    ''' lin = nn.Linear(3, 3) '''
    # TODO
    pass

def modify_lin_6(lin):
    ''' lin = nn.Linear(5, 2) '''
    # TODO
    pass
    


#  ==== Testing code: Tests your solutions ====
def test_1(fnc):
    inp = torch.tensor([1., 5, 11, 20, 21]).reshape([-1, 1])
    tar = torch.tensor([2., 6, 12, 21, 22]).reshape([-1, 1])
    layer = nn.Linear(1, 1)
    fnc(layer)
    out = layer(inp)
    torch.allclose(out, tar)

def test_2(fnc):
    inp = torch.tensor([1.,  5, 11, 20, 21]).reshape([-1, 1])
    tar = torch.tensor([5., 17, 35, 62, 65]).reshape([-1, 1])
    layer = nn.Linear(1, 1)
    fnc(layer)
    out = layer(inp)
    torch.allclose(out, tar)

def test_3(fnc):
    inp = torch.tensor([
        [1., 1, 1, 1],
        [5, 10, 15, 20],
        [11, 20, 21, 25]
    ])
    tar = inp.mean(dim=1, keepdim=True)
    layer = nn.Linear(4, 1)
    fnc(layer)
    out = layer(inp)
    torch.allclose(out, tar)

def test_4(fnc):
    inp = torch.tensor([
        [1., 1, 1, 1],
        [5, 10, 15, 20],
        [11, 20, 21, 25]
    ])
    tar = torch.stack([
        inp.mean(dim=1),
        inp.sum(dim=1)
    ], dim=1)
    layer = nn.Linear(4, 2)
    fnc(layer)
    out = layer(inp)
    torch.allclose(out, tar)

def test_5(fnc):
    inp = torch.tensor([
        [1., 1, 1],
        [5, 10, 15],
        [11, 20, 21],
        [4, 12, 2],
        [6, 5, 4],
    ])
    tar = torch.tensor([
        [1., 1, 1],
        [15, 10, 5],
        [21, 20, 20],
        [2, 12, 4],
        [4, 5, 6],
    ])
    layer = nn.Linear(3, 3)
    fnc(layer)
    out = layer(inp)
    torch.allclose(out, tar)

def test_6(fnc):
    inp = torch.tensor([
        [1., 2, 3, 4, 5],
        [1e10, 2e20, 3e30, 4e40, 5e50],
        [-150, 150, 15, -15, 0.1]
    ])
    tar = torch.tensor([
        [4., 2],
        [4, 2],
        [4, 2],
    ])
    layer = nn.Linear(5, 2)
    fnc(layer)
    out = layer(inp)
    torch.allclose(out, tar)
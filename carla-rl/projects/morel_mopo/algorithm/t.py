import torch

t = torch.Tensor([[5,6,7],[3,4,5]])
print(t.shape)
i = [[0, 1, 1],
     [2, 0, 2]]
v =  [3, 4, 5]
s = torch.sparse_coo_tensor(i, v, (2, 3))
s = s.to_dense()
print(s.shape)

st = torch.cat((t, s))
print(st.shape)
import torch
import torch.nn as nn

m = nn.Softmax(dim=1)
log_m = nn.LogSoftmax(dim=1)
log_criterion = nn.NLLLoss()
criterion = nn.CrossEntropyLoss()
input = torch.randn(2, 3)
output = log_m(input)
# print(input)
# print(output)
# print(torch.exp(output))
# print(m(input))

loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
output = loss(input, target)
print(output)

log_output = log_m(input)
output = log_criterion(input, target)
print(output)
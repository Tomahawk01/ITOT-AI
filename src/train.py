import torch
import torch.nn as nn
import torch.optim as optim

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.layer(x)

def main():
    net = Network()
    optimizer = optim.SGD(net.parameters(), lr=0.00001)
    criterion = nn.MSELoss()
    i = 0

    while True:
        optimizer.zero_grad()
        inputs = torch.rand((4096, 1)) * 1000 - 500
        expected = inputs * 4 + 3
        output = net(inputs)
        loss = criterion(output, expected)
        loss.backward()
        optimizer.step()
        i += 1
        if i % 1000 == 0:
            print(loss)
        if loss < 0.01:
            break

    net.eval()
    print(net(torch.tensor([1.0, 2.0, 3.0, 4.0]).reshape(-1, 1)))
    print(list(net.named_parameters()))

if __name__ == "__main__":
    main()

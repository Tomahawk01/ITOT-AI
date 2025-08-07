import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(1, 3)
        self.linear2 = nn.Linear(3, 1)
    
    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x

def main():
    net = Network()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    i = 0

    while True:
        optimizer.zero_grad()
        inputs = torch.rand((4096, 1)) * math.pi * 2 - math.pi
        expected = torch.sin(inputs)
        output = net(inputs)
        loss = criterion(output, expected)
        loss.backward()
        optimizer.step()
        i += 1
        if i % 5000 == 0:
            print("loss", loss.item())
            net.eval()
            with torch.no_grad():
                with open("output.csv", "w") as f:
                    inputs = torch.arange(500) / 500.0 * math.pi * 2.0 - math.pi
                    expected = torch.sin(inputs)
                    outputs = net(inputs.reshape(-1, 1))

                    for i in range(len(inputs)):
                        f.write("{}, {}, {}\n".format(inputs[i], outputs[i][0], expected[i]))
            net.train()

if __name__ == "__main__":
    main()

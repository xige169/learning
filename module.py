import torch
import torch.nn as nn

class Tudui(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq=nn.Sequential(
            nn.Conv2d(3,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4,64),
            nn.Linear(64,10)
        )
    def forward(self,input):
        output=self.seq(input)
        return output


if __name__=='__main__':
    tudui=Tudui()
    input = torch.ones((64,3,32,32))
    output = tudui(input)
    print(output.shape)
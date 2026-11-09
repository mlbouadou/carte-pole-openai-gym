from torch import nn

class PolicyNetwork(nn.Module):
    def __init__(self,hidden_layers=((16,nn.Tanh),),hidden_bias=False,output_bias=False,input_size=4,output_size=2):
        super().__init__()
        layers=[]
        previous_size=input_size
        for hidden_size,activation_cls in hidden_layers:
            layers.append(nn.Linear(previous_size,hidden_size,bias=hidden_bias))
            layers.append(activation_cls())
            previous_size=hidden_size
        layers.append(nn.Linear(previous_size,output_size,bias=output_bias))
        self.network=nn.Sequential(*layers)

    def forward(self,state):
        return self.network(state)
import torch
from torch.distributions import Categorical
class Agent:
    def __init__(self,model):
        self.model=model
    def act(self,state):
        parameter=next(self.model.parameters())
        state_tensor=torch.as_tensor(state,dtype=parameter.dtype,device=parameter.device)
        logits=self.model(state_tensor)
        distribution=Categorical(logits=logits)
        action=distribution.sample()
        log_prob=distribution.log_prob(action)
        return action.item(),log_prob
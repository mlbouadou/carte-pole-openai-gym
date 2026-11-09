import numpy as np
import torch
from torch.distributions import Categorical
def evaluate(model,env,episodes=100,deterministic=True,seed=10000):
    if episodes<1:
        raise ValueError("episodes must be positive")
    parameter=next(model.parameters())
    previous_mode=model.training
    scores=[]
    lengths=[]
    terminations=[]
    truncations=[]
    model.eval()
    try:
        with torch.no_grad():
            for episode in range(episodes):
                episode_seed=None if seed is None else seed+episode
                state,_=env.reset(seed=episode_seed)
                score=0.
                length=0
                done=False
                while not done:
                    state_tensor=torch.as_tensor(state,dtype=parameter.dtype,device=parameter.device)
                    logits=model(state_tensor)
                    if deterministic:
                        action=logits.argmax(dim=-1).item()
                    else:
                        action=Categorical(logits=logits).sample().item()
                    state,reward,terminated,truncated,_=env.step(action)
                    score+=float(reward)
                    length+=1
                    done=terminated or truncated
                scores.append(score)
                lengths.append(length)
                terminations.append(bool(terminated))
                truncations.append(bool(truncated))
    finally:
        model.train(previous_mode)
    return {
        "scores":scores,
        "lengths":lengths,
        "mean_score":float(np.mean(scores)),
        "std_score":float(np.std(scores)),
        "min_score":float(np.min(scores)),
        "max_score":float(np.max(scores)),
        "terminated":terminations,
        "truncated":truncations,
        "deterministic":deterministic,
        "seed":seed
    }
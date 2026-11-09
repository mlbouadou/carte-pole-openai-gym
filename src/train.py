import torch
from .agent import Agent
def calculate_returns(rewards):
    returns=[]
    total=0.
    for reward in reversed(rewards):
        total+=reward
        returns.append(total)
    return list(reversed(returns))
def train(model,env,optimizer,episodes=1000,batch_size=5,keep_episodes=False,log_every=20):
    if episodes<1 or batch_size<1:
        raise ValueError("episodes and batch_size must be positive")
    parameter=next(model.parameters())
    model.train()
    agent=Agent(model)
    history={"scores":[],"losses":[],"episodes":[]}
    completed=0
    updates=0
    while completed<episodes:
        episode_losses=[]
        current_batch_size=min(batch_size,episodes-completed)
        for _ in range(current_batch_size):
            state,_=env.reset()
            log_probs=[]
            rewards=[]
            states=[]
            actions=[]
            done=False
            while not done:
                action_value,log_prob=agent.act(state)
                next_state,reward,terminated,truncated,_=env.step(action_value)
                log_probs.append(log_prob)
                rewards.append(float(reward))
                if keep_episodes:
                    states.append(state.copy())
                    actions.append(action_value)
                state=next_state
                done=terminated or truncated
            returns=torch.tensor(calculate_returns(rewards),dtype=parameter.dtype,device=parameter.device)
            episode_loss=-(torch.stack(log_probs)*returns).sum()
            episode_losses.append(episode_loss)
            history["scores"].append(sum(rewards))
            if keep_episodes:
                history["episodes"].append({
                    "states":states,
                    "actions":actions,
                    "rewards":rewards,
                    "terminated":terminated,
                    "truncated":truncated
                })
            completed+=1
        loss=torch.stack(episode_losses).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history["losses"].append(loss.item())
        updates+=1
        if log_every and updates%log_every==0:
            mean_score=sum(history["scores"][-current_batch_size:])/current_batch_size
            print(f"episodes={completed} | mean score={mean_score:.1f}")
    return history
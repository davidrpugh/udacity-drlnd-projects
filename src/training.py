import collections
import typing

from unityagents import UnityEnvironment

from agents import UnityAgent


Score = float
Scores = typing.List[float]


def _train_for_at_most(agent: UnityAgent, env: UnityEnvironment, max_timesteps: int) -> Score:
    """Train agent for a maximum number of timesteps."""
    state = env.reset()
    score = 0
    for t in range(max_timesteps):
        action = agent(state)
        next_state, reward, done, _ = env.step(action)
        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward
        if done:
            break
    return score

                
def _train_until_done(agent: UnityAgent, env: UnityEnvironment, brain_name: str) -> Score:
    """Train the agent until the current episode is complete."""
    env_info = env.reset(train_mode=False)[brain_name]
    state = env_info.vector_observations[0]
    score = 0
    done = False
    while not done:
        action = agent(state)
        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward
    return score


def train(agent: UnityAgent,
          env: UnityEnvironment,
          brain_name: str,
          checkpoint_filepath: str,
          target_score: float,
          number_episodes: int,
          maximum_timesteps=None) -> Scores:
    """
    Reinforcement learning training loop.
    
    Parameters:
    -----------
    agent (Agent): an agent to train.
    env (UnityEnvironment): an environment in which to train the agent.
    checkpoint_filepath (str): filepath used to save the state of the trained agent.
    number_episodes (int): maximum number of training episodes.
    maximum_timesteps (int): maximum number of timesteps per episode.
    
    Returns:
    --------
    scores (list): collection of episode scores from training.
    
    """
    scores = []
    most_recent_scores = collections.deque(maxlen=100)
    best_score = 0
    for i in range(number_episodes):
        if maximum_timesteps is None:
            score = _train_until_done(agent, env, brain_name)
        else:
            score = _train_for_at_most(agent, env, maximum_timesteps)         
        scores.append(score)
        most_recent_scores.append(score)
        if score > best_score:
            agent.save(checkpoint_filepath)
            
        
        average_score = sum(most_recent_scores) / len(most_recent_scores)
        if average_score > target_score:
            print(f"\nEnvironment solved in {i:d} episodes!\tAverage Score: {average_score:.2f}")
            agent.save(checkpoint_filepath)
            break
        if average_score > best_score:
            print(f"New top score! {average_score}")
            agent.save(checkpoint_filepath)
            best_score = average_score
        if (i + 1) % 100 == 0:
            print(f"\rEpisode {i + 1}\tAverage Score: {average_score:.2f}")

    return scores
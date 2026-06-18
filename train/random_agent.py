import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.env.mci_env import MCIEnv

def run_random_agent(n_episodes, use_kg=False):
    env = MCIEnv(use_kg_constraint=use_kg)
    label = "KG-constrained" if use_kg else "Vanilla"

    total_rewards = []
    kg_violations = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        ep_violations = 0

        while not done:
            action = env.action_space.sample()
            obs, reward, done, _, info = env.step(action)
            ep_reward += reward
            if reward == env.kg_penalty:
                ep_violations += 1

        total_rewards.append(ep_reward)
        kg_violations.append(ep_violations)

    avg_reward    = sum(total_rewards) / n_episodes
    avg_violations = sum(kg_violations) / n_episodes

    print(f"\n[{label}] Random agent over {n_episodes} episodes:")
    print(f"  Avg reward:     {avg_reward:.2f}")
    print(f"  Avg violations: {avg_violations:.2f}")
    print(f"  Sample episode log:")
    for entry in env.episode_log[:3]:
        print(f"    {entry['patient']:30s} | action={entry['action']:18s} "
              f"| correct={entry['correct']:14s} | r={entry['reward']:+.1f} "
              f"| kg={'OK' if entry['kg_valid'] else 'BLOCKED'}")

if __name__ == "__main__":
    run_random_agent(n_episodes=50, use_kg=False)
    run_random_agent(n_episodes=50, use_kg=True)
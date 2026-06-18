import numpy as np
import torch
import random
import copy

from src.env.mci_env import MCIEnv
from src.env.dqn import DQN, ReplayBuffer
from src.kg.constraint_guard import get_kg_recommended_action
from src.env.mci_env import INDEX_ACTION
from src.eval.exhaustive_eval import exhaustive_accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def select_action(state, model, epsilon, action_dim, mask=None):
    if random.random() < epsilon:
        if mask is not None:
            valid_actions = np.where(mask)[0]
            return int(np.random.choice(valid_actions))
        return random.randint(0, action_dim - 1)

    state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
    q_values = model(state_t).detach().cpu().numpy()[0]

    if mask is not None:
        q_values[~mask] = -1e9

    return int(np.argmax(q_values))


def train(use_kg=False, episodes=500, seed=42, save_path=None, eval_every=20):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = MCIEnv(use_kg_constraint=use_kg)

    # Derive dims from the env instead of hardcoding -- this is what
    # broke last time the observation/action space grew, so the env is
    # now the single source of truth for both.
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model  = DQN(state_dim=state_dim, action_dim=action_dim).to(device)
    target = DQN(state_dim=state_dim, action_dim=action_dim).to(device)
    target.load_state_dict(model.state_dict())

    optimizer     = torch.optim.Adam(model.parameters(), lr=1e-3)
    buffer        = ReplayBuffer()

    gamma         = 0.99
    batch_size    = 32
    epsilon       = 1.0
    epsilon_min   = 0.05
    epsilon_decay = 0.995

    rewards_log      = []
    violation_log    = []
    correct_tag_log  = []
    kg_agreement_log = []

    # Track the BEST snapshot seen during training, scored by the
    # exhaustive 80-profile sweep (see eval_every below) -- not by a
    # rolling average over training episodes, which follow the skewed
    # patient-prior distribution and can under-sample rare categories.
    best_score      = -float("inf")
    best_state_dict = None

    for ep in range(episodes):
        state, _ = env.reset()
        done = False

        ep_reward    = 0
        ep_violations = 0
        ep_correct   = 0
        ep_tags      = 0
        ep_kg_agree  = 0
        ep_kg_total  = 0

        while not done:
            mask   = env.get_valid_action_mask() if use_kg else None
            action = select_action(state, model, epsilon, action_dim, mask)

            # ── KG agreement: checked BEFORE step so current_patient is correct ──
            action_name = INDEX_ACTION[action]
            if action_name.startswith("tag_"):
                kg_rec = get_kg_recommended_action(env.onto, env.current_patient)
                if kg_rec is not None:
                    ep_kg_total += 1
                    if action_name == kg_rec:
                        ep_kg_agree += 1
            # ─────────────────────────────────────────────────────────────────────
            next_state, reward, done, _, info = env.step(action)

            if not env.episode_log[-1]["kg_valid"]:
                ep_violations += 1

            buffer.push(state, action, reward, next_state, done)
            state      = next_state
            ep_reward += reward

            if len(buffer) > batch_size:
                s, a, r, s2, d = buffer.sample(batch_size)

                s  = torch.FloatTensor(np.array(s)).to(device)
                a  = torch.LongTensor(a).to(device)
                r  = torch.FloatTensor(r).to(device)
                s2 = torch.FloatTensor(np.array(s2)).to(device)
                d  = torch.FloatTensor(d).to(device)

                q = model(s).gather(1, a.unsqueeze(1)).squeeze()
                with torch.no_grad():
                    q_next   = target(s2).max(1)[0]
                    target_q = r + gamma * q_next * (1 - d)

                loss = torch.nn.functional.mse_loss(q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Correct tag rate from episode log. IMPORTANT: the denominator
        # is the total number of patients presented this episode
        # (env.EPISODE_LENGTH), not just the ones that received a tag
        # action. A patient who times out without ever being tagged
        # must count as incorrect, not be silently excluded -- otherwise
        # a policy that times out on hard patients but nails the easy
        # ones it does reach can score a misleadingly high "accuracy".
        for entry in env.episode_log:
            if entry["action"].startswith("tag_"):
                ep_tags += 1
                if entry["action"] == entry["correct"]:
                    ep_correct += 1

        correct_rate  = ep_correct  / env.EPISODE_LENGTH
        kg_agree_rate = ep_kg_agree / ep_kg_total  if ep_kg_total  > 0 else 0.0

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        rewards_log.append(ep_reward)
        violation_log.append(ep_violations)
        correct_tag_log.append(correct_rate)
        kg_agreement_log.append(kg_agree_rate)

        if ep % eval_every == 0:
            target.load_state_dict(model.state_dict())

            # Validate against the UNIFORM 80-profile enumeration, not
            # training episodes -- training episodes follow the
            # realistic, skewed patient prior (Expectant patients are
            # ~5% of all patients, ~1% per hazard variant), so a 20-
            # episode window can simply miss a rare category by chance
            # and let a checkpoint that's actually bad at it score a
            # deceptively high "accuracy". Every profile gets equal
            # weight here, every time.
            model.eval()
            exhaustive_acc, exhaustive_viol = exhaustive_accuracy(model, use_masking=use_kg)
            model.train()

            if exhaustive_acc > best_score:
                best_score = exhaustive_acc
                best_state_dict = copy.deepcopy(model.state_dict())

            print(f"EP {ep:>3} | Reward: {ep_reward:>8.2f} | "
                  f"Correct: {correct_rate:.2f} | "
                  f"KG-agree: {kg_agree_rate:.2f} | "
                  f"Violations: {ep_violations} | Eps: {epsilon:.2f} | "
                  f"Exhaustive-acc: {exhaustive_acc:.2f} (viol {exhaustive_viol:.2f})")

    if save_path is not None:
        save_state = best_state_dict if best_state_dict is not None else model.state_dict()
        torch.save(save_state, save_path)
        print(f"Saved BEST checkpoint (exhaustive 80-profile accuracy "
              f"{best_score:.2f}) to {save_path}")

    return rewards_log, violation_log, correct_tag_log, kg_agreement_log
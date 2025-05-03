import numpy as np
import random
import copy
from collections import defaultdict

## Define the MDP with groups seperate, didnt finish but this is an idea
# states = ['Rested', 'Tired', 'Done', 'Undone', '8am', '8pm', '10am', '10pm']
# actions = ['Party', 'Rest', 'Study']
# terminal_states = ['Class']  # Adjust if your terminal state is different

# # Define valid actions per state
# valid_actions = {
#     'Rested': ['Party', 'Study'],
#     'Tired': ['Rest'],
#     'Done': ['Party'],
#     'Undone': ['Study'],
# }

# # Transition probabilities and rewards
# # Format: transitions[state][action] = [(probability, next_state, reward)]
# transitions = {
#     'Rested': {
#         'Party': [(1.0, 'Tired', 10)],
#         'Study': [(1.0, 'Tired', 5)]
#     },
#     'Tired': {
#         'Rest': [(1.0, 'Rested', -2)]
#     },
#     'Done': {
#         'Party': [(1.0, '8pm', 20)]
#     },
#     'Undone': {
#         'Study': [(0.8, 'Done', 10), (0.2, 'Undone', -1)]
#     }
# }

# Grouping states together
states = [
    'RU8', 'TU8', 'RD8', 'TD8',
    'RU9', 'TU9', 'RD9', 'TD9',
    'RU10', 'TU10', 'RD10', 'TD10',
    'RU11', 'TU11', 'RD11', 'TD11',
    'class'
]

terminal_states = ['class']

actions = ['P', 'R', 'S']  # Party, Rest, Study

valid_actions = {
    'RU8': ['P', 'R', 'S'],
    'TU8': ['R'],
    'RD8': ['P'],
    'TD8': ['R'],

    'RU9': ['P', 'R', 'S'],
    'TU9': ['R'],
    'RD9': ['P'],
    'TD9': ['R'],

    'RU10': ['P', 'R', 'S'],
    'TU10': ['R'],
    'RD10': ['P'],
    'TD10': ['R'],

    'RU11': ['P', 'R', 'S'],
    'TU11': ['R'],
    'RD11': ['P'],
    'TD11': ['R']
}

transitions = { #idk if these are supposed to extend past what we see in the graph
    'RU8': {
        'P': [(1.0, 'TU9', 2)],
        'R': [(1.0, 'RU9', 0)],
        'S': [(1.0, 'TU9', -1)]
    },
    'TU8': {
        'R': [(1.0, 'RU9', 0)]
    },
    'RD8': {
        'P': [(1.0, 'TD9', 2)]
    },
    'TD8': {
        'R': [(1.0, 'RD9', 0)]
    },
    'RU9': {
        'P': [(1.0, 'TU10', 2)],
        'R': [(1.0, 'RU10', 0)],
        'S': [(0.5, 'TU10', 2), (0.5, 'TU10', 0)]
    },
    'TU9': {
        'R': [(1.0, 'RU10', -1)]
    },
    'RD9': {
        'P': [(1.0, 'TD10', 2)]
    },
    'TD9': {
        'R': [(1.0, 'RD10', 0)]
    },
    'RU10': {
        'P': [(1.0, 'TU11', 2)],
        'R': [(1.0, 'RU11', 0)],
        'S': [(1.0, 'TU11', -1)]
    },
    'TU10': {
        'R': [(1.0, 'RU11', 0)]
    },
    'RD10': {
        'P': [(1.0, 'TD11', 2)]
    },
    'TD10': {
        'R': [(1.0, 'RD11', 0)]
    },
    'RU11': {
        'P': [(1.0, 'class', -1)],
        'R': [(1.0, 'class', 0)],
        'S': [(1.0, 'class', 4)]
    },
    'TU11': {
        'R': [(1.0, 'class', 0)]
    },
    'RD11': {
        'P': [(1.0, 'class', 3)]
    },
    'TD11': {
        'R': [(1.0, 'class', 2)]
    }
}


DISCOUNT = 0.99
THRESHOLD = 0.001
LEARNING_RATE = 0.1
EPSILON = 0.2 

def value_iteration(): #PART 1 TASK
    V = {state: 0.0 for state in states}
    policy = {state: None for state in states}
    iterations = 0

    while True:
        delta = 0
        iterations += 1
        for state in states:
            #if the terminal state is reached, quit
            if state in terminal_states:
                continue 
            max_value = float('-inf')
            best_action = None
            action_values = {}

            for action in valid_actions.get(state, []):
                value = 0
                for prob, next_state, reward in transitions[state][action]:
                    value += prob * (reward + DISCOUNT * V[next_state])
                action_values[action] = value
                if value > max_value:
                    max_value = value
                    best_action = action

            old_value = V[state]
            V[state] = max_value
            policy[state] = best_action
            delta = max(delta, abs(old_value - V[state]))

            print(f"[VI] State: {state} | Old V: {old_value:.3f} -> New V: {V[state]:.3f} | Action Values: {action_values} | Best: {best_action}")

        if delta < THRESHOLD:
            break

    print("\n[VI] Converged!")
    print(f"Iterations: {iterations}")
    print("Final Values:", V)
    print("Optimal Policy:", policy)

def q_learning(): #PART 2 TASK
    Q = defaultdict(lambda: {action: 0.0 for action in actions})
    policy = {}
    episodes = 0

    while True:
        state = random.choice([s for s in states if s not in terminal_states])
        episode_delta = 0
        episodes += 1

        while state not in terminal_states:
            valid = valid_actions.get(state, [])
            if not valid:
                break

            # Epsilon-greedy action selection
            if random.random() < EPSILON:
                action = random.choice(valid)
            else:
                action = max(valid, key=lambda a: Q[state][a])

            transitions_list = transitions[state][action]
            probs = [t[0] for t in transitions_list]
            chosen = random.choices(transitions_list, weights=probs)[0]
            next_state, reward = chosen[1], chosen[2]

            max_q_next = max(Q[next_state].values()) if next_state not in terminal_states else 0.0
            old_q = Q[state][action]
            new_q = old_q + LEARNING_RATE * (reward + DISCOUNT * max_q_next - old_q)
            Q[state][action] = new_q
            episode_delta = max(episode_delta, abs(new_q - old_q))

            print(f"[QL] State: {state}, Action: {action} | Old Q: {old_q:.3f} -> New Q: {new_q:.3f} | Reward: {reward}, Max Q Next: {max_q_next:.3f}")

            state = next_state

        if episode_delta < THRESHOLD:
            break

    for state in states:
        if state not in terminal_states:
            policy[state] = max(Q[state], key=Q[state].get)

    print("\n[QL] Converged yay!")
    print(f"Episodes: {episodes}")
    print("Final Q Values:", dict(Q))
    print("Optimal Policy:", policy)

if __name__ == "__main__":
    print("\n--- Part 1: Value Iteration  ---")
    value_iteration()

    print("\n--- Part 2: Q-Learning ----")
    q_learning()

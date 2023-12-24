import random

# Define the play_round function to determine the outcome of a single round
def play_round(strategy1, strategy2, history1, history2):
    move1 = strategy1(history2)  # Note that strategy1 looks at history2 and vice versa
    move2 = strategy2(history1)
    history1.append(move1)
    history2.append(move2)
    return move1, move2

# Define each strategy as a function
def tit_for_tat(history):
    return "cooperate" if not history else history[-1]

def tit_for_two_tats(history):
    if len(history) < 2 or history[-1] == "cooperate" or history[-2] == "cooperate":
        return "cooperate"
    return "defect"

def always_defect(_):
    return "defect"

def always_cooperate(_):
    return "cooperate"

def random_strategy(_):
    return random.choice(["cooperate", "defect"])

def suspicious_tit_for_tat(history):
    return "defect" if not history else history[-1]

def grudger(history):
    if "defect" in history:
        return "defect"
    return "cooperate"

def pavlov(history):
    if not history:
        return "cooperate"
    if len(history) % 2 == 0:  # On even turns, repeat last move
        return history[-1]
    else:  # On odd turns, switch if last move was defect, otherwise cooperate
        return "defect" if history[-1] == "defect" else "cooperate"


def gradual(history):
    if "defect" not in history:
        return "cooperate"
    defections = history.count("defect")
    forgiven = history[history.index("defect"):]
    forgiven_cooperations = forgiven.count("cooperate")
    return "defect" if forgiven_cooperations < defections else "cooperate"

def probe(history):
    # This is a simplified version of the probe strategy
    if not history:
        return "defect"
    if history[-1] == "cooperate":
        return "defect"
    return "cooperate"

def tester(history):
    if not history:
        return "defect"
    if history[-1] == "defect":
        return "cooperate"
    return "defect"

def firm_but_fair(history):
    if not history:
        return "cooperate"
    if len(history) < 2:
        return history[-1]
    if history[-1] == "defect" and history[-2] == "defect":
        return "cooperate"
    return "defect" if history[-1] == "defect" else "cooperate"


def joss(history):
    if not history:
        return "cooperate"
    return "defect" if random.random() < 0.1 else history[-1]

def downing(history):
    # This is a placeholder for the Downing strategy which requires complex analysis
    # For the purposes of this example, we'll use a random strategy
    return random.choice(["cooperate", "defect"])

def random_tit_for_tat(history):
    if not history:
        return "cooperate"
    return "defect" if random.random() < 0.05 else history[-1]

# Define a function to run a match between two strategies
def run_match(strategy1, strategy2, rounds=100):
    history1, history2 = [], []
    for _ in range(rounds):
        play_round(strategy1, strategy2, history1, history2)
    return history1, history2

# Define a function to simulate a round-robin tournament
def round_robin_tournament(strategies, rounds_per_match=100):
    scores = {strategy.__name__: 0 for strategy in strategies}

    for i, strategy1 in enumerate(strategies):
        for j, strategy2 in enumerate(strategies):
            if i != j:  # Avoid playing against oneself
                history1, history2 = run_match(strategy1, strategy2, rounds_per_match)
                score1 = history1.count("cooperate")
                score2 = history2.count("cooperate")
                scores[strategy1.__name__] += score1
                scores[strategy2.__name__] += score2
    
    return scores

# List of strategies in the tournament
strategies = [
    tit_for_tat, tit_for_two_tats, always_defect, always_cooperate, random_strategy,
    suspicious_tit_for_tat, grudger, pavlov, gradual, probe, tester, firm_but_fair,
    joss, downing, random_tit_for_tat
]

# Run the tournament
tournament_results = round_robin_tournament(strategies)
print(tournament_results)
import random
from sqlite3 import adapt
import numpy as np


import matplotlib.pyplot as plt

# Define each strategy as a function
def tit_for_tat(history):
    return "cooperate" if not history else history[-1]

def tit_for_two_tats(history):
    if len(history) < 2 or history[-1] == "cooperate" or history[-2] == "cooperate":
        return "cooperate"
    return "defect"

def always_defect(_):
    return "defect"

def pattern_recognition(history):
    pattern_length = 3
    if len(history) < pattern_length * 2:
        return "cooperate"
    last_pattern = history[-pattern_length:]
    if history[-pattern_length*2:-pattern_length] == last_pattern:
        return "defect" if last_pattern[-1] == "cooperate" else "cooperate"
    return "cooperate"


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

def adaptive_strategy(history):
    if not history:
        return "cooperate"
    cooperation_rate = history.count("cooperate") / len(history)
    return "cooperate" if cooperation_rate > 0.5 else "defect"


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

def alternate_cooperate_defect(_):
    return "cooperate" if random.choice([True, False]) else "defect"

def mirror_last_two(history):
    if len(history) < 2:
        return "cooperate"
    return "cooperate" if history[-1] == "cooperate" and history[-2] == "cooperate" else "defect"


# Define a function to run a match between two strategies
def run_match(strategy1, strategy2, min_rounds=50, max_rounds=150):
    rounds = random.randint(min_rounds, max_rounds)
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

# Function to randomly change strategies
def randomize_strategies(strategies, probability=0.1):
    all_strategies = [
        tit_for_tat, tit_for_two_tats, always_defect, always_cooperate, random_strategy,
        suspicious_tit_for_tat, grudger, pavlov, gradual, probe, tester, firm_but_fair,
        joss, downing, random_tit_for_tat, alternate_cooperate_defect, mirror_last_two,
        pattern_recognition, adaptive_strategy
    ]
    for i in range(len(strategies)):
        if random.random() < probability:
            strategies[i] = random.choice(all_strategies)
    return strategies


# Function to simulate the tournament with strategy switching
def simulate_tournament(num_players, strategies, num_rounds):
    # Initialize players with random strategies
    players = [Player(random.choice(strategies)) for _ in range(num_players)]
    
    # Simulate the rounds
    for _ in range(num_rounds):
        # Shuffle the players for random matching
        random.shuffle(players)
        
        # Pair players and simulate the rounds
        for i in range(0, num_players, 2):
            if i+1 < num_players:  # Check if there is a pair to match
                play_round(players[i], players[i+1])
                
    return players





def run_dynamic_match(strategies, strategy1, strategy2, min_rounds=50, max_rounds=150, switch_prob=None):
    # Ensure that switch_prob has a value
    if switch_prob is None:
        switch_prob = 1 / len(strategies)

    rounds = random.randint(min_rounds, max_rounds)
    history1, history2 = [], []
    for _ in range(rounds):
        # Randomly switch strategies with a certain probability
        if random.random() < switch_prob:
            strategy1 = random.choice(strategies)
        if random.random() < switch_prob:
            strategy2 = random.choice(strategies)
        
        play_round(strategy1, strategy2, history1, history2)
    return history1, history2



# Function to simulate a dynamic tournament
def simulate_dynamic_tournament(strategies, rounds_per_match=100):
    results = np.zeros((len(strategies), len(strategies)))
    switch_prob = 1 / len(strategies)  # Set the switch probability based on the number of strategies

    for i, initial_strategy1 in enumerate(strategies):
        for j, initial_strategy2 in enumerate(strategies):
            if i != j:  # Avoid playing against oneself
                history1, history2 = run_dynamic_match(strategies, initial_strategy1, initial_strategy2, rounds_per_match, switch_prob=switch_prob)
                score1 = history1.count("cooperate")
                score2 = history2.count("cooperate")
                results[i, j] += score1
                results[j, i] += score2
    
    return results


class Player:
  def __init__(self, strategy):
      self.strategy = strategy
      self.histories = {} # Dictionary to store history with each opponent
      self.coins = 0 # Actual coins earned
      self.potential_coins = 0 # Potential coins lost

  def update_history(self, opponent_strategy, move):
     # If playing with this opponent for the first time, initialize history
     if opponent_strategy.__name__ not in self.histories:
         self.histories[opponent_strategy.__name__] = []

     # Append the move to the history
     self.histories[opponent_strategy.__name__].append(move)

  def get_history_with_opponent(self, opponent_strategy):
      return self.histories.get(opponent_strategy.__name__, [])

  def get_coins(self):
      return self.coins

  def get_potential_coins(self):
      return self.potential_coins
  

def play_round(player1, player2):
    player1_strategy = player1.strategy
    player2_strategy = player2.strategy
    move1 = player1.strategy(player1.get_history_with_opponent(player2_strategy))
    move2 = player2.strategy(player2.get_history_with_opponent(player1_strategy))
    player1.update_history(player2_strategy, move2)
    player2.update_history(player1_strategy, move1)
    # ... rest of the function remains the same

    if move1 == "cooperate" and move2 == "cooperate":
            player1.coins += 3
            player1.potential_coins += 2
            player2.coins += 3
            player2.potential_coins += 2

    elif move1 == "cooperate" and move2 == "defect":
            player1.coins += 0
            player1.potential_coins += 1
            player2.coins += 5
            player2.potential_coins += 0

    elif move1 == "defect" and move2 == "cooperate":
            player1.coins += 5
            player1.potential_coins += 0
            player2.coins += 0
            player2.potential_coins += 1
    else: # Both players defect
            player1.coins += 1
            player2.coins += 1
            
    return move1, move2

def simulate_round(players):
    for player1 in players:
        player2 = random.choice(players)
        if player1 != player2:  # Make sure a player does not play against themselves
            play_round(player1, player2)

# Function to run the tournament
def run_tournament(strategies, num_players=100, num_rounds=200):
    # Initialize players with random strategies
    players = [Player(random.choice(strategies)) for _ in range(num_players)]

    # Run the rounds
    for _ in range(num_rounds):
        simulate_round(players)

    # Collect the results
    coins = [player.get_coins() for player in players]
    potential_coins = [player.get_potential_coins() for player in players]
    
    return players, coins, potential_coins

# Function to plot the results
def plot_results(players):
    # Sort players by the number of coins
    sorted_players = sorted(players, key=lambda x: x.coins, reverse=True)
    
    # Get the coins and potential coins for plotting
    coins = [player.coins for player in sorted_players]
    potential_coins = [player.potential_coins for player in sorted_players]
    names = [f'Player {i+1}' for i in range(len(sorted_players))]
    
    # Plotting
    fig, ax1 = plt.subplots()
    ax1.bar(names, coins, label='Actual Coins', color='g')
    ax1.set_xlabel('Players')
    ax1.set_ylabel('Actual Coins', color='g')
    ax1.tick_params(axis='y', labelcolor='g')
    ax1.set_xticklabels(names, rotation=45, ha='right')
    
    ax2 = ax1.twinx()
    ax2.bar(names, potential_coins, label='Potential Coins', color='b', alpha=0.5)
    ax2.set_ylabel('Potential Coins', color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    
    fig.tight_layout()
    plt.legend()
    plt.show()


def plot_coins(scores, players):
 fig, ax1 = plt.subplots()

 # Plot actual coins earned
 ax1.set_xlabel('Time')
 ax1.set_ylabel('Actual Coins', color='blue')
 ax1.plot([player.get_coins() for player in players], color='blue')
 ax1.tick_params(axis='y', labelcolor='blue')

 # Plot potential coins lost
 ax2 = ax1.twinx()
 ax2.set_ylabel('Potential Coins', color='red')
 ax2.plot([player.get_potential_coins() for player in players], color='red')
 ax2.tick_params(axis='y', labelcolor='red')

 plt.show()

# List of strategies in the tournament

# List and randomize strategies
strategies = [
   tit_for_tat, tit_for_two_tats, always_defect, always_cooperate, random_strategy,
   suspicious_tit_for_tat, grudger, pavlov, gradual, probe, tester, firm_but_fair,
   joss, downing, random_tit_for_tat, alternate_cooperate_defect, mirror_last_two,
   pattern_recognition, adaptive_strategy
]

players = [Player(strategy) for strategy in strategies]

players = simulate_tournament(players)
coins = [player.get_coins() for player in players]

# Run the tournament
tournament_results = simulate_tournament(strategies, 1000)

# Plotting the results
plt.imshow(tournament_results, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.xticks(range(len(strategies)), [s.__name__ for s in strategies], rotation=45)
plt.yticks(range(len(strategies)), [s.__name__ for s in strategies])
plt.title("Strategy Performance in Iterated Prisoner's Dilemma")
plt.show()

# Get the number of coins for each player
coins = [player.get_coins() for player in players]

# Plot the number of coins
plt.imshow(coins, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.xticks(range(len(players)), [player.__class__.__name__ for player in players], rotation=45)
plt.yticks(range(len(players)), [player.__class__.__name__ for player in players])
plt.title("Number of Coins Earned by Each Player")
plt.show()
# Number of players and rounds
num_players = 100
num_rounds = 200

players = simulate_tournament(num_players, strategies, num_rounds)

players, coins, potential_coins = run_tournament(strategies)
plot_results(players, coins, potential_coins)

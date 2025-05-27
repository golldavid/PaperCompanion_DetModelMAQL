import numpy as np
#import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from abc import ABC


"""
Author: David Goll

This module implements a flexible simulation framework for multi-agent reinforcement learning in repeated normal-form games, as used in the paper:
"Deterministic Model of Incremental Multi-Agent Boltzmann Q-Learning: Transient Cooperation, Metastability, and Oscillations" (D. Goll, J. Heitzig, W. Barfuss, 2024, ArXiv).

Key features:
- Modular agent classes supporting Q-Learning and FrequencyAdjusted Q-learning with extensible design for additional algorithms (SARSA, Expected SARSA, CrossLearning, ...).
- Abstract Agent base class for repeated games, supporting customizable action spaces, reward functions, and observation histories.
- Game and Simulation classes to conduct multi-agent interactions.
- Utilities for analyzing learning dynamics, fixed points, and stability in the context of the Prisoner's Dilemma and similar games.

Note: The code is research-oriented and tailored for generating figures and results in the referenced paper. 
While not structured as a general-purpose library, it is organized for clarity, extensibility, and reproducibility of the depicted experiments.
"""

################################### Agent classes ###################################

class Agent(ABC): 
    """ 
    This class implements an agent that can play a repeated normal-form game and observe past joint actions.
    """

    # parameters of the agent:
    agent_id = None
    """The id of the agent."""
    player_id = None
    """The id of the player that this agent will play.""" 
    num_players = None
    """The number of players of the game."""
    action_space = None
    """The action space of the agent."""
    num_actions = None
    """The number of actions of the agent."""
    discount_factor = None
    """The discount factor of the agent."""
    selection_method = None
    """The selection method of the agent."""
    temperature = None
    """The temperature of the agent."""
    reward_func = None
    """The reward function of the agent."""
    observation_length = None 
    """The observation length of the agent."""

    # learning hyperparameters of the agent:
    learning_rate = None
    """The learning rate of the agent."""
    exploration_rate = None
    """The exploration rate value of the agent."""

    # variables changing during the episode from time step to time step:
    state = None
    """The state of the agent."""
    state_history = None
    """The state history of the agent."""
    action = None
    """The action of the agent."""
    reward = None
    """The utility of the agent."""
    observation = None
    """The observation of the agent.""" 

    # learning-related variables changing during the episode from time step to time step:
    q_table = None
    """The Q-table of the agent."""
    q_table_history = None
    """The Q-table history of the agent."""

    # variables to be saved in history:
    q_table_history = None
    """The Q-table history of the agent."""

    def __init__(self, 
                 player_id = None,
                 action_space = None, 
                 learning_rate = 0.1, 
                 discount_factor=0.0, 
                 exploration_rate=0.1,
                 num_players = None,
                 observation_length = 0,
                 temperature = 1,
                 reward_func = None,
                 state = None, 
                 q_table = None,
                 agent_id = None,
                 selection_method="Boltzmann"):
        """
        This function initializes an agent object.

        Args:
            player_id (int): The id of the player that this agent will play.
            action_space (array): The action space of the agent.
            learning_rate (float, optional): Defaults to 0.1. The learning rate of the agent. 
            discount_factor (float, optional): Defaults to 0.9. The discount factor of the agent.
            exploration_rate (float, optional): Defaults to 0.2. The exploration rate value of the agent.
            cooperation_probability (float, optional): Defaults to None. The cooperation probability of the agent.
            num_players (int, optional): Defaults to None. The number of players of the game.
            observation_length (int, optional): Defaults to 0. The observation length of the agent.
            temperature (int, optional): Defaults to 1. The temperature of the agent.
            reward_func (method, optional): Defaults to None. The reward function of the agent. Requirements: 
                                            Input: an action_vector array and the player_id of the agent. 
                                            Returns: a reward value (float). 
            state (int, optional): Defaults to None. The state of the agent.
            q_table (numpy.array, optional): Defaults to None. The Q-table of the agent.
            agent_id (int, optional): Defaults to None. The id of the agent.
            selection_method (str, optional): Defaults to "epsilon_greedy". The selection method of the agent.
        """     

        assert player_id is not None, "The player_id has to be specified."
        assert action_space is not None, "The action space has to be specified."
        assert num_players is not None, "The number of players has to be specified."
        assert 0 <= player_id < num_players, "The player_id has to be between 0 and the number of players."
        assert num_players is not None, "The number of players has to be specified."
        assert 0 <= discount_factor <= 1, "The discount factor has to be between 0 and 1."
        assert 0 <= exploration_rate <= 1, "The epsilon value has to be between 0 and 1."
        assert 0 <= learning_rate <= 1, "The learning rate has to be between 0 and 1." 
        assert num_players is not None, "The number of players has to be specified."
        assert 0 <= observation_length, "The observation_length has to be a positive integer."
        assert 0 <= temperature, "The temperature has to be a positive float."

        self.player_id = player_id
        self.agent_id = agent_id
        self.num_players = num_players
        self.action_space = action_space 
        self.num_actions = len(action_space) 
        self.observation_length = observation_length
        self.observation = '' # empty string to save the observation of the agent
        self.num_states = (self.num_actions**self.num_players)**self.observation_length
        if state == None:
            self.state = - self.observation_length
        else:
            self.state = state
        self.state_history = []

        self._learning_rate = learning_rate # Initialize learning_rate with the provided value
        self.discount_factor = discount_factor
        self.selection_method = selection_method # by default: Boltzmann if not specified otherwise
        self.exploration_rate = exploration_rate # for epsilon-greedy selection method
        self.temperature = temperature # by default: 1 if not specified otherwise
        self.reward_func = reward_func # by default: None if not specified otherwise

        if q_table is None:
            self.q_table = np.zeros((self.num_states, self.num_actions)) # initialize Q-table, shape: (num_states, num_actions), num_states = (num_actions**num_players)**observation_length
            self.initial_q_table = self.q_table.copy()
        else:
            self.q_table = q_table 
            self.initial_q_table = self.q_table.copy()
        self.q_table_history = [self.initial_q_table.copy()]
        
    # make learning_rate a property to ensure that it stays within [0, 1]
    @property
    def learning_rate(self):
        return self._learning_rate
    # setter method for learning_rate
    @learning_rate.setter
    def learning_rate(self, value):
        # Ensure learning_rate stays within [0, 1]
        if 0 <= value <= 1:
            self._learning_rate = value
        else:
            raise ValueError("Invalid learning rate: {}. Learning rate must be in the range [0, 1].".format(value))
    
    def update_policy(self, current_info):
        pass

    def update_attributes(self, current_info):
        """
        This function updates the attributes of the agent.

        Args:
            current_info (dict): Dictionary containing 'state', 'action_vector', 'reward' and 'next_state'.
        """
        state = current_info['state']
        self.state_history.append(state) # save state in attribute of agent
        next_state = current_info['next_state']
        self.state = next_state

    def reset(self):
        """
        This function resets the agent.
        """
        self.state = - self.observation_length # reset state
        self.state_history = [] # reset state history
        #reset q_table
        self.q_table = self.initial_q_table.copy()
        self.q_table_history = [self.initial_q_table.copy()] # reset Q_history
        
    def get_action_probabilities(self, q_table=None):
        """
        This function calculates the action probabilities of the agent.

        Args:
            q_table (numpy.array): The Q-table of the agent.

        Returns:
            numpy.array: The action probabilities of the agent.
        """
        if q_table is None:
            q_table = self.q_table
        
        if self.selection_method == "epsilon_greedy":
            num_rows, num_cols = q_table.shape
            non_max_prob = self.exploration_rate / (num_cols - 1)
            # Initialize new array
            action_probabilities = np.full((num_rows, num_cols), non_max_prob)
            # Get indices of max values in each row
            max_indices = np.argmax(q_table, axis=1)
            # Set max value positions to (1 - epsilon)
            action_probabilities[np.arange(num_rows), max_indices] = 1 - self.exploration_rate
        
        if self.selection_method == "Boltzmann":
            temperature = self.temperature
            action_probabilities = np.exp(q_table / temperature) / np.sum(np.exp(q_table / temperature), axis=1)[:, np.newaxis] 

        return action_probabilities
    
    def choose_action(self, state): 
        """
        This function chooses an action for the agent. It also updates the action attribute of the agent.
        For the first observation_length time steps, the agent chooses the action according to a fixed strategy (always cooperate / defect, choose randomly).

        Args:
            state (int): The state of the agent.

        Returns:
            action (int): The chosen action of the agent.
        """
        # The first obersvation_length time steps have negative states. For these states, choose the actions according to a fixed strategy (always cooperate / defect, choose randomly)
        if state < 0:
            # always cooperate the first obervation_length steps
            self.action = 0
            return self.action

        action_chosen = np.random.choice(self.action_space, p=self.get_action_probabilities(self.q_table)[state])
        self.action = action_chosen
        return self.action

    def calculate_reward(self, action_vector):
        """
        This function calculates the reward of the agent

        Args:
            action_vector (numpy.array): array of the actions of all players

        Raises:
            ValueError: "The reward function has to be specified."

        Returns:
            reward (float): reward
        """

        ''' 
        This function calculates the utility of the agent. It also updates the utility attribute of the agent.
        :param action_vector: The action vector of the agents. 
        :param utility_function: The utility function of the agent.
        :return: The utility of the agent. 
        '''
        if self.reward_func is None: 
            raise ValueError("The reward function has to be specified.")
        
        self.reward = self.reward_func(action_vector, self.player_id)
        return self.reward
    
    def observe(self, current_info):
        """
        This function updates the policy and the attributes of the agent.

        Args:
            current_info (list): List of dictionaries containing the current information which is presented to the agents.
        """
        self.update_policy(current_info)
        self.update_attributes(current_info)

    def update_observation(self, action_vector):
        """
        This function updates the observation of the agent by appending the action_vector to the observation.
        If the observation_length is zero, no observation is maintained.
        If the number of time steps played is greater than the observation_length, 
        the first num_players digits are cut off from the existing observation before appending the new action_vector.

        Args:
            action_vector (numpy.array): The action vector of the current time step.
        """
        # Return immediately if observation_length is zero
        if self.observation_length == 0:
            return

        action_str = ''.join(map(str, action_vector.astype(int)))
        if len(self.observation) < (self.num_players * self.observation_length):
            self.observation += action_str
        else:
            # Remove the oldest action_vector and append the new one
            self.observation = self.observation[self.num_players:] + action_str

    def get_next_state(self, action_vector):
        if self.state >= 0:
            self.update_observation(action_vector)
            next_state = self.translate_key_to_state(self.observation)
            return next_state
        else:
            next_state = self.state + 1 if self.state < -1 else self.translate_key_to_state(self.observation)
            return next_state

    def translate_key_to_state(self, state_key):
        """
        This function translates a key to a state. A key is a string of 0s and 1s with length num_players * observation_length. It is a binary representation of the actions of the last observation_length time steps of all players.
        The state is an integer between 0 and num_states-1. It is a integer representation of the key. The state can be used to access the Q-table.

        Args:
            state_key (string): binary representation of the actions of the last observation_length time steps of all players

        Raises:
            ValueError: _description_

        Returns:
            int: state as an integer between 0 and num_states-1
        """
        key_length_max = self.num_players * self.observation_length # maximum length of key
        if state_key == '': 
            state = 0 # if the key is empty, the state is 0 because there is only one state 
            return state
        elif len(state_key) == key_length_max:
            state = int(state_key, 2)
            return state
        else:
            raise ValueError(f'There is a problem with the key (key: {state_key}).')
    
    def translate_state_to_key(self, state):
        """
        This function translates a state to a key. A key is a string of 0s and 1s with length num_players * observation_length. It is a binary representation of the actions of the last observation_length time steps of all players.
        The state is an integer between 0 and num_states-1. It is a integer representation of the key. The state can be used to access the Q-table.

        Args:
            state (int): state as an integer between 0 and num_states-1

        Raises:
            ValueError: _description_

        Returns:
            float: binary representation of the actions of the last observation_length time steps of all players
        """
        key_length_max = self.num_players * self.observation_length # maximum length of key
        if state < 0 or state >= 2 ** key_length_max:
            raise ValueError(f'State {state} is out of range for key length {key_length_max}.')

        binary_representation = format(state, f'0{key_length_max}b')
        return binary_representation

    def get_learning_history(self):
        """
        This function returns the learning history of the agent. 

        Returns:
            dictionary: dictionary with the learning history of the agent
        """
        return { "q_table": self.q_table_history }

class QLearningAgent(Agent):
    """
    This class implements a Q-learning agent that can play a multiplayer prisoners dilemma game.

    Args:
        Agent (Agent): parent class.
    """
    def __init__(self, 
                 player_id,
                 action_space, 
                 learning_rate = 0.1, 
                 discount_factor = 0.0, 
                 exploration_rate = 0.2,
                 num_players = None,
                 observation_length = 0,
                 temperature = 1,
                 reward_func = None,
                 state = None, 
                 q_table = None,
                 agent_id = None,
                 selection_method="epsilon_greedy",
                 use_prefactor = False):
        super().__init__(player_id,
                         action_space, 
                         learning_rate, 
                         discount_factor, 
                         exploration_rate,
                         num_players,
                         observation_length,
                         temperature,
                         reward_func,
                         state, 
                         q_table,
                         agent_id,
                         selection_method)
        # Add any QLearningAgent-specific initialization here
        self.name = "QL"

        if use_prefactor:
            self.prefactor = (1 - self.discount_factor)
        else:
            self.prefactor = 1

    def update_policy(self, current_info):
        """
        This function updates the Q-table of the QLearningAgent according to the Q-Learning algorithm.
        The prefactor (1 - self.discount_factor) is missing in the formula in the book (Sutton & Barto, 2018, p. 131. It is taken from 2021 paper by Barfuss et al:
        "[The] prefactor (1 - self.discount_factor) normalizes the state-action values to be on the same numerical scale as the rewards." - Barfuss et al. "Dynamical systems as a level of cognitive analysis of multi-agent learning" 2021, p. 4

        Args:
            current_info (dict): Dictionary containing the current information which is presented to the agents.
        """
        state = current_info['state']
        action = current_info['action']
        reward = current_info['reward']
        next_state = current_info['next_state']
        action_id = np.where(self.action_space == action)

        # update Q-value
        self.q_table[state, action_id] = (1 - self.learning_rate) * self.q_table[state, action_id] + self.learning_rate * ( self.prefactor * reward + self.discount_factor * np.max(self.q_table[next_state, :]) ) 
        self.q_table_history.append(self.q_table.copy()) 

class FreqAdjustedQLearningAgent(QLearningAgent):
    def __init__(self, 
                 player_id,
                 action_space, 
                 learning_rate = 0.1, 
                 discount_factor = 0.9, 
                 exploration_rate = 0.2,
                 num_players = None,
                 observation_length = 0,
                 temperature = 1,
                 reward_func = None,
                 state = None, 
                 q_table = None,
                 agent_id = None,
                 selection_method="epsilon_greedy",
                 use_prefactor = False,
                 learning_rate_adjustment = None):
        super().__init__(player_id,
                         action_space, 
                         learning_rate, 
                         discount_factor, 
                         exploration_rate,
                         num_players,
                         observation_length,
                         temperature,
                         reward_func,
                         state, 
                         q_table,
                         agent_id,
                         selection_method, 
                         use_prefactor)
        self.name = "FreqAdjustedQL"
        if learning_rate_adjustment is None:
            self.learning_rate_adjustment = learning_rate
        else:
            self.learning_rate_adjustment = learning_rate_adjustment 

    def update_policy(self, current_info):
        """
        This function updates the Q-table of the QLearningAgent according to the Q-Learning algorithm.
        The prefactor (1 - self.discount_factor) is missing in the formula in the book (Sutton & Barto, 2018, p. 131. It is taken from 2021 paper by Barfuss:
        "factor (1 - self.discount_factor) normalizes the state- action values to be on the same numerical scale as the rewards." - Barfuss "Dynamical systems as a level of cognitive analysis of multi-agent learning" 2021, p. 4

        Args:
            current_info (dict): Dictionary containing the current information which is presented to the agents.
        """
        state = current_info['state']
        action = current_info['action']
        reward = current_info['reward']
        next_state = current_info['next_state']
        action_id = np.where(self.action_space == action)

        # get probability to choose the action
        action_probability = self.get_action_probabilities(self.q_table)[state][action_id]

        # update Q-value
        self.q_table[state, action_id] = self.q_table[state, action_id] + min(self.learning_rate_adjustment/action_probability, 1) * self.learning_rate * ( self.prefactor * reward + self.discount_factor * np.max(self.q_table[next_state, :]) - self.q_table[state, action_id]) 
        self.q_table_history.append(self.q_table.copy()) 

################################### Game class ###################################

class Game:
    """
    This class implements a game that can be played by multiple agents. It defines the game environment and the rules of the game.
    """

    actions_history = None
    """The actions history of all steps of the game."""
    rewards_history = None
    """The rewards history of all steps of the game."""

    def __init__(self, agents):
        """
        This function initializes a game object.

        Args:
            agents (list): list of agents that play the game. 
        """
        self.agents = agents
        self.num_agents = len(agents)

        self.actions_history = [] # empty list to save actions of all agents
        self.rewards_history = [] # empty list to save rewards of all agents

    def reset(self):
        """
        This function resets the actions history and rewards history of the game.
        """
        # reset all attributes of the game?
        self.actions_history = [] # reset actions_history
        self.rewards_history = [] # reset rewards_history

    def step(self):
        """
        This function executes one time step of the game. It calculates the actions and rewards of the agents.
        The actions and the rewards are saved in the history of the game. 
        It returns a list of dictionaries, each containing the current information which is presented to the agents.
        The current information depends on the type of agent. For example, a Q-learning agent needs the current state, action, reward and next state.

        Returns:
            list: list of dictionaries, each containing the current information which is presented to the agents
        """
        # choose actions based on current states
        action_vector = np.array([agent.choose_action(agent.state) for agent in self.agents])
        reward_vector = np.array([agent.calculate_reward(action_vector) for agent in self.agents])
        # History updates
        self.actions_history.append(action_vector) # save action_vector in attribute of game
        self.rewards_history.append(reward_vector) # save reward_vector in attribute of game

        # calculate next states
        for agent in self.agents:
            agent.next_state = agent.get_next_state(action_vector)

        # prepare current_info_vector for update of policy
        current_info_vector = []
        for agent in self.agents:
            if isinstance(agent, QLearningAgent):
                current_info_vector.append({'state': agent.state, 'action': agent.action, 'reward': agent.reward, 'next_state': agent.next_state})
        
        return current_info_vector
 
################################### Simulation class ###################################

class Simulation:
    """
    This class implements a simulation of a game. It is used to run the learning process of the agents. 
    """

    def __init__(self):
        pass

    def reset(self, game, agents):
        """
        This function resets the history of the game and the agents.

        Args:
            game (Game): game object
            agents (list): list of agents that play the game. One agent is of class Agent.
        """
        game.reset()
        for agent in agents:
            agent.reset()

    def run_time_step(self, game, agents):
        """
        This function runs one time step of the game. 
        After one step is finished, the learning values of the agents are updated via the observe function of the agents.

        Args:
            game (Game): game object
            agents (list): list of agents that play the game. One agent is of class Agent.
        """
        # execute one time step of the game and get the information of the current time step
        current_info_vector = game.step() 

        # update the learning values of the agents
        for agent, current_info in zip(agents, current_info_vector):
            agent.observe(current_info)

    def run(self, game, agents, num_time_steps, learning_rate_func=None, temperature_func=None):
        """
        This function runs multiple time steps of the game.

        Args:
            game (Game): game object
            agents (list): list of agents that play the game. One agent is of class Agent.
            num_episodes (int): number of episodes to run
            learning_rate_func (method, optional): Defaults to None. The learning rate function of the agents. Requirements: episode as args. Returns: learning_rate (float).
            temperature_func (method), optional): Defaults to None. The temperature function of the agents. Requirements: episode as args. Returns: temperature (float).
        """

        self.reset(game, agents) # reset history of game and agents

        for time_step in range(num_time_steps):
            # if learning_rate_func is not None, update learning rate of agents according to given function
            if learning_rate_func is not None:
                for agent in agents:
                    agent.learning_rate = learning_rate_func(time_step)
            
            # if temperature_func is not None, update temperature of agents according to given function
            if temperature_func is not None:
                for agent in agents:
                    agent.temperature = temperature_func(time_step)
            
            self.run_time_step(game, agents)

################################### General Functions ################################### 

def reward_matrix_for_two_player_PD(action_vector, player_id):
    """
    This function calculates the reward of the agent in a two-player Prisoner's Dilemma game.
    The reward function is defined as follows:
    R = 3, P = 1, S = 0, T = 5
    The reward function is defined as a dictionary with the following keys:
    (0, 0): (R, R),  # Both cooperate
    (0, 1): (S, T),  # Agent A defects, agent B cooperates
    (1, 0): (T, S),  # Agent A cooperates, agent B defects
    (1, 1): (P, P),  # Both cooperate
    The reward function takes the action vector and the player id as input and returns the reward of the agent.
    The action vector is a list of two elements, where the first element is the action of agent A and the second element is the action of agent B.
    The player id is an integer, where 0 is agent A and 1 is agent B.
    The reward function returns the reward of the agent as a float.
    """
    action_tuple = tuple(action_vector)
    S, P, R, T = 0, 1, 3, 5
    reward_matrix = {
            (0, 0): (R, R),  # Both cooperate
            (0, 1): (S, T),  # Agent A defects, agent B cooperates
            (1, 0): (T, S),  # Agent A cooperates, agent B defects
            (1, 1): (P, P),  # Both cooperate
        }
    reward = reward_matrix[action_tuple]
    return reward[player_id]

def get_individual_matrices(reward_function):
    """
    This function extracts the individual reward matrices from the reward function.
    """
    # extract the individual reward matrices. A: first agent, B: second agent
    reward_matrix_A = np.array([[reward_function([0, 0], 0), reward_function([0, 1], 0)],
                                [reward_function([1, 0], 0), reward_function([1, 1], 0)]])
    reward_matrix_B = np.array([[reward_function([0, 0], 1), reward_function([0, 1], 1)],
                                [reward_function([1, 0], 1), reward_function([1, 1], 1)]])
    return [reward_matrix_A, reward_matrix_B]

def generate_q_values(prob_to_coop, temperature, base_value):
    """
    This function generates Q-values for the two actions (cooperate and defect) based on the given probability of cooperation, temperature and a parameter called base_value which governs the overall level.
    """
    # Calculate the difference between Q-values
    delta_Q = temperature * np.log(1/prob_to_coop - 1) # difference between Q-values: delta_Q = Q_D - Q_C
    
    # Calculate Q_D and Q_C centered around the base value
    Q_D = base_value + delta_Q / 2
    Q_C = base_value - delta_Q / 2
    
    return np.array([Q_C, Q_D])

def calculate_next_probabilities(agents, initial_probabilities):
    """
    This function calculates the next probabilities of cooperation for two agents based on their current probabilities and the reward matrices according the deterministic BQL model by Barfuss et al. (2019).
    """

    reward_matrix_agent_0 = np.array([[agents[0].reward_func([i, j], agents[0].player_id) for j in range(2)] for i in range(2)])
    reward_matrix_agent_1 = np.array([[agents[1].reward_func([i, j], agents[1].player_id) for j in range(2)] for i in range(2)])

    # get learning rates and temperature of the agents
    learning_rate_agent_0 = agents[0].learning_rate
    learning_rate_agent_1 = agents[1].learning_rate
    temperature_agent_0 = agents[0].temperature
    temperature_agent_1 = agents[1].temperature

    next_probabilities_array = []
    for probabilities in initial_probabilities:
        prop_agent_0, prop_agent_1 = probabilities # prob to cooperate
        # construct probability vectors for both agents
        p_0_vector = np.array([prop_agent_0, 1 - prop_agent_0]) # prob to coop, prob to defect
        p_1_vector = np.array([prop_agent_1, 1 - prop_agent_1]) # prob to coop, prob to defect

        # intermediate calculations
        P_0_vector = p_0_vector * np.exp( learning_rate_agent_0 / temperature_agent_0 * (np.dot(reward_matrix_agent_0, p_1_vector) - temperature_agent_0 * np.log(p_0_vector)))
        P_1_vector = p_1_vector * np.exp( learning_rate_agent_1 / temperature_agent_1 * (np.dot(reward_matrix_agent_1.T, p_0_vector) - temperature_agent_1 * np.log(p_1_vector)))

        # calculate the next probabilities
        prop_agent_0_next = P_0_vector[0] / np.sum(P_0_vector) # expected prob to cooperate for agent 1 in next time step
        prop_agent_1_next = P_1_vector[0] / np.sum(P_1_vector) # expected prob to cooperate for agent 2 in next time step

        next_probabilities_array.append((prop_agent_0_next, prop_agent_1_next))
    
    return next_probabilities_array

################################### CAUTION: the following functions are only valid for the Prisoner's Dilemma with S, P, R, T = 0, 1, 3, 5 ! ################################### 
# ToDo: generalize the functions for any symmetric 2x2 matrix game

# CAUTION: This function is written specifically for the Prisoner's Dilemma with S, P, R, T = 0, 1, 3, 5
def calculate_fixed_point_policy(temperature, discount_factor, initial_guess = [0, 1], print_solution = False, print_additional_info = False):
    """
    CAUTION: This function is written specifically for the Prisoner's Dilemma with S, P, R, T = 0, 1, 3, 5 !
    This function calculates the fixed point policy for the deterministic QL dynamics for the Prisoner's Dilemma game using a numerical solver.

    Args:
        temperature (float): The temperature of the system.
        discount_factor (float): The discount factor of the agent.
        initial_guess (list, optional): Defaults to [0, 1]. Initial guess for the solution.
        print_solution (bool, optional): Defaults to False. If True, the function prints the fixed point policy.
        print_additional_info (bool, optional): Defaults to False. If True, the function prints additional information about the solution process.

    Returns:
        float: The fixed point policy for the Prisoner's Dilemma game.
    """
    # define four-dimensional system of equations
    def equations(variables, T = temperature, discount_factor = discount_factor):
        a, b, c, d = variables

        eq1 = a - 1 / (np.exp(c/T) + np.exp(d/T)) * (3 * np.exp(c/T)) +\
            discount_factor * max(a,b)
        eq2 = b - 1 / (np.exp(c/T) + np.exp(d/T)) * (5 * np.exp(c/T) + np.exp(d/T)) +\
            discount_factor * max(a,b)
        eq3 = c - 1 / (np.exp(a/T) + np.exp(b/T)) * (3 * np.exp(a/T)) +\
            discount_factor * max(c,d)
        eq4 = d - 1 / (np.exp(a/T) + np.exp(b/T)) * (5 * np.exp(a/T) + np.exp(b/T)) +\
            discount_factor * max(c,d)

        return [eq1, eq2, eq3, eq4]

    # Initial guess for the solution
    a_init, b_init = initial_guess
    initial_guess = [a_init, b_init, a_init, b_init]

    # Solve the system numerically and get information
    result, infodict, ier, msg = fsolve(equations, initial_guess, full_output=True)
    a, b, c, d = result
    fixed_point_policy = np.exp(a/temperature) / (np.exp(a/temperature) + np.exp(b/temperature))

    # prints
    if print_solution:
        print("------------------------------------")
        print(f"Numerical Solution: Q^1_C = {a}, Q^1_D = {b}, Q^2_C = {c}, Q^2_D = {d}")
        print("Q^1_D - Q^1_C =", b - a)
        print(f"Fixed point policy at T={temperature} : ", fixed_point_policy)
        print("------------------------------------")
        print()

    if print_additional_info:
        # Additional information about the solution process
        print("\nSolution Process Information:")
        print("Number of iterations:", infodict['nfev'])
        print("fjac\n", infodict['fjac'])
        print("fvec\n", infodict['fvec'])
        print("r", infodict['r'])
        print("qtf", infodict['qtf'])
        print("Exit code:", ier)
        print("Exit message:", msg)

        # Zustandssumme: 
        N1 = np.exp(result[0]) + np.exp(result[1])
        N2 = np.exp(result[2]) + np.exp(result[3])
        print()
        print("Zustandssumme:")
        print("N1 =", N1)
        print("N2 =", N2)

        # Wahrscheinlichkeiten:
        print("Wahrscheinlichkeiten:")
        print("p1 =", np.exp(result[0]) / N1)
        print("p2 =", np.exp(result[1]) / N1)

    return fixed_point_policy

# CAUTION: This function is written specifically for the Prisoner's Dilemma with S, P, R, T = 0, 1, 3, 5
def calculate_target_Q_values(prob_to_coop_j, discount_factor, print_solution = False):
    """ 
    CAUTION: This function is written specifically for the Prisoner's Dilemma with S, P, R, T = 0, 1, 3, 5 !
    This function calculates the target Q-values according to the deterministic QL model for the Prisoner's Dilemma game based on the given probability of cooperation and discount factor.

    Args:
        prob_to_coop_j (float): The probability of cooperation for agent j (opponent of i).
        discount_factor (float): The discount factor of the agent.
        print_solution (bool, optional): Defaults to False. If True, the function prints the target Q-values.
    
    Returns:
        numpy.array: The target Q-values for the two actions (cooperate and defect) for agent i.
    """
    # expected rewards for agent i
    Exp_Reward_C = 3 * prob_to_coop_j + 0 * (1. - prob_to_coop_j)
    Exp_Reward_D = 5 * prob_to_coop_j + 1 * (1. - prob_to_coop_j)

    # geometric series for agent i
    Q_max_C = Exp_Reward_C / (1 - discount_factor)
    Q_max_D = Exp_Reward_D / (1 - discount_factor)

    # target Q-values for agent j
    Q_target_C = Exp_Reward_C + discount_factor * max(Q_max_C, Q_max_D)
    Q_target_D = Exp_Reward_D + discount_factor * max(Q_max_C, Q_max_D)

    if print_solution:
        print("------------------------------------")
        print(f"For prob_to_coop_j = {prob_to_coop_j} and discount_factor = {discount_factor}, the target Q-values are:")
        print(f"Q_target_C = {Q_target_C}")
        print(f"Q_target_D = {Q_target_D}")
        print("------------------------------------")
        print()

    return np.array([Q_target_C, Q_target_D])

# CAUTION: This function is written specifically for the Prisoner's Dilemma with S, P, R, T = 0, 1, 3, 5
def calculate_eigenvalues_and_eigenvectors(discount_factor, temperature, learning_rate, print_solution = False):  
    """ 
    CAUTION: This function is written specifically for the Prisoner's Dilemma with S, P, R, T = 0, 1, 3, 5 !
    This function calculates the eigenvalues and eigenvectors of the Jacobi matrix for the deterministic QL model for the Prisoner's Dilemma game.
    The Jacobi matrix is a 4x4 matrix that describes the dynamics of the Q-values for the two actions (cooperate and defect) for two agents.
    The function uses the fixed point policy to calculate the target Q-values to calculate the Jacobi matrix at the fixed point (the target Q-values).
    The function returns the eigenvalues and eigenvectors of the Jacobi matrix.
    The function also prints the eigenvalues and eigenvectors if print_solution is set to True.
    The function uses the fsolve function from the scipy library to solve the system of equations.
    The function uses the numpy library to calculate the eigenvalues and eigenvectors of the Jacobi matrix.

    Args:
        discount_factor (float): The discount factor of the agent.
        temperature (float): The temperature of the agent.
        learning_rate (float): The learning rate of the agent.
        print_solution (bool, optional): Defaults to False. If True, the function prints the eigenvalues and eigenvectors of the Jacobi matrix.
    
    Returns:
        numpy.array: The eigenvalues of the Jacobi matrix.
        numpy.array: The eigenvectors of the Jacobi matrix.
    """
    # calculate the fixed point policy and the target Q-values
    fixed_point_policy = calculate_fixed_point_policy(temperature, discount_factor, print_solution = False)
    Q_target_C, Q_target_D = calculate_target_Q_values(fixed_point_policy, discount_factor, print_solution = False)

    def p(a, b):
        return np.exp(a/temperature) / (np.exp(a/temperature) + np.exp(b/temperature))

    def partial_p(a, b):
        return np.exp((a+b)/temperature) / ( temperature * (np.exp(a/temperature) + np.exp(b/temperature))**2 )

    def f(a, b, c, d):
        return learning_rate * partial_p(a, b) * ( 3 * p(c, d) + discount_factor * b - a)

    def g(a, b, c, d):
        return 3 * learning_rate * p(a,b) * partial_p(c, d)

    def h(a, b, c, d):
        return learning_rate * partial_p(a, b) * ( 4 * p(c, d) + 1 + (discount_factor - 1) * b)

    def i(a, b, c, d):
        return 4 * learning_rate * (1 - p(a,b)) * partial_p(c, d)

    a, b, c, d = Q_target_C, Q_target_D, Q_target_C, Q_target_D

    # define the Jacobi matrix entries
    if True:
        j_11 = + f(a,b,c,d) - learning_rate * p(a, b) + 1
        j_12 = - f(a,b,c,d) + learning_rate * discount_factor * p(a, b)
        j_13 = + g(a,b,c,d)
        j_14 = - g(a,b,c,d)

        j_21 = - h(a,b,c,d)
        j_22 = + h(a,b,c,d) + learning_rate * (discount_factor - 1) * (1 - p(a, b)) + 1
        j_23 = + i(a,b,c,d)
        j_24 = - i(a,b,c,d)

        j_31 = + g(c,d,a,b)
        j_32 = - g(c,d,a,b)
        j_33 = + f(c,d,a,b) - learning_rate * p(c, d) + 1
        j_34 = - f(c,d,a,b) + learning_rate * discount_factor * p(c, d)

        j_41 = + i(c,d,a,b)
        j_42 = - i(c,d,a,b)
        j_43 = - h(c,d,a,b)
        j_44 = + h(c,d,a,b) + learning_rate * (discount_factor - 1) * (1 - p(c, d)) + 1

    # Define the 4x4 Jacobi matrix 
    J = np.array([[j_11, j_12, j_13, j_14],
                [j_21, j_22, j_23, j_24],
                [j_31, j_32, j_33, j_34],
                [j_41, j_42, j_43, j_44]])

    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(J)
    eigenvalues_betrag = np.abs(eigenvalues)

    # Print the results
    if print_solution:
        print(f"gamma = {discount_factor}")
        print()
        for i in range(len(eigenvalues)):
            print(f"Eigenvalue {i+1}:", eigenvalues[i])
        print()
        for i in range(len(eigenvalues_betrag)):
            print(f"Absolute Eigenvalue {i+1}:", eigenvalues_betrag[i])
        print()
    
    return eigenvalues, eigenvectors
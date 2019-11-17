from pacai.agents.learning.reinforcement import ReinforcementAgent
from pacai.util import counter
from pacai.util import reflection
from pacai.util import probability
import random

class QLearningAgent(ReinforcementAgent):
    """
    An agent that explores an MDP following optimal policy with a probability of random movements.
    This agent stores each Q-value as a running average, which is updated each time it makes the
    corresponding action in the correct state.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

        self.qValues = counter.Counter()

    def getAction(self, state):
        if self.getLegalActions(state) == 0:
            return None
        elif probability.flipCoin(self.getEpsilon()):
            return random.choice(self.getLegalActions(state))
        else:
            return self.getPolicy(state)

    def update(self, state, action, nextState, reward):
        """
        Input: A state, action, successor state, and reward (signed int)

        Looks at the difference between the values of the current and successor state, and
        adds it to the running average of the current state's Q-value.

        Output: None

        """
        currentQ = self.getQValue(state, action)
        alpha = self.getAlpha()
        discount = self.getDiscountRate()
        if self.getValue(nextState) is None:
            print("TRUE")
        sample = reward + (discount * self.getValue(nextState))
        self.qValues[state, action] = currentQ + (alpha * (sample - currentQ))

    def getQValue(self, state, action):
        """
        Get the Q-Value for a `pacai.core.gamestate.AbstractGameState`
        and `pacai.core.directions.Directions`.
        Should return 0.0 if the (state, action) pair has never been seen.
        """
        if action is None:
            return 0.0
        return self.qValues[state, action]

    def getValue(self, state):
        """
        Return the value of the best action in a state.
        I.E., the value of the action that solves: `max_action Q(state, action)`.
        Where the max is over legal actions.
        Note that if there are no legal actions, which is the case at the terminal state,
        you should return a value of 0.0.

        This method pairs with `QLearningAgent.getPolicy`,
        which returns the actual best action.
        Whereas this method returns the value of the best action.
        """
        if len(self.getLegalActions(state)) == 0:
            return 0.0
        value = float("-inf")
        for a in self.getLegalActions(state):
            qVal = self.getQValue(state, a)
            value = max(value, qVal)
        return value

    def getPolicy(self, state):
        """
        Return the best action in a state.
        I.E., the action that solves: `max_action Q(state, action)`.
        Where the max is over legal actions.
        Note that if there are no legal actions, which is the case at the terminal state,
        you should return a value of None.

        This method pairs with `QLearningAgent.getValue`,
        which returns the value of the best action.
        Whereas this method returns the best action itself.
        """
        maxVal = float("-inf")
        bestAction = None
        for a in self.getLegalActions(state):
            qValue = self.getQValue(state, a)
            if maxVal == qValue:
                bestAction = random.choice([bestAction, a])
            elif maxVal < qValue:
                bestAction = a
                maxVal = qValue

        return bestAction

class PacmanQAgent(QLearningAgent):
    """
    Exactly the same as `QLearningAgent`, but with different default parameters.
    """

    def __init__(self, index, epsilon = 0.05, gamma = 0.8, alpha = 0.2, numTraining = 0, **kwargs):
        kwargs['epsilon'] = epsilon
        kwargs['gamma'] = gamma
        kwargs['alpha'] = alpha
        kwargs['numTraining'] = numTraining

        super().__init__(index, **kwargs)

    def getAction(self, state):
        """
        Simply calls the super getAction method and then informs the parent of an action for Pacman.
        Do not change or remove this method.
        """

        action = super().getAction(state)
        self.doAction(state, action)

        return action

class ApproximateQAgent(PacmanQAgent):
    """
    This agent is much like the previous Q-learning agents, except instead of storing and updating
    q-values, it stores and updates weights that correspond to state features.
    Essentially, it creates a better state evaluation function as it learns.
    """

    def __init__(self, index,
            extractor = 'pacai.core.featureExtractors.IdentityExtractor', **kwargs):
        super().__init__(index, **kwargs)
        self.featExtractor = reflection.qualifiedImport(extractor)()
        self.weights = counter.Counter()

    def getQValue(self, state, action):
        """
        Input: A state and action

        Creates a feature vector from the state and calculates a Q value by summing
        the weighted features.

        Output: A Q-value (signed int)

        """
        featureCounter = self.featExtractor.getFeatures(state, action)
        features = featureCounter.sortedKeys()
        qValue = 0
        for f in features:
            qValue += self.weights[f] * featureCounter[f]
        return qValue

    def update(self, state, action, nextState, reward):
        """
        Input: A state, action, successor state, and reward (signed int)

        Looks at the difference between the values of the current and successor state, multiplies
        it to each successor state feature value and adds the total to the running average of
        each weight.

        Output: None

        """
        featureCounter = self.featExtractor.getFeatures(state, action)
        features = featureCounter.sortedKeys()
        discount = self.getDiscountRate()
        alpha = self.getAlpha()
        nextValue = self.getValue(nextState)
        currentQ = self.getQValue(state, action)
        sample = (reward + discount * nextValue) - currentQ
        for f in features:
            self.weights[f] = self.weights[f] + alpha * (sample) * featureCounter[f]

    def final(self, state):
        """
        Called at the end of each game.
        """

        # Call the super-class final method.
        super().final(state)
        """
        # Did we finish training?
        if self.episodesSoFar == self.numTraining:
            # You might want to print your weights here for debugging.
        """

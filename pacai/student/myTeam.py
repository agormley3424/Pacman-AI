from pacai.util import reflection
from pacai.agents.capture.capture import CaptureAgent
from pacai.util import counter
from pacai.util import probability
import random

class LearningAgent(CaptureAgent):
    def __init__(self, index):
        super().__init__(index, 1)

        self.timeLimit = 1
        self.index = index
        self.alpha = 0 #Learning rate
        self.epsilon = 0 #Random exploration probability
        self.discount = 0 #Discounted reward rate, ???
        self.weights = counter.Counter()

    def extractFeatures(self, state):
        featureCounter = counter.Counter()
        featureCounter['selfPosition'] = state.getAgentPosition(self.index)

    def getLegalActions(self, state):
        return state.getLegalActions(state, self.index)

    def chooseAction(self, state):
        if self.getLegalActions(state) == 0:
            return None
        elif probability.flipCoin(self.epsilon):
            return random.choice(self.getLegalActions(state))
        else:
            return self.getPolicy(state)

    def getQValue(self, state, action):
        """
        Input: A state and action

        Creates a feature vector from the state and calculates a Q value by summing
        the weighted features.

        Output: A Q-value (signed int)

        """
        featureCounter = self.extractFeatures(state)
        features = featureCounter.sortedKeys()
        qValue = 0
        for f in featureCounter:
            qValue += self.weights[f] * featureCounter[f]
        return qValue

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
        qValue = self.getValue(state, a)
        if maxVal == qValue:
            bestAction = random.choice([bestAction, a])
        elif maxVal < qValue:
            bestAction = a
            maxVal = qValue
        for a in self.getLegalActions(state):
            qValue = self.getQValue(state, a)
            if maxVal == qValue:
                bestAction = random.choice([bestAction, a])
            else:
                bestAction = a
                maxVal = qValue

        return bestAction

    def update(self, state, action, nextState, reward):
        """
        Input: A state, action, successor state, and reward (signed int)

        Looks at the difference between the values of the current and successor state, multiplies
        it to each successor state feature value and adds the total to the running average of
        each weight.

        Output: None

        """
        featureCounter = self.extractFeatures(state)
        features = featureCounter.sortedKeys()
        nextValue = self.getValue(nextState)
        currentQ = self.getQValue(state, action)
        sample = (reward + self.discount * nextValue) - currentQ
        for f in features:
            self.weights[f] = self.weights[f] + self.alpha * (sample) * featureCounter[f]

def createTeam(firstIndex, secondIndex, isRed,
        first = 'pacai.student.myTeam.LearningAgent',
        second = 'pacai.student.myTeam.LearningAgent'):
    """
    This function should return a list of two agents that will form the capture team,
    initialized using firstIndex and secondIndex as their agent indexed.
    isRed is True if the red team is being created,
    and will be False if the blue team is being created.
    """

    firstAgent = reflection.qualifiedImport(first)
    secondAgent = reflection.qualifiedImport(second)

    return [
        firstAgent(firstIndex),
        secondAgent(secondIndex),
    ]

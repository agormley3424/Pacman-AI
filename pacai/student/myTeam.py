from pacai.util import reflection
from pacai.agents.capture.capture import CaptureAgent
from pacai.util import counter
from pacai.util import probability
import random

#States are CaptureGameStates

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
        """
        Input: A CaptureGameState

        Returns a counter of features and their corresponding values for the given state.
        
        Output: featureCounter (Counter)
        """
        featureCounter = counter.Counter()
        featureCounter['nullFeature'] = 1

        return featureCounter

    def getLegalActions(self, state):
        """
        Input: A CaptureGameState

        Returns a list containing all legal actions possible from this state for this agent.
        
        Output: A list of actions (Strings)
        """
        return state.getLegalActions(self.index)

    def chooseAction(self, state):
        if self.getLegalActions(state) == 0:
            return None
        elif probability.flipCoin(self.epsilon):
            return random.choice(self.getLegalActions(state))
        else:
            return self.getPolicy(state)

    def getQValue(self, state, action):
        """
        Input: A CaptureGameState and action (String)

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
        Input: A CaptureGameState

        Looks through all legal actions for a given state and finds that which corresponds to the
        highest Q-value, then returns that value.

        Returns 0 if there are no legal actions.

        Output: State value (signed int)
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
        Input: A CaptureGameState

        Look through all legal actions for a given state and finds that which corresponds to the
        highest Q-value, then returns that action.

        Returns None if ther are no legal actions.

        Output: An action (String) or None
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

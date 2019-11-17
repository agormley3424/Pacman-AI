from pacai.agents.learning.value import ValueEstimationAgent
from pacai.util import counter

class ValueIterationAgent(ValueEstimationAgent):
    """
    This agent runs value iteration over an MDP a number of times specified by the calling function.
    It can also calculate Q values and optimal policies.
    """

    def __init__(self, index, mdp, discountRate = 0.9, iters = 100, **kwargs):
        """
        Input: Agent index (unsigned int), MarkovDecisionProcess (object),
        number of iterations (unsigned int), unspecified keyword arguments

        For each state, finds the highest Q-value corresponding to the optimal policy action
        and sets value to it.

        Output: None
        """
        super().__init__(index)

        self.mdp = mdp
        self.discountRate = discountRate
        self.iters = iters
        self.values = counter.Counter()  # A Counter is a dict with default 0

        # Compute the values here.
        for i in range(iters):
            newValues = counter.Counter()
            for s in mdp.getStates():
                policy = self.getPolicy(s)
                newValues[s] = float("-inf")
                newValues[s] = max(self.getQValue(s, policy), newValues[s])
            self.values = newValues

    def getQValue(self, state, action):
        """
        Input: An MDP state (object), a legal action (string)

        Calculates a Q-value for the appropriate state-action pair.

        Output: Q-value (floating point)
        """
        if action is None:
            return 0.0
        qValue = 0.0
        tranStates = self.mdp.getTransitionStatesAndProbs(state, action)
        for t in tranStates:
            newState = t[0]
            transProb = t[1]
            reward = self.mdp.getReward(state, action, newState)
            qValue += transProb * (reward + self.discountRate * self.getValue(newState))

        return qValue

    def getPolicy(self, state):
        """
        Input: An MDP state (object)

        Finds the action that corresponds to the state's highest Q value and returns it

        Output: An action (String)
        """
        bestAction = None
        maxQ = float("-inf")
        for a in self.mdp.getPossibleActions(state):
            qVal = self.getQValue(state, a)
            if qVal > maxQ:
                maxQ = qVal
                bestAction = a

        return bestAction

    def getValue(self, state):
        """
        Return the value of the state (computed in __init__).
        """

        return self.values[state]

    def getAction(self, state):
        """
        Returns the policy at the state (no exploration).
        """

        return self.getPolicy(state)

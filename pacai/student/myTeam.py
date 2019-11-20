from pacai.util import reflection
from pacai.agents.capture.capture import CaptureAgent
from pacai.util import counter
from pacai.util import probability
import random

#States are CaptureGameStates
#Ask about training
#Ask about features from states vs. features from states and actions
#Make update actually get called
#Have learning and playing mode flags
#python3 -m pacai.bin.capture --red pacai.student.myTeam --blue pacai.core.baselineTeam 
#reinforcement agent calls update from one of its own functions

#final is being called twice at the end, once for each agent

#Removed **args from the createTeam function in capture.py loadAgents
#Changed numGames > 0 to numGames - numTraining > 0 in capture.py runGames

class LearningAgent(CaptureAgent):
    def __init__(self, index):
        super().__init__(index)

        self.timeLimit = 1
        self.index = index
        self.alpha = 0.01 #Learning rate
        self.epsilon = 0 #Random exploration probability
        self.discount = 0.9 #Discounted reward rate, ???
        self.weights = counter.Counter()
        self.visitedStates = counter.Counter()
        self.strategy = 'offensive'
        self.features = ['minDistanceToEnemyFood',
                         'minDistanceToEnemyCapsules',
                         'successorScore',
                         'minDistanceToEnemyScared',
                         'stateRedundancy',
                         'minDistanceToFriendlyCapsules',
                         'minDistanceToEnemyBrave',
                         'minDistanceToEnemyPac']
        self.weights['minDistanceToEnemyFood'] = -1
        self.weights['successorReward'] = 100

    def getOpponentPositions(self, state):
        scaredEnemies = []
        braveEnemies = []
        enemyPacPositions = []
        for o in self.getOpponents(state):
            enemy = state.getAgentState(o)
            if enemy.isGhost():
                if enemy.isScared():
                    scaredEnemies.append(enemy.getPosition())
                else:
                    braveEnemies.append(enemy.getPosition())
            else:
                enemyPacPositions.append(enemy.getPosition())

        return scaredEnemies, braveEnemies, enemyPacPositions

    def getAgentPosition(self, state):
        return state.getAgentPosition(self.index)

    def getFriendPosition(self, state):
        friendIndex = None
        for i in self.getTeam(state):
            if i != self.index:
                friendIndex = i
        return state.getAgentPosition(friendIndex)

    def extractFeatures(self, state, action):
        """
        Input: A CaptureGameState

        Returns a counter of features and their corresponding values for the given state.
        
        Output: featureCounter (Counter)
        """
        featureCounter = counter.Counter()
        if not state.isOver():
            newState = state.generateSuccessor(self.index, action)
            walls = state.getWalls()
            area = walls.getWidth() * walls.getHeight()
            enemyFoodList = self.getFood(state).asList()
            enemyCapsules = self.getCapsules(state)
            friendlyFood = self.getFoodYouAreDefending(newState).asList()
            friendlyCapsules = self.getCapsulesYouAreDefending(newState)
            agentPos = self.getAgentPosition(state)
            ghostTuple = self.getOpponentPositions(state)
            scaredEnemies = ghostTuple[0]
            braveEnemies = ghostTuple[1]
            enemyPacPositions = ghostTuple[2]
            thisAgentState = newState.getAgentState(self.index)

            # Offensive Features
            if len(enemyFoodList) > 0:
                featureCounter['minDistanceToEnemyFood'] = self.minDistance(enemyFoodList, agentPos) / area
            if len(enemyCapsules) > 0:
                featureCounter['minDistanceToEnemyCapsules'] = self.minDistance(enemyCapsules, agentPos) / area
            if len(scaredEnemies) > 0:
                featureCounter['minDistanceToEnemyScared'] = self.minDistance(scaredEnemies, agentPos) / area

            featureCounter['successorReward'] = self.getReward(state, newState)
            featureCounter['stateRedundancy'] = self.visitedStates[newState]
            """
            featureCounter['onEnemySide'] = 1 if thisAgentState.isPacman() else 0
            
            # Defensive Features
            if len(friendlyCapsules) > 0:
                featureCounter['minDistanceToFriendlyCapsules'] = self.minDistance(friendlyCapsules, agentPos) / area
            if len(friendlyFood) > 0:
                featureCounter['minDistanceToFriendlyFood'] = self.minDistance(friendlyFood, agentPos) / area
            if len(enemyPacPositions) > 0:
                featureCounter['minDistanceToEnemyPacmen'] = self.minDistance(enemyPacPositions, agentPos) / area

            # Offensive and Defensive Features
            if len(braveEnemies) > 0:
                featureCounter['minDistanceToEnemyBrave'] = self.minDistance(braveEnemies, agentPos) / area
            """

        return featureCounter

    def minDistance(self, positionList, originPoint):
        minDist = float("inf")
        for p in positionList:
            minDist = min(minDist, self.getMazeDistance(originPoint, p))
        return minDist

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
            action = random.choice(self.getLegalActions(state))
        else:
            action = self.getPolicy(state)
        nextState = state.generateSuccessor(self.index, action)
        reward = self.getReward(state, nextState)
        self.update(state, action, nextState, reward)
        return action

    def getReward(self, oldState, newState):
        newScore = self.getScore(newState)
        oldScore = self.getScore(oldState)

        agentState = newState.getAgentState(self.index)
        agentPosition = self.getAgentPosition(newState)
        ghostTuple = self.getOpponentPositions(newState)
        scaredEnemies = ghostTuple[0]
        braveEnemies = ghostTuple[1]
        enemyPacPositions = ghostTuple[2]
        combatValue = 0
        if agentState.isGhost():
            if agentState.isScared():
                if agentPosition in enemyPacPositions:
                    combatValue = -5
            else:
                if agentPosition in enemyPacPositions:
                    combatValue = 3
        else:
            if agentPosition in scaredEnemies:
                combatValue = 3
            elif agentPosition in braveEnemies:
                combatValue = -5

        #reward = newScore - oldScore + combatValue - newState.getTimeleft()
        reward = newScore - oldScore
        return reward

    def getQValue(self, state, action):
        """
        Input: A CaptureGameState and action (String)

        Creates a feature vector from the state and calculates a Q value by summing
        the weighted features.

        Output: A Q-value (signed int)

        """
        featureCounter = self.extractFeatures(state, action)
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
        #The discount makes a positive number smaller, but makes a negative number larger (resolved by setting discount to 1/discount if nextValue is negative)
        #Weight for minimum distance is becoming a gigantic negative number, overshadowing all other features
        featureCounter = self.extractFeatures(state, action)
        features = featureCounter.sortedKeys()
        nextValue = self.getValue(nextState)
        if nextValue < 0:
            discount = 1 / self.discount
        else:
            discount = self.discount
        currentQ = self.getQValue(state, action)
        sample = (reward + discount * nextValue) - currentQ
        for f in features:
            self.weights[f] = self.weights[f] + self.alpha * (sample) * featureCounter[f]
        self.visitedStates[state] += 1

    def final(self, gameState):
        for f in self.features:
            print(f + ' ' + str(self.weights[f]))

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

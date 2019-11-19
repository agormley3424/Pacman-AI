from pacai.util import reflection
from pacai.agents.capture.reflex import ReflexCaptureAgent
from pacai.core.directions import Directions
from pacai.util import counter
from pacai.util import probability
import random

#States are CaptureGameStates

class StrategyAgentA(ReflexCaptureAgent):
    """
    A reflex agent that seeks food.
    This agent will give you an idea of what an offensive agent might look like,
    but it is by no means the best or only way to build an offensive agent.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index)

        self.offensive = 1
        self.defensive = 0

    def getDefensive(self):
        return self.defensive

    def updateStrategy(self, gameState, action):
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        agents = [gameState.getAgentState(i) for i in self.getTeam(gameState)]

        agentPos = [agents[0].getPosition(), agents[1].getPosition()]

        invaders = [a for a in enemies if a.isPacman() and a.getPosition() is not None]
        defenders = [a for a in agents if not a.isPacman() and a.getPosition() is not None]

        # best-case scenario
        if (len(invaders) is 0):
            self.offensive = 1
            self.defensive = 0

        # worse scenario
        else:
            self.offensive = 0
            self.defensive = 1


        # print(enemyPos, agentPos)

    def getFeatures(self, gameState, action):
        self.updateStrategy(gameState, action)

        # OFFENSE FEATURES
        features = counter.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)
        
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Compute distance to the nearest food.
        foodList = self.getFood(successor).asList()

        # This should always be True, but better safe than sorry.
        if (len(foodList) > 0):
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = minDistance

        agents = [gameState.getAgentState(i) for i in self.getTeam(gameState)]
        agentPos = [agents[0].getPosition(), agents[1].getPosition()]
        # features['tooClose'] = 0

        # if (abs(agentPos[0][0] - agentPos[1][0]) + abs(agentPos[0][1] - agentPos[1][1]) < 3):
        #    features['tooClose'] = 1


        # DEFENSE FEATURES

        # Computes whether we're on defense (1) or offense (0).
        features['onDefense'] = 1
        if (myState.isPacman()):
            features['onDefense'] = 0

        # Computes distance to invaders we can see.
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman() and a.getPosition() is not None]
        features['numInvaders'] = len(invaders)

        if (not myState.isPacman()):
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in enemies]
            features['enemyDistance'] = min(dists)

        else:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in enemies]
            features['danger'] = 1/min(dists)

        if (len(invaders) > 0):
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        if (action == Directions.STOP):
            features['stop'] = 1

        rev = Directions.REVERSE[gameState.getAgentState(self.index).getDirection()]
        if (action == rev):
            features['reverse'] = 1

        return features


    def getWeights(self, gameState, action):
        ourWeights = {
            # 'tooClose': -2,
        }
        
        offensiveWeights = {
            'successorScore': 100,
            'distanceToFood': -1,
            'danger': -5,
        }
        
        defensiveWeights = {
            'numInvaders': -1000,
            'onDefense': 100,
            'invaderDistance': -10,
            'enemyDistance': -2,
            'stop': -100,
            'reverse': -2
        }

        # ourWeights = (offensiveWeights * self.offensive) + (defensiveWeights * self.defensive)
        
        for key in offensiveWeights.keys():
            feature = offensiveWeights[key]

            ourWeights[key] = feature * self.offensive;

        for key in defensiveWeights.keys():
            feature = defensiveWeights[key]

            ourWeights[key] = feature * self.defensive;

        return ourWeights


class StrategyAgentB(ReflexCaptureAgent):
    """
    A reflex agent that seeks food.
    This agent will give you an idea of what an offensive agent might look like,
    but it is by no means the best or only way to build an offensive agent.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index)

        self.offensive = 1
        self.defensive = 0

    def getDefensive(self):
        return self.defensive

    def updateStrategy(self, gameState, action):
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        agents = [gameState.getAgentState(i) for i in self.getTeam(gameState)]

        agentPos = [agents[0].getPosition(), agents[1].getPosition()]

        invaders = [a for a in enemies if a.isPacman() and a.getPosition() is not None]
        defenders = [a for a in agents if not a.isPacman() and a.getPosition() is not None]

        # best-case scenario
        if (len(invaders) is 0):
            self.offensive = 1
            self.defensive = 0

        # average scenario
        if (len(invaders) is 1):
            self.offensive = 0
            self.defensive = 1

        # worst-case scenario
        else:
            self.offensive = 0
            self.defensive = 1


        # print(enemyPos, agentPos)

    def getFeatures(self, gameState, action):
        self.updateStrategy(gameState, action)

        # OFFENSE FEATURES
        features = counter.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Compute distance to the nearest food.
        foodList = self.getFood(successor).asList()

        # This should always be True, but better safe than sorry.
        if (len(foodList) > 0):
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = minDistance

        agents = [gameState.getAgentState(i) for i in self.getTeam(gameState)]
        agentPos = [agents[0].getPosition(), agents[1].getPosition()]
        # features['tooClose'] = 0

        # if (abs(agentPos[0][0] - agentPos[1][0]) + abs(agentPos[0][1] - agentPos[1][1]) < 3):
        #     features['tooClose'] = 1


        # DEFENSE FEATURES

        # Computes whether we're on defense (1) or offense (0).
        features['onDefense'] = 1
        if (myState.isPacman()):
            features['onDefense'] = 0

        # Computes distance to invaders we can see.
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman() and a.getPosition() is not None]
        features['numInvaders'] = len(invaders)

        if (not myState.isPacman()):
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in enemies]
            features['enemyDistance'] = min(dists)

        else:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in enemies]
            features['danger'] = 1/min(dists)

        if (len(invaders) > 0):
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        if (action == Directions.STOP):
            features['stop'] = 1

        rev = Directions.REVERSE[gameState.getAgentState(self.index).getDirection()]
        if (action == rev):
            features['reverse'] = 1

        return features


    def getWeights(self, gameState, action):
        ourWeights = {
            # 'tooClose': -2,
        }
        
        offensiveWeights = {
            'successorScore': 100,
            'distanceToFood': -1,
            'danger': -5,
        }
        
        defensiveWeights = {
            'numInvaders': -1000,
            'onDefense': 100,
            'invaderDistance': -10,
            'enemyDistance': -2,
            'stop': -100,
            'reverse': -2
        }

        # ourWeights = (offensiveWeights * self.offensive) + (defensiveWeights * self.defensive)
        
        for key in offensiveWeights.keys():
            feature = offensiveWeights[key]

            ourWeights[key] = feature * self.offensive;

        for key in defensiveWeights.keys():
            feature = defensiveWeights[key]

            ourWeights[key] = feature * self.defensive;

        return ourWeights


def createTeam(firstIndex, secondIndex, isRed,
        first = 'pacai.student.myTeam.StrategyAgentA',
        second = 'pacai.student.myTeam.StrategyAgentB'):
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

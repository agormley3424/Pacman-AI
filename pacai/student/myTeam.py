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
        
        self.statesReached = {}
        
        self.offenseDetector = [1, 1]

    def getDefensive(self):
        return self.defensive

    def updateStrategy(self, gameState):
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

    def updateExpectation(self, gameState):
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]

        index = 0
        offenseLerp = 0.1
        defenseLerp = 0.99

        for a in enemies:
            if a.isPacman():
                lastValue = self.offenseDetector[index]
                newValue = (lastValue * offenseLerp) + (1 * (1 - offenseLerp))
                self.offenseDetector[index] = newValue

            else:
                lastValue = self.offenseDetector[index]
                newValue = (lastValue * defenseLerp) + (0 * (1 - defenseLerp))
                self.offenseDetector[index] = newValue

            index += 1


    def minHorizDist(self, myPos, enemyPos):
        minDist = 999999

        for ePos in enemyPos:
            dist = abs(myPos[0] - ePos[0])

            if (dist < minDist):
                minDist = dist

        return minDist

    def getFeatures(self, gameState, action):
        self.updateStrategy(gameState)
        self.updateExpectation(gameState)

        # OFFENSE FEATURES
        features = counter.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)
        
        myState = successor.getAgentState(self.index)
        thisPos = myState.getPosition()
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        
        if (myState.getPosition() not in self.statesReached):
            self.statesReached.update({thisPos : 1})

        else:
            lastValue = self.statesReached[thisPos]
            self.statesReached.update({thisPos : lastValue + 1})

        features['repeatState'] = self.statesReached[thisPos]

        # Compute distance to the nearest food.
        foodList = self.getFood(successor).asList()

        # This should always be True, but better safe than sorry.
        if (len(foodList) > 0):
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            """
            minDistance = 999999
            for food in foodList:
                pathDist = self.getMazeDistance(myPos, food)
                
                index = 0
                tooShort = False
                for a in enemies:
                    if not a.isPacman() and a.isBraveGhost():
                        enemyDist = self.getMazeDistance(myPos, a.getPosition())

                        if (enemyDist <= pathDist * 2):
                            tooShort = True

                    index += 1

                if not tooShort:
                    minDistance = min(minDistance, pathDist)
            """
            features['distanceToFood'] = minDistance

            sumFoodX = 0
            sumFoodY = 0
            for food in foodList:
                sumFoodX += food[0]
                sumFoodY += food[1]

            averageFood = [sumFoodX/len(foodList), sumFoodY/len(foodList)]
            
            features['distToAvgFood'] = (abs(myPos[0] - averageFood[0]) + abs(myPos[1] - averageFood[1])) ** 1.2
            # print(features['distToAvgFood'])

        capsuleList = self.getCapsules(successor)
        
        if (len(capsuleList) > 0):
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, capsule) for capsule in capsuleList])
            """
            minDistance = 999999
            for capsule in capsuleList:
                pathDist = self.getMazeDistance(myPos, capsule)
                
                index = 0
                tooShort = False
                for a in enemies:
                    if not a.isPacman() and a.isBraveGhost():
                        enemyDist = self.getMazeDistance(myPos, a.getPosition())

                        if (enemyDist <= pathDist * 2):
                            tooShort = True

                    index += 1

                if not tooShort:
                    minDistance = min(minDistance, pathDist)
            """
            features['distanceToCapsule'] = minDistance
            
            features['onCapsule'] = 0
            if (len(capsuleList) < len(self.getCapsules(gameState))):
                features['onCapsule'] = 1

        agents = [gameState.getAgentState(i) for i in self.getTeam(gameState)]
        agentPos = [agents[0].getPosition(), agents[1].getPosition()]


        # DEFENSE FEATURES
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0).
        features['onDefense'] = 1
        if (myState.isPacman()):
            features['onDefense'] = 0

        # Computes distance to invaders we can see.
        
        invaders = [a for a in enemies if a.isPacman() and a.getPosition() is not None]
        features['numInvaders'] = len(invaders)

        if (not myState.isPacman()):
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in enemies]
            features['enemyDistance'] = min(dists)

            enemyPos = [a.getPosition() for a in enemies]
            features['horizEnemyDistance'] = self.minHorizDist(myPos, enemyPos)

        else:
            brave = []
            scared = []
            for a in enemies:
                if a.isBraveGhost():
                    brave.append(a)
                
                else:
                    scared.append(a)
            
            features['danger'] = 0
            features['distToScared'] = 0
            
            if (len(brave) > 0):
                dists = [self.getMazeDistance(myPos, a.getPosition()) for a in brave]
                features['danger'] = 1/min(dists)

            if (len(scared) > 0):
                smallestDist = 999999
                timer = 0
                
                for a in scared:
                    dist = self.getMazeDistance(myPos, a.getPosition())
                    
                    if (dist < smallestDist):
                        smallestDist = dist
                        timer = a.getScaredTimer()
                
                features['distToScared'] = timer/smallestDist
            
            currentEnemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
            scaredNum = 0
            
            for a in currentEnemies:
                if not a.isBraveGhost():
                    scaredNum += 1
            
            features['eatenGhost'] = 0
            if (len(scared) < scaredNum):
                features['eatenGhost'] = 1

        if (len(invaders) > 0):
            if myState.isBraveGhost():
                dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
                features['invaderDistance'] = min(dists)
                features['runaway'] = 0

            else:
                dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
                features['runAway'] = min(dists)
                features['invaderDistance'] = 0

        if (action == Directions.STOP):
            features['stop'] = 1

        rev = Directions.REVERSE[gameState.getAgentState(self.index).getDirection()]
        if (action == rev):
            features['reverse'] = 1

        ourFood = self.getFoodYouAreDefending(successor).asList()

        closestFoodPos = None
        shortestFoodDist = 999999
        index = 0

        for a in enemies:
            if self.offenseDetector[index] > 0.1:
                for food in ourFood:
                    foodDist = self.getMazeDistance(food, a.getPosition())

                    if foodDist < shortestFoodDist:
                        closestFoodPos = food
                        shortestFoodDist = foodDist

            index += 1

        features['targetedFoodDist'] = 0

        if (closestFoodPos is not None):
            targetedDist = self.getMazeDistance(myPos, closestFoodPos)
            features['targetedFoodDist'] = targetedDist


        return features


    def getWeights(self, gameState, action):
        ourWeights = {}

        offensiveWeights = {
            'successorScore': 100,
            'distanceToFood': -2,
            'distanceToCapsule': -1,
            'danger': -7,
            'distToAvgFood': -0.2,
            'onCapsule': 10000,
            'repeatState': -0.2,
            'distToScared': 0.3,
            'eatenGhost': 10000,
        }

        defensiveWeights = {
            'numInvaders': -1000,
            'onDefense': 100,
            'invaderDistance': -10,
            'runAway': 10,
            'enemyDistance': -2,
            'horizEnemyDistance': 0,
            'targetedFoodDist': -3,
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

        self.statesReached = {}

        self.offenseDetector = [1, 1]

    def getDefensive(self):
        return self.defensive

    def updateStrategy(self, gameState):
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

    def updateExpectation(self, gameState):
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]

        index = 0
        offenseLerp = 0.1
        defenseLerp = 0.99

        for a in enemies:
            if a.isPacman():
                lastValue = self.offenseDetector[index]
                newValue = (lastValue * offenseLerp) + (1 * (1 - offenseLerp))
                self.offenseDetector[index] = newValue

            else:
                lastValue = self.offenseDetector[index]
                newValue = (lastValue * defenseLerp) + (0 * (1 - defenseLerp))
                self.offenseDetector[index] = newValue

            index += 1


    def minHorizDist(self, myPos, enemyPos):
        minDist = 999999

        for ePos in enemyPos:
            if (self.red):
                dist = 30 - ePos[0]
            
            else:
                dist = ePos[0]

            if (dist < minDist):
                minDist = dist

        return minDist

    def getFeatures(self, gameState, action):
        self.updateStrategy(gameState)
        self.updateExpectation(gameState)
        # print(self.offenseDetector)

        # OFFENSE FEATURES
        features = counter.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)

        myState = successor.getAgentState(self.index)
        thisPos = myState.getPosition()
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        
        if (myState.getPosition() not in self.statesReached):
            self.statesReached.update({thisPos : 1})

        else:
            lastValue = self.statesReached[thisPos]
            self.statesReached.update({thisPos : lastValue + 1})

        features['repeatState'] = self.statesReached[thisPos]

        # Compute distance to the nearest food.
        foodList = self.getFood(successor).asList()

        # This should always be True, but better safe than sorry.
        if (len(foodList) > 0):
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            """
            minDistance = 999999
            for food in foodList:
                pathDist = self.getMazeDistance(myPos, food)
                
                index = 0
                tooShort = False
                for a in enemies:
                    if self.offenseDetector[index] < 0.2:
                        enemyDist = self.getMazeDistance(myPos, a.getPosition())

                        if (enemyDist <= pathDist * 2):
                            tooShort = True

                    index += 1

                if not tooShort:
                    minDistance = min(minDistance, pathDist)
            """
            features['distanceToFood'] = minDistance

            sumFoodX = 0
            sumFoodY = 0
            for food in foodList:
                sumFoodX += food[0]
                sumFoodY += food[1]

            averageFood = [sumFoodX/len(foodList), sumFoodY/len(foodList)]
            
            features['distToAvgFood'] = (abs(myPos[0] - averageFood[0]) + abs(myPos[1] - averageFood[1])) ** 1.2
            # print(features['distToAvgFood'])

        capsuleList = self.getCapsules(successor)
        
        if (len(capsuleList) > 0):
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, capsule) for capsule in capsuleList])
            """
            minDistance = 999999
            for capsule in capsuleList:
                pathDist = self.getMazeDistance(myPos, capsule)
                
                index = 0
                tooShort = False
                for a in enemies:
                    if self.offenseDetector[index] < 0.2:
                        enemyDist = self.getMazeDistance(myPos, a.getPosition())

                        if (enemyDist <= pathDist * 2):
                            tooShort = True

                    index += 1

                if not tooShort:
                    minDistance = min(minDistance, pathDist)
            """
            features['distanceToCapsule'] = minDistance
            
            features['onCapsule'] = 0
            if (len(capsuleList) < len(self.getCapsules(gameState))):
                features['onCapsule'] = 1

        agents = [gameState.getAgentState(i) for i in self.getTeam(gameState)]
        agentPos = [agents[0].getPosition(), agents[1].getPosition()]



        # DEFENSE FEATURES
        myPos = myState.getPosition()

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

            enemyPos = [a.getPosition() for a in enemies]
            features['horizEnemyDistance'] = self.minHorizDist(myPos, enemyPos)

        else:
            brave = []
            scared = []
            for a in enemies:
                if a.isBraveGhost():
                    brave.append(a)
                
                else:
                    scared.append(a)
            
            features['danger'] = 0
            features['distToScared'] = 0
            
            if (len(brave) > 0):
                dists = [self.getMazeDistance(myPos, a.getPosition()) for a in brave]
                features['danger'] = 1/min(dists)

            if (len(scared) > 0):
                smallestDist = 999999
                timer = 0
                
                for a in scared:
                    dist = self.getMazeDistance(myPos, a.getPosition())
                    
                    if (dist < smallestDist):
                        smallestDist = dist
                        timer = a.getScaredTimer()
                
                features['distToScared'] = timer/smallestDist
            
            currentEnemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
            scaredNum = 0
            
            for a in currentEnemies:
                if not a.isBraveGhost():
                    scaredNum += 1
            
            features['eatenGhost'] = 0
            if (len(scared) < scaredNum):
                features['eatenGhost'] = 1


        if (len(invaders) > 0):
            if myState.isBraveGhost():
                dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
                features['invaderDistance'] = min(dists)
                features['runaway'] = 0

            else:
                dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
                features['runAway'] = min(dists)
                features['invaderDistance'] = 0

        if (action == Directions.STOP):
            features['stop'] = 1

        rev = Directions.REVERSE[gameState.getAgentState(self.index).getDirection()]
        if (action == rev):
            features['reverse'] = 1


        ourFood = self.getFoodYouAreDefending(successor).asList()

        closestFoodPos = None
        shortestFoodDist = 999999
        index = 0

        for a in enemies:
            if self.offenseDetector[index] > 0.1:
                for food in ourFood:
                    foodDist = self.getMazeDistance(food, a.getPosition())

                    if foodDist < shortestFoodDist:
                        closestFoodPos = food
                        shortestFoodDist = foodDist

            index += 1

        features['targetedFoodDist'] = 0

        if (closestFoodPos is not None):
            targetedDist = self.getMazeDistance(myPos, closestFoodPos)
            features['targetedFoodDist'] = targetedDist


        return features


    def getWeights(self, gameState, action):
        ourWeights = {}

        offensiveWeights = {
            'successorScore': 100,
            'distanceToFood': -2,
            'distanceToCapsule': -1,
            'danger': -7,
            'distToAvgFood': -0.2,
            'onCapsule': 10000,
            'repeatState': -0.2,
            'distToScared': 0.3,
            'eatenGhost': 10000,
        }
        
        defensiveWeights = {
            'numInvaders': -1000,
            'onDefense': 100,
            'invaderDistance': -10,
            'runAway': 10,
            'enemyDistance': -2,
            'horizEnemyDistance': 0,
            'targetedFoodDist': -3,
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

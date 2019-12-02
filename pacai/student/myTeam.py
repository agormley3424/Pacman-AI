# from pacai.util import reflection
from pacai.agents.capture.reflex import ReflexCaptureAgent
from pacai.core.directions import Directions
from pacai.util import counter
# from pacai.util import probability
# import random

# States are CaptureGameStates

class HybridAgent(ReflexCaptureAgent):
    """
    This is an offense agent.

    It will seek out as many food pellets as it can while attempting to avoid ghosts
    within a certain threshold. It will also seek out conveniently-placed power pellets
    and make an attempt to eat scared ghosts.

    The agent is both greedy and will attempt to keep a good distance away from ghosts.

    Fatal flaw: This agent can be deadlocked on very specific maps, and it does not
    use minimax search, so it is prone to making poorly-informed movements in tight spots.
    """

    """
    Features to add:

    Discourage agents from grouping up (Reduce vertical distance?)
    Discourage agents from approaching enemy pacmen while scared
    Play safer after one agent is killed (check distance of other agent to border)
    """

    def __init__(self, index, defaultOffense, cautionThreshold, **kwargs):
        super().__init__(index)

        # This can help the agent detect the potential behavior of the opposing agents.
        self.offenseDetector = [1, 1]
        self.defaultOffense = defaultOffense
        self.offense = defaultOffense
        self.cautionThreshold = cautionThreshold

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights.
        """
        self.offense = self.evaluateStrategy(gameState)
        if self.offense:
            features = self.offenseFeatures(gameState, action)
        else:
            features = self.defenseFeatures(gameState, action)
        weights = self.getWeights(gameState, action)

        return features * weights

    def evaluateStrategy(self, gameState):
        numInvaders = 0
        for i in self.getOpponents(gameState):
            foodDist = self.closestDist(self.getFoodYouAreDefending(gameState).asList(), gameState)
            if foodDist  < self.cautionThreshold:
                numInvaders += 1
        if numInvaders == 0:
            return True
        elif numInvaders > 1:
            return False
        else:
            return self.defaultOffense

    def updateExpectation(self, gameState):
        # Update our estimate of how we expect our opponents to behave.
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]

        index = 0

        # We want to keep a close eye on offensive opponents.
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

    def minDistance(self, positionList, originPoint):
        minDist = float("inf")
        for p in positionList:
            minDist = min(minDist, self.getMazeDistance(originPoint, p))
        return minDist
    """
    def openness(self, walls, pointPos):
        x, y = pointPos
        openWest = 0
        while not walls[x][y]:
            x -= 1
            openWest += 1

        x, y = pointPos
        openNorth = 0
        while not walls[x][y]:
            y += 1
            openNorth += 1

        x, y = pointPos
        openEast = 0
        while not walls[x][y]:
            x += 1
            openEast += 1

        x, y = pointPos
        openSouth = 0
        while not walls[x][y]:
            y -= 1
            openSouth += 1

        cardinalList = [openWest, openNorth, openEast, openSouth]
        cardinalList.remove(max(cardinalList))

        return sum(cardinalList) / 3

    def weightedMinDistance(self, ourPosition, foodPositions, ghostPositions, walls):
        minDist = float("inf")
        minEnemyDist = float("inf")
        minPoint = None
        for p in foodPositions:
            foodDist = self.getMazeDistance(ourPosition, p) / 200
            open = self.openness(walls, p)
            foodDist /= open
            minDist = min(minDist, foodDist)
        return minDist
    """

    def closestDist(self, foodList, gameState):
        closestFoodPos = None
        shortestFoodDist = float("inf")
        index = 0
        enemyStates = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        scaredPos = []
        bravePos = []
        for en in enemyStates:
            if en.isScared():
                scaredPos.append(en.getPosition())
            else:
                bravePos.append(en.getPosition())
        for e in bravePos:
            for food in foodList:
                foodDist = self.getMazeDistance(food, e)
                if foodDist < shortestFoodDist:
                    closestFoodPos = food
                    shortestFoodDist = foodDist

        for e in scaredPos:
            for food in foodList:
                foodDist = self.getMazeDistance(food, e) / 2
                if foodDist < shortestFoodDist:
                    closestFoodPos = food
                    shortestFoodDist = foodDist

        return shortestFoodDist

    def offenseFeatures(self, oldState, action):
        self.updateExpectation(oldState)

        features = counter.Counter()
        newState = self.getSuccessor(oldState, action)
        newAgentState = newState.getAgentState(self.index)
        enemyStates = [newState.getAgentState(i) for i in self.getOpponents(newState)]
        bravePositions = [s.getPosition() for s in enemyStates if s.isBraveGhost()]
        teamPos = [newState.getAgentState(i).getPosition() for i in self.getTeam(newState)]
        enemyFood = self.getFood(oldState).asList()  # Compute distance to the nearest food.
        newPos = newState.getAgentState(self.index).getPosition()
        oldPos = oldState.getAgentState(self.index).getPosition()
        walls = oldState.getWalls()

        features['newStateScore'] = self.getScore(newState) - self.getScore(oldState)  

        if len(enemyFood) > 0:
            enemyFoodDist = self.minDistance(enemyFood, newPos)
            # Individual food distances are a bit irrelevant from far away
            features['distanceToFood'] = enemyFoodDist ** 0.7

            sumFoodX = 0
            sumFoodY = 0
            for food in enemyFood:
                sumFoodX += food[0]
                sumFoodY += food[1]

            averageFood = [sumFoodX / len(enemyFood), sumFoodY / len(enemyFood)]

            # The average of all food distances is helpful when not many food pellets.
            # are immediately nearby.
            features['distToAvgFood'] = (abs(newPos[0] - averageFood[0])
                                         + abs(newPos[1] - averageFood[1])) ** 1.2

        if (enemyStates[0].isBraveGhost()) and (enemyStates[1].isBraveGhost()):
            enemyCapsules = self.getCapsules(oldState)

            if (len(enemyCapsules) > 0):
                capsuleDist = self.minDistance(enemyCapsules, newPos)

                # Same thing as calculating distance to power pellets.
                # The only difference is that power pellets are relevant within a larger radius.
                features['distanceToCapsule'] = capsuleDist ** 0.9

                if (len(enemyCapsules) < len(self.getCapsules(oldState))):
                    features['onCapsule'] = 1

        # Computes whether we're on defense (1) or offense (0).
        if not self.offense:
            features['onDefense'] = 1

        # Determine whether the agent should be afraid of ghosts or seeking out afraid ones.
        if newAgentState.isPacman():
            braveEnemies = []
            scaredEnemies = []

            # Check which ghosts are scared.
            for a in enemyStates:
                if a.isBraveGhost():
                    braveEnemies.append(a.getPosition())
    
                else:
                    scaredEnemies.append(a.getPosition())

            # The agent should be wary of ghosts within a tight radius.
            if len(braveEnemies) > 0:
                features['distToBrave'] = self.minDistance(braveEnemies, newPos) ** -3

            if (len(scaredEnemies) > 0):
                minDist = float("inf")
                timer = 0

                for s in scaredEnemies:
                    dist = self.getMazeDistance(newPos, s)
                    if (dist < minDist):
                        minDist = dist
                        timer = a.getScaredTimer()

                # Eating vulnerable ghosts involves less danger, so the radius is relaxed.
                features['distToScared'] = timer / minDist
            
            oldScaredies = []
            oldBravies = []
            oldEnemyStates = [oldState.getAgentState(i) for i in self.getOpponents(oldState)]
            for s in oldEnemyStates:
                if s.isScared():
                    oldScaredies.append(s.getPosition())
                else:
                    oldBravies.append(s.getPosition())

            # Reward the agent for eating a ghost.
            if (newPos in oldScaredies):
                features['eatenGhost'] = 1
            elif (newPos in oldBravies):
                features['killedbyGhost'] = 1

        return features

    def defenseFeatures(self, gameState, action):
        self.updateExpectation(gameState)

        features = counter.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]

        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0).
        if self.offense:
            features['notOnDefense'] = 1

        # Computes distance to invaders we can see.
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman() and a.getPosition() is not None]
        features['numInvaders'] = len(invaders)

        # Check whether we should be chasing an opposing Pac-Man or running away.
        if (len(invaders) > 0):
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)
            if myState.isScared():
                features['invaderDistance'] *= -1

        # Slightly discourage stalling and indecisiveness.
        if (action == Directions.STOP):
            features['stop'] = 1

        rev = Directions.REVERSE[gameState.getAgentState(self.index).getDirection()]
        if (action == rev):
            features['reverse'] = 1

        ourFood = self.getFoodYouAreDefending(successor).asList()

        closestFoodPos = None
        shortestFoodDist = float("inf")
        index = 0


        # Attempt to predict the next food pellet the opponent will try to eat.
        # For simple agents, this will be the closest food pellet to them.
        for a in enemies:
            if self.offenseDetector[index] > 0.1:
                for food in ourFood:
                    foodDist = self.getMazeDistance(food, a.getPosition())

                    if foodDist < shortestFoodDist:
                        closestFoodPos = food
                        shortestFoodDist = foodDist

            index += 1

        if (closestFoodPos is not None):
            targetedDist = self.getMazeDistance(myPos, closestFoodPos)
            features['targetedFoodDist'] = targetedDist

        ourCapsules = self.getCapsulesYouAreDefending(successor)

        closestCapsulePos = None
        shortestCapsuleDist = 999999
        index = 0

        # Try to predict the next power pellet the opponent will try to eat.
        for a in enemies:
            if self.offenseDetector[index] > 0.1:
                for capsule in ourCapsules:
                    capsuleDist = self.getMazeDistance(capsule, a.getPosition())

                    if capsuleDist < shortestCapsuleDist:
                        closestCapsulePos = capsule
                        shortestCapsuleDist = capsuleDist

            index += 1

        if (closestCapsulePos is not None):
            targetedDist = self.getMazeDistance(myPos, closestCapsulePos)
            features['targetedCapsuleDist'] = targetedDist

        # If our defensive agent is already in danger, attempt to camp the next power pellet.
        # Best-case scenario: the scared timer runs out and the agent defends the pellet.
        # Worst-case scenario: the opponent eats the second pellet with our agent and the
        # scared timer immediately ends.
        if myState.isScared():
            features['targetedCapsuleDist'] = targetedDist * 1.2

        return features

    def getWeights(self, gameState, action):

        offenseWeights = {
            'newStateScore': 100,
            'distanceToFood': -5,
            'distanceToCapsule': -8,
            'distToAvgFood': -0.1,
            'onCapsule': 100000,
            'distToScared': 0.1,
            'eatenGhost': 10000,
            'onDefense': 0,
            'distToBrave': -90,
            'killedByGhost': -1000,
            'teamDist': 1
        }
        defenseWeights = {
            'numInvaders': -1000,
            'notOnDefense': 0,
            'invaderDistance': -14,
            'targetedFoodDist': -20,
            'targetedCapsuleDist': -18,
            'stop': -10,
            'reverse': -0.1
        }

        if self.offense:
            return offenseWeights
        else:
            return defenseWeights

def createTeam(firstIndex, secondIndex, isRed):
    """
    This function should return a list of two agents that will form the capture team,
    initialized using firstIndex and secondIndex as their agent indexed.
    isRed is True if the red team is being created,
    and will be False if the blue team is being created.
    """

    # firstAgent = StrategyAgentA
    # secondAgent = StrategyAgentB

    return [
        HybridAgent(index = firstIndex, defaultOffense = True, cautionThreshold = 5),
        HybridAgent(index = secondIndex, defaultOffense = False, cautionThreshold = 10),
    ]

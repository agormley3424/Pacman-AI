from pacai.util import reflection
from pacai.agents.capture.reflex import ReflexCaptureAgent
from pacai.core.directions import Directions
from pacai.util import counter
# from pacai.util import probability
# import random

# States are CaptureGameStates

class StrategyAgentA(ReflexCaptureAgent):
    """
    This is an offense agent.

    It will seek out as many food pellets as it can while attempting to avoid ghosts
    within a certain threshold. It will also seek out conveniently-placed power pellets
    and make an attempt to eat scared ghosts.

    The agent is both greedy and will attempt to keep a good distance away from ghosts.

    Fatal flaw: This agent can be deadlocked on very specific maps, and it does not
    use minimax search, so it is prone to making poorly-informed movements in tight spots.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index)

        # This can help the agent detect the potential behavior of the opposing agents.
        self.offenseDetector = [1, 1]

        self.currentSearchFood = None
        self.lastPos = None
        self.foodPenalty = 0

        self.dangerousFood = []

        self.walls = None

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

    def evaluateFood(self, gameState, pos):
        foodList = self.getFood(gameState).asList()
        bestDist = 999999
        bestFood = None

        if self.currentSearchFood is not None and self.lastPos is not None:
            oldPathDist = self.getMazeDistance(self.lastPos, self.currentSearchFood)
            currentPathDist = self.getMazeDistance(pos, self.currentSearchFood)

            if pos is self.lastPos:
                self.foodPenalty += 1

            if currentPathDist > oldPathDist:
                self.foodPenalty += 5

        if self.foodPenalty > 4:
            self.dangerousFood.append(self.currentSearchFood)
            self.currentSearchFood = None
            self.foodPenalty = 0
            # print("update dangerousFood: ", self.dangerousFood)

        if self.currentSearchFood is None:
            for food in foodList:
                # check if current food is not considered dangerous
                if food not in self.dangerousFood:
                    isFaraway = True

                    # make sure the food pellet is not close to a dangerous one
                    for dangerFood in self.dangerousFood:
                        if (abs(food[0] - dangerFood[0]) + abs(food[1] - dangerFood[1])) < 3:
                            isFaraway = False

                    # the next food pellet is far enough away that we can count it
                    if isFaraway is True:
                        dist = self.getMazeDistance(pos, food)

                        if dist < bestDist:
                            bestDist = dist
                            bestFood = food

        # print("bestFood: ", bestFood)
        self.currentSearchFood = bestFood

    def getFeatures(self, gameState, action):
        if (self.walls is None):
            self.walls = gameState.getWalls().asList()

        self.updateExpectation(gameState)

        features = counter.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)

        myState = successor.getAgentState(self.index)
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]

        # Compute distance to the nearest food.
        foodList = self.getFood(successor).asList()

        # This should always be True, but better safe than sorry.
        if (len(foodList) > 0):
            oldPos = gameState.getAgentState(self.index).getPosition()
            
            myPos = successor.getAgentState(self.index).getPosition()
            # minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            # features['distanceToFood'] = minDistance ** 0.7

            self.evaluateFood(gameState, oldPos)

            if (self.currentSearchFood is None):
                # print("refresh")
                self.dangerousFood = []
                self.evaluateFood(gameState, oldPos)

            if (self.currentSearchFood is not None):
                minDistance = self.getMazeDistance(myPos, self.currentSearchFood)
                # print("current food: ", self.currentSearchFood)
                # print("min distance: ", minDistance)

                # Individual food distances are a bit irrelevant from far away
                features['distanceToFood'] = minDistance ** 0.7

            else:
                print("no food found")

            sumFoodX = 0
            sumFoodY = 0
            for food in foodList:
                sumFoodX += food[0]
                sumFoodY += food[1]

            averageFood = [sumFoodX / len(foodList), sumFoodY / len(foodList)]
            
            if (len(foodList) < len(self.getFood(gameState).asList())):
                self.dangerousFood = []

            # The average of all food distances is helpful when not many food pellets.
            # are immediately nearby.
            features['distToAvgFood'] = (abs(myPos[0] - averageFood[0])
                                        + abs(myPos[1] - averageFood[1])) ** 1.2

        if (enemies[0].isBraveGhost()) and (enemies[1].isBraveGhost()):
            capsuleList = self.getCapsules(successor)

            if (len(capsuleList) > 0):
                myPos = successor.getAgentState(self.index).getPosition()
                minDistance = min([self.getMazeDistance(myPos, capsule) for capsule in capsuleList])

                # Same thing as calculating distance to power pellets.
                # The only difference is that power pellets are relevant within a larger radius.
                features['distanceToCapsule'] = minDistance ** 0.9

                features['onCapsule'] = 0
                if (len(capsuleList) < len(self.getCapsules(gameState))):
                    features['onCapsule'] = 1

        myPos = myState.getPosition()


        if (self.walls is not None):
            openness = 0
            otherPos = (myPos[0] + 1, myPos[1])

            if (otherPos not in self.walls):
                openness += 1

            otherPos = (myPos[0] - 1, myPos[1])
            
            if (otherPos not in self.walls):
                openness += 1

            otherPos = (myPos[0], myPos[1] + 1)

            if (otherPos not in self.walls):
                openness += 1

            otherPos = (myPos[0], myPos[1] - 1)

            if (otherPos not in self.walls):
                openness += 1

            features['openness'] = openness ** 1.05
            # print(openness)

        # Computes whether we're on defense (1) or offense (0).
        features['onDefense'] = 1
        if (myState.isPacman()):
            features['onDefense'] = 0

        # Determine whether the agent should be afraid of ghosts or seeking out afraid ones.
        if myState.isPacman():
            brave = []
            scared = []

            # Check which ghosts are scared.
            for a in enemies:
                if a.isBraveGhost():
                    brave.append(a)

                else:
                    scared.append(a)

            features['danger'] = 0
            features['distToScared'] = 0

            dists = []

            for a in enemies:
                if a.isBraveGhost():
                    dists.append(self.getMazeDistance(myPos, a.getPosition()))

            # The agent should be wary of ghosts within a tight radius.
            if (len(dists)) > 0:
                features['danger'] = (min(dists)) ** -3.5

            if (len(scared) > 0):
                smallestDist = 999999
                timer = 0

                for a in scared:
                    dist = self.getMazeDistance(myPos, a.getPosition())

                    if (dist < smallestDist):
                        smallestDist = dist
                        timer = a.getScaredTimer()

                # Eating vulnerable ghosts involves less danger, so the radius is relaxed.
                features['distToScared'] = timer / smallestDist

            currentEnemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
            scaredNum = 0

            for a in currentEnemies:
                if not a.isBraveGhost():
                    scaredNum += 1

            # Reward the agent for eating a ghost.
            features['eatenGhost'] = 0
            if (len(scared) < scaredNum):
                features['eatenGhost'] = 1

        self.lastPos = gameState.getAgentState(self.index).getPosition()

        return features

    def getWeights(self, gameState, action):
        ourWeights = {
            'successorScore': 10,
            'distanceToFood': -6,
            'distanceToCapsule': -8,
            'danger': -90,
            'distToAvgFood': -0.1,
            'onCapsule': 100000,
            'distToScared': 0.1,
            'eatenGhost': 10000,
            'onDefense': -1,
            'openness': 1,
        }

        return ourWeights

class StrategyAgentB(ReflexCaptureAgent):
    """
    This is a defense agent.

    This agent attempts to find the most likely food pellet the enemy may eat and
    proceeds to sit at that pellet. The agent will try to track down incoming invaders
    but prefers to sit atop of pellets to stall the opponent's progress. When this agent
    is in danger of being eaten, it will attempt to be efficient by sitting on the other
    remaining power pellet, either waiting for the vulnerability to end or tricking the
    opponent into wasting the next power pellet.

    Even if the opponent manages to make some progress, it will have a lot of difficulty
    eating most of the pellets without first eating a power pellet. Fortunately, power
    pellets are typically hidden deeper into a map most of the time, and there's a good
    chance that agents will be greedy enough to approach the nearest food pellet first,
    since eating pellets is the only way to increase their score. In many cases, it
    becomes easier to camp pellets as more are collected, buying valuable time for the
    offense agent.

    Fatal flaw: This agent always assumes the opponent will approach the food pellet
    closest to it. A minimax agent may be able to outmaneuver it. Any agent that can
    reach power pellets before this agent will put this agent in serious danger.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index)

        # This can help the agent detect the potential behavior of the opposing agents.
        self.offenseDetector = [1, 1]

    def updateExpectation(self, gameState):
        # Update our estimate of how we expect our opponents to behave
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]

        index = 0

        # We want to keep a close eye on offensive opponents.
        # Opponents that stick around the middle of the map are also potentially dangerous.
        defenseLerp = 0.99
        mediumLerp = 0.98

        for a in enemies:
            enemyPos = a.getPosition()

            if a.isPacman():
                self.offenseDetector[index] = 1

            else:
                lastValue = self.offenseDetector[index]
                newValue = (lastValue * defenseLerp) + (0 * (1 - defenseLerp))
                self.offenseDetector[index] = newValue

            if enemyPos[0] > 13 and enemyPos[0] < 17:
                lastValue = self.offenseDetector[index]
                newValue = (lastValue * mediumLerp) + (1 * (1 - mediumLerp))
                self.offenseDetector[index] = newValue

            index += 1

    def getFeatures(self, gameState, action):
        self.updateExpectation(gameState)

        features = counter.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]

        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0).
        features['notOnDefense'] = 0
        if (myState.isPacman()):
            features['notOnDefense'] = 1

        # Computes distance to invaders we can see.
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman() and a.getPosition() is not None]
        features['numInvaders'] = len(invaders)

        # Check whether we should be chasing an opposing Pac-Man or running away.
        if (len(invaders) > 0):
            if myState.isBraveGhost():
                dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
                features['invaderDistance'] = min(dists)
                features['runAway'] = 0

            else:
                dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
                features['runAway'] = (1 / min(dists))
                features['invaderDistance'] = 0

        # Slightly discourage stalling and indecisiveness.
        if (action == Directions.STOP):
            features['stop'] = 1

        rev = Directions.REVERSE[gameState.getAgentState(self.index).getDirection()]
        if (action == rev):
            features['reverse'] = 1

        ourFood = self.getFoodYouAreDefending(successor).asList()

        closestFoodPos = None
        shortestFoodDist = 999999
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

        features['targetedFoodDist'] = 0

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

        features['targetedCapsuleDist'] = 0

        if (closestCapsulePos is not None):
            targetedDist = self.getMazeDistance(myPos, closestCapsulePos)
            features['targetedCapsuleDist'] = targetedDist

        # If our defensive agent is already in danger, attempt to camp the next power pellet.
        # Best-case scenario: the scared timer runs out and the agent defends the pellet.
        # Worst-case scenario: the opponent eats the second pellet with our agent and the
        # scared timer immediately ends.
        if (features['runAway'] > 0):
            features['targetedCapsuleDist'] = targetedDist * 1.2

        # This is scrapped code for finding the path to the nearest food pellet that this
        # agent can reach before the opponent can, under the assumption that the opponent
        # will always collect the closest food pellet to it. This code is left here in
        # case we want to use some version of it in the future, and also because it took
        # a while to write it all out.
        """
        ourFood = self.getFoodYouAreDefending(successor).asList()

        bestFoodTarget = None
        bestFoodDist = 999999
        index = 0

        for a in enemies:
            foodChecked = []
            enemyPos = a.getPosition()
            lastPos = enemyPos
            foodInRange = None
            pathLength = 0
            ourPathLength = 999999

            if self.offenseDetector[index] > 0.15:
                while foodInRange is None:
                    closestFood = None
                    shortestDist = 999999

                    for food in ourFood:
                        if food not in foodChecked:
                            dist = self.getMazeDistance(food, lastPos)

                            if dist < shortestDist:
                                shortestDist = dist
                                closestFood = food

                    pathLength += self.getMazeDistance(closestFood, lastPos)
                    # print("new path length: ", pathLength)
                    distToFood = self.getMazeDistance(myPos, closestFood)
                    #print("our path length: ", distToFood)

                    if (distToFood <= (pathLength + 1)):
                        foodInRange = closestFood
                        ourPathLength = distToFood
                        # print("found best path: ", ourPathLength)

                    lastPos = closestFood
                    foodChecked.append(closestFood)

            if ourPathLength < bestFoodDist:
                bestFoodDist = ourPathLength

            index += 1


        # features['targetedFoodDist'] = 0

        if bestFoodTarget is not None:
            features['targetedFoodDist'] = bestFoodDist

        # print(bestFoodDist)
        """

        return features

    def getWeights(self, gameState, action):
        ourWeights = {
            'numInvaders': -1000,
            'notOnDefense': -200,
            'invaderDistance': -9,
            'runAway': -100,
            'targetedFoodDist': -22,
            'targetedCapsuleDist': -17,
            'stop': -10,
            'reverse': -0.1
        }

        return ourWeights

def createTeam(firstIndex, secondIndex, isRed):
    """
    This function should return a list of two agents that will form the capture team,
    initialized using firstIndex and secondIndex as their agent indexed.
    isRed is True if the red team is being created,
    and will be False if the blue team is being created.
    """

    firstAgent = StrategyAgentA
    secondAgent = StrategyAgentB

    return [
        StrategyAgentA(firstIndex),
        StrategyAgentB(secondIndex),
    ]

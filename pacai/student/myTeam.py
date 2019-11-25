from pacai.util import reflection
from pacai.agents.capture.reflex import ReflexCaptureAgent
from pacai.core.directions import Directions
from pacai.util import counter
import random
# from pacai.util import probability
# import random

# States are CaptureGameStates

class OffenseAgent(ReflexCaptureAgent):
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
        self.simulatedIntrospection = False

    def makeProgramWork(self, everything):
        if not programWork():
            self.makeProgramWork(everything)
        else:
            return None

    def chooseActionIndex(self, gameState, index):
        """
        Picks among the actions with the highest return from `ReflexCaptureAgent.evaluate`.
        """
        
        actions = gameState.getLegalActions(index)
        self.simulatedIntrospection = True
        values = [self.evaluate(gameState, a, index) for a in actions]

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        return random.choice(bestActions)


    def updateExpectation(self, gameState):
        # Update our estimate of how we expect our opponents to behave.
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState, self.index)]

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
    def abMaxValue(agent, gameState, treeDepth, agentIndex, alpha, beta, levelCount, evalFn):
        legalActions = gameState.getLegalActions(agentIndex)

        # remove STOP as a direction for pac-man
        if (Directions.STOP in legalActions):
            legalActions.remove(Directions.STOP)

        # end search if game is over or max tree depth has been reached
        if (levelCount == treeDepth * 4) or (gameState.isOver()):
            return (evalFn(gameState), Directions.STOP)

        if agentIndex > len(self.getOpponents()) + len(self.getTeam) - 1:
            agentIndex = 0

        # max value
        maxScore = float("-inf")
        bestAction = Directions.STOP

        # (score, action) pair
         scorePair = abMinValue(agent, gameState.generateSuccessor(agentIndex, action), treeDepth, agentIndex + 1, alpha,
                                beta, levelCount + 1)

        # update best action and score
        if (scorePair[0] > maxScore):
            maxScore = scorePair[0]

        # return if score exceeeds bounds
        if (maxScore >= beta):
            return (maxScore, bestAction)

        # update alpha
        alpha = max(alpha, maxScore)

        return (maxScore, bestAction)

    def abMinValue(agent, gameState, treeDepth, agentIndex, alpha, beta, levelCount, evalFn):
        legalActions = gameState.getLegalActions(agentIndex)

        # end search if game is over
        if (gameState.isOver()):
            return evalFn(gameState)

        numAgents = gameState.getNumAgents()

        # min value
        minScore = float("inf")
        worstAction = Directions.STOP

        # iterate through all possible actions
        for action in legalActions:
            scorePair = None

            # (score, action) pair
            # continue iterating through ghosts or search pac-man's actions next
            if (agentIndex < numAgents - 1):
                scorePair = abMinValue(agent, gameState.generateSuccessor(agentIndex, action),
                                    treeDepth, agentIndex + 1, alpha, beta)
            else:
                scorePair = abMaxValue(agent, gameState.generateSuccessor(agentIndex, action),
                                    treeDepth - 1, 0, alpha, beta)

            # update worst action and score
            if (scorePair[0] < minScore):
                minScore = scorePair[0]
                worstAction = action

            # return if score exceeds bounds
            if (minScore <= alpha):
                return(minScore, worstAction)

            # update beta
            beta = min(beta, minScore)

        return (minScore, worstAction)
    """
    def gonnaDie(self, oldState, newState, depth, level, index):
        if index > max(self.getOpponents(newState, index) + self.getTeam(newState, index)):
            index = min(self.getOpponents(newState, index) + self.getTeam(newState, index))
        if self.isEaten(oldState, newState, index):
            return True
        elif level == depth * 4:
            return False
        else:
            newerState = newState.generateSuccessor(index, self.chooseActionIndex(newState, index))
            return self.gonnaDie(newState, newerState, depth, level + 1, index + 1)

    def isEaten(self, oldState, newState, index):
        agentPos = oldState.getAgentPosition(self.index)
        enemyPositions = [newState.getAgentPosition(i) for i in self.getOpponents(newState, index)]
        if agentPos in enemyPositions:
            return True
        else:
            return False

    def getScore(self, gameState, index):
        if index % 2 == 0:
            return gameState.getScore()
        else:
            return gameState.getScore() * -1

    def getOpponents(self, gameState, index):
        if index % 2 == 0:
            return gameState.getBlueTeamIndices()
        else:
            return gameState.getRedTeamIndices()

    def getCapsules(self, gameState, index):
        if index % 2 == 0:
            return gameState.getBlueCapsules()
        else:
            return gameState.getRedCapsules()

    def getTeam(self, gameState, index):
        if index % 2 == 0:
            return gameState.getRedTeamIndices()
        else:
            return gameState.getBlueTeamIndices()

    def evaluate(self, gameState, action, index = None):
        """
        Computes a linear combination of features and feature weights.
        """
        if index is None:
            index = self.index
        features = self.getFeatures(gameState, action, index)
        weights = self.getWeights(gameState, action)

        return features * weights

    def getFeatures(self, oldState, action, index = None):
        # Made score take difference rather than the new score only
        # Using old positions of enemy food
        # Removed setting features['onCapsule] = 0
        # Commented out features['onDefense']
        if index is None:
            index = self.index

        self.updateExpectation(oldState)

        features = counter.Counter()
        newState = oldState.generateSuccessor(index, action)
        newAgentState = newState.getAgentState(index)
        enemyStates = [newState.getAgentState(i) for i in self.getOpponents(newState, index)]
        enemyFood = self.getFood(oldState).asList()  # Compute distance to the nearest food.
        newPos = newState.getAgentState(index).getPosition()

        features['newStateScore'] = self.getScore(newState, index) - self.getScore(oldState, index)  

        if (len(enemyFood) > 0):
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
            enemyCapsules = self.getCapsules(oldState, index)

            if (len(enemyCapsules) > 0):
                capsuleDist = self.minDistance(enemyCapsules, newPos)

                # Same thing as calculating distance to power pellets.
                # The only difference is that power pellets are relevant within a larger radius.
                features['distanceToCapsule'] = capsuleDist ** 0.9

                if (len(enemyCapsules) > len(self.getCapsules(newState, index))):
                    features['onCapsule'] = 1

        # Computes whether we're on defense (1) or offense (0).
        """
        features['onDefense'] = 1
        if (myState.isPacman()):
            features['onDefense'] = 0
        """

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

            features['distToBrave'] = 0
            features['distToScared'] = 0

            # The agent should be wary of ghosts within a tight radius.
            if len(braveEnemies) > 0:
                features['distToBrave'] = self.minDistance(braveEnemies, newPos)

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
            oldEnemyStates = [oldState.getAgentState(i) for i in self.getOpponents(oldState, index)]
            for s in oldEnemyStates:
                if s.isScared():
                    oldScaredies.append(s.getPosition())
                else:
                    oldBravies.append(s.getPosition())

            # Reward the agent for eating a ghost.
            features['eatenGhost'] = 0
            features['killedbyGhost'] = 0
            if (newPos in oldScaredies):
                features['eatenGhost'] = 1
            elif (newPos in oldBravies):
                features['killedbyGhost'] = 1

            if not self.simulatedIntrospection:
                features['gonnaDie'] = 1 if self.gonnaDie(oldState, newState, 1, 1, index) else 0
                self.simulatedIntrospection = False

        return features

    def getWeights(self, gameState, action):
        ourWeights = {
            'newStateScore': 100,
            'distanceToFood': -5,
            'distanceToCapsule': -7,
            'distToAvgFood': -0.1,
            'distToBrave': 0,
            'onCapsule': 100000,
            'distToScared': 10,
            'eatenGhost': 10000,
            'onDefense': -10,
            'gonnaDie': -10
        }

        return ourWeights

class DefenseAgent(ReflexCaptureAgent):
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

            if enemyPos[0] > 12 and enemyPos[0] < 18:
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
            'invaderDistance': -8,
            'runAway': -100,
            'targetedFoodDist': -21,
            'targetedCapsuleDist': -19,
            'stop': -0.5,
            'reverse': -1
        }

        return ourWeights

def createTeam(firstIndex, secondIndex, isRed,
        first = 'pacai.student.myTeam.OffenseAgent',
        second = 'pacai.student.myTeam.DefenseAgent'):
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

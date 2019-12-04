from pacai.util import reflection
from pacai.agents.capture.reflex import ReflexCaptureAgent
from pacai.core.directions import Directions
from pacai.core import distanceCalculator
from pacai.util import counter
from pacai.core import distance
from pacai.core.search.position import PositionSearchProblem
#from pacai.core.gamestate import   #EXPERIMENTAL
from pacai.student import search
from pacai.core.actions import Actions
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
        self.introspection = False
        self.currentSearchFood = None
        self.lastPos = None
        self.foodPenalty = 0

        self.dangerousFood = []

        self.walls = None

    def simpleEval(self, gameState, action, introspection = False):
        """
        Computes a linear combination of features and feature weights.
        """
        self.introspection = introspection
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features['distToBrave'] * weights['distToBrave']

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

    def getFeatures(self, oldState, action):
        # Made score take difference rather than the new score only
        # Using old positions of enemy food
        # Removed setting features['onCapsule] = 0
        # Commented out features['onDefense']
        self.updateExpectation(oldState)

        features = counter.Counter()
        newState = self.getSuccessor(oldState, action)
        newAgentState = newState.getAgentState(self.index)
        enemyStates = [newState.getAgentState(i) for i in self.getOpponents(newState)]
        enemyFood = self.getFood(oldState).asList()  # Compute distance to the nearest food.
        newPos = newState.getAgentState(self.index).getPosition()
        oldPos = oldState.getAgentState(self.index).getPosition()

        features['newStateScore'] = self.getScore(newState) - self.getScore(oldState)

        if (len(enemyFood) > 0):
            self.evaluateFood(oldState, oldPos)

            if (self.currentSearchFood is None):
                # print("refresh")
                self.dangerousFood = []
                self.evaluateFood(oldState, oldPos)

            if (self.currentSearchFood is not None):
                minDistance = self.getMazeDistance(newPos, self.currentSearchFood)
                # print("current food: ", self.currentSearchFood)
                # print("min distance: ", minDistance)

                # Individual food distances are a bit irrelevant from far away
                features['distanceToFood'] = minDistance ** 0.7

            else:
                print("no food found")

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

    def getWeights(self, gameState, action):
        ourWeights = {
            'newStateScore': 100,
            'distanceToFood': -5,
            'distanceToCapsule': -7,
            'distToAvgFood': -0.1,
            'distToBrave': -90,
            'onCapsule': 100000,
            'distToScared': 10,
            'eatenGhost': 10000,
            'onDefense': -10,
            'minMaxEstimate': 1
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

        self.optimalIdlePosition = None
        self.chokePoints = None

    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the agent and populates useful fields,
        such as the team the agent is on and the `pacai.core.distanceCalculator.Distancer`.
        """

        self.red = gameState.isOnRedTeam(self.index)

        self.distancer = distanceCalculator.Distancer(gameState.getInitialLayout())

        self.distancer.getMazeDistances()

        self.chokePoints = self.findChokes(gameState.getInitialLayout())
        self.optimalIdlePosition = self.findOptimalIdlePosition(gameState)

        print("Choke Points: ", self.chokePoints)
        print("Optimal Idle Position: ", self.optimalIdlePosition)

    def updateExpectation(self, gameState):
        # Update our estimate of how we expect our opponents to behave
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]

        index = 0

        # We want to keep a close eye on offensive opponents.
        # Opponents that stick around the middle of the map are also potentially dangerous.
        defenseLerp = 0.99
        mediumLerp = 0.98

        for enemy in enemies:
            enemyPos = enemy.getPosition()

            if enemy.isPacman():
                self.offenseDetector[index] = 1

            else:
                lastValue = self.offenseDetector[index]
                newValue = (lastValue * defenseLerp) + (0 * (1 - defenseLerp))  # Dafaq? why is (1 - defenseLerp) * 0?
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
        # enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]

        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0).
        features['notOnDefense'] = 0
        if (myState.isPacman()):
            features['notOnDefense'] = 1

        # Computes distance to invaders we can see.
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [enemy for enemy in enemies if enemy.isPacman() and enemy.getPosition() is not None]
        features['numInvaders'] = len(invaders)

        # Check whether we should be chasing an opposing Pac-Man or running away.
        if (len(invaders) > 0):
            if myState.isBraveGhost():
                dists = [self.getMazeDistance(myPos, enemy.getPosition()) for enemy in invaders]
                features['invaderDistance'] = min(dists)
                features['runAway'] = 0

            else:
                dists = [self.getMazeDistance(myPos, enemy.getPosition()) for enemy in invaders]
                features['runAway'] = (1 / min(dists))
                features['invaderDistance'] = 0

        index = 0

        """
        i = 0
        for enemy in enemies:
            (x, y) = enemy.getPosition()
            initLayout = gameState.getInitialLayout()
            border = int(initLayout.getWidth())/2

            if self.offenseDetector[i] > 0.1 and x > border:

                for choke in self.chokePoints:
                    enemyDistToChoke = self.getMazeDistance(enemy.getPosition(), choke)
                    myDistToChoke = self.getMazeDistance(myPos, choke)

                    # Enemy is approaching a choke, time to mobilize
                    if enemyDistToChoke <= myDistToChoke:
                        features['distanceToAttackedChoke'] = myDistToChoke
                    else:   # Enemy is not approaching a choke, so don't worry about it
                        features['distanceToAttackedChoke'] = self.getMazeDistance(myPos, self.optimalIdlePosition)

            i += 1
            """
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
        for enemy in enemies:
            (x, nothing) = self.chokePoints[0]
            (b, alsoNothing) = enemy.getPosition()

            if self.red == True:
                invaderIsPastChokes = x > b - 2
            else:
                invaderIsPastChokes = x < b - 2

            if self.offenseDetector[index] > 0.1 and invaderIsPastChokes:
                for food in ourFood:
                    foodDist = self.getMazeDistance(food, enemy.getPosition())

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
        for enemy in enemies:

            if self.offenseDetector[index] > 0.1:
                for capsule in ourCapsules:
                    capsuleDist = self.getMazeDistance(capsule, enemy.getPosition())

                    if capsuleDist < shortestCapsuleDist:
                        closestCapsulePos = capsule
                        shortestCapsuleDist = capsuleDist

            index += 1

        features['targetedCapsuleDist'] = 0

        (x, y) = self.chokePoints[0]
        (b, ugh) = enemy.getPosition()  # Issue

        if self.red == True:
            invaderIsPastChokes = x > b - 2
        else:
            invaderIsPastChokes = x < b - 2

        if (closestCapsulePos is not None):
            targetedDist = self.getMazeDistance(myPos, closestCapsulePos)
            if invaderIsPastChokes:
                features['targetedCapsuleDist'] = targetedDist
            else:
                features['targetedCapsuleDist'] = 0

        # Find out which choke point the invader is attempting to access our base from
        index = 0
        myDistToChoke = 0
        distOfClosestChokeToEnemy = float("inf")
        invadersTargetedChoke = None

        for a in enemies:
            (x, y) = a.getPosition()
            (border, whatever) = self.chokePoints[0]

            if self.red == True:
                invaderIsPastChokes = x < border
            else:
                invaderIsPastChokes = x > border

            if self.offenseDetector[index] > 0.1 and not invaderIsPastChokes:

                # Find the closest choke point to the enemy.
                # This will tell us where they are trying to get in from
                for choke in self.chokePoints:
                    enemyDistToChoke = self.getMazeDistance(a.getPosition(), choke)

                    if enemyDistToChoke < distOfClosestChokeToEnemy:
                        invadersTargetedChoke = choke
                        distOfClosestChokeToEnemy = enemyDistToChoke
                        myDistToChoke = self.getMazeDistance(myPos, invadersTargetedChoke)

            index += 1

        if (invadersTargetedChoke is not None):
            if myDistToChoke >= distOfClosestChokeToEnemy:
                features['distanceToAttackedChoke'] = myDistToChoke - distOfClosestChokeToEnemy
            else:
                features['distanceToAttackedChoke'] = self.getMazeDistance(myPos, self.optimalIdlePosition)

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
            'distanceToAttackedChoke': -27,
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

    def findChokes(self, layout):
        """
        Inputs = layout of the map.

        Output = List containing the states that the algorithm found as choke points

        Purpose: This algorithm finds out where the choke points on the map the defender must cross
        in order to enter our territory and steal our food. This alogrithm is useful for helping
        defense-focusd agents find which entry points they need to defend so that they can
        hopefully position themselves to respond to any incoming invaders as promptly as possible,
        regardless of where the invaders attack from.

        ISSUES: Currently, this strategy is flawed and does not work very well when the enemy team
        attacks with more than one of their agents. Granted, this is expected, but this strategy
        CAN but does NOT mitigate a all-out-offensive strategy UNLESS both agents attack from the
        same vector.
        """

        chokePoints = []
        possibleChokes = 0  # Pretty sure I can remove this, but want to commit just in case it breaks something.

        width = (layout.getWidth())
        height = (layout.getHeight())

        border = int(width/2)   # Truncated down, since this will be an improper fraction

        if self.red == True:
            numChokes = 99999999

            # Looking at the left-hand side of the map
            for row in range((border - 1), 0, -1):
                chokesOnThisIteration = []  # Reset the list of choke points with each iteration

                # For each space in this column of the map...
                for column in range(height - 1, 0, -1):
                    observedPosition = (row, column)

                    if not layout.isWall(observedPosition):

                        # Need to do some tweaking of the return values for east and west here.
                        # For some reason, getSuccessor returns a pair of floats, but we NEED ints.
                        (a, b) = Actions.getSuccessor(observedPosition, Directions.EAST)
                        (x, y) = Actions.getSuccessor(observedPosition, Directions.WEST)

                        east = (int(a), int(b))
                        west = (int(x), int(y))

                        # Deciding what position IS a choke point:

                        # Clear corridor
                        if not layout.isWall(east) and not layout.isWall(west):
                            chokesOnThisIteration.append(observedPosition)

                        # No idea how this fixes the alg, but don't touch it.
                        # Seriously, this was an experiment that worked and I don't know why.
                        if not layout.isWall(east) and layout.isWall(west):
                            chokesOnThisIteration.append(observedPosition)

                    # If it wasn't a choke, we don't care about this position, move along.
                    else:
                        continue

                # Need to check to make sure we even found chokes. Move on if we didn't
                if len(chokesOnThisIteration) != 0:
                    if possibleChokes > 0:  # Pretty sure I can remove this, but want to commit first
                        if len(chokesOnThisIteration) <= numChokes + possibleChokes:
                            numChokes = len(chokesOnThisIteration)
                            chokePoints = chokesOnThisIteration
                            possibleChokes = 0

                    # Found a new, easier to defend position. Use set of chokes instead.
                    if len(chokesOnThisIteration) < numChokes:
                        numChokes = len(chokesOnThisIteration)
                        chokePoints = chokesOnThisIteration

                    # We have probably gotten to the best defensive position possible. Return
                    if len(chokesOnThisIteration) > numChokes:
                        return chokePoints

                    # Change nothing if numChokes didn't change.
                    if len(chokesOnThisIteration) == numChokes:
                        continue

                if len(chokesOnThisIteration) == numChokes or len(chokesOnThisIteration) == 0:
                    continue

        # Same algorithm with some slight differences to accommodate for looking at a flipped map
        if self.red == False:
            numChokes = 99999999

            for row in range(border, width):
                chokesOnThisIteration = []
                for column in range(height - 1, 0, -1):
                    observedPosition = (row, column)

                    if not layout.isWall(observedPosition):
                        (a, b) = Actions.getSuccessor(observedPosition, Directions.EAST)
                        (x, y) = Actions.getSuccessor(observedPosition, Directions.WEST)

                        east = (int(a), int(b))
                        west = (int(x), int(y))

                        if not layout.isWall(east) and not layout.isWall(west):
                            chokesOnThisIteration.append(observedPosition)

                        if not layout.isWall(west) and layout.isWall(east):
                            chokesOnThisIteration.append(observedPosition)

                    else:
                        continue

                if len(chokesOnThisIteration) != 0:
                    if possibleChokes > 0:  # Pretty sure I can kill this, but want to commit first.
                        if len(chokesOnThisIteration) <= numChokes + possibleChokes:
                            numChokes = len(chokesOnThisIteration)
                            chokePoints = chokesOnThisIteration
                            possibleChokes = 0

                    if len(chokesOnThisIteration) < numChokes:
                        numChokes = len(chokesOnThisIteration)
                        chokePoints = chokesOnThisIteration

                    if len(chokesOnThisIteration) > numChokes:
                        return chokePoints

                    if len(chokesOnThisIteration) == numChokes:
                        continue

                if len(chokesOnThisIteration) == numChokes or len(chokesOnThisIteration) == 0:
                    continue

    def findOptimalIdlePosition(self, gameState):
        """
        Will look for the optimal location to idle at when no enemy invaders are present. This
        position is found by finding which position on the path between chokes positions the agent
        such that their distance to the farthest choke point relative to their current position is
        minimized. What I mean by this is that the agent will position itself so that its distance
        between all choke points is roughly equal. This is done because our defense can only be as
        strong as our most crippling weakness, so we need to distribute our "distance budget"
        between ALL choke points on the map and make sure that the agent is not too far, and also
        not too close to any choke point on the map.
        """

        # This bit of code just gets the distance between any one pair of choke points
        chokePoints = self.findChokes(gameState.getInitialLayout())
        routes = []

        # Getting the routes to each choke point pair (top to bottom) and putting them in routes
        for chokePoint in chokePoints:
            index = chokePoints.index(chokePoint)

            # This is here so that we don't get an ArrayOutOfBounds exception
            if index == len(chokePoints) - 1:
                break

            nextChokePoint = chokePoints[index + 1]
            searchProblem = \
                PositionSearchProblem(gameState, start=chokePoint, goal=nextChokePoint)

            # To get the shortest path to the next choke, we run BFS. Not too taxing.
            optimalRoute = search.breadthFirstSearch(searchProblem)
            routes.append((optimalRoute, chokePoint, nextChokePoint))

        """
        Now we are done getting the chokes and the routes to get from one to another. Now, we need
        to find out where we can idle so that we are not too far from any given choke point. In
        order to accomplish this, we need to look through each path and find the maze dist to each
        choke from that position. We then take the max of those distances and check if it is less
        than the distance we have recorded, and replace this inequality is true.
        
        THIS IS PRETTY TAXING: This is becausee we are not checking manhattan distance, but the
        maze distance to each choke at EVERY POINT ON THE PATH BETWEEN CHOKES. This is, however,
        a necessary evil, since we cannot make estimates here; a bad estimate can lead to a
        very exploitable hole in our defense.
        """

        # Two vars that store the farthestChokeDist and the optimalIdlePos between iterations
        minimumDistanceToFarthestChoke = 999999
        optimalIdlePosition = None

        # For each path from choke to choke in routes...
        for route in routes:
            (plan, currentPos, NOTHING) = route     # 3rd variable in the tuple is unused.

            # For each action we take in that path...
            for action in plan:
                currentPosDistances = {}

                # For every choke point on the map...
                for choke in chokePoints:

                    if currentPos == choke:
                        currentPosDistances.update({choke: 0})  # distance.maze() cannot return 0
                    else:
                        currentPosDistances.update({choke:
                            distance.maze(currentPos, choke, gameState)})

                # Get the farthest choke point from the current observed position
                currentDistToFarthestChoke = max(currentPosDistances.values())

                # Update the optimalIdle position.
                if currentDistToFarthestChoke < minimumDistanceToFarthestChoke:
                    minimumDistanceToFarthestChoke = int(currentDistToFarthestChoke)
                    optimalIdlePosition = currentPos

                # Updating the current position along the path for next iteration
                (x, y) = Actions.getSuccessor(currentPos, action)
                currentPos = (int(x), int(y))

        """
        Legacy version of this. Doesn't work as well, but have thi here just in case.
        Will likely remove this code block after committing the first time.
        
        for route in routes:
            (plan, chokeOne, chokeTwo) = route
            currentPos = chokeOne
            for action in plan:
                #currentPosDistance = 0
                currentPosDistance = 0

                for choke in chokePoints:
                    #currentPosDistance += distance.maze(currentPos, choke, gameState)
                    if currentPos == choke:
                        continue
                    currentPosDistance += distance.maze(currentPos, choke, gameState)

                averageCurrentPosDist = currentPosDistance/len(chokePoints)

                if averageCurrentPosDist < minDistance:
                    minDistance = int(averageCurrentPosDist)
                    minDistancePoint = currentPos

                (x, y) = Actions.getSuccessor(currentPos, action)
                currentPos = (int(x), int(y))
        """

        return optimalIdlePosition

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

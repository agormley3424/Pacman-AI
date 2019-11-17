"""
This file contains incomplete versions of some agents that can be selected to control Pacman.
You will complete their implementations.

Good luck and happy searching!
"""

import logging
import copy

from pacai.core.actions import Actions
from pacai.core import distance
from pacai.core.directions import Directions
from pacai.core.search.position import PositionSearchProblem
from pacai.core.search.problem import SearchProblem
from pacai.agents.base import BaseAgent
from pacai.agents.search.base import SearchAgent
from pacai.student.search import uniformCostSearch
from pacai.student.search import aStarSearch
from pacai.core.distance import maze

class CornersProblem(SearchProblem):
    """
    This search problem is to discover all four corners of the map.

    States in the CornersProblem are represented by a tuple containing a list of
    corners which have been discovered by that point and a tuple with pacman's
    coordinates.
    """

    def __init__(self, startingGameState):
        super().__init__()

        self.walls = startingGameState.getWalls()
        self.startingPosition = startingGameState.getPacmanPosition()
        top = self.walls.getHeight() - 2
        right = self.walls.getWidth() - 2

        self.corners = ((1, 1), (1, top), (right, 1), (right, top))
        for corner in self.corners:
            if not startingGameState.hasFood(*corner):
                logging.warning('Warning: no food in corner ' + str(corner))

        self.exCorners = []

    def startingState(self):
        """
        Returns the initial problem state, given the game state.

        The initial state is an empty list paired with pacman's starting position,
        which is determined by the game state.
        """

        return copy.deepcopy(self.exCorners), self.startingPosition

    def isGoal(self, state):
        """
        Returns true if all corners have been discovered in the state, otherwise false.

        The goal test checks whether the state's discovered corners list includes all corners.
        """
        for i in self.corners:
            if i not in state[0]:
                return False

        # Register the locations we have visited.
        # This allows the GUI to highlight them.
        self._visitedLocations.add(state[1])
        self._visitHistory.append(state[1])
        return True

    def successorStates(self, state):
        """
        Returns a list of successors states, each bundled in a tuple including
        an action and costs from the previous state.

        The function checks whether any child states have discovered a new corner,
        and appends all discoveries to the corresponding states. Pacman's location
        is also updated.
        """

        successors = []

        for action in Directions.CARDINAL:
            x, y = state[1]
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            newCorners = copy.deepcopy(state[0])

            if not self.walls[nextx][nexty]:
                if (nextx, nexty) in self.corners and (nextx, nexty) not in state[0]:
                    newCorners.append((nextx, nexty),)
                nextState = newCorners, (nextx, nexty)
                cost = 1
                successors.append((nextState, action, cost))

        # Bookkeeping for display purposes (the highlight in the GUI).
        self._numExpanded += 1
        if (state[1] not in self._visitedLocations):
            self._visitedLocations.add(state[1])
            self._visitHistory.append(state[1])

        return successors

    def actionsCost(self, actions):
        """
        Returns the cost of a particular sequence of actions.
        If those actions include an illegal move, return 999999.
        This is implemented for you.
        """

        if (actions is None):
            return 999999

        x, y = self.startingPosition
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999

        return len(actions)

def cornersHeuristic(state, problem):
    """
    A function which returns an estimate of distance to the goal of
    an instance of CornersProblem, given a particular state.

    My implementation uses either the distance from pacman to the
    farthest corner or the distance from the farthest corner
    to its own farthest corner (whichever is greater) to estimate
    distance to the goal, given a particular state.
    """

    # Useful information.
    # corners = problem.corners  # These are the corner coordinates
    # walls = problem.walls  # These are the walls of the maze, as a Grid.

    pacDist = 0
    for c in problem.corners:
        if c not in state[0]:
            if distance.manhattan(state[1], c) > pacDist:
                pacDist = distance.manhattan(state[1], c)
                farCorner = c
    maxDist = 0
    for c in problem.corners:
        if c not in state[0]:
            if distance.manhattan(farCorner, c) > maxDist:
                maxDist = distance.manhattan(farCorner, c)
    return max(maxDist, pacDist)

def foodHeuristic(state, problem):
    """
    A function that returns an estimated distance from the goal,
    given a particular state for the FoodSearchProblem.

    The distance between pacman and any undiscovered food is
    necessarily less than or equal to the actual solution,
    since it's a path he will need to take (or take an even
    longer path). Similarly, the path between any two undiscovered
    food pellets is less than or equal to the solution, since
    at best pacman will need to travel that length.

    It varies depending on circumstances which of these values is greater,
    so I take the maximum of the three as my heuristic. Since the food
    is organized in a grid, the most distant pellets will always be in
    opposite corners, which I got by looking at the first and last
    values of the food grid.
    """

    posList = state[1].asList()
    distList = []
    if len(posList) == 0:
        return 0
    else:
        distList.append(maze(posList[0], posList[len(posList) - 1]))
        distList.append(maze(state[0], posList[len(posList) - 1]))
        distList.append(maze(state[0], posList[0]))
        return max(distList)

class ClosestDotSearchAgent(SearchAgent):
    """
    Search for all food using a sequence of searches.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index)

    def registerInitialState(self, state):
        self._actions = []
        self._actionIndex = 0

        currentState = state

        while currentState.getFood().count() > 0:
            nextPathSegment = self.findPathToClosestDot(currentState)  # The missing piece
            self._actions += nextPathSegment

            for action in nextPathSegment:
                legal = currentState.getLegalActions()
                if action not in legal:
                    raise Exception('findPathToClosestDot returned an illegal move: %s!\n%s' %
                        (str(action), str(currentState)))

                currentState = currentState.generateSuccessor(0, action)

        logging.info('Path found with cost %d.' % len(self._actions))

    def findPathToClosestDot(self, gameState):
        """
        Returns a path (a list of actions) to the closest dot, starting from gameState.

        For every piece of food, this agent creates an instance of the AnyFoodSearchProblem
        and uses UCS to solve it for the closest piece each time. This continues until
        there is no more food remaining.
        """

        # Here are some useful elements of the startState
        # startPosition = gameState.getPacmanPosition()
        # food = gameState.getFood()
        # walls = gameState.getWalls()
        # problem = AnyFoodSearchProblem(gameState)

        problem = AnyFoodSearchProblem(gameState = gameState, start = gameState.getPacmanPosition())
        return uniformCostSearch(problem)

class AnyFoodSearchProblem(PositionSearchProblem):
    """
    This problem is to find an undiscovered piece of food. Any piece.

    States are represented as tuples containing pacman's x-y coordinates.

    The problem stores an unchanging list of all original food coordinates as 'food'.
    """

    def __init__(self, gameState, start = None):
        super().__init__(gameState, goal = None, start = start)

        # Store the food for later reference.
        self.food = gameState.getFood().asList()

    def isGoal(self, state):
        """
        Returns true if pacman's current position matches one of the food coordinates
        stored in the problem's food list.
        """
        if state not in self.food:
            return False

        # Register the locations we have visited.
        # This allows the GUI to highlight them.
        self._visitedLocations.add(state[0])
        self._visitHistory.append(state[0])
        return True

class ApproximateSearchAgent(BaseAgent):
    """
    Implement your contest entry here.

    Additional methods to implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Get a `pacai.bin.pacman.PacmanGameState`
    and return a `pacai.core.directions.Directions`.

    `pacai.agents.base.BaseAgent.registerInitialState`:
    This method is called before any moves are made.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index)

    def registerInitialState(self, state):
        self._actions = []
        self._actionIndex = 0

        currentState = state

        while currentState.getFood().count() > 0:
            nextPathSegment = self.getAction(currentState)  # The missing piece
            self._actions += nextPathSegment

            for action in nextPathSegment:
                legal = currentState.getLegalActions()
                if action not in legal:
                    raise Exception('getAction returned an illegal move: %s!\n%s' %
                        (str(action), str(currentState)))

                currentState = currentState.generateSuccessor(0, action)

    def getAction(self, gameState):
        problem = AnyPositionProblem(gameState = gameState, start = gameState.getPacmanPosition())
        return aStarSearch(problem, distance.manhattan)

class AnyPositionProblem(PositionSearchProblem):
    """
    A `pacai.core.search.problem.SearchProblem` for finding a specific location on the board.
    The state space consists of (x, y) positions.

    Note that this search problem is fully specified and should be used as an example.
    """
    def __init__(self, gameState, start = None):
        super().__init__(gameState, goal = None, start = start)

        # Store the food for later reference.
        self.startState = start

    def isGoal(self, state):
        if (state == self.startState):
            return False

        # Register the locations we have visited.
        # This allows the GUI to highlight them.
        self._visitedLocations.add(state)
        self._visitHistory.append(state)

        return True

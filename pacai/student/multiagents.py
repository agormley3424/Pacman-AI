import random

from pacai.agents.base import BaseAgent
from pacai.agents.search.multiagent import MultiAgentSearchAgent
from pacai.core.distance import manhattan


class ReflexAgent(BaseAgent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.
    You are welcome to change it in any way you see fit,
    so long as you don't touch the method headers.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index)
        self.cache = []

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        `ReflexAgent.getAction` chooses among the best options according to the evaluation function.

        Just like in the previous project, this method takes a
        `pacai.core.gamestate.AbstractGameState` and returns some value from
        `pacai.core.directions.Directions`.
        """

        # Collect legal moves.
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions.
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best.
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Takes as input a pacman game state and an action string.

        A function that evaluates the relative quality of an action given a game state.
        Adds the weighted inverse of the minimum distance from pacman to the nearest pellet,
        the inverse minimum distance between pacman and scared ghosts (weighted for
        how soon until they stop being scared), the difference in score between the current
        and successor state, and the inverse distance between pacman and the nearest capsule.

        Returns a number corresponding to the quality of the action.
        """

        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPosition = successorGameState.getPacmanPosition()
        oldFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        capsules = currentGameState.getCapsules()

        scaredyCats = []
        for g in newGhostStates:
            if g.isScared():
                scaredyCats.append(g)

        braveyCats = []
        for g in newGhostStates:
            if not g.isScared():
                braveyCats.append(g)

        foodDist = 10 / self.minMat(newPosition, oldFood.asList())
        scaredDist = 2 * self.minScared(newPosition, scaredyCats)
        scoreDif = 3 * (successorGameState.getScore() - currentGameState.getScore())
        ghostDist = self.minGhost(newPosition, braveyCats)
        capDist = self.minMat(newPosition, capsules) / 10

        return foodDist + scaredDist + scoreDif + ghostDist + capDist

    def minMat(self, pacPos, list):
        minDist = 1000
        for e in list:
            manHat = manhattan(pacPos, e)
            if manHat < minDist:
                minDist = manHat
        if minDist == 0:
            return 0.1
        return minDist

    def minGhost(self, pacPos, list):
        minDist = 1000
        for e in list:
            manHat = manhattan(pacPos, e.getPosition())
            if manHat < minDist:
                minDist = manHat
        return minDist

    def minScared(self, pacPos, list):
        minDist = 100
        flag = False
        for e in list:
            manHat = manhattan(pacPos, e.getPosition())
            if manHat < minDist:
                minDist = manHat
                minGhost = e
                flag = True
        if not flag:
            return 0
        if minDist == 0:
            return 100
        minDist = minGhost.getScaredTimer() / minDist
        return minDist


class MinimaxAgent(MultiAgentSearchAgent):
    """
    A class containing an algorithm that returns the optimal action for pacman to take in a
    given state, assuming rational ghosts. Values are recursively calculated down the search tree
    until a terminal state is reached or maximum depth is reached.

    At every terminal / max depth node, the quality of the state is
    evaluated. Its score is then propagated back through the tree, until an optimal successor
    state and corresponding action can be identified.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
        self.eval = self.getEvaluationFunction()

    def getAction(self, gameState):
        """
        Input: A pacman game state.

        Output: An optimal action for pacman to make.
        """
        return self.minimaxDecision(gameState)

    def minimaxDecision(self, state):
        """
        Input: A pacman game state

        Returns the optimal action for pacman to take by calling the appropriate successor
        evaluator function on its successors for all actions, which recursively finds scores from
        deeper in the game tree and propagates them backwards.
        """
        if state.isOver():
            return 'Stop'
        maxScore = float("-inf")
        ply = 1
        level = 0
        bestAction = state.getLegalActions(0)[0]
        for a in state.getLegalActions(0):
            if level == state.getNumAgents() - 1:
                result = self.maxValue(state.generateSuccessor(0, a), ply + 1, 0)
            else:
                result = self.minValue(state.generateSuccessor(0, a), ply, level + 1)
            if result > maxScore:
                maxScore = result
                bestAction = a
        return bestAction

    def maxValue(self, state, ply, level):
        """
        Input: A pacman game state, a tree ply integer, and a level integer.

        Recursively calculates the minimax value of a pacman game state.

        Output: The value of the state.
        """
        if ply == self.getTreeDepth() and level == state.getNumAgents() - 1 or state.isOver():
            return self.eval(state)
        v = float("-inf")
        for a in state.getLegalActions(level):
            if a == 'Stop':
                continue
            if level == state.getNumAgents() - 1:
                v = max(v, self.maxValue(state.generateSuccessor(level, a), ply + 1, 0))
            else:
                v = max(v, self.minValue(state.generateSuccessor(level, a), ply, level + 1))
        return v

    def minValue(self, state, ply, level):
        """
        Input: A pacman game state, a tree ply integer, and a level integer.

        Recursively calculates the minimax value of a ghost game state.

        Output: The value of the state.
        """
        if ply == self.getTreeDepth() and level == state.getNumAgents() - 1 or state.isOver():
            return self.eval(state)
        v = float("inf")
        for a in state.getLegalActions(level):
            if a == 'Stop':
                continue
            if level == state.getNumAgents() - 1:
                v = min(v, self.maxValue(state.generateSuccessor(level, a), ply + 1, 0))
            else:
                v = min(v, self.minValue(state.generateSuccessor(level, a), ply, level + 1))
        return v


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    A class containing an algorithm that returns the optimal action for pacman to take in
    a given state, assuming rational ghosts. Values are recursively calculated down the search tree
    until a terminal state is reached or maximum depth is reached.

    At every terminal / max depth node, the quality of the state is evaluated.
    Its score is then propagated back through the tree, until an optimal successor state and
    corresponding action can be identified.

    This differs fron minimax by pruning nodes of the tree which are unnecessary to explore.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
        self.eval = self.getEvaluationFunction()

    def getAction(self, gameState):
        """
        Input: A pacman game state.

        Output: An optimal action for pacman to make.
        """
        return self.alphaBeta(gameState)

    def alphaBeta(self, state):
        """
        Input: A pacman game state

        Returns the optimal action for pacman to take by calling the appropriate successor
        evaluator function on its successors for all actions, which recursively finds scores from
        deeper in the game tree and propagates them backwards.
        """
        maxScore = float("-inf")
        ply = 1
        level = 0
        bestAction = state.getLegalActions(0)[0]
        for a in state.getLegalActions(0):
            result = self.minValue(state.generateSuccessor(0, a), ply, level + 1, float("-inf"),
                                   float("inf"))
            if result > maxScore:
                maxScore = result
                bestAction = a
        return bestAction

    def maxValue(self, state, ply, level, alpha, beta):
        """
        Input: A pacman game state, a tree ply integer, and a level integer.

        Recursively calculates the minimax value of a pacman game state.

        Output: The value of the state.
        """
        if ply == self.getTreeDepth() and level == state.getNumAgents() - 1 or state.isOver():
            return self.eval(state)
        v = float("-inf")
        for a in state.getLegalActions(level):
            if a == 'Stop':
                continue
            v = max(v, self.minValue(state.generateSuccessor(level, a), ply, level + 1, alpha,
                                     beta))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def minValue(self, state, ply, level, alpha, beta):
        """
        Input: A pacman game state, a tree ply integer, and a level integer.

        Recursively calculates the minimax value of a ghost game state.

        Output: The value of the state.
        """
        if ply == self.getTreeDepth() and level == state.getNumAgents() - 1 or state.isOver():
            return self.eval(state)
        v = float("inf")
        # If on the final level of the ply
        if level == state.getNumAgents() - 1:
            for a in state.getLegalActions(level):
                if a == 'Stop':
                    continue
                v = min(v, self.maxValue(state.generateSuccessor(level, a), ply + 1, 0, alpha,
                                         beta))
                if v <= alpha:
                    return v
                beta = min(beta, v)
        else:
            for a in state.getLegalActions(level):
                if a == 'Stop':
                    continue
                v = min(v, self.minValue(state.generateSuccessor(level, a), ply, level + 1, alpha,
                                         beta))
                if v <= alpha:
                    return v
                beta = min(beta, v)

        return v


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    A class containing an algorithm that returns the optimal action for pacman to take in a
    given state, assuming random ghosts.

    Values are recursively calculated down the search tree until a terminal state is reached or
    maximum depth is reached. At every terminal / max depth node, the quality of the state is
    evaluated.

    Its score is then propagated back through the tree, until an optimal successor state and
    corresponding action can be identified.

    This differs fron minimax by treating ghosts as irrational, randomly acting agents.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
        self.eval = self.getEvaluationFunction()

    def getAction(self, gameState):
        """
        Input: A pacman game state.

        Output: An optimal action for pacman to make.
        """
        return self.expectiMax(gameState)

    def expectiMax(self, state):
        """
        Input: A pacman game state

        Returns the optimal action for pacman to take by calling the appropriate successor
        evaluator function on its successors for all actions, which recursively finds scores from
        deeper in the game tree and propagates them backwards.
        """
        maxScore = float("-inf")
        ply = 1
        level = 0
        bestAction = state.getLegalActions(0)[0]
        for a in state.getLegalActions(0):
            result = self.minValue(state.generateSuccessor(0, a), ply, level + 1)
            if result > maxScore:
                maxScore = result
                bestAction = a
        return bestAction

    def maxValue(self, state, ply, level):
        """
        Input: A pacman game state, a tree ply integer, and a level integer.

        Recursively calculates the expected minimax value of a pacman game state.

        Output: The value of the state.
        """
        if ply == self.getTreeDepth() and level == state.getNumAgents() - 1 or state.isOver():
            return self.eval(state)
        v = float("-inf")
        for a in state.getLegalActions(level):
            if a == 'Stop':
                continue
            v = max(v, self.minValue(state.generateSuccessor(level, a), ply, level + 1))
        return v

    def minValue(self, state, ply, level):
        """
        Input: A pacman game state, a tree ply integer, and a level integer.

        Recursively calculates the expected minimax value of a ghost game state.

        Output: The value of the state.
        """
        if ply == self.getTreeDepth() and level == state.getNumAgents() - 1 or state.isOver():
            return self.eval(state)
        v = 0
        probability = 1 / len(state.getLegalActions(level))
        # If on the final level of the ply
        if level == state.getNumAgents() - 1:
            for a in state.getLegalActions(level):
                if a == 'Stop':
                    continue
                v += self.maxValue(state.generateSuccessor(level, a), ply + 1, 0) * probability
        else:
            for a in state.getLegalActions(level):
                if a == 'Stop':
                    continue
                v += self.minValue(state.generateSuccessor(level, a), ply, level + 1) * probability
        return v


def betterEvaluationFunction(currentGameState):
    """
    Takes as input a pacman game state.

    This function evaluates the relative quality of a a game state.

    It adds the score of the game, the inverse minimum distance of pacman to the nearest pellet,
    and the inverse minimum distance between pacman and the scared ghosts, weighted by their
    timers.

    Returns a number corresponding to the quality of the state.
    """

    def minMat(pacPos, List):
        minDist = 1000
        for e in List:
            manHat = manhattan(pacPos, e)
            if manHat < minDist:
                minDist = manHat
        if minDist == 0:
            return 0.1
        return minDist

    pacPos = currentGameState.getPacmanPosition()
    ghosts = currentGameState.getGhostStates()

    scaredyCats = []
    for g in ghosts:
        if g.isScared():
            scaredyCats.append(g)

    def minScared(pacPos, list):
        minDist = 100
        flag = False
        for e in list:
            manHat = manhattan(pacPos, e.getPosition())
            if manHat < minDist:
                minDist = manHat
                minGhost = e
                flag = True
        if not flag:
            return 0
        if minDist == 0:
            return 100
        minDist = minGhost.getScaredTimer() / minDist
        return minDist

    score = currentGameState.getScore()
    foodDist = 3 / minMat(pacPos, currentGameState.getFood().asList())
    scaredDist = minScared(pacPos, scaredyCats)
    return score + foodDist + scaredDist


class ContestAgent(MultiAgentSearchAgent):
    """
    Your agent for the mini-contest.

    You can use any method you want and search to any depth you want.
    Just remember that the mini-contest is timed, so you have to trade off speed and computation.

    Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
    just make a beeline straight towards Pacman (or away if they're scared!)

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`
    """

    def __init__(self, index, **kwargs):
        super().__init__(index)

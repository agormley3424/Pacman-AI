"""
In this file, you will implement generic search algorithms which are called by Pacman agents.
"""

from pacai.util import stack
from pacai.util import queue
from pacai.util import priorityQueue

def depthFirstSearch(problem):
    """
    Returns a list of actions which a search agent uses to solve
    the specified problem.

    This implementation of DFS stores newly discovered
    states on a stack fringe, where the most recent additions
    are first to be expanded. Thus, it quickly goes to the
    deepest level of the search tree. Each node is bundled with an action,
    a cost from the previous state, and a sequence of actions representing
    the path to reach each node from the source.
    """

    fringe = stack.Stack()
    explored = [problem.startingState()]

    fringe.push((explored[0], 0, 0, ""))
    while not fringe.isEmpty():
        parent_node = fringe.pop()
        if problem.isGoal(parent_node[0]):
            return list(parent_node[3:])
        child_nodes = problem.successorStates(parent_node[0])
        for child in child_nodes:
            if child[0] not in explored:
                # If parent node isn't source
                if parent_node[0] != problem.startingState():
                    # Add action list of parent
                    child += parent_node[3:]
                # Add child's new action to list
                child += (child[1],)
                fringe.push(child)
                explored.append(child[0])
    return None

def breadthFirstSearch(problem):
    """
    Returns a list of actions which a search agent uses to solve
    the specified problem.

    This implementation of BFS stores newly discovered
    states on a queue fringe, where nodes are expanded in the order
    in which they were added. Thus, it thoroughly explores each level
    of the search tree before diving down. Each node is bundled with an action,
    a cost from the previous state, and a sequence of actions representing
    the path to reach each node from the source.
    """

    fringe = queue.Queue()
    explored = [problem.startingState()]

    fringe.push((explored[0], 0, 0, ""))
    while not fringe.isEmpty():
        parent_node = fringe.pop()
        if problem.isGoal(parent_node[0]):
            return list(parent_node[3:])
        child_nodes = problem.successorStates(parent_node[0])
        for child in child_nodes:
            if child[0] not in explored:
                # If parent node isn't source...
                if parent_node[0] != problem.startingState():
                    # Add action list of parent.
                    child += parent_node[3:]
                # Add child's new action to list.
                child += (child[1],)
                fringe.push(child)
                explored.append(child[0])
    return None

def uniformCostSearch(problem):
    """
    Returns a list of actions which a search agent uses to solve
    the specified problem.

    This implementation of UCS stores newly discovered
    states on a priority queue fringe. The priority of each state
    is equal to the total cost of reaching it from the source.
    Thus, it explores paths in increasing order of continuing expenses.
    Each node is bundled with an action, a cost from the previous state,
    a running cost from the source, and a sequence of actions representing
    the path to reach each node from the source.
    """

    fringe = priorityQueue.PriorityQueue()
    explored = [problem.startingState()]

    fringe.push((explored[0], "Stop", 0, [0], "Stop"), 0)
    while not fringe.isEmpty():
        parent_node = fringe.pop()
        if problem.isGoal(parent_node[0]):
            return list(parent_node[4:])
        child_nodes = problem.successorStates(parent_node[0])
        for child in child_nodes:
            if child[0] not in explored:
                child += ([child[2]],)
                # If parent node isn't source...
                if parent_node[0] != problem.startingState():
                    # Add cost of parent node.
                    child[3][0] += parent_node[3][0]
                    # Add action list of parent
                    child += parent_node[4:]
                # Add child's new action to list
                child += (child[1],)
                fringe.push(child, child[3][0])
                explored.append(child[0])
    return None

def aStarSearch(problem, heuristic):
    """
    Returns a list of actions which a search agent uses to solve
    the specified problem.

    This implementation of A* stores newly discovered
    states on a priority queue fringe. The priority of each state
    is equal to the total cost of reaching it from the source plus a
    heuristic estimate of how close the state is to a goal state.
    Thus, it explores paths in increasing order of estimated expenses.
    Each node is bundled with an action, a cost from the previous state,
    a running cost from the source, and a sequence of actions representing
    the path to reach each node from the source.
    """

    fringe = priorityQueue.PriorityQueue()
    explored = [problem.startingState()]

    fringe.push((explored[0], "Stop", 0, [0], "Stop"), 0)
    while not fringe.isEmpty():
        parent_node = fringe.pop()
        if problem.isGoal(parent_node[0]):
            return list(parent_node[4:])
        child_nodes = problem.successorStates(parent_node[0])
        for child in child_nodes:
            if child[0] not in explored:
                child += ([child[2]],)
                # If parent node isn't source...
                if parent_node[0] != problem.startingState():
                    # Add cost of parent node.
                    child[3][0] += parent_node[3][0]
                    # Add action list of parent.
                    child += parent_node[4:]
                # Add child's new action to list
                child += (child[1],)
                fringe.push(child, child[3][0] + heuristic(child[0], problem))
                explored.append(child[0])
    return None

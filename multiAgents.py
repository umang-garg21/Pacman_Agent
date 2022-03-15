# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """

        print(" >>>>>>>>>>>>>>>>>>>>> PACMAN CHOOSING MOVE <<<<<<<<<<<<<<<<<<<<<<<<")
        print("The number of active ghosts in the environment are:", gameState.getNumGhosts())
        # Collect legal moves and successor states
        legalpacmanMoves = gameState.getLegalActions(self.index)

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalpacmanMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"
        print("PACMAN chose", legalpacmanMoves[chosenIndex])
        return legalpacmanMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """

        print("---------", action, "----------")
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(self.index, action)

        """
        Get the possible ghost moves
        """
        legalghostMoves = []
        for i in range(currentGameState.getNumGhosts()):
            legalghostMoves.append(currentGameState.getLegalActions(i + currentGameState.getNumPacman()))
        print("Legal ghost moves are:", legalghostMoves)

        """
        Get the new position of the pacman
        """
        currentPacmanPos = currentGameState.getPacmanPosition(self.index)
        newPos = successorGameState.getPacmanPosition(self.index)
        # print(newPos)

        """
         Get the legal possible ghost positions for all ghosts
        """
        worst_new_distances_to_ghosts = []
        new_Possible_ghosts_positions = []
        for i in range(currentGameState.getNumGhosts()):
            new_Possible_ghosts_positions.append([])
            for legalghostactions in legalghostMoves[i]:
                # print(legalghostactions)
                successorghostStates = currentGameState.generateSuccessor(i + currentGameState.getNumPacman(),
                                                                          legalghostactions)
                new_Possible_ghosts_positions[i].append(
                    successorghostStates.getGhostPosition(i + currentGameState.getNumPacman()))
            #  print("Possible new position of ghost number", i + 1, "are", new_Possible_ghosts_positions[i])
            worst_new_distances_to_ghosts.append(util.manhattanDistance(newPos, min(new_Possible_ghosts_positions[i])))
        print("Worst new distances to the ghost are", worst_new_distances_to_ghosts)

        ghostDistances = []
        currentghostPos = successorGameState.getGhostPositions()
        # print("The current ghost position is", currentghostPos)

        newFood = successorGameState.getFood()
        # print(newFood.asList())

        newGhostStates = successorGameState.getGhostStates()

        # Iterating over all available food list locations in the environment

        fooddist_temp = []
        if len(newFood.asList()):
            for food in newFood.asList():
                fooddist_temp.append(util.manhattanDistance(newPos, food))

        sorted_food_dist = sorted(fooddist_temp)
        # print("The sorted food distance list is", sorted_food_dist)

        if fooddist_temp:
            closest_food_distance = min(fooddist_temp)
        else:
            closest_food_distance = 1

        # print("Closest food distance is at", closest_food_distance)
        currentghostdist = []
        for ghostposition in currentghostPos:
            currentghostdist.append(util.manhattanDistance(newPos, ghostposition))
        # print("Ghosts are currently at following distances from Pacman", currentghostdist)

        futureghostdist = []
        for ghostposition in currentghostPos:
            futureghostdist.append(util.manhattanDistance(newPos, ghostposition))

        """
         Check if ghost(s) are near which scales with number of ghosts
        """
        near_ghosts = 0
        for ghost_distance_item in currentghostdist:
            if ghost_distance_item <= 2 + currentGameState.getNumGhosts():
                ghost_near = True
                break
            else:
                ghost_near = False

        if ghost_near:
            print("Ghosts are near")
        elif not ghost_near:
            print("Ghosts are far")

        for ghost_distance_item in currentghostdist:
            if ghost_distance_item <= 1 + currentGameState.getNumGhosts():
                near_ghosts += 1

        """
         Find capsule locations and target them first
        """
        capsule_locations = currentGameState.getCapsules()
        # print("Capsules are at", capsule_locations)

        closest_capsule_distance = 0
        capsuledist = []
        if capsule_locations:
            for locations in capsule_locations:
                capsuledist.append(util.manhattanDistance(newPos, locations))
            closest_capsule_distance = min(capsuledist)

        print("The closest capsule will be", closest_capsule_distance, "units far")

        ######################################################
        #  Score evaluation starts  #
        ######################################################
        eval_score = 0

        flat_list = [item for sublist in new_Possible_ghosts_positions for item in sublist]

        """
         check possibility of pacman and ghost collision in succesive step
        """

        for i in range(currentGameState.getNumGhosts()):
            for location in new_Possible_ghosts_positions[i]:
                # print("location variable value is ", location)
                if newPos == location:
                    print("Lethal Move")
                    eval_score = eval_score - (
                            1000000 / len(new_Possible_ghosts_positions[i]))  # More options to ghost -- less threat

        """
         check if ghosts are near, and take measures w.r.t how many are near.
        """

        if capsule_locations:
            eval_score = eval_score + 50 * (1 / (closest_capsule_distance + 1))

        if ghost_near:
            for i in range(len(fooddist_temp)):
                eval_score = eval_score + 2 * float(1 / fooddist_temp[i]) - float(
                    10 * near_ghosts / sum(worst_new_distances_to_ghosts)) - len(newFood.asList())

        else:
            for i in range(len(fooddist_temp)):
                if len(fooddist_temp) >= 3:
                    eval_score = eval_score + 2 * float(1 / closest_food_distance) + 1 * float(
                        1 / sorted_food_dist[1]) + 0.5 * float(1 / sorted_food_dist[2]) - len(newFood.asList())

                elif len(fooddist_temp) >= 2:
                    eval_score = eval_score + 2 * float(1 / closest_food_distance) + 1 * float(
                        1 / sorted_food_dist[1]) - len(newFood.asList())

                else:
                    eval_score = eval_score + 2 * float(1 / closest_food_distance) - len(newFood.asList())

        if action == 'Stop':
            if ghost_near:
                eval_score = eval_score - 5
            else:
                eval_score = eval_score - 50

        print("Here")
        walls = currentGameState.getWalls()
        print(walls)
        print(currentGameState.getPacmanPosition(self.index))
        print("Working")

        print("Evaluation score for action", action, "is", eval_score)
        return successorGameState.getScore()[self.index] + eval_score


def scoreEvaluationFunction(currentGameState, index):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """

    """
    Get the new position of the pacman
    """
    # currentPacmanPos = currentGameState.getPacmanPosition(0)
    # # print("The current pacman position in the terminal states is", currentPacmanPos)
    #
    # """
    #     Get the legal possible ghost positions for all ghosts
    # """
    #
    # ghosts_positions = []
    # for i in range(currentGameState.getNumGhosts()):
    #     ghosts_positions.append([])
    #
    # ghostDistances = []
    # currentghostPos = currentGameState.getGhostPositions()
    # # print("The ghost positions under the score evaluations function are", currentghostPos)
    #
    # Food = currentGameState.getFood()
    # # print(Food.asList())
    #
    # # Iterating over all food list locations in the environment
    #
    # fooddist_temp = []
    # if len(Food.asList()):
    #     for food in Food.asList():
    #         fooddist_temp.append(util.manhattanDistance(currentPacmanPos, food))
    #
    # sorted_food_dist = sorted(fooddist_temp)
    # # print("The sorted food distance list is", sorted_food_dist)
    #
    # if fooddist_temp:
    #     closest_food_distance = min(fooddist_temp)
    # else:
    #     closest_food_distance = 1
    #
    # # print("Closest food distance is at", closest_food_distance)
    # currentghostdist = []
    # for ghostposition in currentghostPos:
    #     currentghostdist.append(util.manhattanDistance(currentPacmanPos, ghostposition))
    # # print("Ghosts are currently at following distances from Pacman", currentghostdist)
    #
    # """
    #  Check if ghost(s) are near which scales with number of ghosts
    # """
    # near_ghosts = 0
    # for ghost_distance_item in currentghostdist:
    #     if ghost_distance_item <= 5 + currentGameState.getNumGhosts():
    #         ghost_near = True
    #         break
    #     else:
    #         ghost_near = False
    #
    # # if ghost_near:
    # # print("Ghosts are near")
    # # elif not ghost_near:
    # # print("Ghosts are far")
    #
    # for ghost_distance_item in currentghostdist:
    #     if ghost_distance_item <= 3 + currentGameState.getNumGhosts():
    #         near_ghosts += 1
    #
    # """
    #  Find capsule locations and target them first
    # """
    # capsule_locations = currentGameState.getCapsules()
    # # print("Capsules are at", capsule_locations)
    #
    # closest_capsule_distance = 0
    # capsuledist = []
    # if capsule_locations:
    #     for locations in capsule_locations:
    #         capsuledist.append(util.manhattanDistance(currentPacmanPos, locations))
    #     closest_capsule_distance = min(capsuledist)
    #
    # # print("The closest capsule will be", closest_capsule_distance, "units far")
    #
    # ######################################################
    # #  Score evaluation starts  #
    # ######################################################
    # eval_score = 0
    #
    # """
    #  check possibility of pacman and ghost collision
    # """
    #
    # if currentGameState.isLose():
    #     return -100000
    # elif currentGameState.isWin():
    #     return 100000
    #
    # """
    #  check if ghosts are near, and take measures w.r.t how many are near.
    # """
    #
    # if capsule_locations:
    #     eval_score = eval_score + 500 * (1 / (closest_capsule_distance + 1))
    #
    #
    # if ghost_near:
    #     if capsule_locations:
    #         eval_score = eval_score + 2*len(Food.asList())
    #     else:
    #          eval_score = eval_score - 2*len(Food.asList())
    #     for i in range(len(fooddist_temp)):
    #         if capsule_locations:
    #             eval_score = eval_score + 10 / (sorted_food_dist[i])
    #         else:
    #             eval_score = eval_score + 10 / (float(sorted_food_dist[i]) ** (1 - 1/(1+len(fooddist_temp))))
    #
    #     # if len(fooddist_temp) >= 3:
    #     #         eval_score = eval_score + 2 * float(1 / closest_food_distance) + 1 * float(
    #     #             1 / sorted_food_dist[1]) + 0.5 * float(1 / sorted_food_dist[2]) - len(Food.asList())
    #     #
    #     # elif len(fooddist_temp) >= 2:
    #     #         eval_score = eval_score + 2 * float(1 / closest_food_distance) + 1 * float(
    #     #                 1 / sorted_food_dist[1]) - len(Food.asList())
    #     #
    #     # else:
    #     #         eval_score = eval_score + 2 * float(1 / closest_food_distance) - len(Food.asList())
    #
    #     for ghost_distance_item in currentghostdist:
    #         eval_score = eval_score - float(15*(1.5-(1/len(Food.asList()))) * near_ghosts / ghost_distance_item)
    #
    # else:
    #     if capsule_locations:
    #         eval_score = eval_score + 2*len(Food.asList())
    #     else:
    #          eval_score = eval_score - 5*len(Food.asList())
    #     for i in range(len(fooddist_temp)):
    #         if capsule_locations:
    #             eval_score = eval_score + 1 / (sorted_food_dist[i])
    #         else:
    #             eval_score = eval_score + 100 / (float(sorted_food_dist[i]) ** (1 - 1/(1+len(fooddist_temp))))
    #
    #     for ghost_distance_item in currentghostdist:
    #         eval_score = eval_score - float(10*(1.5-(1/len(Food.asList())) / ghost_distance_item))
    #
    # # Account of location of pacman w.r.t center of maze: Pacman tries to not get cornered, but have stochasticity for evasion from center
    #
    # interghostdist =0
    # # Together ghosts are better
    # for ghostposition in currentghostPos:
    #     for otherghostspositions in currentghostPos:
    #         interghostdist += manhattanDistance(otherghostspositions, ghostposition)
    # interghostdist = interghostdist/2       # Doublecounting remove
    #
    # if sum(currentghostdist) > 3 and sum(currentghostdist)< 15 :
    #     eval_score = eval_score - (5 + 5/sum(currentghostdist))*(currentGameState.getNumGhosts())*(interghostdist)
    #
    # # print("The pacman position is", currentGameState.getPacmanPosition(0))
    #
    # if len(fooddist_temp) <=2:
    #     if not capsule_locations:
    #         eval_score = eval_score - 5*closest_food_distance
    #
    #
    # return currentGameState.getScore()[index] + eval_score

    """
    Get the new position of the pacman
    """
    currentPacmanPos = currentGameState.getPacmanPosition(0)
    # print("The current pacman position in the terminal states is", currentPacmanPos)

    """
        Get the legal possible ghost positions for all ghosts
    """

    ghosts_positions = []
    for i in range(currentGameState.getNumGhosts()):
        ghosts_positions.append([])

    ghostDistances = []
    currentghostPos = currentGameState.getGhostPositions()
    # print("The ghost positions under the score evaluations function are", currentghostPos)

    Food = currentGameState.getFood()
    # print(Food.asList())

    # Iterating over all food list locations in the environment

    fooddist_temp = []
    if len(Food.asList()):
        for food in Food.asList():
            fooddist_temp.append(util.manhattanDistance(currentPacmanPos, food))

    sorted_food_dist = sorted(fooddist_temp)
    # print("The sorted food distance list is", sorted_food_dist)

    if fooddist_temp:
        closest_food_distance = min(fooddist_temp)
    else:
        closest_food_distance = 1

    # print("Closest food distance is at", closest_food_distance)
    currentghostdist = []
    for ghostposition in currentghostPos:
        currentghostdist.append(util.manhattanDistance(currentPacmanPos, ghostposition))
    # print("Ghosts are currently at following distances from Pacman", currentghostdist)

    """
     Check if ghost(s) are near which scales with number of ghosts
    """
    near_ghosts = 0
    for ghost_distance_item in currentghostdist:
        if ghost_distance_item <= 3 + currentGameState.getNumGhosts():
            ghost_near = True
            break
        else:
            ghost_near = False

    # if ghost_near:
    # print("Ghosts are near")
    # elif not ghost_near:
    # print("Ghosts are far")

    for ghost_distance_item in currentghostdist:
        if ghost_distance_item <= 1 + currentGameState.getNumGhosts():
            near_ghosts += 1

    """
     Find capsule locations and target them first
    """
    capsule_locations = currentGameState.getCapsules()
    # print("Capsules are at", capsule_locations)

    closest_capsule_distance = 0
    capsuledist = []
    if capsule_locations:
        for locations in capsule_locations:
            capsuledist.append(util.manhattanDistance(currentPacmanPos, locations))
        closest_capsule_distance = min(capsuledist)

    # print("The closest capsule will be", closest_capsule_distance, "units far")

    ######################################################
    #  Score evaluation starts  #
    ######################################################
    eval_score = 0

    """
     check possibility of pacman and ghost collision 
    """

    if currentGameState.isLose():
        return -100000
    elif currentGameState.isWin():
        return 100000

    """
     check if ghosts are near, and take measures w.r.t how many are near.
    """

    if capsule_locations:
        eval_score = eval_score + 500 * (1 / (closest_capsule_distance + 1))

    if ghost_near:
        eval_score = eval_score - len(Food.asList())
        for i in range(len(fooddist_temp)):
            if capsule_locations:
                eval_score = eval_score + 2 / (sorted_food_dist[i])
            else:
                eval_score = eval_score + 2 / (float(sorted_food_dist[i]) ** (1 - 1/(1+len(fooddist_temp))))

        # if len(fooddist_temp) >= 3:
        #         eval_score = eval_score + 2 * float(1 / closest_food_distance) + 1 * float(
        #             1 / sorted_food_dist[1]) + 0.5 * float(1 / sorted_food_dist[2]) - len(Food.asList())
        #
        # elif len(fooddist_temp) >= 2:
        #         eval_score = eval_score + 2 * float(1 / closest_food_distance) + 1 * float(
        #                 1 / sorted_food_dist[1]) - len(Food.asList())
        #
        # else:
        #         eval_score = eval_score + 2 * float(1 / closest_food_distance) - len(Food.asList())

        for ghost_distance_item in currentghostdist:
            eval_score = eval_score - float(50 * near_ghosts / ghost_distance_item)

    else:

        eval_score = eval_score - len(Food.asList())
        for i in range(len(fooddist_temp)):
            if capsule_locations:
                eval_score = eval_score + 2 / (sorted_food_dist[i])
            else:
                eval_score = eval_score + 2 / (float(sorted_food_dist[i]) ** (0.5))

        for ghost_distance_item in currentghostdist:
            eval_score = eval_score - float(50 / ghost_distance_item)

    # Account of location of pacman w.r.t center of maze: Pacman tries to not get cornered, but have stochasticity for evasion from center

    interghostdist =0
    # Together ghosts are better
    for ghostposition in currentghostPos:
        for otherghostspositions in currentghostPos:
            interghostdist += manhattanDistance(otherghostspositions, ghostposition)
    interghostdist = interghostdist/2       # Doublecounting remove

    eval_score = eval_score - 2*(interghostdist)

    # print("Evaluation score is", eval_score)
    walls = currentGameState.getWalls()
    # print(walls)
    # print("Here")

    # print("The pacman position is", currentGameState.getPacmanPosition(0))

    interghostdist =0
    # Together ghosts are better
    for ghostposition in currentghostPos:
        for otherghostspositions in currentghostPos:
            interghostdist += manhattanDistance(otherghostspositions, ghostposition)
    interghostdist = interghostdist/2       # Doublecounting remove

    if sum(currentghostdist) >= 4 and sum(currentghostdist) <= 16 :
        eval_score = eval_score - (0 + 0/sum(currentghostdist))*(currentGameState.getNumGhosts())*(interghostdist)

    return currentGameState.getScore()[index] + eval_score

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent & AlphaBetaPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, index=0, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = index  # Pacman is always agent index 0
        self.evaluationFunction = lambda state: util.lookup(evalFn, globals())(state, self.index)
        self.depth = int(depth)


class MultiPacmanAgent(MultiAgentSearchAgent):
    """
      Expectimax agent
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.
          Here are some method calls that might be useful when implementing minimax.
          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1
          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action
          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        # print(" MINIMAX AGENT ON WORK")

        # print(" >>>>>>>>>>>>>>>>>>>>> PACMAN CHOOSING MOVE <<<<<<<<<<<<<<<<<<<<<<<<")
        # print("The number of active ghosts in the environment are:", gameState.getNumGhosts())
        # Collect legal moves and successor states

        initialdepth = 0
        legalpacmanMoves = gameState.getLegalActions(self.index)

        score, bestaction = self.expectimax(gameState, 0, 0)

        return bestaction

    def expectimax(self, gameState, whoseturn, depth):

        # print(" Whoseturn =", whoseturn)
        currentdepth = depth

        # If fates are locked win/ lose
        if gameState.isLose() or gameState.isWin():
            return scoreEvaluationFunction(gameState, 0), ""

        # base case : targetDepth reached
        if self.depth == currentdepth:
            # print("Max depth reached. Depth = ", currentdepth)
            # print(" ########### Recursion starts now #################")
            return scoreEvaluationFunction(gameState, 0), ""  # Evaluation of pacman score at the target depth

        # No legal move available :: blocked
        if not gameState.getLegalActions(whoseturn):
            return scoreEvaluationFunction(gameState, 0), ""

        # Choose next agent
        if whoseturn < gameState.getNumAgents() - 1:
            nextagent = whoseturn + 1
        else:
            nextagent = 0

        # Increment depth condition, if all agents have moved
        if nextagent == 0:
            depth += 1
           # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^ The depth was increased to:", depth, "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

        ####### Recursive Conditions ########

        available_evals = []

        # print("All legal actions allowed for agent", whoseturn, "are:", gameState.getLegalActions(whoseturn) )
        # Check which agent's turn
        if whoseturn == 0:  # Pacman is choosing
            max_score = float("-inf")
            best_action = ""

            for legalaction in gameState.getLegalActions(whoseturn):

                # Selection probability of fully rational choice: 0.5 + 0.5/ len(gameState.getLegalActions(whoseturn))

                # print(" >>>>>>>>>>>>>>>>>>>>>>>>>>New Actions for pacman starting here <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                # print("The action under consideration for current agent", whoseturn, "is", legalaction)
                # Generate next state for different actions taken from the current state
                successorGameState = gameState.generateSuccessor(whoseturn, legalaction)
                # print("Successor state successfully generated for this action")
                # print("The next agent to decide move is", nextagent)
                score = self.expectimax(successorGameState, nextagent, depth)[0]

                # Penalize for stopping
                if legalaction == 'Stop':
                    score -= 100

                available_evals.append(score)

                # print("Score is", score)
                if float(score) >= max_score:
                    max_score = score
                    best_action = legalaction
                # print("The current Best Pacman action at depth", depth, "is:", best_action)

            # Return max score and best action after iterating over all legal actions
            # print("Available evaluations for following legalmoves", gameState.getLegalActions(whoseturn), "are", available_evals)
            # print("The final Best action for Pacman at depth", depth, "is:", best_action)
            return max_score, best_action

        else:

            min_score = float("inf")
            best_action = ""
            for legalghostaction in gameState.getLegalActions(whoseturn):

                # print(" >>>>>>>>>>>>>>>>>>>>>>>>>>>>New Action for ghost starting here <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                # print("The action under consideration for agent", whoseturn, "is", legalghostaction)
                successorghostState = gameState.generateSuccessor(whoseturn, legalghostaction)
                # print("Successor state successfully generated for this action")
                score = self.expectimax(successorghostState, nextagent, depth)[0]

                # print("score", score)
                if float(score) < min_score:
                    min_score = score
                    best_action = legalghostaction
                # print("The current best action chosen by ghost", whoseturn, "at depth", depth, "is:", best_action)
                available_evals.append(score)

            expectiscore = 0.5 * min_score + (0.5 / len(gameState.getLegalActions(whoseturn))) * sum(available_evals)
            # print("The expectimax score is:", expectiscore)
            # Return max score and best action after iterating over all legal actions
            # print("Available evaluations for following legalmoves", gameState.getLegalActions(whoseturn), "are", available_evals)
            # print("The final Best action for ghost", whoseturn, " selected at depth", depth, "is:", best_action)
            return expectiscore, best_action


class RandomAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        legalMoves = gameState.getLegalActions(self.index)
        return random.choice(legalMoves)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Minimax agent
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.
          Here are some method calls that might be useful when implementing minimax.
          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1
          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action
          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        # print(" MINIMAX AGENT ON WORK")

        # print(" >>>>>>>>>>>>>>>>>>>>> PACMAN CHOOSING MOVE <<<<<<<<<<<<<<<<<<<<<<<<")
        # print("The number of active ghosts in the environment are:", gameState.getNumGhosts())
        # Collect legal moves and successor states

        initialdepth = 0
        legalpacmanMoves = gameState.getLegalActions(self.index)

        score, bestaction = self.minimax(gameState, 0, 0)

        return bestaction

    def minimax(self, gameState, whoseturn, depth):

        print(" Whoseturn =", whoseturn)
        currentdepth = depth

        # If fates are locked win/ lose
        if gameState.isLose() or gameState.isWin():
            return scoreEvaluationFunction(gameState, 0), ""

        # base case : targetDepth reached
        if self.depth == currentdepth:
            print("Max depth reached. Depth = ", currentdepth)
            print(" ########### Recursion starts now #################")
            return scoreEvaluationFunction(gameState, 0), ""  # Evaluation of pacman score at the target depth

        # No legal move available :: blocked
        if not gameState.getLegalActions(whoseturn):
            return scoreEvaluationFunction(gameState, 0), ""

        # Choose next agent
        if whoseturn < gameState.getNumAgents() - 1:
            nextagent = whoseturn + 1
        else:
            nextagent = 0

        # Increment depth condition, if all agents have moved
        if nextagent == 0:
            depth += 1
            print("^^^^^^^^^^^^^^^^^^^^^^^^^^^ The depth was increased to:", depth, "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

        ####### Recursive Conditions ########

        available_evals = []

        print("All legal actions allowed for agent", whoseturn, "are:", gameState.getLegalActions(whoseturn))
        # Check which agent's turn
        if whoseturn == 0:  # Pacman is choosing
            max_score = float("-inf")
            best_action = ""

            for legalaction in gameState.getLegalActions(whoseturn):

                print(
                    " >>>>>>>>>>>>>>>>>>>>>>>>>>New Actions for pacman starting here <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                print("The action under consideration for current agent", whoseturn, "is", legalaction)
                # Generate next state for different actions taken from the current state
                successorGameState = gameState.generateSuccessor(whoseturn, legalaction)
                print("Successor state successfully generated for this action")
                print("The next agent to decide move is", nextagent)
                score = self.minimax(successorGameState, nextagent, depth)[0]

                # Penalize for stopping
                if legalaction == 'Stop':
                    score -= 1000

                available_evals.append(score)

                # print("Score is", score)
                if float(score) >= max_score:
                    max_score = score
                    best_action = legalaction
                print("The current Best Pacman action at depth", depth, "is:", best_action)

            # Return max score and best action after iterating over all legal actions
            print("Available evaluations for following legalmoves", gameState.getLegalActions(whoseturn), "are",
                  available_evals)
            print("The final Best action for Pacman at depth", depth, "is:", best_action)
            return max_score, best_action

        else:

            min_score = float("inf")
            best_action = ""
            for legalghostaction in gameState.getLegalActions(whoseturn):

                print(
                    " >>>>>>>>>>>>>>>>>>>>>>>>>>>>New Action for ghost starting here <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                print("The action under consideration for agent", whoseturn, "is", legalghostaction)
                successorghostState = gameState.generateSuccessor(whoseturn, legalghostaction)
                print("Successor state successfully generated for this action")
                score = self.minimax(successorghostState, nextagent, depth)[0]
                print("score", score)
                if float(score) < min_score:
                    min_score = score
                    best_action = legalghostaction
                print("The current best action chosen by ghost", whoseturn, "at depth", depth, "is:", best_action)

            # Return max score and best action after iterating over all legal actions
            print("The final Best action for ghost", whoseturn, " selected at depth", depth, "is:", best_action)
            return min_score, best_action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Expectimax agent
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.
          Here are some method calls that might be useful when implementing minimax.
          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1
          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action
          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        # print(" MINIMAX AGENT ON WORK")

        print(" >>>>>>>>>>>>>>>>>>>>> PACMAN CHOOSING MOVE <<<<<<<<<<<<<<<<<<<<<<<<")
        print("The number of active ghosts in the environment are:", gameState.getNumGhosts())
        # Collect legal moves and successor states

        initialdepth = 0
        legalpacmanMoves = gameState.getLegalActions(self.index)

        score, bestaction = self.expectimax(gameState, 0, 0)

        return bestaction

    def expectimax(self, gameState, whoseturn, depth):

        print(" Whoseturn =", whoseturn)
        currentdepth = depth

        # If fates are locked win/ lose
        if gameState.isLose() or gameState.isWin():
            return scoreEvaluationFunction(gameState, 0), ""

        # base case : targetDepth reached
        if self.depth == currentdepth:
            print("Max depth reached. Depth = ", currentdepth)
            print(" ########### Recursion starts now #################")
            return scoreEvaluationFunction(gameState, 0), ""  # Evaluation of pacman score at the target depth

        # No legal move available :: blocked
        if not gameState.getLegalActions(whoseturn):
            return scoreEvaluationFunction(gameState, 0), ""

        # Choose next agent
        if whoseturn < gameState.getNumAgents() - 1:
            nextagent = whoseturn + 1
        else:
            nextagent = 0

        # Increment depth condition, if all agents have moved
        if nextagent == 0:
            depth += 1
            print("^^^^^^^^^^^^^^^^^^^^^^^^^^^ The depth was increased to:", depth, "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

        ####### Recursive Conditions ########

        available_evals = []

        print("All legal actions allowed for agent", whoseturn, "are:", gameState.getLegalActions(whoseturn))
        # Check which agent's turn
        if whoseturn == 0:  # Pacman is choosing
            max_score = float("-inf")
            best_action = ""

            for legalaction in gameState.getLegalActions(whoseturn):

                # Selection probability of fully rational choice: 0.5 + 0.5/ len(gameState.getLegalActions(whoseturn))

                print(
                    " >>>>>>>>>>>>>>>>>>>>>>>>>>New Actions for pacman starting here <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                print("The action under consideration for current agent", whoseturn, "is", legalaction)
                # Generate next state for different actions taken from the current state
                successorGameState = gameState.generateSuccessor(whoseturn, legalaction)
                print("Successor state successfully generated for this action")
                print("The next agent to decide move is", nextagent)
                score = self.expectimax(successorGameState, nextagent, depth)[0]

                # Penalize for stopping
                if legalaction == 'Stop':
                    score -= 1000

                available_evals.append(score)

                print("Score is", score)
                if float(score) >= max_score:
                    max_score = score
                    best_action = legalaction
                print("The current Best Pacman action at depth", depth, "is:", best_action)

            # Return max score and best action after iterating over all legal actions
            print("Available evaluations for following legalmoves", gameState.getLegalActions(whoseturn), "are",
                  available_evals)
            print("The final Best action for Pacman at depth", depth, "is:", best_action)
            return max_score, best_action

        else:

            min_score = float("inf")
            best_action = ""
            for legalghostaction in gameState.getLegalActions(whoseturn):

                print(
                    " >>>>>>>>>>>>>>>>>>>>>>>>>>>>New Action for ghost starting here <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                print("The action under consideration for agent", whoseturn, "is", legalghostaction)
                successorghostState = gameState.generateSuccessor(whoseturn, legalghostaction)
                print("Successor state successfully generated for this action")
                score = self.expectimax(successorghostState, nextagent, depth)[0]
                print("score", score)
                if float(score) < min_score:
                    min_score = score
                    best_action = legalghostaction
                print("The current best action chosen by ghost", whoseturn, "at depth", depth, "is:", best_action)
                available_evals.append(score)

            expectiscore = 0.5 * min_score + (0.5 / len(gameState.getLegalActions(whoseturn))) * sum(available_evals)
            print("The expectimax score is:", expectiscore)
            # Return max score and best action after iterating over all legal actions
            print("Available evaluations for following legalmoves", gameState.getLegalActions(whoseturn), "are",
                  available_evals)
            print("The final Best action for ghost", whoseturn, " selected at depth", depth, "is:", best_action)
            return expectiscore, best_action

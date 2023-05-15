# myTeam.py
# ---------
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


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
import numpy as np
from util import nearestPoint


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'SwitchAgent', second = 'SwitchAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]


#################
#               #
# Dynamic Agent # 
#               #
#################

DEPTH = 30 # adversarial search tree depth

class DynamicAgent(CaptureAgent):
  """
  Parent class for multi agent search agents
  """

  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    self.depth = DEPTH
    CaptureAgent.registerInitialState(self, gameState)

    self.currentMission = ""
    self.currentMissionCounter = 0
    self.pointToGoTo = (0, 0)
    self.walls = gameState.getWalls().asList()
    self.middleX = gameState.data.layout.width // 2
    self.previouselyExistingFood = \
      self.getFoodYouAreDefending(gameState).asList()
    
    # Switch 
    indices = []
    if (gameState.isOnRedTeam(self.index)):
      indices = gameState.getRedTeamIndices()
    
    else:
      indices = gameState.getBlueTeamIndices()

    defensiveIndex = min(indices)
    offensiveIndex = max(indices)

    self.isDefense = self.index==defensiveIndex
    self.isOffense = self.index==offensiveIndex


  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights
  
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()    
    features['successorScore'] = -len(foodList)#self.getScore(successor)

    # Compute distance to the nearest food

    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToFood': -1}
  
  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor
    
  
  
class SwitchAgent(DynamicAgent):

  def chooseAction(self, gameState):
    # Check offense or defense
    if (self.isDefense):
      return self.chooseActionDefensiveBehaviour(gameState)
    
    else:
    # Expectimax
      if (gameState.getAgentState(self.index).isPacman):
        #print("Expectimax Agent")
        return self.chooseActionExpectimax(gameState)

      else:
        #print("Reflex Agent")
        return self.chooseActionReflex(gameState)

  ###############
  #             #
  # EXPECTIMAX 
  #             #
  ###############
  # Expectimax mode
  def chooseActionExpectimax(self, gameState):
    
    """
    Chooses pacman action based on expectimaz with DEPTH
    """
    possibleActions = gameState.getLegalActions(self.index)
    action_scores = [self.expectimax(0, 0, self.getSuccessor(gameState,action)) for action in possibleActions]

    max_action = max(action_scores)
    max_indices = [index for index in range(len(action_scores)) if action_scores[index] == max_action]

    chosenIndex = random.choice(max_indices)

    return possibleActions[chosenIndex]

  def expectimax(self, agent, depth, gameState):

    if gameState.isOver() or depth == self.depth:
        return max(self.getSuccessor(gameState, action) for action in gameState.getLegalActions(self.index))

    if agent == self.index:  # maximize for our team
        return max(self.expectimax(agent+1, depth, self.getSuccessor(gameState, action)) for action in gameState.getLegalActions(self.index))

    else:  # minimize for other team
        if gameState.getNumAgents() == agent:
            agent = 0
            depth += 1

        if (not gameState.getAgentState(agent).isPacman) and (gameState.getAgentPosition(agent) is not None):  # minimize for ghosts
            ghost_state = gameState.getAgentState(agent)
            ghost_pos = ghost_state.getPosition()
            actions = gameState.getLegalActions(self.index)
            if Directions.STOP in actions and len(actions)>1:
                actions.remove(Directions.STOP)
            min_dist = float("inf")
            bestAction = actions[0]
            for action in actions:
                successor = self.getSuccessor(gameState, action)
                pos2 = successor.getAgentState(agent).getPosition()
                dist = self.getMazeDistance(ghost_pos, pos2)
                if dist < min_dist:
                  bestAction = action
                  min_dist = dist
            if min_dist <= 1:
                return -float("inf")
            return sum(self.expectimax(agent+1, depth, self.getSuccessor(gameState, bestAction)) for action in actions) / float(len(actions))

        else:  # ignore Pacman agents
            actions = gameState.getLegalActions(self.index)
            action = random.choice(actions)
            return self.evaluate(gameState, action)

  
  ###############
  #             #
  # REFLEX 
  #             #
  ###############
  def chooseActionReflex(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluateReflex(gameState, a) for a in actions]
    # print('eval time for agent %d: %.4f' % (self.index, time.time() - start))

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    return random.choice(bestActions)
  
  def evaluateReflex(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights
  
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()    
    features['successorScore'] = -len(foodList)#self.getScore(successor)

    # Compute distance to the nearest food

    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToFood': -1}
  
  
  ###############
  #             #
  # DEFENSIVE BAHVIOUR TREE 
  #             #
  ###############

  def bestActionToGetToPoint(self, gameState, point):
      bestDistance = float('inf')
      bestAction = ""
      for action in gameState.getLegalActions(self.index):
        d = self.distancer.getDistance(gameState.generateSuccessor(self.index, action).getAgentState(self.index).getPosition(), point)
        if d < bestDistance:
          bestDistance = d
          bestAction = action

      return bestAction
    

  def patrolFood(self, gameState):
      if self.currentMission != "GUARD FOOD":
        self.currentMission = "GUARD FOOD"
        self.currentMissionCounter = 0

      existingFoodPositions = self.getFoodYouAreDefending(gameState).asList()
      existingFoodPositions.sort(key=lambda x: abs(x[0] - self.middleX))

      self.pointToGoTo = existingFoodPositions[0]
      if len(existingFoodPositions) > 0 and gameState.getAgentPosition(self.index) == self.pointToGoTo:
        self.pointToGoTo = existingFoodPositions[1]


  def chooseActionDefensiveBehaviour(self, gameState):
      startTime = time.time()

      enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
      invaders = [enemy.getPosition() for enemy in enemies if enemy.isPacman]
      seenInvaders = [enemy.getPosition() for enemy in enemies if enemy.isPacman and enemy.getPosition() != None]
      foodEaten = list(set(self.previouselyExistingFood) - set(self.getFoodYouAreDefending(gameState).asList()))


      """ ------------------- Behavior Tree ------------------- """

      if len(invaders) != 0:
        if len(seenInvaders) != 0: # Invader exists, and is visible -> chase invader
          if self.currentMission != "CHASE":
            self.currentMission = "CHASE"
            self.currentMissionCounter = 0
          self.pointToGoTo = invaders[0]
        
        else: 
          if len(foodEaten) != 0: # Invader exists, is not visible, but food has been eaten -> go to eaten food
            if self.currentMission != "GO TO EATEN FOOD":
              self.currentMission = "GO TO EATEN FOOD"
              self.currentMissionCounter = 0
            self.pointToGoTo = foodEaten[0]

          else:
            if len(self.getCapsulesYouAreDefending(gameState)) != 0: # Invader exists, is not visible -> try go guard capsule
              if self.currentMission != "GUARD CAPSULE":
                self.currentMission = "GUARD CAPSULE"
                self.currentMissionCounter = 0
              self.pointToGoTo = self.getCapsulesYouAreDefending(gameState)[0]


            elif self.currentMission != "GUARD FOOD" or  gameState.getAgentPosition(self.index) == self.pointToGoTo: # Invader exists, is not visible, capsule don't exist -> guard food
              self.patrolFood(gameState)

      elif self.currentMission != "GUARD FOOD" or  gameState.getAgentPosition(self.index) == self.pointToGoTo:
        self.patrolFood(gameState)

      """ ----------------------------------------------------- """

      while time.time() - startTime < 0.1:
        pass

      self.currentMissionCounter += 1
      self.previouselyExistingFood = self.getFoodYouAreDefending(gameState).asList()

      return self.bestActionToGetToPoint(gameState, self.pointToGoTo)
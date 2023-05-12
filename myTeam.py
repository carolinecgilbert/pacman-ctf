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
from game import Directions, Grid
import game
import numpy as np

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed, first='DummyAgent', second='GhostAgent'):
  return [eval(first)(firstIndex), eval(second)(secondIndex)]


class GhostAgent(CaptureAgent):
    """
    A Dummy agent to serve as an example of the necessary agent structure.
    You should look at baselineTeam.py for more details about how to
    create an agent as this is the bare minimum.
    """

    def registerInitialState(self, gameState):
      CaptureAgent.registerInitialState(self, gameState)

      '''
      Your initialization code goes here, if you need any.
      '''
      self.currentMission = ""
      self.currentMissionCounter = 0
      self.pointToGoTo = (0, 0)


      self.walls = gameState.getWalls().asList()
      print(self.walls)


      """
      self.createGhostDomain()
      self.start = gameState.getAgentPosition(self.index)
      self.walls = gameState.getWalls().asList()
      self.masterCapsule = self.getCapsulesYouAreDefending(gameState)[0]
      self.currentScore = self.getScore(gameState)
      """

      self.middleX = gameState.data.layout.width // 2
      self.previouselyExistingFood = self.getFoodYouAreDefending(gameState).asList()

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



    def chooseAction(self, gameState):
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

      # print(self.currentMission, self.currentMissionCounter)
      return self.bestActionToGetToPoint(gameState, self.pointToGoTo)








class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)

    '''
    Your initialization code goes here, if you need any.
    '''


  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)

    '''
    You should change this in your own agent.
    '''

    return random.choice(actions)





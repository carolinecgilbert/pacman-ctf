from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game

class defensivePDDLAgent(CaptureAgent):
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
    self.createGhostDomain()
    self.start = gameState.getAgentPosition(self.index)
    self.pddlFluentGrid = self.generatePDDLFluentStatic(gameState)
    self.pddlObject = self.generatePddlObject(gameState)
    self.boundaryPos = self.getBoundaryPos(gameState, 1)
    self.masterCapsules = self.getCapsulesYouAreDefending(gameState)
    self.masterFoods = self.getFoodYouAreDefending(gameState).asList()
    self.currScore = self.getScore(gameState)
    self.numFoodDef = len(self.masterFoods)
    self.target = list()

  def createGhostDomain(self):
    ghost_domain_file = open(GHOST_DOMAIN_FILE, "w")
    domain_statement = """
      (define (domain ghost)

          (:requirements
              :typing
              :negative-preconditions
          )

          (:types
              invaders cells
          )

          (:predicates
              (cell ?p)

              ;pacman's cell location
              (at-ghost ?loc - cells)

              ;food cell location
              (at-invader ?i - invaders ?loc - cells)

              ;Indicated if a cell location has a capsule
              (has-capsule ?loc - cells)

              ;connects cells
              (connected ?from ?to - cells)

          )

          ; move ghost towards the goal state of invader
          (:action move
              :parameters (?from ?to - cells)
              :precondition (and 
                  (at-ghost ?from)
                  (connected ?from ?to)
              )
              :effect (and
                          ;; add
                          (at-ghost ?to)
                          ;; del
                          (not (at-ghost ?from))       
                      )
          )

          ; kill invader
          (:action kill-invader
              :parameters (?loc - cells ?i - invaders)
              :precondition (and 
                              (at-ghost ?loc)
                              (at-invader ?i ?loc)
                            )
              :effect (and
                          ;; add

                          ;; del
                          (not (at-invader ?i ?loc))
                      )
          )
      )
      """
    ghost_domain_file.write(domain_statement)
    ghost_domain_file.close()

  def generatePddlObject(self, gameState):
    """
    Function for creating PDDL objects for the problem file.
    """

    # Get Cell Locations without walls and Food count for object setup.
    allPos = gameState.getWalls().asList(False)
    invader_len = len(self.getOpponents(gameState))

    # Create Object PDDl line definition of objects.
    objects = list()
    cells = [f'cell{pos[0]}_{pos[1]}' for pos in allPos]
    cells.append("- cells\n")
    invaders = [f'invader{i+1}' for i in range(invader_len)]
    invaders.append("- invaders\n")

    objects.append("\t(:objects \n")
    objects.append(f'\t\t{" ".join(cells)}')
    objects.append(f'\t\t{" ".join(invaders)}')
    objects.append("\t)\n")

    return "".join(objects)

  def getBoundaryPos(self, gameState, span=4):
    """
    Get Boundary Position for Home to set as return when chased by ghost
    """
    layout = gameState.data.layout
    x = layout.width / 2 - 1 if self.red else layout.width / 2
    enemy = 1 if self.red else -1
    xSpan = [x - i for i in range(span)] if self.red else [x + i for i in range(span)]
    walls = gameState.getWalls().asList()
    homeBound = list()
    for x in xSpan:
      pos = [(int(x), y) for y in range(layout.height) if (x, y) not in walls and (x+enemy, y) not in walls]
      homeBound.extend(pos)
    return homeBound

  def generatePDDLFluentStatic(self, gameState):
    # Set Adjacency Position
    allPos = gameState.getWalls().asList(False)
    connected = list()
    for pos in allPos:
      if (pos[0] + 1, pos[1]) in allPos:
        connected.append(f'\t\t(connected cell{pos[0]}_{pos[1]} cell{pos[0]+1}_{pos[1]})\n')
      if (pos[0] - 1, pos[1]) in allPos:
        connected.append(f'\t\t(connected cell{pos[0]}_{pos[1]} cell{pos[0]-1}_{pos[1]})\n')
      if (pos[0], pos[1] + 1) in allPos:
        connected.append(f'\t\t(connected cell{pos[0]}_{pos[1]} cell{pos[0]}_{pos[1]+1})\n')
      if (pos[0], pos[1] - 1) in allPos:
        connected.append(f'\t\t(connected cell{pos[0]}_{pos[1]} cell{pos[0]}_{pos[1]-1})\n')

    return "".join(connected)

  def generatePddlFluent(self, gameState):
    """
    Function for creating PDDL fluents for the problem file.
    """

    # Set Self Position
    pacmanPos = gameState.getAgentPosition(self.index)
    at_ghost = f'\t\t(at-ghost cell{pacmanPos[0]}_{pacmanPos[1]})\n'

    # Set Invader(s) positions
    has_invaders = list()

    # if len(ANTICIPATER) == 0:
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    for i, invader in enumerate(invaders):
      invaderPos = invader.getPosition()
      has_invaders.append(f'\t\t(at-invader invader{i+1} cell{int(invaderPos[0])}_{int(invaderPos[1])})\n')
    # else:
    #   for i, invaderVal in enumerate(ANTICIPATER):
    #     invaderState, invaderPos = invaderVal
    #     if invaderState.isPacman:
    #       has_invaders.append(f'\t\t(at-invader invader{i+1} cell{int(invaderPos[0])}_{int(invaderPos[1])})\n')

    # Set Capsule Position
    capsules = self.getCapsulesYouAreDefending(gameState)
    has_capsule = [f'\t\t(has-capsule cell{capsule[0]}_{capsule[1]})\n' for capsule in capsules]

    fluents = list()
    fluents.append("\t(:init \n")
    fluents.append(at_ghost)
    fluents.append("".join(has_invaders))
    fluents.append("".join(has_capsule))
    fluents.append(self.pddlFluentGrid)
    fluents.append("\t)\n")

    return "".join(fluents)

  def generatePddlGoal(self, gameState):
    """
    Function for creating PDDL goals for the problem file.
    """
    print(f'======New Defensive Action{self.index}========')
    goal = list()
    goal.append('\t(:goal (and\n')

    myPos = gameState.getAgentPosition(self.index)
    print(f'Ghost at {myPos}')
    foods = self.getFoodYouAreDefending(gameState).asList()
    prevFoods = self.getFoodYouAreDefending(self.getPreviousObservation()).asList() \
      if self.getPreviousObservation() is not None else list()
    targetFood = list()
    invaders = list()
    Eaten = False

    # Get Food Defending Calculation based on current Game Score
    newScore = self.getScore(gameState)
    if newScore < self.currScore:
      self.numFoodDef -= self.currScore - newScore
      self.currScore = newScore
    else:
      self.currScore = newScore

    # myState = gameState.getAgentState(self.index)
    # scared = True if myState.scaredTimer > 2 else False
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    enemyHere = [a for a in enemies if a.isPacman]
    # if len(ANTICIPATER) == 0:
      # Find Invaders and set their location.
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    for i, invader in enumerate(invaders):
      invaderPos = invader.getPosition()
      goal.append(f'\t\t(not (at-invader invader{i+1} cell{int(invaderPos[0])}_{int(invaderPos[1])}))\n')
    # else:
    #   for i, invaderVal in enumerate(ANTICIPATER):
    #     invaderState, invaderPos = invaderVal
    #     if invaderState.isPacman:
    #       invaders.append(invaderPos)
    #       goal.append(f'\t\t(not (at-invader invader{i+1} cell{int(invaderPos[0])}_{int(invaderPos[1])}))\n')

    if len(foods) < self.numFoodDef:
      Eaten = True
      targetFood = list(set(prevFoods) - set(foods))
      if targetFood:
        self.target = targetFood
    elif self.numFoodDef == len(foods):
      Eaten = False
      self.target = list()
      print(f'Handling #1')

    # If No Invaders are detected (Seen 5 steps)
    if not invaders:
      # If Food has not been eaten, Guard the Capsules or Foods
      if not Eaten:
        if myPos not in self.boundaryPos and len(enemyHere) == 0:
          print(f'Going to #1')
          goal.extend(self.generateRedundantGoal(self.boundaryPos, myPos))
        elif myPos not in self.masterCapsules and len(self.getCapsulesYouAreDefending(gameState)) > 0:
          print(f'Going to #2')
          capsules = self.getCapsulesYouAreDefending(gameState)
          goal.extend(self.shufflePddlGoal(capsules, myPos))
        else:
          print(f'Going to #3')
          goal.extend(self.generateRedundantGoal(foods, myPos))
      # If Food have been eaten Rush to the food location.
      else:
        print(f'Going to #4')
        if myPos in self.target:
          self.target.remove(myPos)
        goal.extend(self.shufflePddlGoal(self.target, myPos))


    goal.append('\t))\n')
    return "".join(goal)

  def generateRedundantGoal(self,compare,myPos):
    goal = list()
    goal.append('\t\t(or\n')
    for pos in compare:
      if myPos != pos:
        goal.append(f'\t\t\t(at-ghost cell{pos[0]}_{pos[1]})\n')
    goal.append('\t\t)\n')
    return goal

  def shufflePddlGoal(self, target, myPos):
    goal = list()
    if len(target) > 1:
      goal.append('\t\t(or\n')
      goal.extend([f'\t\t\t(at-ghost cell{pos[0]}_{pos[1]})\n' for pos in target])
      goal.append('\t\t)\n')
    elif len(target) == 1:
      goal.append(f'\t\t(at-ghost cell{target[0][0]}_{target[0][1]})\n')
    else:
      goal.extend(self.generateRedundantGoal(self.boundaryPos, myPos))
    return goal

  def generatePddlProblem(self, gameState):
    """
    Generates a file for Creating PDDL problem file for current state.
    """
    problem = list()
    problem.append(f'(define (problem p{self.index}-ghost)\n')
    problem.append('\t(:domain ghost)\n')
    # problem.append(self.pddlObject)
    problem.append(self.generatePddlObject(gameState))
    problem.append(self.generatePddlFluent(gameState))
    problem.append(self.generatePddlGoal(gameState))
    problem.append(')')

    problem_file = open(f"{CD}/ghost-problem-{self.index}.pddl", "w")
    problem_statement = "".join(problem)
    problem_file.write(problem_statement)
    problem_file.close()
    return f"ghost-problem-{self.index}.pddl"

  def chooseAction(self, gameState):
    # global ANTICIPATER
    agentPosition = gameState.getAgentPosition(self.index)
    problem_file = self.generatePddlProblem(gameState)
    planner = PlannerFF(GHOST_DOMAIN_FILE, problem_file)
    output = planner.run_planner()
    plannerPosition, plan = planner.parse_solution(output)
    action = planner.get_legal_action(agentPosition, plannerPosition)
    print(f'Action Planner: {action}')
    # actions = gameState.getLegalActions(self.index)
    return action
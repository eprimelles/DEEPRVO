import numpy as np
import rvo2
from Circle import Circle
from typing import Any
class DeepNav():    
    def __init__(self, n_agents : int, scenario : int, discrete: bool = True, width : int = 255, height : int = 255, timestep : float = 0.25 , neighbor_dists : float = 1.0, 
                 time_horizont : float=10.0, time_horizont_obst : float = 20.0, radius : float=2.0, 
                 max_speed : float=3.5) -> None:
        super().__init__()
        
        
        self.n_agents = n_agents
        self.scenario = scenario
        self.width = width
        self.height = height
        self.timestep = timestep
        self.neighbor_dists = neighbor_dists
        self.max_neig = n_agents
        self.time_horizont = time_horizont
        self.time_horizont_obst = time_horizont_obst
        self.radius = radius
        self.max_speed = max_speed
        self. sim = rvo2.PyRVOSimulator(self.timestep, self.neighbor_dists, self.max_neig, self.time_horizont, self.time_horizont_obst, self.radius, self.max_speed)
        self.time = 0.0
        self.T = 0
               
        
        self.positions, self.goals, self.obstacles = self.getScenario().getAgentPosition()
        self.__state = np.zeros((self.n_agents, 4 * self.n_agents))
        self.__episode_ended = False
        self.__setupScenario()
        self.success = True
        self.max_dis = [self.calculateDist(self.positions[i], self.goals[i]) for i in range(self.n_agents)]

        self.discrete = discrete

        if self.discrete:
            self.p_actions = {
                0 : (0, 0),
                1 : (1, 0),
                2 : (0, 1),
                3 : (1, 1),
                4 : (-1, 0),
                5 : (0, -1),
                6 : (-1, -1),
                7 : (-1, 1),
                8 : (1, -1),

            } 
   

    def act(self, actions):
        if self.discrete:
            return self.setDiscreteActions(actions)
        return self.setContActions(actions)
    
    def getState(self):
        pass

    

    # Utility functions
    def normalize(self, x):
        x = np.array(x)
        norm = np.linalg(x)

        if norm == 0:
            return x
        return x / norm
    
    def getScenario(self) -> Any:
        if self.scenario == 0:
            return Circle(self.n_agents)
    
    def __setupScenario(self) -> None:
        [
            self.sim.addAgent(i)
            for i in self.positions
        ]

    def calculateDist(self, a : tuple, b : tuple):
        return np.hypot(a[0] - b[0], a[1] - b[1])
    
    def isLegal(self, agent, action):
        pos = np.array(self.sim.getAgentPos(agent))
        final_pos = pos + action * self.timestep

        if final_pos[0] > self.width or final_pos[1] > self.height:
            return (0, 0)
        return action
    def setDiscreteActions(self, actions):
        assert self.discrete

        for i, a in enumerate (actions):
            action = self.normalize(self.p_actions[a])
            action = self.isLeagal(i, np.array(action))
            self.sim.setAgentPrefVelocity(i, tuple(action))

    def setContActions(self , actions):
        assert not self.discrete

        for i, a in enumerate (actions):
            action = self.isLegal(i, np.array(action))
            self.sim.setAgentPrefVelocity(i, tuple(action))

    

DeepNav(2, 0)
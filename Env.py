import numpy as np
import rvo2
from Circle import Circle
from typing import Any
class DeepNav():    
    def __init__(self, n_agents : int, scenario : int, width : int = 255, height : int = 255, timestep : float = 0.25 , neighbor_dists : float = 1.0, 
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

    
DeepNav(2, 0)
import numpy as np
import rvo2
from environment.Circle import Circle
from typing import Any
class DeepNav():    
    def __init__(self, n_agents : int, scenario : int, discrete: bool = True, width : int = 255, height : int = 255, timestep : float = 0.25 , neighbor_dists : float = 1.0, 
                 time_horizont : float=10.0, time_horizont_obst : float = 20.0, radius : float=2.0, 
                 max_speed : float=3.5, opt : str = 'full') -> None:
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
   
        self.opt = opt
    def act(self, actions):
        if self.discrete:
            return self.setDiscreteActions(actions)
        return self.setContActions(actions)    
    

    def getState(self):
        
        return {
            'full' : self.getFullState,
            'sensor' : self.getSensorialState,
            'min' : self.getMinimalState
        }[self.opt]

    def reset(self):
        self.success = False
        
        for i in range(self.n_agents):
            self.sim.setAgentPosition(i, self.positions[i])
        self.__episode_ended = False
        self.T = 0
        self.__state = self.getState()()
        return self.__state
    
    def step(self, actions):
        self.act(actions)
        self.sim.doStep()
        self.setState()
        rwd = self.calculate_global_rwd() + self.calculate_local_rwd()
        self.T += 1
        
        return self.__state, rwd, self.isDone()

    def getObservationSpace(self):
        return len(self.getState()()[0])
    
    def getActionSpace(self):
        if self.discrete:
            return 9
        return 2
    # Utility functions
    def calculate_local_rwd(self):
        rwd = [0] * self.n_agents
        
        for i in range(self.n_agents):
            rwd[i] -= self.calculateDist(self.sim.getAgentPosition(i), self.goals[i]) / self.max_dis[i]
            
            if self.agentDone(i):
                rwd[i] += 1

            
        return np.array(rwd)
        
    def calculate_global_rwd(self):
        if self.success:
            return - self.T
        return 0
    def isDone(self):
        
        for i in range(self.n_agents):
            if not self.agentDone(i):
                self.success == False
                return False
        self.success == True
        return True
    

    def agentDone(self, agent):
        pos = self.sim.getAgentPosition(agent)
        return self.calculateDist(pos, self.goals[agent]) < self.radius
    def setState(self):
        self.__state = self.getState()()

    def normalize(self, x):
        x = np.array(x)
        norm = np.linalg.norm(x)

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
        pos = np.array(self.sim.getAgentPosition(agent))
        final_pos = pos + action * self.timestep

        if np.abs(final_pos[0]) > self.width or np.abs(final_pos[1]) > self.height:
            return (0, 0)
        return action
    def setDiscreteActions(self, actions):
        assert self.discrete
        actions = tuple(actions)
        for i, a in enumerate (actions):
            
            action = self.normalize(self.p_actions[a])
            action = self.isLegal(i, np.array(action))
            self.sim.setAgentPrefVelocity(i, tuple(action))

    def setContActions(self , actions):
        assert not self.discrete

        for i, a in enumerate (actions):
            action = self.isLegal(i, np.array(action))
            self.sim.setAgentPrefVelocity(i, tuple(action))

    def getFullState(self):
        '''Return every agent position and pref vel'''
        return [
            self.getAgentFullState(i)
            for i in range(self.n_agents)
        ]
    def getSensorialState(self):
        return [
            self.getAgentSensorialState(i)
            for i in range(self.n_agents)
        ]
    def getMinimalState(self):
        return [
            self.getAgentMinimalState(i)
            for i in range(self.n_agents)
        ]
    def getAgentFullState(self, agent):
        
        state = []
        state.append(self.sim.getAgentPosition(agent)[0])
        state.append(self.sim.getAgentPosition(agent)[1])
        state.append(self.sim.getAgentVelocity(agent)[0])
        state.append(self.sim.getAgentVelocity(agent)[1])

        for  i in range(self.n_agents):
            if i == agent:
                continue

            state.append(self.sim.getAgentPosition(i)[0])
            state.append(self.sim.getAgentPosition(i)[1])
            state.append(self.sim.getAgentVelocity(i)[0])
            state.append(self.sim.getAgentVelocity(i)[1])
        return state

    def getAgentSensorialState(self, agent):
        dirs = [0, 45, 90, 135, 180, 225, 270, 315]
        a_pos = self.sim.getAgentPosition(agent)
        
        state = [None] * 10
        state [0] = a_pos[0]
        state [1] = a_pos[1]
        positions = [self.sim.getAgentPosition(a) for a in range(self.n_agents) if a != agent]
        for poss in positions:
            x = a_pos[0] - poss[0]
            y = a_pos[1] - poss[1]
            ang = np.degrees(np.arctan(y/x))
            
            norm = np.linalg.norm((x, y))
            ang_int = np.degrees(np.arcsin(self.radius/norm))
            if np.isnan(ang_int):
                continue
            ang_inf = np.round((ang - ang_int) + 0.5)
            ang_sup = np.round((ang + ang_int) - 0.5)
            
            ang_range = np.arange(ang_inf, ang_sup)
            
            # ang_range = list(range(ang_inf, ang_sup + 1))
            for indx, a in enumerate(dirs):
                if a in ang_range:
                    state[indx + 2] = norm - self.radius
        # Missing no collision measurements
        for i in range(len(state)):
            if state[i] is None:
                x_dist = a_pos[0] - self.width
                y_dist = a_pos[1] - self.height
                state[i] = np.linalg.norm([x_dist, y_dist])
        return state

    def getAgentMinimalState(self, agent):
        state = []
        state.append(self.sim.getAgentPosition(agent)[0])
        state.append(self.sim.getAgentPosition(agent)[1])
        state.append(self.sim.getAgentVelocity(agent)[0])
        state.append(self.sim.getAgentVelocity(agent)[1])
        return state


if __name__ == '__main__':

    env = DeepNav(2, 0)
    s = env.reset()

    s, r, done = env.step((1, 0))
    print(r)

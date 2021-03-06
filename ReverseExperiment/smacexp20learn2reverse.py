# -*- coding: utf-8 -*-
"""
https://github.com/oxwhirl/smac

"""
from smac.env import StarCraft2Env
import numpy as np
#import sys
import random 
import pickle

#from gym.spaces import Discrete, Box, Dict


#Вывод массива целиком
np.set_printoptions(threshold=np.inf)

#определяем может ли агент сделать заданнное действие action_is
def is_possible_action (avail_actions_ind, action_is):
     ia=0
     #print ("in def len(avail_actions_ind) = ", len(avail_actions_ind))
     while ia<len(avail_actions_ind):
         #print ("ia = ", ia)
         if avail_actions_ind[ia] == action_is:
             ia = len(avail_actions_ind)+1
             return True
         else:
             ia = ia+1
         
     return False


#получаем состояние агента как позицию на карте
def get_stateFox(agent_id, agent_posX, agent_posY):

        if agent_id == 0:
            state = 3

            if 23 < agent_posX < 24 and 15 < agent_posY < 16.5:
                state = 0
            elif 22 < agent_posX < 23 and 15 < agent_posY < 16.5:
                state = 1
            elif 21.1 < agent_posX < 22 and 15 < agent_posY < 16.5:
                state = 2
            elif 19.9 < agent_posX < 21.1 and 15 < agent_posY < 16.5:
                state = 3
            elif 19 < agent_posX < 19.9 and 15 < agent_posY < 16.5:
                state = 4
            elif 18 < agent_posX < 19 and 15 < agent_posY < 16.5:
                state = 5
            elif 17 < agent_posX < 18 and 15 < agent_posY < 16.5:
                state = 6
            elif 16 < agent_posX < 17 and 15 < agent_posY < 16.5:
                state = 7

            elif 23 < agent_posX < 24 and 14 < agent_posY < 15:
                state = 8
            elif 22 < agent_posX < 23 and 14 < agent_posY < 15:
                state = 9
            elif 21.1 < agent_posX < 22 and 14 < agent_posY < 15:
                state = 10
            elif 19.9 < agent_posX < 21.1 and 14 < agent_posY < 15:
                state = 11
            elif 19 < agent_posX < 19.9 and 14 < agent_posY < 15:
                state = 12
            elif 18 < agent_posX < 19 and 14 < agent_posY < 15:
                state = 13
            elif 17 < agent_posX < 18 and 14 < agent_posY < 15:
                state = 14
            elif 16 < agent_posX < 17 and 14 < agent_posY < 15:
                state = 15

        if agent_id == 1:
            state = 11

            if 23 < agent_posX < 24 and 16.2 < agent_posY < 17:
                state = 0
            elif 22 < agent_posX < 23 and 16.2 < agent_posY < 17:
                state = 1
            elif 21.1 < agent_posX < 22 and 16.2 < agent_posY < 17:
                state = 2
            elif 19.9 < agent_posX < 21.1 and 16.2 < agent_posY < 17:
                state = 3
            elif 19 < agent_posX < 19.9 and 16.2 < agent_posY < 17:
                state = 4
            elif 18 < agent_posX < 19 and 16.2 < agent_posY < 17:
                state = 5
            elif 17 < agent_posX < 18 and 16.2 < agent_posY < 17:
                state = 6
            elif 16 < agent_posX < 17 and 16.2 < agent_posY < 17:
                state = 7

            elif 23 < agent_posX < 24 and 15.5 < agent_posY < 16.2:
                state = 8
            elif 22 < agent_posX < 23 and 15.5 < agent_posY < 16.2:
                state = 9
            elif 21.1 < agent_posX < 22 and 15.5 < agent_posY < 16.2:
                state = 10
            elif 19.9 < agent_posX < 21.1 and 15.5 < agent_posY < 16.2:
                state = 11
            elif 19 < agent_posX < 19.9 and 15.5 < agent_posY < 16.2:
                state = 12
            elif 18 < agent_posX < 19 and 15.5 < agent_posY < 16.2:
                state = 13
            elif 17 < agent_posX < 18 and 15.5 < agent_posY < 16.2:
                state = 14
            elif 16 < agent_posX < 17 and 15.5 < agent_posY < 16.2:
                state = 15

        return state

"""    
keys = [0 1 2 3 4 5]
act_ind_decode= {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}
qt_arr[act_ind]= 0.0
qt_arr[act_ind]= 0.0
qt_arr[act_ind]= 0.0
qt_arr[act_ind]= 0.0
qt_arr[act_ind]= 0.0
qt_arr[act_ind]= 0.0
"""

def select_actionFox(agent_id, state, avail_actions_ind, n_actionsFox, epsilon, Q_table):
    
    if random.uniform(0, 1) < (1 - epsilon):
        action = np.random.choice(avail_actions_ind)  # Explore action space
    else:

        qt_arr = np.zeros(len(avail_actions_ind))
        
        #Функция arange() возвращает одномерный массив с равномерно разнесенными значениями внутри заданного интервала. 
        keys = np.arange(len(avail_actions_ind))
        #print ("keys =", keys)
        #act_ind_decode= {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}
        #Функция zip объединяет в кортежи элементы из последовательностей переданных в качестве аргументов.
        act_ind_decode = dict(zip(keys, avail_actions_ind))
        #print ("act_ind_decode=", act_ind_decode)
        stateFoxint = int(state)
        
        #print ('len(avail_actions_ind)=',len(avail_actions_ind))


        for act_ind in range(len(avail_actions_ind)):
            qt_arr[act_ind] = Q_table[agent_id, stateFoxint, act_ind_decode[act_ind]]
            #print ("qt_arr[act_ind]=",qt_arr[act_ind])

        #Returns the indices of the maximum values along an axis.
        action = act_ind_decode[np.argmax(qt_arr)]  # Exploit learned values
    
    return action      
        
    



#MAIN
def main():
    """The StarCraft II environment for decentralised multi-agent micromanagement scenarios."""
    '''difficulty ="1" is VeryEasy'''
    #replay_dir="D:\StarCraft II\Replays\smacfox"
    env = StarCraft2Env(map_name="2m2mFOXReverse", difficulty="1")
    
    '''env_info= {'state_shape': 48, 'obs_shape': 30, 'n_actions': 9, 'n_agents': 3, 'episode_limit': 60}'''
    env_info = env.get_env_info()
    #print("env_info = ", env_info)
    

    
    """Returns the size of the observation."""
    """obssize =  10"""
    """obs= [array([ 1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
        0.63521415,  0.63517255, -0.00726997,  0.06666667,  0.06666667],
      dtype=float32)]"""
    obssize=env.get_obs_size()
    #print("obssize = ", obssize)
    
    ######################################################################
    """
    ready_agents = []
    #observation_space= Dict(action_mask:Box(9,), obs:Box(30,))
    observation_space = Dict({
            "obs": Box(-1, 1, shape=(env.get_obs_size())),
            "action_mask": Box(0, 1, shape=(env.get_total_actions()))  })
    #print ("observation_space=", observation_space)
    
    #action_space= Discrete(9)
    action_space = Discrete(env.get_total_actions())
    #print ("action_space=", action_space)
    """
    ########################################################################
    
    n_actions = env_info["n_actions"]
    #print ("n_actions=", n_actions)
    n_agents = env_info["n_agents"]
   
    n_episodes = 300 # количество эпизодов lapan = 20
    
   
    alpha = 0.3    #learning rate sayon - 0.5 больш - 0.9 Lapan = 0.2
    gamma = 0.3   #discount factor sayon - 0.9 больш - 0.5 lapan = 0.9
    epsilon = 0.7 #e-greedy sayon - 0.3 больш - 0.7 lapan = = 1.0 (100% random actions)
    
    n_statesFox = 16 # количество состояний нашего мира-сетки
    #n_statesFox1 = 16 # количество состояний нашего мира-сетки
    n_actionsFox = 7 # вводим свое количество действий, которые понадобятся
    
    Q_table = np.zeros([n_agents, n_statesFox, n_actions]) #задаем пустую q таблицу
    #Q_table1 = np.zeros([n_statesFox1, n_actionsFox])
    #Q_table = np.zeros([32, n_actions]) 
    #print (Q_table)

    for e in range(n_episodes):
        #print("n_episode = ", e)
        """Reset the environment. Required after each full episode.Returns initial observations and states."""
        env.reset()
        ''' Battle is over terminated = True'''
        terminated = False
        episode_reward = 0
        
        #n_steps = 1 #пока не берем это количество шагов для уменьгения награды за долгий поиск
        
        """
        # вывод в файл
        fileobj = open("файл.txt", "wt")
        print("text",file=fileobj)
        fileobj.close()
        """
       
        #динамический epsilon - только при большом количестве эпизодов имеет смысл!!!
        
        if e % 15 == 0:
            epsilon += (1 - epsilon) * 10 / n_episodes
            print("epsilon = ", epsilon)
        

        #stoprun = [0,0,0,0,0] 
      
        
        while not terminated:
            """Returns observation for agent_id."""
            obs = env.get_obs()
            #print ("obs=", obs)
            """Returns the global state."""
            #state = env.get_state()
         
            
            actions = []
            action = 0
            stateFox= np.zeros([n_agents])
           
           
            '''agent_id= 0, agent_id= 1'''
            for agent_id in range(n_agents):
                
                #получаем характеристики юнита
                unit = env.get_unit_by_id(agent_id)
                #получаем состояние по координатам юнита
                stateFox[agent_id] = get_stateFox(agent_id, unit.pos.x, unit.pos.y)
                #print ("agent_id =", agent_id)
                #print ("stateFox[agent_id] =", stateFox[agent_id])
                
                '''
                tag = unit.tag #много разных характеристик юнита
                x = unit.pos.x
                y = unit.pos.y
                '''
                """Returns the available actions for agent_id."""
                """avail_actions= [0, 1, 1, 1, 1, 1, 0, 0, 0]"""
                avail_actions = env.get_avail_agent_actions(agent_id)
                '''Функция nonzero() возвращает индексы ненулевых элементов массива.'''
                """avail_actions_ind of agent_id == 0: [1 2 3 4 5]"""   
                avail_actions_ind = np.nonzero(avail_actions)[0]
                # выбираем действие
                action = select_actionFox(agent_id, stateFox[agent_id], avail_actions_ind, n_actionsFox, epsilon, Q_table)
                #собираем действия от разных агентов
                actions.append(action)
                
                
                ###############_Бежим вправо и стреляем_################################
                """
                if is_possible_action(avail_actions_ind, 6) == True:
                    action = 6
                else:
                    if is_possible_action(avail_actions_ind, 4) == True:
                        action = 4
                    else:
                        action = np.random.choice(avail_actions_ind)
                        #Случайная выборка из значений заданного одномерного массива
                  """ 
                #####################################################################    
                """Функция append() добавляет элементы в конец массива."""
                #print("agent_id=",agent_id,"avail_actions_ind=", avail_actions_ind, "action = ", action, "actions = ", actions)
                #f.write(agent_id)
                #f.write(avail_actions_ind)
                #собираем действия от разных агентов
                #actions.append(action)
               
            #как узнать куда стрелять? в определенного человека?
            #как узнать что делают другие агенты? самому создавать для них глобальное состояние 
            #раз я ими управляю?
            """A single environment step. Returns reward, terminated, info."""
            reward, terminated, _ = env.step(actions)
            #print ('actions=', actions)
            episode_reward += reward
            
            ###################_Обучаем_##############################################
            
            for agent_id in range(n_agents):
                #получаем характеристики юнита
                unit = env.get_unit_by_id(agent_id)
                #получаем состояние по координатам юнита
                stateFox_next = get_stateFox(agent_id, unit.pos.x, unit.pos.y)
                stateFoxint= int(stateFox[agent_id])
                
                Q_table[agent_id, stateFoxint, action] = Q_table[agent_id, stateFoxint, action] + alpha * \
                             (reward + gamma * np.max(Q_table[agent_id, stateFox_next, :]) - Q_table[agent_id, stateFoxint, action])
            
            ##########################################################################            
       
        #Total reward in episode 4 = 20.0
        print("Total reward in episode {} = {}".format(e, episode_reward))
        #get_stats()= {'battles_won': 2, 'battles_game': 5, 'battles_draw': 0, 'win_rate': 0.4, 'timeouts': 0, 'restarts': 0}
        print ("get_stats()=", env.get_stats())
    
    #env.save_replay() """Save a replay."""
    """"Close StarCraft II.""""" 
    env.close()
    print(Q_table)
    with open("se21.pkl", 'wb') as f:
        pickle.dump(Q_table, f)
    
    
if __name__ == "__main__":
    main()
 
    
    
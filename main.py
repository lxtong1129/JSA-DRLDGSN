import matplotlib.pyplot as plt
from drl_dgsn import Agent
import numpy as np
import math
import pandas as pd
import warnings
import copy

ServiceZone_X = 500
ServiceZone_Y = 500
Hight_limit_Z = 120


MAXUserspeed = 0.5
UAV_Speed = 5

UserNumberPerCell = 2
NumberOfUAVs = 3
NumberOfCells = NumberOfUAVs
NumberOfUsers = NumberOfUAVs * UserNumberPerCell
F_c = 2  # carrier frequency
Bandwidth = 30
R_require = 0.1
Power_level = 3
amplification_constant = 10000
UAV_power_unit = 100 * amplification_constant  # 100mW=20dBm
NoisePower = 10 ** (-9) * amplification_constant



def main():
    Episodes_number = 200
    Test_episodes_number = 150
    T = 60
    T_AS = np.arange(0, T, 5)
    env = SystemModel()  # crate an environment
    agent = Agent()  # crate an agent

    Epsilon = 0.9
    datarate_seq = np.zeros(T)
    WorstuserRate_seq = np.zeros(T)
    Through_put_seq = np.zeros(Episodes_number)
    Worstuser_TP_seq = np.zeros(Episodes_number)
    user_position = np.zeros([Episodes_number * 3 * env.User_number])
    uav_position = np.zeros([Episodes_number * 3 * env.UAV_number])

    for episode in range(Episodes_number):
        env.Reset_position()
        Epsilon -= 0.9 / (Episodes_number - Test_episodes_number)  # decaying epsilon


        for t in range(T):

            if t in T_AS:
                User_AS_List = agent.User_association(env.PositionOfUAVs, env.PositionOfUsers, NumberOfUAVs,
                                                      NumberOfUsers)

            for UAV in range(NumberOfUAVs):

                Distence_CG = env.Get_Distance_U2K(env.PositionOfUAVs, env.PositionOfUsers, NumberOfUAVs,
                                                   NumberOfUsers)
                PL_for_CG = env.Get_Propergation_Loss(Distence_CG, env.PositionOfUAVs, NumberOfUAVs, NumberOfUsers,
                                                      F_c)
                CG = env.Get_Channel_Gain_NOMA(NumberOfUAVs, NumberOfUsers, PL_for_CG, User_AS_List,
                                               NoisePower)
                Eq_CG = env.Get_Eq_CG(NumberOfUAVs, NumberOfUsers, PL_for_CG, User_AS_List,
                                      NoisePower)

                State = env.Create_state_Noposition(UAV, User_AS_List,
                                                    CG)
                action_name = agent.Choose_action(State, Epsilon)
                env.take_action_NOMA(action_name, UAV, User_AS_List, Eq_CG)

                Distence = env.Get_Distance_U2K(env.PositionOfUAVs, env.PositionOfUsers, NumberOfUAVs,
                                                NumberOfUsers)
                P_L = env.Get_Propergation_Loss(Distence, env.PositionOfUAVs, NumberOfUAVs, NumberOfUsers,
                                                F_c)
                SINR = env.Get_SINR_NNOMA(NumberOfUAVs, NumberOfUsers, P_L, User_AS_List, Eq_CG,
                                          NoisePower)
                DataRate, SumRate, WorstuserRate = env.Calcullate_Datarate(SINR, NumberOfUsers,
                                                                           Bandwidth)



                Reward = SumRate
                if WorstuserRate < R_require:
                    Reward = -50

                CG_next = env.Get_Channel_Gain_NOMA(NumberOfUAVs, NumberOfUsers, P_L, User_AS_List,
                                                    NoisePower)
                Next_state = env.Create_state_Noposition(UAV, User_AS_List, CG_next)


                State_for_memory = copy.deepcopy(State[0])
                Action_for_memory = copy.deepcopy(action_name)
                Next_state_for_memory = copy.deepcopy(Next_state[0])
                Reward_for_memory = copy.deepcopy(Reward)

                agent.remember(State_for_memory, Action_for_memory, Next_state_for_memory,
                               Reward_for_memory)
                agent.train()
                env.User_randomMove(MAXUserspeed, NumberOfUsers)



                # save data after all UAVs moved
                if UAV == (NumberOfUAVs - 1):
                    Rate_during_t = copy.deepcopy(SumRate)
                    datarate_seq[t] = Rate_during_t
                    

        Through_put = np.sum(datarate_seq)
        Through_put_seq[episode] = Through_put
        print('Episode=', episode, 'Epsilon=', Epsilon, 'Through_put=', Through_put)

    # save data


    # print throughput
    x_axis = range(1, Episodes_number + 1)
    plt.plot(x_axis, Through_put_seq)
    plt.xlabel('Episodes')
    plt.ylabel('Throughput')
    plt.savefig('./ Throughput.png')
    plt.show()


if __name__ == '__main__':
    main()



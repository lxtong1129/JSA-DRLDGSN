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

class SystemModel(object):
    def __init__(
            self,
    ):

        self.Zone_border_X = ServiceZone_X
        self.Zone_border_Y = ServiceZone_Y
        self.Zone_border_Z = Hight_limit_Z


        self.UAVspeed = UAV_Speed
        self.UAV_number = NumberOfUAVs
        self.UserperCell = UserNumberPerCell
        self.U_idx = np.arange(NumberOfUAVs)
        self.PositionOfUAVs = pd.DataFrame(
            np.zeros((3, NumberOfUAVs)),
            columns=self.U_idx.tolist(),
        )
        self.PositionOfUAVs.iloc[0, :] = [100, 200, 400]
        self.PositionOfUAVs.iloc[1, :] = [100, 400, 100]
        self.PositionOfUAVs.iloc[2, :] = [100, 100, 100]


        self.User_number = NumberOfUsers
        self.K_idx = np.arange(NumberOfUsers)
        self.PositionOfUsers = pd.DataFrame(
            np.random.random((3, NumberOfUsers)),
            columns=self.K_idx.tolist(),
        )

        coords = generate_coordinates(self.PositionOfUAVs, NumberOfUsers)

        self.PositionOfUsers.iloc[0, :] = coords[0]
        self.PositionOfUsers.iloc[1, :] = coords[1]
        self.PositionOfUsers.iloc[2, :] = 0


        self.Init_PositionOfUsers = copy.deepcopy(self.PositionOfUsers)
        self.Init_PositionOfUAVs = copy.deepcopy(self.PositionOfUAVs)


        self.State = np.zeros([1, NumberOfUAVs * 3 + NumberOfUsers], dtype=float)


        self.Power_allocation_list = pd.DataFrame(
            np.ones((1, NumberOfUsers)),
            columns=np.arange(NumberOfUsers).tolist(),
        )
        self.Power_unit = UAV_power_unit
        self.Power_allocation_list = self.Power_allocation_list * self.Power_unit

        self.Distence = pd.DataFrame(
            np.zeros((self.UAV_number, self.User_number)),
            columns=np.arange(self.User_number).tolist(), )


        self.Propergation_Loss = pd.DataFrame(
            np.zeros((self.UAV_number, self.User_number)),
            columns=np.arange(self.User_number).tolist(), )

        self.ChannelGain_list = pd.DataFrame(
            np.zeros((1, self.User_number)),
            columns=np.arange(self.User_number).tolist(), )

        self.Eq_CG_list = pd.DataFrame(
            np.zeros((1, self.User_number)),
            columns=np.arange(self.User_number).tolist(), )

        self.SINR_list = pd.DataFrame(
            np.zeros((1, self.User_number)),
            columns=np.arange(self.User_number).tolist(), )

        self.Daterate = pd.DataFrame(
            np.zeros((1, self.User_number)),
            columns=np.arange(self.User_number).tolist(), )

        self.amplification_constant = amplification_constant

    def User_randomMove(self, MAXspeed, NumberofUsers):
        self.PositionOfUsers.iloc[[0, 1], :] += np.random.randn(2, NumberofUsers) * MAXspeed
        return

    def Get_Distance_U2K(self, UAV_Position, User_Position, UAVsnumber,
                         Usersnumber):

        for i in range(UAVsnumber):
            for j in range(Usersnumber):
                self.Distence.iloc[i, j] = np.linalg.norm(
                    UAV_Position.iloc[:, i] - User_Position.iloc[:, j])

        return self.Distence

    def Get_Propergation_Loss(self, distence_U2K, UAV_Position, UAVsnumber, Usersnumber,
                              f_c):  # calculating the pathloss

        for i in range(
                UAVsnumber):
            for j in range(Usersnumber):
                UAV_Hight = UAV_Position.iloc[2, i]
                D_H = np.sqrt(np.square(distence_U2K.iloc[i, j]) - np.square(UAV_Hight))

                d_0 = np.max([(294.05 * math.log(UAV_Hight, 10) - 432.94), 18])
                p_1 = 233.98 * math.log(UAV_Hight, 10) - 0.95
                if D_H <= d_0:
                    P_Los = 1.0
                else:
                    P_Los = d_0 / D_H + math.exp(-(D_H / p_1) * (1 - (d_0 / D_H)))

                if P_Los > 1:
                    P_Los = 1

                P_NLos = 1 - P_Los


                L_Los = 30.9 + (22.25 - 0.5 * math.log(UAV_Hight, 10)) * math.log(distence_U2K.iloc[i, j],
                                                                                  10) + 20 * math.log(f_c, 10)
                L_NLos = np.max([L_Los,
                                 32.4 + (43.2 - 7.6 * math.log(UAV_Hight, 10)) * math.log(distence_U2K.iloc[i, j],
                                                                                          10) + 20 * math.log(f_c, 10)])

                Avg_Los = P_Los * L_Los + P_NLos * L_NLos
                gain = np.random.rayleigh(scale=1, size=None) * pow(10, (-Avg_Los / 10))
                self.Propergation_Loss.iloc[i, j] = gain

        return self.Propergation_Loss

    def Get_Channel_Gain_NOMA(self, UAVsnumber, Usersnumber, PropergationLosslist, UserAssociationlist,
                              Noise_Power):  #  calculating channel gain

        for j in range(
                Usersnumber):
            i_Server_UAV = UserAssociationlist.iloc[0, j]
            Signal_power = self.amplification_constant * PropergationLosslist.iloc[i_Server_UAV, j]
            ChannelGain = Signal_power / (Noise_Power)
            self.ChannelGain_list.iloc[0, j] = ChannelGain

        return self.ChannelGain_list

    def Get_Eq_CG(self, UAVsnumber, Usersnumber, PropergationLosslist, UserAssociationlist,
                  Noise_Power):  #  calculate the equivalent channel gain to determine SIC decoding order

        for j in range(
                Usersnumber):
            i_Server_UAV = UserAssociationlist.iloc[0, j]
            Signal_power = 100 * self.amplification_constant * PropergationLosslist.iloc[
                0, j]
            I_inter_cluster = 0

            for j_idx in range(Usersnumber):
                if UserAssociationlist.iloc[0, j_idx] == i_Server_UAV:
                    pass
                else:
                    Inter_UAV = UserAssociationlist.iloc[0, j_idx]
                    I_inter_cluster = I_inter_cluster + (
                            100 * self.amplification_constant * PropergationLosslist.iloc[
                        Inter_UAV, j])

            Eq_CG = Signal_power / (I_inter_cluster + Noise_Power)
            self.Eq_CG_list.iloc[0, j] = Eq_CG

        return self.Eq_CG_list

    def Get_SINR_NNOMA(self, UAVsnumber, Usersnumber, PropergationLosslist, UserAssociationlist, ChannelGain_list,
                       Noise_Power):
        #  calculate the SINR

        for j in range(
                Usersnumber):
            i_Server_UAV = UserAssociationlist.iloc[0, j]
            Signal_power = self.Power_allocation_list.iloc[0, j] * PropergationLosslist.iloc[
                i_Server_UAV, j]
            I_inter_cluster = 0

            for j_idx in range(Usersnumber):
                if UserAssociationlist.iloc[0, j_idx] == i_Server_UAV:
                    if ChannelGain_list.iloc[0, j] < ChannelGain_list.iloc[
                        0, j_idx] and j != j_idx:
                        I_inter_cluster = I_inter_cluster + (
                                self.Power_allocation_list.iloc[0, j_idx] * PropergationLosslist.iloc[
                            i_Server_UAV, j])

                else:
                    Inter_UAV = UserAssociationlist.iloc[
                        0, j_idx]
                    I_inter_cluster = I_inter_cluster + (
                                self.Power_allocation_list.iloc[0, j_idx] * PropergationLosslist.iloc[Inter_UAV, j])  #

            SINR = Signal_power / (I_inter_cluster + Noise_Power)
            self.SINR_list.iloc[0, j] = SINR

        return self.SINR_list

    def Calcullate_Datarate(self, SINRlist, Usersnumber, B):
        # calculate data rate
        for j in range(Usersnumber):

            if SINRlist.iloc[0, j] <= 0:
                print(SINRlist)
                warnings.warn(
                    'SINR wrong')

            self.Daterate.iloc[0, j] = B * math.log((1 + SINRlist.iloc[0, j]), 2)

        SumDataRate = sum(self.Daterate.iloc[0, :])
        Worst_user_rate = min(self.Daterate.iloc[0, :])
        return self.Daterate, SumDataRate, Worst_user_rate

    def Reset_position(self):
        self.PositionOfUsers = copy.deepcopy(self.Init_PositionOfUsers)
        self.PositionOfUAVs = copy.deepcopy(self.Init_PositionOfUAVs)
        return

    def Create_state_Noposition(self, serving_UAV, User_association_list, User_Channel_Gain):

        UAV_position_copy = copy.deepcopy(self.PositionOfUAVs.values)
        UAV_position_copy[:, [0, serving_UAV]] = UAV_position_copy[:, [serving_UAV,
                                                                       0]]
        User_Channel_Gain_copy = copy.deepcopy(User_Channel_Gain.values[0])

        for UAV in range(NumberOfUAVs):
            self.State[0, 3 * UAV:3 * UAV + 3] = UAV_position_copy[:,
                                                 UAV].T

        User_association_copy = copy.deepcopy(User_association_list.values)
        desirable_user = np.where(User_association_copy[0] == serving_UAV)[0]

        for i in range(len(desirable_user)):
            User_Channel_Gain_copy[i], User_Channel_Gain_copy[desirable_user[i]] = User_Channel_Gain_copy[
                desirable_user[i]], User_Channel_Gain_copy[
                i]

        for User in range(NumberOfUsers):
            self.State[0, (3 * UAV + 3) + User] = User_Channel_Gain_copy[User].T

        Stat_for_return = copy.deepcopy(self.State)
        return Stat_for_return

    def take_action_NOMA(self, action_number, acting_UAV, User_asso_list, ChannelGain_list):
        UAV_move_direction = action_number % 7
        if UAV_move_direction == 0:
            self.PositionOfUAVs.iloc[0, acting_UAV] += self.UAVspeed
            if self.PositionOfUAVs.iloc[0, acting_UAV] > self.Zone_border_X:
                self.PositionOfUAVs.iloc[0, acting_UAV] = self.Zone_border_X
        elif UAV_move_direction == 1:
            self.PositionOfUAVs.iloc[0, acting_UAV] -= self.UAVspeed
            if self.PositionOfUAVs.iloc[0, acting_UAV] < 0:
                self.PositionOfUAVs.iloc[0, acting_UAV] = 0
        elif UAV_move_direction == 2:
            self.PositionOfUAVs.iloc[1, acting_UAV] += self.UAVspeed
            if self.PositionOfUAVs.iloc[1, acting_UAV] > self.Zone_border_Y:
                self.PositionOfUAVs.iloc[1, acting_UAV] = self.Zone_border_Y
        elif UAV_move_direction == 3:
            self.PositionOfUAVs.iloc[1, acting_UAV] -= self.UAVspeed
            if self.PositionOfUAVs.iloc[1, acting_UAV] < 0:
                self.PositionOfUAVs.iloc[1, acting_UAV] = 0
        elif UAV_move_direction == 4:
            self.PositionOfUAVs.iloc[2, acting_UAV] += self.UAVspeed
            if self.PositionOfUAVs.iloc[2, acting_UAV] > self.Zone_border_Z:
                self.PositionOfUAVs.iloc[2, acting_UAV] = self.Zone_border_Z
        elif UAV_move_direction == 5:
            self.PositionOfUAVs.iloc[2, acting_UAV] -= self.UAVspeed
            if self.PositionOfUAVs.iloc[2, acting_UAV] < 20:
                self.PositionOfUAVs.iloc[2, acting_UAV] = 20
        elif UAV_move_direction == 6:
            pass

        # Power allocation
        power_allocation_scheme = action_number // 7
        acting_user_list = np.where(User_asso_list.iloc[0, :] == acting_UAV)[0]
        First_user = acting_user_list[0]
        Second_user = acting_user_list[1]

        # SIC
        first_user_CG = ChannelGain_list.iloc[0, First_user]
        second_user_CG = ChannelGain_list.iloc[0, Second_user]
        if first_user_CG >= second_user_CG:
            User0 = Second_user
            User1 = First_user
        else:
            User0 = First_user
            User1 = Second_user

        if power_allocation_scheme % 3 == 0:
            self.Power_allocation_list.iloc[0, User0] = self.Power_unit * 2
        elif power_allocation_scheme % 3 == 1:
            self.Power_allocation_list.iloc[0, User0] = self.Power_unit * 4
        elif power_allocation_scheme % 3 == 2:
            self.Power_allocation_list.iloc[0, User0] = self.Power_unit * 7
        # for the strong user, the power levels can be 1, 1/2, 1/4 * power unit
        if power_allocation_scheme // 3 == 0:
            self.Power_allocation_list.iloc[0, User1] = self.Power_unit
        elif power_allocation_scheme // 3 == 1:
            self.Power_allocation_list.iloc[0, User1] = self.Power_unit / 2
        elif power_allocation_scheme // 3 == 2:
            self.Power_allocation_list.iloc[0, User1] = self.Power_unit / 4

def generate_coordinates(PositionOfUAVs, num_coords, std_dev=10):
    coords_x = []
    coords_y = []
    for i in range(num_coords):

        idx = i % len(PositionOfUAVs.columns)
        new_x = np.random.normal(PositionOfUAVs.iloc[0, idx], std_dev)
        new_y = np.random.normal(PositionOfUAVs.iloc[1, idx], std_dev)

        coords_x.append(new_x)
        coords_y.append(new_y)
        coords = [coords_x, coords_y]
    return coords

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
                    WorstuserRate_seq[t] = WorstuserRate

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



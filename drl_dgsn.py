import numpy as np
import pandas as pd
import warnings
from collections import deque
import random
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Lambda
import tensorflow as tf
from sklearn.mixture import GaussianMixture



UserNumberPerCell = 2  # user number per UAV
NumberOfUAVs = 3  # number of UAVs
NumberOfCells = NumberOfUAVs  # Each UAV is responsible for one cell
NumberOfUsers = NumberOfUAVs * UserNumberPerCell

# 291199,1234567
tf.random.set_seed(1234567)

class Agent(object):
    def __init__(self):
        self.update_freq = 300
        self.replay_size = 50000
        self.step = 0
        self.replay_queue = deque(maxlen=self.replay_size)

        self.power_number = 3 ** UserNumberPerCell
        self.action_number = 7 * self.power_number

        self.model = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):

        STATE_DIM = NumberOfUAVs * 3 + NumberOfUsers
        ACTION_DIM = 7 * self.power_number

        input_layer = Input(shape=(STATE_DIM,))


        shared_layer = Dense(128, activation='relu')(input_layer)


        value_layer = Dense(64, activation='relu')(shared_layer)
        value = Dense(1)(value_layer)


        advantage_layer = Dense(64, activation='relu')(shared_layer)
        advantage = Dense(ACTION_DIM)(advantage_layer)


        def combine_value_advantage(inputs):
            value, advantage = inputs
            advantage_mean = tf.reduce_mean(advantage, axis=1, keepdims=True)
            q_values = value + (advantage - advantage_mean)
            return q_values

        q_values = Lambda(combine_value_advantage)([value, advantage])


        model = Model(inputs=input_layer, outputs=q_values)

        model.compile(optimizer='adam', loss='mean_squared_error')

        return model

    def Choose_action(self, s, epsilon):

        if np.random.uniform() < epsilon:
            return np.random.choice(self.action_number)
        else:
            return np.argmax(self.model.predict(s))

    def remember(self, s, a, next_s, reward):
        # save MDP transitions
        self.replay_queue.append((s, a, next_s, reward))

    def train(self, batch_size=64, lr=0.00001, factor=0.999):
        if len(self.replay_queue) < self.replay_size:
            return
        self.step += 1


        if self.step % self.update_freq == 0:
            self.target_model.set_weights(self.model.get_weights())

        replay_batch = random.sample(self.replay_queue, batch_size)
        s_batch = np.array([replay[0] for replay in replay_batch])
        a_batch = np.array([replay[1] for replay in replay_batch])
        next_s_batch = np.array([replay[2] for replay in replay_batch])
        reward_batch = np.array([replay[3] for replay in replay_batch])

        Q = self.model.predict(s_batch)
        Q_next_online = self.model.predict(next_s_batch)
        Q_next_target = self.target_model.predict(next_s_batch)


        next_actions = np.argmax(Q_next_online, axis=1)

        for i in range(len(replay_batch)):
            a = a_batch[i]
            reward = reward_batch[i]

            Q[i][a] = (1 - lr) * Q[i][a] + lr * (reward + factor * Q_next_target[i][next_actions[i]])

        self.model.fit(s_batch, Q, verbose=0)  # DNN training

    def User_association(self, UAV_Position, User_Position, UAVsnumber, Usersnumber):
        # this function is DUA-GMM
        User_Position_array = np.zeros([Usersnumber, 2])

        User_Position_array[:, 0] = User_Position.iloc[0, :].T
        User_Position_array[:, 1] = User_Position.iloc[1, :].T

        gmm = GaussianMixture(n_components=UAVsnumber, random_state=42)
        gmm.fit(User_Position_array)
        User_cluster = gmm.predict(User_Position_array)
        Cluster_center = gmm.means_


        for dectecter in range(UAVsnumber):
            user_numberincluster = np.where(User_cluster == dectecter)[0]
            if len(user_numberincluster) == (Usersnumber / UAVsnumber):
                pass
            else:
                cluster_redun = []
                cluster_lack = []
                Cluster_center_of_lack = []

                for ck_i in range(len(Cluster_center)):
                    User_for_cluster_i = np.where(User_cluster == ck_i)

                    if np.size(User_for_cluster_i) > (
                            Usersnumber / UAVsnumber):
                        for i in range(int(np.size(User_for_cluster_i) - (Usersnumber / UAVsnumber))):
                            cluster_redun.append(ck_i)

                    if np.size(User_for_cluster_i) < (Usersnumber / UAVsnumber):
                        for i in range(int((Usersnumber / UAVsnumber) - np.size(User_for_cluster_i))):
                            cluster_lack.append(ck_i)
                            Cluster_center_of_lack.append(Cluster_center[ck_i, :])

                # this function is UCA
                for fixer_i in range(np.size(cluster_lack)):

                    current_lack_cluster = cluster_lack[fixer_i]
                    lack_cluster_center = Cluster_center_of_lack[fixer_i]
                    current_redun_cluster = cluster_redun[fixer_i]


                    redun_user_indices = np.where(User_cluster == current_redun_cluster)[0]
                    if len(redun_user_indices) == 0:
                        continue


                    redun_user_positions = User_Position_array[redun_user_indices, :]


                    distances = np.zeros(len(redun_user_indices))
                    for i in range(len(redun_user_indices)):
                        distances[i] = np.linalg.norm(redun_user_positions[i] - lack_cluster_center)


                    min_dist_idx = np.argmin(distances)
                    target_user_idx = redun_user_indices[min_dist_idx]


                    User_cluster_fixed = User_cluster.copy()
                    User_cluster_fixed[target_user_idx] = current_lack_cluster
                    User_cluster = User_cluster_fixed



        # UAV - cluster
        UAV_Position_array = np.zeros([UAVsnumber, 2])
        UAV_Position_array[:, 0] = UAV_Position.iloc[0, :].T
        UAV_Position_array[:, 1] = UAV_Position.iloc[1, :].T

        User_association_list = pd.DataFrame(
            np.zeros((1, Usersnumber)),
            columns=np.arange(Usersnumber).tolist(),
        )

        for UAV_name in range(UAVsnumber):
            distence_UAVi2C = np.zeros(UAVsnumber)
            for cluster_center_i in range(UAVsnumber):
                distence_UAVi2C[cluster_center_i] = np.linalg.norm(
                    UAV_Position_array[UAV_name, :] - Cluster_center[cluster_center_i])
            Servied_cluster = np.where(
                distence_UAVi2C == np.min(distence_UAVi2C))
            Cluster_center[Servied_cluster] = 9999
            Servied_cluster_list = Servied_cluster[0]
            Servied_users = np.where(User_cluster == Servied_cluster_list)
            Servied_users_list = Servied_users[0]

            for i in range(np.size(Servied_users)):
                User_association_list.iloc[0, Servied_users_list[i]] = int(
                    UAV_name)

            User_association_list = User_association_list.astype('int')

        return User_association_list


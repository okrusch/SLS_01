import random

from agents.AbstractAgent import AbstractAgent
import pandas as pd
import numpy as np

def get_row_index_in_string_format(state):
    """
        Returns a state (row index) as q-table row index string.

                Parameters:
                        state ([int, int]): The distance between agent (marine) and beacon.

                Returns:
                        state (str): Transformed state, so it can be used as index in the q-table.
    """
    return "("+str(state[0]) + "," + str(state[1]) + ")"


class QLearningAgent(AbstractAgent):
    def __init__(self, train, screen_size, explore=1):
        super(QLearningAgent, self).__init__(screen_size)
        # TODO Initialize all hyperparameter and the q-table (with the helper function below)
        self.train = train
        self.explore = explore
        self.actions = self._DIRECTIONS.keys()
        self.states = []
        for x in range(-64, 65):
            for y in range(-64, 65):
                self.states.append("("+str(x) + "," + str(y) + ")")
        self.q_table = self.init_q_table()
        self.alpha = 0.1
        self.gamma = 0.9
        self.old_state = None
        self.old_action = None

    def step(self, obs, epsilon):
        # TODO step method
        if self._MOVE_SCREEN.id in obs.observation.available_actions:
            # get q_state from position
            marine = self._get_marine(obs)
            if marine is None:
                return self._NO_OP
            marine_coordinates = self._get_unit_pos(marine)
            beacon = self._get_beacon(obs)
            if beacon is None:
                return self._NO_OP
            beacon_coordinates = self._get_unit_pos(beacon)

            q_state = self.get_q_state_from_position(marine_position=marine_coordinates,
                                                     beacon_position=beacon_coordinates)

            # epsilon integration
            rnd = random.random()
            if rnd > epsilon:
                action = self.get_new_action(q_state)
            else:
                action = random.choice(list(self.actions))

            if self.train:
                if self.old_state == None and self.old_action == None:
                    # first step where there is no previous state
                    self.old_state = get_row_index_in_string_format(q_state)
                    self.old_action = action
                else:
                    t = obs.reward == 1 # terminate when beacon reached
                    self.update_q_value(self.old_state, self.old_action, marine_coordinates, obs.reward, t) # update q_value

                    # set previous state and action
                    self.old_state = get_row_index_in_string_format(q_state)
                    self.old_action = action

                return self._dir_to_sc2_action(action, marine_coordinates)
            else:
                return self._dir_to_sc2_action(action, marine_coordinates)
        else:
            self.old_state = None
            self.old_action = None
            return self._SELECT_ARMY    # initialize army in first step

    def save_model(self, path):
        # save model as pkl
        self.q_table.to_pickle(path + ".pkl")

    def load_model(self, path):
        # load model from pkl
        self.q_table = pd.read_pickle(path + ".pkl")

    def get_new_action(self, state):
        """
            Returns the action to execute.

                    Parameters:
                            state ([int, int]): A row index (a state) of the q-table.

                    Returns:
                            action (str): e.g. 'N', 'NW', 'NO', ...
        """
        # TODO get_new_action method
        index = get_row_index_in_string_format(state)
        options = self.q_table.loc[index]
        m = max(options)
        indices = [index for index, value in enumerate(options) if value == m]
        choice = random.choice(indices)
        action = list(self.actions)[choice]
        return action

    def get_q_value(self, q_table_column_index, q_table_row_index):
        """
            Returns a q-value.

                    Parameters:
                            q_table_column_index (str): The column index of the searched value (the action).
                            q_table_row_index (str): The row index of the searched value (the state).

                    Returns:
                            action (float): The value for the given indices.
        """
        # TODO get_new_action method
        q_value = self.q_table.loc[q_table_column_index][q_table_row_index]
        return float(q_value)

    def update_q_value(self, old_state, old_action, new_state, reward, terminal):
        # TODO update_q_value method
        new_state_str = get_row_index_in_string_format(new_state)
        q_value = self.get_q_value(q_table_column_index=old_state,
                                   q_table_row_index=old_action)
        if not terminal:
            max_new = max(self.q_table.loc[new_state_str])
            new_q_value = q_value + self.alpha * (reward + (self.gamma * max_new) - q_value)
        else:
            new_q_value = q_value + self.alpha * (reward - q_value)
            print("final", old_state, new_q_value)

        self.q_table.at[old_state, old_action] = new_q_value



    def get_q_state_from_position(self, marine_position, beacon_position):
        """
            Transforms the position of agent (marine) and beacon into a q-table row index.

                    Parameters:
                            marine_position ([int, int]): The position of the agent (marine).
                            beacon_position ([int, int]): The position of the beacon.

                    Returns:
                            state ([int, int]): A row index (a state) of the q-table.
        """
        # TODO get_q_state_from_position method
        x_dif = marine_position[0] - beacon_position[0]
        y_dif = marine_position[1] - beacon_position[1]
        return (x_dif, y_dif)

    def init_q_table(self):
        """
            Initializes the q-table

                    Returns:
                            q_table (panda.Dataframe): The q-table.
                                                       The row indices must be in the format '(x,y)'
                                                       The column indices must be in the format 'action' (e.g. 'W')
        """
        return pd.DataFrame(np.random.rand(len(self.states), len(self.actions)), index=self.states, columns=self.actions)

        #return pd.DataFrame(0, index=self.states, columns=self.actions)
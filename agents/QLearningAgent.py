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
                self.states.append((x, y))
        self.q_table = self.init_q_table()

        pass


    def step(self, obs):
        # TODO step method
        if self._MOVE_SCREEN.id in obs.observation.available_actions:
            pass
        else:
            return self._SELECT_ARMY    # initialize army in first step

    def save_model(self, path):
        self.q_table.to_pickle(path)

    def load_model(self, path):
        self.q_table = pd.read_pickle(path)

    def get_new_action(self, state):
        """
            Returns the action to execute.

                    Parameters:
                            state ([int, int]): A row index (a state) of the q-table.

                    Returns:
                            action (str): e.g. 'N', 'NW', 'NO', ...
        """
        # TODO get_new_action method
        ...

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
        ...

    def update_q_value(self, old_state, old_action, new_state, reward, terminal):
        # TODO update_q_value method
        ...

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
        ...

    def init_q_table(self):
        """
            Initializes the q-table

                    Returns:
                            q_table (panda.Dataframe): The q-table.
                                                       The row indices must be in the format '(x,y)'
                                                       The column indices must be in the format 'action' (e.g. 'W')
        """
        return pd.DataFrame(np.random.rand(len(self.states), len(self.actions)), index=self.states, columns=self.actions)
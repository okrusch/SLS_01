from agents.AbstractAgent import AbstractAgent


class BasicAgent(AbstractAgent):
    def __init__(self, train, screen_size):
        super(BasicAgent, self).__init__(screen_size)
        self.old_state = None
        self.old_action = None

    def step(self, obs):
        if self._MOVE_SCREEN.id in obs.observation.available_actions:
            marine = self._get_marine(obs)
            if marine is None:
                return self._NO_OP
            marine_coordinates = self._get_unit_pos(marine)
            marine_x, marine_y = marine_coordinates[0], marine_coordinates[1]

            beacon = self._get_beacon(obs)
            if beacon is None:
                return self._NO_OP
            beacon_coordinates = self._get_unit_pos(beacon)
            beacon_x, beacon_y = beacon_coordinates[0], beacon_coordinates[1]

            move = ""
            if(marine_y < beacon_y):
                move += "S"
            elif (marine_y > beacon_y):
                move += "N"

            if(marine_x < beacon_x):
                move += "E"
            elif(marine_x > beacon_x):
                move += "W"

            assert move != ""

            self.old_state = marine_coordinates
            self.old_action = move
            x = obs.reward

            return self._dir_to_sc2_action(move, marine_coordinates)
        else:
            return self._SELECT_ARMY

    def save_model(self, filename):
        pass

    def load_model(self, filename):
        pass

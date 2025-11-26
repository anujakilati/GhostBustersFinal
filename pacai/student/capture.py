import typing

import pacai.agents.greedy
import pacai.capture.gamestate
import pacai.core.action
import pacai.core.agent
import pacai.core.agentinfo
import pacai.core.board
import pacai.core.features
import pacai.core.gamestate
import pacai.pacman.board
import pacai.search.distance

GHOST_IGNORE_RANGE: float = 2.5


def create_team() -> list[pacai.core.agentinfo.AgentInfo]:
    """
    Return the agent information used to create a capture team.

    We use one OffensiveAgent and one DefensiveAgent.
    Using __name__ keeps things working even if the file is renamed
    by the tournament harness.
    """
    offensive_info = pacai.core.agentinfo.AgentInfo(
        name = f"{__name__}.OffensiveAgent"
    )
    defensive_info = pacai.core.agentinfo.AgentInfo(
        name = f"{__name__}.DefensiveAgent"
    )

    return [offensive_info, defensive_info]


class DefensiveAgent(pacai.agents.greedy.GreedyFeatureAgent):
    """
    A capture agent that prioritizes defending its own territory.
    """

    def __init__(
            self,
            override_weights: dict[str, float] | None = None,
            **kwargs: typing.Any) -> None:

        kwargs['feature_extractor_func'] = _extract_baseline_defensive_features
        super().__init__(**kwargs)

        self._distances: pacai.search.distance.DistancePreComputer = (
            pacai.search.distance.DistancePreComputer()
        )

        # Base defensive weights.
        self.weights['on_home_side'] = 100.0
        self.weights['stopped'] = -100.0
        self.weights['reverse'] = -0.5
        self.weights['num_invaders'] = -1000.0
        self.weights['distance_to_invader'] = -40.0

        # Patrol toward the center line when no invaders are visible.
        self.weights['distance_to_home_center'] = -3.0

        # When scared, prefer to increase distance from invaders.
        self.weights['scared_distance_to_invader'] = 25.0

        if override_weights is None:
            override_weights = {}

        for (key, weight) in override_weights.items():
            self.weights[key] = weight

    def game_start(self, initial_state: pacai.core.gamestate.GameState) -> None:
        """
        Precompute distances for this board at the start of the game.
        """
        self._distances.compute(initial_state.board)


class OffensiveAgent(pacai.agents.greedy.GreedyFeatureAgent):
    """
    A capture agent that prioritizes offense on the opponent side.
    """

    def __init__(
            self,
            override_weights: dict[str, float] | None = None,
            **kwargs: typing.Any) -> None:

        kwargs['feature_extractor_func'] = _extract_baseline_offensive_features
        super().__init__(**kwargs)

        self._distances: pacai.search.distance.DistancePreComputer = (
            pacai.search.distance.DistancePreComputer()
        )
        self._last_food_count: int | None = None

        # Base offensive weights.

        # Care a lot about overall score.
        self.weights['score'] = 125.0
        # Smart food choice (closer food is strongly preferred).
        self.weights['distance_to_food'] = -4.5
        # Movement smoothness.
        self.weights['stopped'] = -50.0
        # Allow backtracking for safety.
        self.weights['reverse'] = -0.5

        # Ghost avoidance.
        # Positive weight means larger distance is better.
        self.weights['ghost_too_close'] = 50.0
        # Extra penalty when very close (moderate spike).
        self.weights['distance_to_ghost_squared'] = -1.5
        self.weights['distance_to_home_if_ghost_close'] = -9.0

        # Capsule logic.
        self.weights['distance_to_capsule'] = -2.0
        # Extra pull toward capsule when ghost is close.
        self.weights['escape_capsule_distance'] = -9.0

        # In general, being on home side is slightly bad for an offensive agent.
        self.weights['on_home_side'] = -2.0

        # Prefer food that is quick to grab and quick to return.
        self.weights['food_return_cost'] = -2.5

        # Additional pressure when multiple ghosts are close.
        self.weights['ghost_too_close'] = -35.0

        # Strong pull to finish the final pellet.
        self.weights['last_food'] = 150.0

        if override_weights is None:
            override_weights = {}

        for (key, weight) in override_weights.items():
            self.weights[key] = weight

    def game_start(self, initial_state: pacai.core.gamestate.GameState) -> None:
        """
        Precompute distances for this board at the start of the game.
        """
        self._distances.compute(initial_state.board)
        self._last_food_count = None

    def get_action(self, state: pacai.core.gamestate.GameState) -> pacai.core.action.Action:
        """
        Log food/capsule counts once per real turn (not per successor).
        """
        food_count = len(state.get_food(agent_index = self.agent_index))
        # Count only opponent-side capsules.
        capsule_count = 0
        for cap_pos in state.board.get_marker_positions(pacai.pacman.board.MARKER_CAPSULE):
            if (state._team_side(position = cap_pos) != state._team_modifier(agent_index = self.agent_index)):
                capsule_count += 1

        total_targets = food_count + capsule_count
        if (self._last_food_count is None):
            self._last_food_count = total_targets
        elif (total_targets < self._last_food_count):
            print(f"[Offense] Food remaining: {total_targets}")
            self._last_food_count = total_targets

        # If exactly one edible target remains, log it and greedily move toward it.
        remaining_food = state.get_food(agent_index = self.agent_index)
        if (total_targets == 1) and (len(remaining_food) == 1):
            only_food = next(iter(remaining_food))
            print(f"[Offense] Last food at: {only_food}")

            legal_actions = state.get_legal_actions()
            if (pacai.core.action.STOP in legal_actions and len(legal_actions) > 1):
                legal_actions.remove(pacai.core.action.STOP)

            best_actions: list[pacai.core.action.Action] = []
            best_distance: int | None = None

            for action in legal_actions:
                succ = state.generate_successor(action, self.rng)
                succ_pos = succ.get_agent_position(self.agent_index)
                if succ_pos is None:
                    continue
                d = self._distances.get_distance(succ_pos, only_food)
                if d is None:
                    continue
                if (best_distance is None) or (d < best_distance):
                    best_distance = d
                    best_actions = [action]
                elif d == best_distance:
                    best_actions.append(action)

            if best_actions:
                return self.rng.choice(best_actions)

        return super().get_action(state)


def _extract_baseline_defensive_features(
        state: pacai.core.gamestate.GameState,
        action: pacai.core.action.Action,
        agent: pacai.core.agent.Agent | None = None,
        **kwargs: typing.Any) -> pacai.core.features.FeatureDict:
    agent = typing.cast(DefensiveAgent, agent)
    state = typing.cast(pacai.capture.gamestate.GameState, state)

    features: pacai.core.features.FeatureDict = pacai.core.features.FeatureDict()

    current_position = state.get_agent_position(agent.agent_index)
    if (current_position is None):
        # We are dead and waiting to respawn.
        return features

    # Note the side of the board we are on.
    features['on_home_side'] = int(state.is_ghost(agent_index = agent.agent_index))

    # Prefer moving over stopping.
    features['stopped'] = int(action == pacai.core.action.STOP)

    # Prefer not turning around.
    # Remember that the state we get is already a successor, so we have to look two actions back.
    agent_actions = state.get_agent_actions(agent.agent_index)
    if (len(agent_actions) > 1):
        features['reverse'] = int(action == state.get_reverse_action(agent_actions[-2]))
    else:
        features['reverse'] = 0

    # We don't like any invaders on our side.
    invader_positions = state.get_invader_positions(agent_index = agent.agent_index)
    features['num_invaders'] = len(invader_positions)

    # Hunt down the closest invader!
    if invader_positions:
        distances: list[int] = []
        for inv_pos in invader_positions.values():
            d = agent._distances.get_distance(current_position, inv_pos)
            if d is not None:
                distances.append(d)

        if distances:
            features['distance_to_invader'] = min(distances)
        else:
            # If no distances could be computed, fall back to 0.
            features['distance_to_invader'] = 0
    else:
        # No invaders right now.
        features['distance_to_invader'] = 0

    # patrol if no invaders
    # compute distance to center of home territory
    board = state.board
    mid_x = board.width // 2
    # if red agent, home is left side; if blue, home is right side
    # Even agent indexes (modifier -1) are the red team on the left; odds are blue on the right.
    if state._team_modifier(agent.agent_index) == -1:  # CHANGED: replace missing is_red with team modifier
        home_center = pacai.core.board.Position(current_position.row, mid_x - 1)  # CHANGED: build Position not tuple
    else:
        home_center = pacai.core.board.Position(current_position.row, mid_x + 1)  # CHANGED: build Position not tuple

    d_center = agent._distances.get_distance(current_position, home_center)
    features['distance_to_home_center'] = d_center if d_center is not None else 0

    if state.is_scared(agent.agent_index) and invader_positions:  # CHANGED: use is_scared (get_agent_state missing)
        scared_distances: list[int] = []
        for inv_pos in invader_positions.values():
            d = agent._distances.get_distance(current_position, inv_pos)
            if d is not None:
                scared_distances.append(d)

        if scared_distances:
            features['scared_distance_to_invader'] = min(scared_distances)
        else:
            features['scared_distance_to_invader'] = 0
    else:
        features['scared_distance_to_invader'] = 0

    return features

def _extract_baseline_offensive_features(
        state: pacai.core.gamestate.GameState,
        action: pacai.core.action.Action,
        agent: pacai.core.agent.Agent | None = None,
        **kwargs: typing.Any) -> pacai.core.features.FeatureDict:
    agent = typing.cast(OffensiveAgent, agent)
    state = typing.cast(pacai.capture.gamestate.GameState, state)

    features: pacai.core.features.FeatureDict = pacai.core.features.FeatureDict()
    features['score'] = state.get_normalized_score(agent.agent_index)

    # Note the side of the board we are on.
    features['on_home_side'] = int(state.is_ghost(agent_index = agent.agent_index))

    # Prefer moving over stopping.
    features['stopped'] = int(action == pacai.core.action.STOP)

    # Prefer not turning around.
    # Remember that the state we get is already a successor, so we have to look two actions back.
    agent_actions = state.get_agent_actions(agent.agent_index)
    if (len(agent_actions) > 1):
        features['reverse'] = int(action == state.get_reverse_action(agent_actions[-2]))
    else:
        features['reverse'] = 0

    current_position = state.get_agent_position(agent.agent_index) 
    if (current_position is None): # We are dead and waiting to respawn. 
        return features

    features['on_home_side'] = int(state.is_ghost(agent_index = agent.agent_index))

    food_positions = state.get_food(agent_index = agent.agent_index)
    food_list = list(food_positions)  # CHANGED: convert set to list (was .as_list())
    features['last_food'] = int(len(food_list) == 1)  # CHANGED: reward grabbing the final pellet
    if (len(food_list) > 0):
        food_distances = [agent._distances.get_distance(current_position, f) for f in food_list if agent._distances.get_distance(current_position, f) is not None]
        min_food_dist = min(food_distances) if food_distances else 9999
        features['distance_to_food'] = min_food_dist
        if len(food_list) == 1:
            features['distance_to_food'] = -1000.0
            features['reverse'] = 0
            features['stopped'] = 0
            features["ghost_too_close"] = 0
            features["distance_to_home_if_ghost_close"] = 0
            features["distance_to_ghost_squared"] = 0
    else:
        # There is no food left, give a large score.
        features['distance_to_food'] = 9999

    ghost_positions = state.get_nonscared_opponent_positions(agent_index = agent.agent_index)
    if (len(ghost_positions) > 0):
        ghost_distances = [agent._distances.get_distance(current_position, g) for g in ghost_positions.values() if agent._distances.get_distance(current_position, g) is not None]

        if ghost_distances:
            d = min(ghost_distances)
            features["distance_to_ghost_squared"] = d ** 2
            features["ghost_too_close"] = 1 if d < GHOST_IGNORE_RANGE else 0
            features["distance_to_home_if_ghost_close"] = 0 if d >= GHOST_IGNORE_RANGE else d
        else:
            features["distance_to_ghost_squared"] = 0
            features["ghost_too_close"] = 0
            features["distance_to_home_if_ghost_close"] = 0

    else:
        features["distance_to_ghost_squared"] = 0
        features["ghost_too_close"] = 0
        features["distance_to_home_if_ghost_close"] = 0

    return features

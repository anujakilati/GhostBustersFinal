import typing

import pacai.agents.greedy
import pacai.capture.gamestate
import pacai.core.action
import pacai.core.agent
import pacai.core.agentinfo
import pacai.core.features
import pacai.core.gamestate
import pacai.search.distance

GHOST_IGNORE_RANGE: float = 3.0


def create_team() -> list[pacai.core.agentinfo.AgentInfo]:
    """
    Return the agent information used to create a capture team.

    We use one OffensiveAgent and one DefensiveAgent.
    Using __name__ keeps things working even if the file is renamed
    by the tournament harness.
    """
    offensive_info = pacai.core.agentinfo.AgentInfo(
        name=f"{__name__}.OffensiveAgent"
    )
    defensive_info = pacai.core.agentinfo.AgentInfo(
        name=f"{__name__}.DefensiveAgent"
    )

    return [offensive_info, defensive_info]


class DefensiveAgent(pacai.agents.greedy.GreedyFeatureAgent):
    """
    Simple defensive agent:
    - Stay on home side.
    - Chase visible invaders.
    - If no invaders, drift toward nearest opponent (pressure).
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
        self.weights['reverse'] = -2.0
        self.weights['num_invaders'] = -1000.0
        self.weights['distance_to_invader'] = -10.0

        # When no invaders, move a bit toward opponents.
        self.weights['distance_to_opponent'] = -1.0

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
    Simple offensive agent (Reflex-style):
    - Maximize score.
    - Move toward nearest food.
    - Avoid getting too close to ghosts.
    - Penalize stopping and tight back-and-forth reversals to reduce loops.
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

        # Base offensive weights (tuned to reduce looping).
        self.weights['score'] = 100.0

        # Smart food choice: nearer food is better.
        self.weights['distance_to_food'] = -2.0

        # Movement smoothness / anti-loop.
        self.weights['stopped'] = -150.0
        self.weights['reverse'] = -4.0

        # Ghost avoidance:
        # ghost_too_close is 1 when a ghost is within GHOST_IGNORE_RANGE.
        self.weights['ghost_too_close'] = -200.0
        # Mild penalty for being generally closer to ghosts.
        self.weights['distance_to_ghost_squared'] = -0.1

        # On home side is slightly bad for an offensive agent.
        self.weights['on_home_side'] = -5.0

        # Keep a tiny bias to eat remaining food and capsules.
        self.weights['num_food'] = -3.0
        self.weights['num_capsules'] = -2.0

        if override_weights is None:
            override_weights = {}

        for (key, weight) in override_weights.items():
            self.weights[key] = weight

    def game_start(self, initial_state: pacai.core.gamestate.GameState) -> None:
        """
        Precompute distances for this board at the start of the game.
        """
        self._distances.compute(initial_state.board)


def _extract_baseline_defensive_features(
        state: pacai.core.gamestate.GameState,
        action: pacai.core.action.Action,
        agent: pacai.core.agent.Agent | None = None,
        **kwargs: typing.Any) -> pacai.core.features.FeatureDict:
    """
    Defensive feature extractor:
    - on_home_side
    - stopped
    - reverse
    - num_invaders
    - distance_to_invader
    - distance_to_opponent (when there are no invaders)
    """
    agent = typing.cast(DefensiveAgent, agent)
    state = typing.cast(pacai.capture.gamestate.GameState, state)

    features: pacai.core.features.FeatureDict = pacai.core.features.FeatureDict()

    current_position = state.get_agent_position(agent.agent_index)
    if current_position is None:
        # Dead and waiting to respawn.
        return features

    # Stay on home side as a ghost.
    features['on_home_side'] = int(state.is_ghost(agent_index=agent.agent_index))

    # Avoid stopping.
    features['stopped'] = int(action == pacai.core.action.STOP)

    # Avoid reversing if possible.
    agent_actions = state.get_agent_actions(agent.agent_index)
    if len(agent_actions) > 1:
        features['reverse'] = int(action == state.get_reverse_action(agent_actions[-2]))
    else:
        features['reverse'] = 0

    # Invaders (enemy pacmen on our side).
    invader_positions = state.get_invader_positions(agent_index=agent.agent_index)
    features['num_invaders'] = len(invader_positions)

    if invader_positions:
        # Chase closest invader.
        distances: list[int] = []
        for inv_pos in invader_positions.values():
            d = agent._distances.get_distance(current_position, inv_pos)
            if d is not None:
                distances.append(d)

        features['distance_to_invader'] = min(distances) if distances else 0
        # When invaders exist, we do not care about distance_to_opponent.
        features['distance_to_opponent'] = 0
    else:
        features['distance_to_invader'] = 0

        # No invaders: lightly pressure nearest opponent (on their side).
        opponent_positions = state.get_opponent_positions(agent_index=agent.agent_index)
        if opponent_positions:
            opp_dists: list[int] = []
            for opp_pos in opponent_positions.values():
                d = agent._distances.get_distance(current_position, opp_pos)
                if d is not None:
                    opp_dists.append(d)
            features['distance_to_opponent'] = min(opp_dists) if opp_dists else 0
        else:
            features['distance_to_opponent'] = 0

    return features


def _extract_baseline_offensive_features(
        state: pacai.core.gamestate.GameState,
        action: pacai.core.action.Action,
        agent: pacai.core.agent.Agent | None = None,
        **kwargs: typing.Any) -> pacai.core.features.FeatureDict:
    """
    Offensive feature extractor:
    - score
    - on_home_side
    - stopped
    - reverse
    - distance_to_food
    - num_food
    - num_capsules
    - ghost_too_close
    - distance_to_ghost_squared
    """
    agent = typing.cast(OffensiveAgent, agent)
    state = typing.cast(pacai.capture.gamestate.GameState, state)

    features: pacai.core.features.FeatureDict = pacai.core.features.FeatureDict()

    # Overall game score (already normalized by capture rules).
    features['score'] = state.get_normalized_score(agent.agent_index)

    # Home side (0 when we are pacman on opponent side, which we like).
    features['on_home_side'] = int(state.is_ghost(agent_index=agent.agent_index))

    # Avoid stopping.
    features['stopped'] = int(action == pacai.core.action.STOP)

    # Avoid reversing (loop breaker).
    agent_actions = state.get_agent_actions(agent.agent_index)
    if len(agent_actions) > 1:
        features['reverse'] = int(action == state.get_reverse_action(agent_actions[-2]))
    else:
        features['reverse'] = 0

    current_position = state.get_agent_position(agent.agent_index)
    if current_position is None:
        # Dead, waiting to respawn.
        return features

    if not hasattr(agent, 'distance_cache') or agent._last_position != current_position:
        agent.distance_cache = {}

        for i in state.get_food(agent.agent_index):
            d = agent._distances.get_distance(current_position, i)
            if d is not None:
                agent.distance_cache[i] = d

        ghost = state.get_nonscared_opponent_positions(
            agent_index=agent.agent_index
        )
        for i in ghost.values():
            d = agent._distances.get_distance(current_position, i)
            if d is not None:
                agent.distance_cache[i] = d
        agent._last_position = current_position

    # --- Food features ---
    food_positions = state.get_food(agent_index=agent.agent_index)
    food_list = list(food_positions)
    features['num_food'] = len(food_list)

    if food_list:
        food_distances = [agent.distance_cache[i] for i in food_list if i in agent.distance_cache]
        # for f in food_list:
        #     d = agent._distances.get_distance(current_position, f)
        #     if d is not None:
        #         food_distances.append(d)
        features['distance_to_food'] = min(food_distances) if food_distances else 0
    else:
        # No food left; treat distance_to_food as 0 (doesn't matter anymore).
        features['distance_to_food'] = 0

    # --- Capsules (just count, no fancy routing) ---
    # We only care weakly about clearing capsules.
    capsule_positions = state.board.get_marker_positions(pacai.pacman.board.MARKER_CAPSULE)
    features['num_capsules'] = len(capsule_positions)

    # --- Ghost avoidance ---
    ghost_positions = state.get_nonscared_opponent_positions(
        agent_index=agent.agent_index
    )

    ghost_too_close = 0
    ghost_dist_sq = 0.0

    if ghost_positions:
        ghost_distances: list[int] = [agent.distance_cache[i] for i in ghost_positions.values() if i in agent.distance_cache]
        # for gpos in ghost_positions.values():
        #     d = agent._distances.get_distance(current_position, gpos)
        #     if d is not None:
        #         ghost_distances.append(d)

        if ghost_distances:
            d = min(ghost_distances)
            ghost_dist_sq = float(d * d)
            if d < GHOST_IGNORE_RANGE:
                ghost_too_close = 1
    # If no ghosts visible, both stay 0.

    features['ghost_too_close'] = ghost_too_close
    features['distance_to_ghost_squared'] = ghost_dist_sq

    return features

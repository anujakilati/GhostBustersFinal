import pacai.core.agentinfo
import pacai.util.alias

def create_team() -> list[pacai.core.agentinfo.AgentInfo]:
    agent1_info = pacai.core.agentinfo.AgentInfo(name = pacai.util.alias.AGENT_DUMMY.long)
    agent2_info = pacai.core.agentinfo.AgentInfo(name = pacai.util.alias.AGENT_DUMMY.long)

    return [agent1_info, agent2_info]

def _extract_baseline_defensive_features(
        state: pacai.core.gamestate.GameState,
        action: pacai.core.action.Action,
        agent: pacai.core.agent.Agent | None = None,
        **kwargs: typing.Any) -> pacai.core.features.FeatureDict:

    agent = typing.cast(DefensiveAgent, agent)
    state = typing.cast(pacai.capture.gamestate.GameState, state)

    features: pacai.core.features.FeatureDict = pacai.core.features.FeatureDict()

    # normalized score (defenders also want to keep score stable / low for opponents)
    features['score'] = state.get_normalized_score(agent.agent_index)

    # whether we are on our home side (defenders prefer to stay home)
    features['on_home_side'] = int(state.is_ghost(agent_index = agent.agent_index))

    # prefer moving over stopping
    features['stopped'] = int(action == pacai.core.action.STOP)

    # prefer not turning around
    agent_actions = state.get_agent_actions(agent.agent_index)
    if (len(agent_actions) > 1):
        features['reverse'] = int(action == state.get_reverse_action(agent_actions[-2]))
    else:
        features['reverse'] = 0

    # grab our current position
    current_position = state.get_agent_position(agent.agent_index)
    if current_position is None:
        # dead / respawning: no spatial defensive features
        return features

    #identify invaders
    enemy_positions = state.get_opponent_positions(agent_index = agent.agent_index)
    enemy_is_pacman = state.get_opponent_is_pacman(agent_index = agent.agent_index)

    invader_positions = [
        pos for idx, pos in enemy_positions.items()
        if enemy_is_pacman.get(idx, False) and pos is not None
    ]

    # count how many invaders we have
    features['num_invaders'] = len(invader_positions)

    # distance to the closest invader
    if len(invader_positions) > 0:
        distances = [
            agent._distances.get_distance(current_position, inv)
            for inv in invader_positions
            if agent._distances.get_distance(current_position, inv) is not None
        ]

        if len(distances) == 0:
            #means invader is known but no distance computed (rare edge case)
            features['distance_to_invader'] = 1000
            features['distance_to_invader_squared'] = 1000000
        else:
            d = min(distances)
            features['distance_to_invader'] = d
            features['distance_to_invader_squared'] = d * d
    else:
        #no invaders right now
        features['distance_to_invader'] = 1000
        features['distance_to_invader_squared'] = 1000000

    #additional defensive logic?

    # track distance to defended food (so we stay near our resources)
    defended_food = state.get_food(agent_index = agent.agent_index)
    defended_food_list = defended_food.as_list()

    if len(defended_food_list) > 0:
        d_foods = [
            agent._distances.get_distance(current_position, f)
            for f in defended_food_list
            if agent._distances.get_distance(current_position, f) is not None
        ]
        features['distance_to_defended_food'] = min(d_foods) if d_foods else 0
    else:
        # all our food is gone,dangerous so push this high
        features['distance_to_defended_food'] = 50

    return features

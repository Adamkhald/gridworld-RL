# featureExtractors.py
# --------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"Feature extractors for Pacman game states"

from game import Directions, Actions
import util

class FeatureExtractor:
    def getFeatures(self, state, action):
        """
          Returns a dict from features to counts
          Usually, the count will just be 1.0 for
          indicator functions.
        """
        util.raiseNotDefined()

class IdentityExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[(state,action)] = 1.0
        return feats

class CoordinateExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[state] = 1.0
        feats['x=%d' % state[0]] = 1.0
        feats['y=%d' % state[0]] = 1.0
        feats['action=%s' % action] = 1.0
        return feats

def closestFood(pos, food, walls):
    """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if food[pos_x][pos_y]:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no food found
    return None

def manhattanDistance(pos1, pos2):
    """Calculate Manhattan distance between two positions"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

class SimpleExtractor(FeatureExtractor):
    """
    Returns simple features for a basic reflex Pacman:
    - whether food will be eaten
    - how far away the next food is
    - whether a ghost collision is imminent
    - whether a ghost is one step away
    """

    def getFeatures(self, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()

        features = util.Counter()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # count the number of ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

        # if there is no danger of ghosts then add the food feature
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height)
        features.divideAll(10.0)
        return features


class EnhancedExtractor(FeatureExtractor):
    """
    Enhanced feature extractor with comprehensive features
    """

    def getFeatures(self, state, action):
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()
        capsules = state.getCapsules()
        scared_times = [state.getGhostState(i).scaredTimer for i in range(1, state.getNumAgents())]
        
        features = util.Counter()
        features["bias"] = 1.0

        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # Food features
        if food[next_x][next_y]:
            features["eats-food"] = 1.0
        
        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            features["closest-food"] = float(dist) / (walls.width * walls.height)
        
        features["food-remaining"] = state.getNumFood() / (walls.width * walls.height)
        
        food_nearby = 0
        for i in range(-2, 3):
            for j in range(-2, 3):
                fx, fy = next_x + i, next_y + j
                if 0 <= fx < walls.width and 0 <= fy < walls.height:
                    if food[fx][fy]:
                        food_nearby += 1
        features["food-density"] = food_nearby / 25.0

        # Ghost features
        ghost_distances = [manhattanDistance((next_x, next_y), g) for g in ghosts]
        
        features["ghosts-in-danger-zone"] = sum(1 for d in ghost_distances if d <= 3) / float(len(ghosts)) if ghosts else 0.0
        
        if ghost_distances:
            min_ghost_dist = min(ghost_distances)
            features["closest-ghost"] = min_ghost_dist / (walls.width * walls.height)
            
            if min_ghost_dist <= 1:
                features["ghost-collision-imminent"] = 1.0
            elif min_ghost_dist <= 2:
                features["ghost-very-close"] = 1.0
        
        scared_ghosts = sum(1 for t in scared_times if t > 0)
        features["scared-ghosts"] = scared_ghosts / float(len(scared_times)) if scared_times else 0.0
        
        if scared_ghosts > 0 and ghost_distances:
            scared_dists = [ghost_distances[i] for i in range(len(ghost_distances)) if i < len(scared_times) and scared_times[i] > 0]
            if scared_dists:
                min_scared_dist = min(scared_dists)
                features["closest-scared-ghost"] = min_scared_dist / (walls.width * walls.height)
                if min_scared_dist <= 1:
                    features["eats-scared-ghost"] = 1.0

        # Capsule features
        if capsules:
            features["capsules-remaining"] = len(capsules)
            min_capsule_dist = min([manhattanDistance((next_x, next_y), c) for c in capsules])
            features["closest-capsule"] = float(min_capsule_dist) / (walls.width * walls.height)
            
            if ghost_distances and min(ghost_distances) <= 5:
                features["capsule-when-danger"] = 1.0 / (min_capsule_dist + 1)

        # Spatial features
        legal_actions = 0
        for a in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            ax, ay = Actions.directionToVector(a)
            check_x, check_y = int(next_x + ax), int(next_y + ay)
            if not walls[check_x][check_y]:
                legal_actions += 1
        
        features["num-legal-actions"] = legal_actions / 4.0
        
        if legal_actions <= 1:
            features["dead-end"] = 1.0
        
        if legal_actions == 2:
            features["corner"] = 1.0

        # Action features
        if action == Directions.STOP:
            features["stop-action"] = 1.0
        
        pac_dir = state.getPacmanState().configuration.direction
        if pac_dir in Directions.REVERSE and action == Directions.REVERSE[pac_dir]:
            features["reverse-direction"] = 1.0

        # Strategic features
        if food[next_x][next_y] and ghost_distances and min(ghost_distances) > 3:
            features["safe-food-eat"] = 1.0
        
        if food[next_x][next_y] and ghost_distances and min(ghost_distances) <= 2:
            features["risky-food-eat"] = 1.0

        features.divideAll(10.0)
        return features


class AdvancedExtractor(FeatureExtractor):
    """
    Most advanced feature extractor with strategic planning
    """

    def getFeatures(self, state, action):
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()
        capsules = state.getCapsules()
        scared_times = [state.getGhostState(i).scaredTimer for i in range(1, state.getNumAgents())]
        
        features = util.Counter()
        
        # Get all enhanced features first
        enhanced = EnhancedExtractor()
        enhanced_features = enhanced.getFeatures(state, action)
        for key, value in enhanced_features.items():
            features[key] = value

        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # Escape route analysis
        escape_routes = 0
        for escape_action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            ex, ey = Actions.directionToVector(escape_action)
            escape_x, escape_y = int(next_x + ex), int(next_y + ey)
            if 0 <= escape_x < walls.width and 0 <= escape_y < walls.height and not walls[escape_x][escape_y]:
                safe = True
                for g in ghosts:
                    if manhattanDistance((escape_x, escape_y), g) <= 2:
                        safe = False
                        break
                if safe:
                    escape_routes += 1
        features["escape-routes"] = escape_routes / 4.0

        # Food clustering
        dist = closestFood((next_x, next_y), food, walls)
        if dist:
            food_in_direction = 0
            for i in range(1, int(dist) + 2):
                for angle in [-1, 0, 1]:
                    check_x = next_x + i * (1 if dx > 0 else -1 if dx < 0 else angle)
                    check_y = next_y + i * (1 if dy > 0 else -1 if dy < 0 else angle)
                    if 0 <= check_x < walls.width and 0 <= check_y < walls.height:
                        if food[check_x][check_y]:
                            food_in_direction += 1
            features["food-cluster-ahead"] = food_in_direction / 10.0

        # Ghost prediction
        ghost_threats = 0
        for i, g in enumerate(ghosts):
            current_dist = manhattanDistance((x, y), g)
            next_dist = manhattanDistance((next_x, next_y), g)
            
            if i < len(scared_times):
                if next_dist < current_dist and scared_times[i] == 0:
                    ghost_threats += 1
                elif next_dist < current_dist and scared_times[i] > 0:
                    features["approaching-scared-ghost"] = 1.0
        
        features["ghost-threat-level"] = ghost_threats / float(len(ghosts)) if ghosts else 0.0

        # Two-step lookahead
        good_second_moves = 0
        for second_action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            sx, sy = Actions.directionToVector(second_action)
            second_x, second_y = int(next_x + sx), int(next_y + sy)
            if 0 <= second_x < walls.width and 0 <= second_y < walls.height and not walls[second_x][second_y]:
                if food[second_x][second_y]:
                    good_second_moves += 1
                
                second_food_dist = closestFood((second_x, second_y), food, walls)
                if second_food_dist is not None and dist is not None and second_food_dist < dist:
                    good_second_moves += 0.5
        features["good-followup-moves"] = good_second_moves / 4.0

        # Strategic positioning
        center_x, center_y = walls.width // 2, walls.height // 2
        distance_to_center = manhattanDistance((next_x, next_y), (center_x, center_y))
        features["distance-to-center"] = distance_to_center / (walls.width + walls.height)

        features.divideAll(10.0)
        return features
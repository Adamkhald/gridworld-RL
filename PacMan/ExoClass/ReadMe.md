# Enhanced Feature Extractors for Pacman

## Overview

This file contains three feature extractors for Pacman approximate Q-learning:
- `SimpleExtractor` (original)
- `EnhancedExtractor` (new)
- `AdvancedExtractor` (new)

## Usage

```bash
# With SimpleExtractor
python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumGrid

# With EnhancedExtractor
python pacman.py -p ApproximateQAgent -a extractor=EnhancedExtractor -x 50 -n 60 -l mediumGrid

# With AdvancedExtractor
python pacman.py -p ApproximateQAgent -a extractor=AdvancedExtractor -x 100 -n 110 -l mediumGrid
```

## SimpleExtractor (Original)

**Features extracted:**
- `bias` - Constant bias term (always 1.0)
- `#-of-ghosts-1-step-away` - Count of ghosts one step away
- `eats-food` - Will the action eat food? (1.0 or 0.0)
- `closest-food` - Distance to nearest food (normalized)

## EnhancedExtractor (New)

**Food Features:**
- `bias` - Constant bias term (always 1.0)
- `eats-food` - Indicator if action will eat food
- `closest-food` - Normalized distance to nearest food
- `food-remaining` - Ratio of remaining food to total grid size
- `food-density` - Food count in 5x5 area around next position (normalized by 25)
- `safe-food-eat` - Eating food when nearest ghost is > 3 steps away
- `risky-food-eat` - Eating food when nearest ghost is ≤ 2 steps away

**Ghost Features:**
- `ghosts-in-danger-zone` - Ratio of ghosts within 3 steps
- `closest-ghost` - Normalized distance to nearest ghost
- `ghost-collision-imminent` - Ghost at distance ≤ 1
- `ghost-very-close` - Ghost at distance ≤ 2
- `scared-ghosts` - Ratio of scared ghosts to total ghosts
- `closest-scared-ghost` - Normalized distance to nearest scared ghost
- `eats-scared-ghost` - Will the action eat a scared ghost?

**Capsule Features:**
- `capsules-remaining` - Number of power capsules left
- `closest-capsule` - Normalized distance to nearest capsule
- `capsule-when-danger` - Inverse distance to capsule when ghosts are within 5 steps

**Spatial Features:**
- `num-legal-actions` - Ratio of legal moves from next position (out of 4)
- `dead-end` - Only one legal action available from next position
- `corner` - Exactly two legal actions available (corner detection)

**Action Features:**
- `stop-action` - Penalty indicator for STOP action
- `reverse-direction` - Penalty indicator for reversing direction

**Total: 21 features**

## AdvancedExtractor (New)

**Includes all EnhancedExtractor features plus:**

**Escape Route Analysis:**
- `escape-routes` - Ratio of safe escape paths (no ghosts within 2 steps)

**Food Clustering:**
- `food-cluster-ahead` - Food density in the direction of movement

**Ghost Prediction:**
- `approaching-scared-ghost` - Moving closer to a scared ghost
- `ghost-threat-level` - Ratio of non-scared ghosts moving toward Pacman

**Lookahead Planning:**
- `good-followup-moves` - Quality score of second-step actions (0-4 scale)

**Strategic Positioning:**
- `distance-to-center` - Normalized distance to map center

**Total: 27 features**

## Feature Normalization

All features are divided by 10.0 at the end of extraction to prevent divergence during Q-learning updates.

## Implementation Notes

- All distance features are normalized by `(walls.width * walls.height)` to keep values between 0 and 1
- All ratio features are calculated as counts divided by totals
- Binary indicator features are either 0.0 or 1.0
- The `manhattanDistance` function is used for all distance calculations
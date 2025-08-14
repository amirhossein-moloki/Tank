from collections import namedtuple

# --- Technical Specifications ---
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
FPS = 60

# --- Color Palette ---
COLOR_BACKGROUND = (40, 40, 40)
COLOR_WALL = (150, 150, 150)
COLOR_TANK_1 = (50, 150, 255)  # Team A - Dark Blue
COLOR_TANK_2 = (255, 50, 50)   # Team B - Dark Red
COLOR_TANK_3 = (80, 180, 255)  # Team A - Light Blue
COLOR_TANK_4 = (255, 80, 80)   # Team B - Light Red
COLOR_BULLET = (255, 255, 0)   # Yellow
COLOR_TEXT = (255, 255, 255)

# --- Game Map ---
# Map layouts are now in maps.py

# --- Game Physics & Rules ---
# TANK_SIZE is now calculated in the simulator based on TILE_SIZE
TANK_SPEED = 2
TANK_ROTATION_SPEED = 3
BULLET_SIZE = 8
BULLET_SPEED = 10
MAX_BULLETS = 1 # Max active bullets per tank

# --- AI & Learning Parameters ---
# For 2v2 MARL: Own Angle (1) + Wall Distances (8) + 3x Other Tank Info (3 * 8 = 24) = 33
# Other Tank Info: Rel Pos (2), Angle (1), Bullet Active (1), Bullet Rel Pos (2), Bullet Vel (2)
STATE_SIZE = 33
ACTION_SIZE = 5  # [Forward, Turn Left, Turn Right, Shoot, Idle]
REPLAY_MEMORY_SIZE = 20000 # Increased memory for more complex environment
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TARGET_UPDATE_FREQ = 10 # update target network every 10 episodes
LEARNING_RATE = 0.001

# Transition tuple for Replay Memory
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# --- PER Parameters ---
PER_ALPHA = 0.6  # How much prioritization to use (0: no prioritization, 1: full prioritization)
PER_BETA_START = 0.4 # Initial value of beta for importance sampling
PER_BETA_FRAMES = 100000 # Number of frames over which beta will be annealed to 1.0
PER_EPSILON = 1e-6 # Small value to ensure no transition has zero priority

# --- Power-up Parameters ---
POWERUP_TYPES = ['speed_boost', 'shield', 'rapid_fire']
POWERUP_COLORS = {
    'speed_boost': (0, 255, 128),  # A bright green
    'shield': (0, 128, 255),      # A strong blue
    'rapid_fire': (255, 128, 0)    # A fiery orange
}
POWERUP_SIZE_RATIO = 0.5 # Relative to tile size
POWERUP_SPAWN_DELAY = 10000 # 10 seconds between spawns
POWERUP_DURATION = 5000 # 5 seconds effect duration for speed and rapid fire

# --- N-step Learning Parameters ---
N_STEP_RETURN = 3

# --- Wall Parameters ---
WALL_HEALTH = 3
WALL_DAMAGE_COLORS = {
    3: (150, 150, 150), # Full health
    2: (120, 120, 120), # Damaged
    1: (90, 90, 90),   # Heavily damaged
}

import pygame
import torch
import random
import numpy as np
from collections import deque
from constants import (
    SCREEN_WIDTH, SCREEN_HEIGHT, FPS, COLOR_BACKGROUND, COLOR_WALL,
    COLOR_TANK_1, COLOR_TANK_2, COLOR_TANK_3, COLOR_TANK_4,
    COLOR_TEXT, BULLET_SPEED, STATE_SIZE, ACTION_SIZE, N_STEP_RETURN, GAMMA,
    TARGET_UPDATE_FREQ,
    REWARD_WIN, REWARD_LOSE, REWARD_SUCCESSFUL_HIT, REWARD_TEAM_ASSIST,
    REWARD_HIT_SHIELD, REWARD_POWERUP_PICKUP, PENALTY_WALL_HIT,
    PENALTY_SHOT_FIRED, PENALTY_SURVIVAL, PENALTY_MISSED_SHOT
)
from game_objects import Tank, Wall, PowerUp
from ai_components import DQNAgent, device
from maps import ALL_MAPS

class GameSimulator:
    def __init__(self):
        pygame.mixer.pre_init(44100, -16, 2, 512)
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("AI Tank Battle Simulator")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 36)
        self.running = True
        self.paused = False
        self.muted = False

        self.all_sprites = pygame.sprite.Group()
        self.walls = pygame.sprite.Group()
        self.tanks = pygame.sprite.Group()
        self.bullets = pygame.sprite.Group()
        self.powerups = pygame.sprite.Group()

        self.maps = ALL_MAPS
        self.current_map_layout = None
        self.tile_size = 0
        self.tank_size = 0
        self.map_offset_x = 0
        self.map_offset_y = 0
        self.select_new_map() # Select initial map

        # Create 4 agents for 2v2 gameplay
        self.agents = {
            1: DQNAgent(1, STATE_SIZE, ACTION_SIZE, N_STEP_RETURN, GAMMA),
            2: DQNAgent(2, STATE_SIZE, ACTION_SIZE, N_STEP_RETURN, GAMMA),
            3: DQNAgent(3, STATE_SIZE, ACTION_SIZE, N_STEP_RETURN, GAMMA),
            4: DQNAgent(4, STATE_SIZE, ACTION_SIZE, N_STEP_RETURN, GAMMA),
        }

        self.n_step_buffers = {i: deque(maxlen=N_STEP_RETURN) for i in self.agents}

        spawn_points = self._get_spawn_points()

        # Team A: 1 (Dark Blue), 3 (Light Blue)
        # Team B: 2 (Dark Red), 4 (Light Red)
        self.tanks_dict = {
            1: Tank(spawn_points[0], COLOR_TANK_1, 1, 'A', self.all_sprites, self.bullets, self.tank_size),
            3: Tank(spawn_points[1], COLOR_TANK_3, 3, 'A', self.all_sprites, self.bullets, self.tank_size),
            2: Tank(spawn_points[2], COLOR_TANK_2, 2, 'B', self.all_sprites, self.bullets, self.tank_size),
            4: Tank(spawn_points[3], COLOR_TANK_4, 4, 'B', self.all_sprites, self.bullets, self.tank_size),
        }

        for tank in self.tanks_dict.values():
            self.all_sprites.add(tank)
            self.tanks.add(tank)

        self.score = {'A': 0, 'B': 0}
        self.episode_count = 0
        self.next_powerup_spawn_time = 0
        self.game_speed = 1.0

        self._create_buttons()
        self._create_sounds()

    def _create_sounds(self):
        self.sounds = {}

        def generate_beep(frequency, duration_ms, volume=0.1):
            sample_rate = pygame.mixer.get_init()[0]
            n_samples = int(round(duration_ms / 1000 * sample_rate))
            buf = np.zeros((n_samples, 2), dtype=np.int16)
            max_sample = 2**(16 - 1) - 1

            amplitude = max_sample * volume

            arr = np.array([amplitude * np.sin(2.0 * np.pi * frequency * x / sample_rate) for x in range(n_samples)])

            buf[:, 0] = arr
            buf[:, 1] = arr

            return pygame.sndarray.make_sound(buf)

        # Generate different sounds for different events
        self.sounds['shoot'] = generate_beep(880, 50) # A5 note, short
        self.sounds['explosion'] = generate_beep(220, 400) # A3 note, long
        self.sounds['powerup'] = generate_beep(1320, 100) # E6 note, short and high
        self.sounds['shield_up'] = generate_beep(660, 200) # E5 note
        self.sounds['shield_hit'] = generate_beep(440, 150) # A4 note
        self.sounds['wall_hit'] = generate_beep(330, 75) # E4 note, very short

    def _play_sound(self, name):
        if not self.muted and self.sounds.get(name):
            self.sounds[name].play()

    def select_new_map(self):
        self.current_map_layout = random.choice(self.maps)

        map_rows = len(self.current_map_layout)
        map_cols = len(self.current_map_layout[0])

        # Calculate tile size based on the screen dimensions
        tile_size_h = SCREEN_HEIGHT // map_rows
        tile_size_w = SCREEN_WIDTH // map_cols
        self.tile_size = min(tile_size_h, tile_size_w)

        # Calculate offsets to center the map
        map_width = self.tile_size * map_cols
        map_height = self.tile_size * map_rows
        self.map_offset_x = (SCREEN_WIDTH - map_width) // 2
        self.map_offset_y = (SCREEN_HEIGHT - map_height) // 2

        self.tank_size = int(self.tile_size * 0.8)
        self._create_map()

    def _spawn_powerup(self):
        # Find all empty tiles to spawn a powerup
        empty_tiles = []
        for r, row in enumerate(self.current_map_layout):
            for c, tile in enumerate(row):
                if tile == ' ':
                    # Check if it's not too close to a wall
                    is_open = all(self.current_map_layout[r+dr][c+dc] == ' ' for dr in [-1,0,1] for dc in [-1,0,1] if 0 <= r+dr < len(self.current_map_layout) and 0 <= c+dc < len(row))
                    if is_open:
                        x = self.map_offset_x + (c + 0.5) * self.tile_size
                        y = self.map_offset_y + (r + 0.5) * self.tile_size
                        empty_tiles.append((x, y))

        if empty_tiles:
            pos = random.choice(empty_tiles)
            powerup_size = self.tile_size * POWERUP_SIZE_RATIO
            powerup = PowerUp(pos, powerup_size)
            self.all_sprites.add(powerup)
            self.powerups.add(powerup)
            self.next_powerup_spawn_time = pygame.time.get_ticks() + POWERUP_SPAWN_DELAY

    def _create_map(self):
        # Clear existing walls before creating a new map
        self.walls.empty()
        for sprite in self.all_sprites:
            if isinstance(sprite, Wall):
                sprite.kill()

        for row_idx, row in enumerate(self.current_map_layout):
            for col_idx, tile in enumerate(row):
                if tile == '#':
                    x = self.map_offset_x + col_idx * self.tile_size
                    y = self.map_offset_y + row_idx * self.tile_size
                    wall = Wall(x, y, self.tile_size)
                    self.all_sprites.add(wall)
                    self.walls.add(wall)

    def _get_spawn_points(self):
        points = []
        for r, row in enumerate(self.current_map_layout):
            for c, tile in enumerate(row):
                if tile == ' ' and r > 0 and c > 0 and r < len(self.current_map_layout)-1 and c < len(self.current_map_layout[0])-1:
                    is_open = all(self.current_map_layout[r+dr][c+dc] == ' ' for dr in [-1,0,1] for dc in [-1,0,1])
                    if is_open:
                        x = self.map_offset_x + (c + 0.5) * self.tile_size
                        y = self.map_offset_y + (r + 0.5) * self.tile_size
                        points.append((x, y))
        if len(points) < 4:
            # Fallback for small maps
            return [(100, 100), (100, SCREEN_HEIGHT - 100), (SCREEN_WIDTH - 100, 100), (SCREEN_WIDTH - 100, SCREEN_HEIGHT - 100)]
        return random.sample(points, 4)

    def _cast_rays(self, start_pos, walls):
        distances = []
        for angle in range(0, 360, 45): # 8 directions
            ray_dir = pygame.math.Vector2(1, 0).rotate(angle)
            dist = 0
            while dist < SCREEN_WIDTH: # Max distance
                check_pos = start_pos + ray_dir * dist
                check_rect = pygame.Rect(check_pos.x, check_pos.y, 1, 1)
                if any(wall.rect.colliderect(check_rect) for wall in walls):
                    distances.append(dist)
                    break
                dist += self.tile_size / 4
            else:
                distances.append(dist)
        return distances

    def _get_state(self, agent_tank):
        # Helper function to get info for one other tank
        def get_other_tank_info(other_tank):
            rel_pos = other_tank.pos - agent_tank.pos
            angle = other_tank.angle / 360.0

            bullet = next((b for b in self.bullets if b.owner == other_tank), None)
            bullet_active = 1.0 if bullet else 0.0
            bullet_rel_pos = (bullet.pos - agent_tank.pos) if bullet else pygame.math.Vector2(0, 0)
            bullet_vel = bullet.velocity if bullet else pygame.math.Vector2(0, 0)

            return [
                rel_pos.x / SCREEN_WIDTH, rel_pos.y / SCREEN_HEIGHT,
                angle, bullet_active,
                bullet_rel_pos.x / SCREEN_WIDTH, bullet_rel_pos.y / SCREEN_HEIGHT,
                bullet_vel.x / BULLET_SPEED, bullet_vel.y / BULLET_SPEED
            ]

        # Identify teammate and opponents
        allies = [t for t in self.tanks if t.team == agent_tank.team and t != agent_tank]
        opponents = sorted([t for t in self.tanks if t.team != agent_tank.team], key=lambda x: x.agent_id)

        # Build the state vector
        state_list = []

        # Add teammate info (or zeros if no teammate is alive)
        if allies:
            state_list.extend(get_other_tank_info(allies[0]))
        else:
            state_list.extend([0.0] * 8)

        # Add opponents' info (or zeros if they are not alive)
        for i in range(2):
            if i < len(opponents):
                state_list.extend(get_other_tank_info(opponents[i]))
            else:
                state_list.extend([0.0] * 8)

        # Add self info
        state_list.append(agent_tank.angle / 360.0)
        state_list.extend([d / SCREEN_WIDTH for d in self._cast_rays(agent_tank.pos, self.walls)])

        return torch.tensor([state_list], dtype=torch.float32, device=device)

    def _create_buttons(self):
        self.buttons = {}
        button_font = pygame.font.SysFont(None, 30)
        button_w, button_h = 100, 40
        margin = 10

        # Positions for buttons at the bottom center
        button_labels = ["Speed -", "Speed +", "Pause", "Mute", "Save", "Restart", "Quit"]
        total_width = (button_w + margin) * len(button_labels) - margin
        start_x = (SCREEN_WIDTH - total_width) / 2

        for i, label in enumerate(button_labels):
            x = start_x + i * (button_w + margin)
            y = SCREEN_HEIGHT - button_h - margin
            self.buttons[label] = {
                "rect": pygame.Rect(x, y, button_w, button_h),
                "text": button_font.render(label, True, COLOR_TEXT),
                "label": label
            }

    def _draw_buttons(self):
        for label, button in self.buttons.items():
            # Special case for Pause/Resume button text
            if label == "Pause":
                text_label = "Resume" if self.paused else "Pause"
                button["text"] = self.font.render(text_label, True, COLOR_TEXT)
            elif label == "Mute":
                text_label = "Unmute" if self.muted else "Mute"
                button["text"] = self.font.render(text_label, True, COLOR_TEXT)

            pygame.draw.rect(self.screen, COLOR_WALL, button["rect"], border_radius=5)
            text_rect = button["text"].get_rect(center=button["rect"].center)
            self.screen.blit(button["text"], text_rect)

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # Left mouse button
                    for label, button in self.buttons.items():
                        if button["rect"].collidepoint(event.pos):
                            if label == "Pause":
                                self.paused = not self.paused
                            elif label == "Mute":
                                self.muted = not self.muted
                            elif label == "Save":
                                for i, agent in self.agents.items():
                                    agent.save_model(f"agent{i}_dqn.pth")
                                print("All models saved!")
                            elif label == "Restart":
                                self.reset_round()
                            elif label == "Quit":
                                self.running = False
                            elif label == "Speed +":
                                self.game_speed = min(10.0, self.game_speed + 1.0)
                            elif label == "Speed -":
                                self.game_speed = max(1.0, self.game_speed - 1.0)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                if event.key == pygame.K_p:
                    self.paused = not self.paused
                if event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.game_speed = min(10.0, self.game_speed + 1.0)
                if event.key == pygame.K_MINUS:
                    self.game_speed = max(1.0, self.game_speed - 1.0)
                if event.key == pygame.K_s:
                    for i, agent in self.agents.items():
                        agent.save_model(f"agent{i}_dqn.pth")
                    print("All models saved!")
                if event.key == pygame.K_l:
                    try:
                        for i, agent in self.agents.items():
                            agent.load_model(f"agent{i}_dqn.pth")
                        print("All models loaded!")
                    except FileNotFoundError:
                        print("Could not find all saved models.")

    def run(self):
        while self.running:
            self.clock.tick(FPS * self.game_speed)
            self._handle_events()
            if self.paused:
                self._draw()
                continue

            # --- Main AI Loop for 4 Agents ---
            # 1. Observe state for all living agents
            states = {i: self._get_state(tank) for i, tank in self.tanks_dict.items() if tank.alive()}

            # 2. Decide action for all living agents
            actions = {i: self.agents[i].select_action(states[i]) for i in states}

            # 3. Execute actions and calculate immediate rewards
            rewards = {i: PENALTY_SURVIVAL for i in self.agents} # Survival penalty
            for i, action in actions.items():
                shot, wall_hit = self.tanks_dict[i].update(action.item(), self.walls)
                if shot:
                    self._play_sound('shoot')
                if wall_hit:
                    rewards[i] += PENALTY_WALL_HIT # Apply penalty for hitting a wall

            self.bullets.update(self.walls, self.screen.get_rect(), rewards)
            self._check_powerup_collisions(rewards)
            self._check_bullet_wall_collisions(rewards)

            # 4. Check for eliminations and team-based win conditions
            eliminated_tanks = self._check_tank_eliminations(rewards)

            done = False
            if eliminated_tanks:
                self._play_sound('explosion')
                team_a_alive = any(t.alive() for t in self.tanks if t.team == 'A')
                team_b_alive = any(t.alive() for t in self.tanks if t.team == 'B')

                if not team_a_alive or not team_b_alive:
                    done = True
                    winner_team = 'B' if not team_a_alive else 'A'
                    self.score[winner_team] += 1
                    print(f"Round {self.episode_count} Over. Winner: Team {winner_team}")

                    # Assign team-based rewards
                    for tank in self.tanks_dict.values():
                        if tank.team == winner_team: rewards[tank.agent_id] += REWARD_WIN
                        else: rewards[tank.agent_id] += REWARD_LOSE

            # 5. Store experiences in N-step buffers
            next_states = {i: self._get_state(tank) if tank.alive() else None for i, tank in self.tanks_dict.items()}
            for i in self.agents:
                if i in states: # If agent was alive at the start of the frame
                    is_done = done or not self.tanks_dict[i].alive()
                    self._process_n_step_buffer(self.agents[i], self.n_step_buffers[i], states[i], actions[i], rewards[i], is_done)

            # 6. Train models
            for agent in self.agents.values():
                agent.optimize_model()

            # 7. Reset round if done
            if done:
                self.reset_round()
                self.episode_count += 1

                # Save models after each match
                for i, agent in self.agents.items():
                    agent.save_model(f"agent{i}_dqn.pth")
                print(f"Models saved after round {self.episode_count - 1}.")

                if self.episode_count % TARGET_UPDATE_FREQ == 0:
                    for agent in self.agents.values():
                        agent.target_net.load_state_dict(agent.policy_net.state_dict())
                    print("Target networks updated.")

            self._draw()
        pygame.quit()

    def _check_powerup_collisions(self, rewards):
        collided_powerups = pygame.sprite.groupcollide(self.tanks, self.powerups, False, True)
        for tank, powerups in collided_powerups.items():
            for powerup in powerups:
                tank.activate_powerup(powerup.type)
                self._play_sound('powerup')
                if powerup.type == 'shield': self._play_sound('shield_up')
                print(f"Tank {tank.agent_id} collected a {powerup.type} power-up!")
                rewards[tank.agent_id] += REWARD_POWERUP_PICKUP

    def _check_tank_eliminations(self, rewards):
        eliminated_tanks = []
        for bullet in self.bullets:
            # We use a copy of the tanks group because a tank might be killed, which modifies the group
            hit_tanks = pygame.sprite.spritecollide(bullet, self.tanks, False, pygame.sprite.collide_rect_ratio(0.8))
            for tank in hit_tanks:
                # Check for friendly fire
                if bullet.owner.team != tank.team:
                    bullet.kill()
                    attacker = bullet.owner
                    if tank.shield_active:
                        tank.shield_active = False
                        self._play_sound('shield_hit')
                        print(f"Tank {tank.agent_id}'s shield absorbed a hit!")
                        rewards[attacker.agent_id] += REWARD_HIT_SHIELD
                    else:
                        tank.kill() # Remove tank from all sprite groups
                        eliminated_tanks.append(tank)
                        rewards[attacker.agent_id] += REWARD_SUCCESSFUL_HIT
                        # Find the teammate and give them an assist reward
                        teammates = [t for t in self.tanks if t.team == attacker.team and t != attacker]
                        if teammates:
                            # Assuming only one other teammate
                            rewards[teammates[0].agent_id] += REWARD_TEAM_ASSIST
        return eliminated_tanks

    def _check_bullet_wall_collisions(self, rewards):
        # We don't kill the bullet on collision anymore, we handle it manually
        collided_walls = pygame.sprite.groupcollide(self.bullets, self.walls, False, False)

        for bullet, walls_hit in collided_walls.items():
            self._play_sound('wall_hit')
            bullet.ricochets += 1

            if bullet.ricochets >= 3:
                rewards[bullet.owner.agent_id] += PENALTY_MISSED_SHOT
                bullet.kill()
                continue # Move to the next bullet

            # Simple bounce logic
            wall = walls_hit[0] # Assume collision with one wall at a time for simplicity

            # To prevent getting stuck, move bullet back
            bullet.pos -= bullet.velocity
            bullet.rect.center = bullet.pos

            # Determine bounce direction
            dx = abs(bullet.rect.centerx - wall.rect.centerx)
            dy = abs(bullet.rect.centery - wall.rect.centery)

            if dx > wall.rect.width / 2: # Horizontal collision
                bullet.velocity.x *= -1
            if dy > wall.rect.height / 2: # Vertical collision
                bullet.velocity.y *= -1

    def _process_n_step_buffer(self, agent, buffer, state, action, reward, done):
        experience = (state, action, reward, done)
        buffer.append(experience)

        if len(buffer) < N_STEP_RETURN and not done:
            return

        n_step_reward = 0
        for i in range(len(buffer)):
            n_step_reward += (agent.gamma ** i) * buffer[i][2]

        initial_state, initial_action, _, _ = buffer[0]
        final_state = state if not done else None

        agent.store_experience(initial_state, initial_action, final_state, torch.tensor([n_step_reward], device=device))

        if done:
            while len(buffer) > 1:
                buffer.popleft()
                n_step_reward = 0
                for i in range(len(buffer)):
                    n_step_reward += (agent.gamma ** i) * buffer[i][2]
                initial_state, initial_action, _, _ = buffer[0]
                agent.store_experience(initial_state, initial_action, None, torch.tensor([n_step_reward], device=device))
            buffer.clear()

    def reset_round(self):
        self.select_new_map()
        for buffer in self.n_step_buffers.values():
            buffer.clear()

        spawn_points = self._get_spawn_points()

        self.tanks.empty()
        for sprite in self.all_sprites:
            if isinstance(sprite, Tank):
                sprite.kill()

        self.tanks_dict = {
            1: Tank(spawn_points[0], COLOR_TANK_1, 1, 'A', self.all_sprites, self.bullets, self.tank_size),
            3: Tank(spawn_points[1], COLOR_TANK_3, 3, 'A', self.all_sprites, self.bullets, self.tank_size),
            2: Tank(spawn_points[2], COLOR_TANK_2, 2, 'B', self.all_sprites, self.bullets, self.tank_size),
            4: Tank(spawn_points[3], COLOR_TANK_4, 4, 'B', self.all_sprites, self.bullets, self.tank_size),
        }
        for tank in self.tanks_dict.values():
            self.all_sprites.add(tank)
            self.tanks.add(tank)

        for bullet in self.bullets:
            bullet.kill()
        for powerup in self.powerups:
            powerup.kill()

    def _draw(self):
        self.screen.fill(COLOR_BACKGROUND)
        self.all_sprites.draw(self.screen)

        score_a_text = self.font.render(f"Team Blue Score: {self.score['A']}", True, COLOR_TANK_1)
        score_b_text = self.font.render(f"Team Red Score: {self.score['B']}", True, COLOR_TANK_2)
        self.screen.blit(score_a_text, (10, 10))
        self.screen.blit(score_b_text, (SCREEN_WIDTH - score_b_text.get_width() - 10, 10))

        speed_text = self.font.render(f"Speed: {int(self.game_speed)}x", True, COLOR_TEXT)
        self.screen.blit(speed_text, (SCREEN_WIDTH // 2 - speed_text.get_width() // 2, 10))

        if self.paused:
            pause_text = self.font.render("PAUSED", True, COLOR_TEXT)
            self.screen.blit(pause_text, (SCREEN_WIDTH // 2 - pause_text.get_width() // 2, SCREEN_HEIGHT // 2 - pause_text.get_height() // 2))

        self._draw_buttons()
        pygame.display.flip()

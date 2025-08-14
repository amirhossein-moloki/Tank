import pygame
from constants import *

class Tank(pygame.sprite.Sprite):
    def __init__(self, pos, color, agent_id, team, all_sprites, bullets_group, tank_size):
        super().__init__()
        self.agent_id = agent_id
        self.team = team
        self.all_sprites = all_sprites
        self.bullets_group = bullets_group
        self.color = color

        self.image_orig = pygame.Surface((tank_size, tank_size), pygame.SRCALPHA)
        pygame.draw.rect(self.image_orig, self.color, self.image_orig.get_rect())
        # Add a "barrel" to see direction
        barrel = pygame.Surface((tank_size // 2, tank_size // 4))
        barrel.fill(COLOR_WALL)
        self.image_orig.blit(barrel, (tank_size // 2, tank_size // 2 - tank_size // 8))

        self.image = self.image_orig
        self.rect = self.image.get_rect(center=pos)

        self.pos = pygame.math.Vector2(pos)
        self.angle = 0
        self.speed = 0
        self.last_shot_time = 0
        self.powerup_end_time = { 'speed_boost': 0, 'rapid_fire': 0 }
        self.shield_active = False
        self.shot_cooldown = 500 # ms

    def activate_powerup(self, powerup_type):
        if powerup_type == 'speed_boost' or powerup_type == 'rapid_fire':
            self.powerup_end_time[powerup_type] = pygame.time.get_ticks() + POWERUP_DURATION
        elif powerup_type == 'shield':
            self.shield_active = True

    def update(self, action, walls):
        current_time = pygame.time.get_ticks()
        shot_fired = False

        # Check for active power-ups
        current_speed = TANK_SPEED
        if self.powerup_end_time['speed_boost'] > current_time:
            current_speed *= 2

        self.shot_cooldown = 250 if self.powerup_end_time['rapid_fire'] > current_time else 500

        # Action: 0:Forward, 1:Left, 2:Right, 3:Shoot, 4:Idle
        if action == 0: # Move Forward
            self.speed = current_speed
        else:
            self.speed = 0 # Stop if not moving forward

        if action == 1: # Turn Left
            self.angle = (self.angle + TANK_ROTATION_SPEED) % 360
        elif action == 2: # Turn Right
            self.angle = (self.angle - TANK_ROTATION_SPEED) % 360

        if action == 3: # Shoot
            shot_fired = self.shoot()

        # Rotate image
        self.image = pygame.transform.rotate(self.image_orig, self.angle)
        self.rect = self.image.get_rect(center=self.pos)

        # Move tank
        wall_hit = False
        if self.speed != 0:
            move_vec = pygame.math.Vector2(1, 0).rotate(-self.angle) * self.speed
            self.pos += move_vec
            self.rect.center = self.pos

            # Collision with walls
            if pygame.sprite.spritecollideany(self, walls):
                self.pos -= move_vec # Revert move
                self.rect.center = self.pos
                wall_hit = True
        return shot_fired, wall_hit

    def shoot(self):
        now = pygame.time.get_ticks()
        if now - self.last_shot_time > self.shot_cooldown:
            if len([b for b in self.bullets_group if b.owner == self]) < MAX_BULLETS:
                self.last_shot_time = now
                bullet = Bullet(self.pos, self.angle, self)
                self.all_sprites.add(bullet)
                self.bullets_group.add(bullet)
                return True # Shot fired
        return False # No shot

    def draw(self, surface):
        surface.blit(self.image, self.rect)

class Bullet(pygame.sprite.Sprite):
    def __init__(self, pos, angle, owner):
        super().__init__()
        self.owner = owner
        self.image = pygame.Surface((BULLET_SIZE, BULLET_SIZE))
        self.image.fill(COLOR_BULLET)
        self.rect = self.image.get_rect(center=pos)
        self.pos = pygame.math.Vector2(pos)
        self.velocity = pygame.math.Vector2(1, 0).rotate(-angle) * BULLET_SPEED

    def update(self, walls, screen_rect):
        self.pos += self.velocity
        self.rect.center = self.pos

        # Ricochet logic is removed. Collision is now handled in the simulator.
        # Kill bullet if it goes way off-screen
        if not screen_rect.colliderect(self.rect):
             self.kill()

    def draw(self, surface):
        surface.blit(self.image, self.rect)

class Wall(pygame.sprite.Sprite):
    def __init__(self, x, y, tile_size):
        super().__init__()
        self.tile_size = tile_size
        self.health = WALL_HEALTH
        self.image = pygame.Surface((self.tile_size, self.tile_size))
        self.image.fill(WALL_DAMAGE_COLORS[self.health])
        self.rect = self.image.get_rect(topleft=(x, y))

    def hit(self):
        self.health -= 1
        if self.health <= 0:
            self.kill()
        else:
            self.image.fill(WALL_DAMAGE_COLORS[self.health])

class PowerUp(pygame.sprite.Sprite):
    def __init__(self, pos, size):
        super().__init__()
        self.type = random.choice(POWERUP_TYPES)
        self.image = pygame.Surface((size, size))
        self.image.fill(POWERUP_COLORS[self.type])
        self.rect = self.image.get_rect(center=pos)

import pygame, sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import ode
import math
import random
import time

#Global values definition
BLACK = (0, 0, 0)
GREY = (192, 192, 192)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
win_width = 1400
win_height = 700
fwRadius = 5

# clock object that ensure that animation has the same
# on all machines, regardless of the actual machine speed.
clock = pygame.time.Clock()

def to_screen(x, y):
    '''flipping y, since we want our y to increase as we move up'''
    x += 10
    y += 10

    return x, win_height - y

def normalize(v):
    return v / np.linalg.norm(v)

def randomVel():
    #X-Velocity set to 3 for firework to have normality.
    vel = random.uniform(90, 110)
    return vel

def sparkCount():
    return random.randint(14,24)

def getVelocityList(count):
    half1 = count // 2
    half2 = count - half1
    velList = []
    dist1 = float(1.0 / (half1 + 1))
    dist2 = float(1.0 / (half2 + 1))

    temp = dist1
    tempHalf = half1//2
    #
    quad4 = []
    for i in range(half1):
        vel = random.uniform(20, 25)
        if (i < tempHalf):
            quad4.append(((temp)*vel))
        else:
            quad4.append(((-temp)*vel))
        temp += dist1
    quad2 = list(reversed(quad4))

    for i in range(half1):
        velList.append([quad4[i], quad2[i]])


    temp = dist2
    tempHalf = half2//2
    quad1 = []
    for i in range(half2):
        vel = random.uniform(20, 25)
        quad1.append(((temp)*vel))
        temp += dist2
    quad3 = list(reversed(quad1))

    for i in range(half2):
        if (i < tempHalf):
            velList.append([quad1[i], quad3[i]])
        else:
            velList.append([-quad1[i], -quad3[i]])
    return velList


class Firework(pygame.sprite.Sprite):
    def __init__(self, colour, radius):
        pygame.sprite.Sprite.__init__(self)

        self.image = pygame.Surface([radius*2, radius*2])
        self.image.fill(BLACK)
        pygame.draw.circle(self.image, colour, (radius, radius), radius, radius)

        self.rect = self.image.get_rect()

    def update(self):
        pass


class FireworkStand(pygame.sprite.Sprite):
    def __init__(self, colour, width, height):
        pygame.sprite.Sprite.__init__(self)

        self.image = pygame.Surface([width, height])
        self.rect = self.image.get_rect()
        self.image.fill(colour)
        pygame.draw.rect(self.image, colour, (700, 0, width, height), 0)

    def update(self):
        pass


class Simulation:
    def __init__(self):
        #Firework is by default centered
        self.pos = [700,10]
        self.velocity = [0,0]
        self.angle = 90
        self.fuse = 10.0
        #True = right
        #False = left
        self.direction = True
        self.cur_time = 0.0
        self.g = -9.8
        self.c = 0.0001
        self.dt = 0.05
        self.t = 0
        self.m = 1
        self.paused = True

        self.solver = ode(self.f)
        self.solver.set_integrator('dop853')
        self.solver.set_f_params(self.c,self.g)

    def f(self, t, state, friction, gravity):
        dx = state[2]
        dy = state[3]
        dvx = -state[2]*friction
        dvy = gravity

        return [dx, dy, dvx, dvy]

    def setup(self):
        #Lauching with a positive X-Value
        if (self.direction):
            self.velocity[0] =  math.cos(math.radians(self.angle))*randomVel()
        #Launching with a negative X-Value
        else:
            self.velocity[0] =  -math.cos(math.radians(self.angle))*randomVel()

        self.velocity[1] = math.sin(math.radians(self.angle))*randomVel()

        self.solver.set_initial_value([self.pos[0], self.pos[1], self.velocity[0], self.velocity[1]], self.cur_time)

        self.trace_x = [self.pos[0]]
        self.trace_y = [self.pos[1]]


    def step(self):
        self.cur_time += self.dt

        if self.solver.successful():
            self.solver.integrate(self.cur_time)
            self.pos = self.solver.y[0:2]
            self.velocity = self.solver.y[2:4]

            self.trace_x.append(self.pos[0])
            self.trace_y.append(self.pos[1])

    def check_explosion(self):
        if (self.fuse < self.cur_time):
            return True
        return False

    def shift_left(self, win_width):
        #Stop shifting off-screen, given 5 pixels of border grace
        if (self.pos[0] > 10):
            self.pos[0] -= 5

    def shift_right(self, win_width):
        #Stop shifting off-screen, given 5 pixels of border grace
        if (self.pos[0] < win_width - 25):
            self.pos[0] += 5

    def angle_left(self):
        if not(self.direction):
            if (self.angle > 45):
                self.angle -= 5
        else:
            if (self.angle == 90):
                self.direction = False
                self.angle -= 5
            else:
                self.angle += 5
        print (self.direction ,self.angle)

    def angle_right(self):
        if (self.direction):
            if (self.angle > 45):
                self.angle -= 5
        else:
            if (self.angle == 90):
                self.direction = True
                self.angle -= 5
            else:
                self.angle += 5
        print (self.direction ,self.angle)

    def drawAngleAim(self):
        #Add 5 to center the line to the middle of the stand
        if (self.direction):
            x = self.pos[0] + 200*math.cos(math.radians(self.angle)) + 5
            y = self.pos[1] + 200*math.sin(math.radians(self.angle))
        else:
            x = self.pos[0] - 200*math.cos(math.radians(self.angle)) + 5
            y = self.pos[1] + 200*math.sin(math.radians(self.angle))
        return [self.pos[0] + 5, self.pos[1], x, y]


    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def get_pos(self):
        return self.pos[0], self.pos[1]

    def get_final_pos(self):
        return [self.pos[0], self.pos[1]]

    def get_angle(self):
        return [self.angle, self.direction]

    def launch_setup(self, pos, angle):
        self.pos = pos
        self.angle = angle[0]
        self.direction = angle[1]

class ExplosionObj(pygame.sprite.Sprite):

    def __init__(self, radius, colour):
        pygame.sprite.Sprite.__init__(self)

        self.image = pygame.Surface([radius*2, radius*2])
        self.image.fill(BLACK)
        pygame.draw.circle(self.image, colour, (radius, radius), radius, radius)
        self.rect = self.image.get_rect()

        self.state = [0, 0, 0, 0]
        self.t = 0
        self.g = -9.8
        self.mass = 0.5
        self.radius = radius

        self.solver = ode(self.f)
        self.solver.set_integrator('dop853')
        self.solver.set_initial_value(self.state, self.t)

    def f(self, t, state):
        dx = state[2]
        dy = state[3]
        dvy = (self.g * self.mass)

        return [dx, dy, 0, dvy]

    def set_pos(self, pos):
        self.state[0:2] = pos
        self.solver.set_initial_value(self.state, self.t)
        return self

    def set_vel(self, vel):
        self.state[2:] = vel
        self.solver.set_initial_value(self.state, self.t)
        return self


    def update(self, dt):
        self.t += dt
        self.state = self.solver.integrate(self.t)

    def draw(self, surface):
        rect = self.image.get_rect()
        rect.center = (self.state[0], win_height-self.state[1]) # Flipping y
        surface.blit(self.image, rect)

class Explosion:
    def __init__(self):
        #Change name to sparks
        self.sparks = []
        self.spark_group = pygame.sprite.Group()
        self.e = 1. # Coefficient of restitution
        self.dt = 0.05
        self.burn_time = 5

    def set_sparks(self, radius, count, pos, colour):
        velList = getVelocityList(count)
        for i in range(count):
            disk = ExplosionObj(3, colour)
            disk.set_pos(pos)
            disk.set_vel(velList[i])
            self.sparks.append(disk)
            self.spark_group.add(disk)

    def update(self):
        if (self.burn_time < 4.5):
            self.check_spark_coll()
        self.burn_time -= self.dt
        for d in self.sparks:
            d.update(self.dt)

        #self.spark_group.update()
    def draw(self, screen):
        for d in self.sparks:
            d.draw(screen)

    def check_burnout(self):
        if(self.burn_time < 0.1):
            return True
        return False

    def compute_collision_response(self, i, j):
        pass

    def check_spark_coll(self):
        for i in range(0, len(self.sparks)):
            #Do Single Disk to Wall Collision here
            for j in range(i + 1, len(self.sparks)):
                if i == j:
                    continue
                #print 'Checking sparks', i, 'and', j
                iPos = np.array(self.sparks[i].state[0:2])
                jPos = np.array(self.sparks[j].state[0:2])
                dist = np.sqrt(np.sum((iPos - jPos)**2))

                iRad = self.sparks[i].radius
                jRad = self.sparks[j].radius

                if dist > iRad + jRad:
                    continue

                iVel = np.array(self.sparks[i].state[2:])
                jVel = np.array(self.sparks[j].state[2:])
                relativeVel = iVel - jVel
                norm = normalize(iPos - jPos)

                if np.dot(relativeVel, norm) >= 0:
                    continue

                iMass = self.sparks[i].mass
                jMass = self.sparks[j].mass

                # Don't confuse this J with j
                J = -( 1 + self.e) * np.dot(relativeVel, norm) / ((1.0 / iMass) + (1.0 / jMass))

                iFinalVel = iVel + norm * J / iMass
                jFinalVel = jVel - norm * J / jMass

                self.sparks[i].set_vel(iFinalVel)
                self.sparks[j].set_vel(jFinalVel)

                break


def main():

    # initializing pygame
    pygame.init()
    clock.tick(60)

    screen = pygame.display.set_mode((win_width, win_height))
    pygame.display.set_caption('Firework Simulation')

    # setting up a sprite group, which will be drawn on the
    # screen
    fw = Firework(RED, fwRadius)
    stand = FireworkStand(GREY, 12, 80)
    fw_group = pygame.sprite.GroupSingle(fw)
    stand_group = pygame.sprite.GroupSingle(stand)

    # setting up simulation
    sim = Simulation()

    posComplete = False

    print '--------------------------------'
    print 'Usage:'
    print 'Press (d) to shift firework right'
    print 'Press (a) to shift firework left'
    print 'Press (q) to direct firework on a left angle'
    print 'Press (e) to direct firework on a right angle'
    print 'Press (space) to confirm location of firework'
    print '--------------------------------'

    posComplete = False

    while not posComplete:
        stand.rect.x, stand.rect.y = to_screen(sim.pos[0], sim.pos[1])

        event = pygame.event.poll()
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit(0)

        if event.type == pygame.KEYDOWN and event.key == pygame.K_a:
            sim.shift_left(win_width)
            continue

        elif event.type == pygame.KEYDOWN and event.key == pygame.K_d:
            sim.shift_right(win_width)
            continue

        elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
            sim.angle_left()
            continue

        elif event.type == pygame.KEYDOWN and event.key == pygame.K_e:
            sim.angle_right()
            continue

        elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            posComplete = True
            continue
        else:
            pass

        screen.fill(BLACK)
        stand_group.update()
        stand_group.draw(screen)
        fwAimPos = sim.drawAngleAim()
        pygame.draw.line(screen, (0, 0, 255), to_screen(fwAimPos[0], fwAimPos[1]), to_screen(fwAimPos[2], fwAimPos[3]))
        pygame.display.flip()


    finalStandPos = sim.get_pos()
    finalAngle = sim.get_angle()
    #Need to loop until launch button is pressed, check for left and right arrow
    #events and update the sim_class: ie the firework location

    sim.pause()

    launch = Simulation()
    launch.launch_setup(finalStandPos, finalAngle)
    launch.setup()

    fuseComplete = False
    launch.resume()
    while not fuseComplete:

        # update sprite x, y position using values
        # returned from the simulation
        fw.rect.x, fw.rect.y = to_screen(launch.pos[0], launch.pos[1])
        stand.rect.x, stand.rect.y = to_screen(sim.pos[0], sim.pos[1])

        event = pygame.event.poll()
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit(0)

        # clear the background, and draw the sprites
        screen.fill(BLACK)
        fw_group.update()
        fw_group.draw(screen)
        stand_group.update()
        stand_group.draw(screen)
        pygame.display.flip()

        #Where the simulation ends, determines height
        #Changes to be made to this: add collision detection if the firework comes back down
        if launch.pos[1] <= -1.:
            pygame.quit()
            break

        # update simulation
        if not launch.paused:
            launch.step()
        else:
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                launch.step()

        fuseComplete = launch.check_explosion()
    fw.kill()
    launch.pause()
    explPos = launch.get_final_pos()

    expl = Explosion()
    # radius, count, pos):
    expl.set_sparks(fwRadius, 15, explPos, RED)

    burnout = False

    while not burnout:
        event = pygame.event.poll()
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit(0)

        #screen.fill(BLACK)
        expl.update()
        expl.draw(screen)
        pygame.display.flip()

        burnout = expl.check_burnout()


    time.sleep(2)
    '''
    plt.figure(1)
    plt.plot(launch.trace_x, launch.trace_y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.title('2D projectile trajectory')
    plt.show()
    '''


if __name__ == '__main__':
    main()

import pygame
import random
import numpy as np
from collections import deque

# TODO MAKE IT LEARN SLOWLY
# TODO MAKE IT SAVE DATA

# Initialize Pygame
pygame.init()
pygame.font.init()

# Constants
WIDTH, HEIGHT = 750,750
FPS = 60

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)
CYAN = (0, 255, 255)
ORANGE = (255, 165, 0)
MAGENTA = (255, 0, 255)

# TRAP PROPTERTIES:
# 0 = wall
# 1 = path
# 2 = start
# 3 = end
# 4 = PURPLE = motion sensor (has to wait for 1 second before it can move)
# 5 = CYAN = spin sensor (has to rotate at least 1 time before it can move)
# 6 = ORANGE = 

class NeuralNetwork():
    def __init__(self):
        self.input_size = 4
        self.hidden_size = 6
        self.output_size = 3

        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.1
        self.b1 = np.zeros((1, self.hidden_size))

        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.1
        self.b2 = np.zeros((1, self.output_size))

    def relu(self, x):
        return np.maximum(0, x)

    def predict_q_values(self, state):
        z1 = np.dot(state, self.W1) + self.b1
        a1 = self.relu(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        return z2  # Raw Q-values
    
    def softmax(self, x):
        x = x - np.max(x, axis=1, keepdims=True)  # helps avoid big exponentials
        e_x = np.exp(x)
        return e_x / e_x.sum(axis=1, keepdims=True)

    def forward(self, state):
        """
        state: numpy array of shape (1, 3)
        returns: action probabilities (1, 3)
        """
        z1 = np.dot(state, self.W1) + self.b1
        a1 = self.relu(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        probs = self.softmax(z2)
        return probs
    
    def train(self, state, action, reward, next_state, done, learning_rate=0.01, gamma=0.99):
        
        # Forward pass on current state
        z1 = np.dot(state, self.W1) + self.b1
        a1 = self.relu(z1)
        q_values = np.dot(a1, self.W2) + self.b2

        # Forward pass on next state
        z1_next = np.dot(next_state, self.W1) + self.b1
        a1_next = self.relu(z1_next)
        q_values_next = np.dot(a1_next, self.W2) + self.b2

        # Compute target Q-value
        target_q = q_values.copy()
        if done:
            target_q[0][action] = reward
        else:
            target_q[0][action] = reward + gamma * np.max(q_values_next)

        # --- BACKPROPAGATION ---

        # Output layer gradient
        dZ2 = q_values - target_q        # (1, 3)
        dW2 = np.dot(a1.T, dZ2)          # (6, 3)
        db2 = dZ2                        # (1, 3)

        # Hidden layer gradient
        dA1 = np.dot(dZ2, self.W2.T)     # (1, 6)
        dZ1 = dA1 * (z1 > 0)             # ReLU derivative
        dW1 = np.dot(state.T, dZ1)       # (3, 6)
        db1 = dZ1                        # (1, 6)

        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        

        loss = np.mean((q_values - target_q) ** 2)
        return loss


class Maze:
    def __init__(self, size, cell_size):
        self.rows = size
        self.cols = size
        self.offset = [100,100]
        self.cell_size = cell_size
        self.generateNewMaze()
    
    def generateNewMaze(self):
        self.exit_made = False
        self.maze = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        self.visited = [[False for _ in range(self.cols)] for _ in range(self.rows)]
        self.generateMaze(1, 1)
        self.maze[1][1] = 2
        self.generateRandomTrap()
        self.printMaze()

    def generateMaze(self, x, y):
        self.maze[x][y] = 1
        self.visited[x][y] = True

        directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
        random.shuffle(directions)
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 < nx < self.rows and 0 < ny < self.cols and not self.visited[nx][ny]:
                self.maze[x + dx // 2][y + dy // 2] = 1
                self.generateMaze(nx, ny)
        if not self.exit_made:
            self.exit_made = True
            self.maze[x][y] = 3

    def generateRandomTrap(self):
        random_x = random.randint(1, self.rows - 2)
        random_y = random.randint(1, self.cols - 2)
        while self.maze[random_x][random_y] != 1:
            random_x = random.randint(1, self.rows - 2)
            random_y = random.randint(1, self.cols - 2)
        self.maze[random_x][random_y] = 4

    def printMaze(self):
        for row in range(self.rows):
            new_string = ""
            for col in range(self.cols):
                if self.maze[row][col] == 1:
                    new_string += ". "
                else:
                    new_string += "X "
            print(new_string)
    
    def draw(self, screen):
        for row in range(self.rows):
            for col in range(self.cols):
                match self.maze[row][col]:
                    case 0:
                        color = BLACK
                    case 1:
                        color = WHITE
                    case 2:
                        color = YELLOW
                    case 3:
                        color = GREEN
                    case 4:
                        color = PURPLE
                    case _:
                        color = WHITE
                pygame.draw.rect(screen, color, (col * self.cell_size + self.offset[0], row * self.cell_size + self.offset[1], self.cell_size, self.cell_size))

class Minion:
    def __init__(self, x: int, y: int, color: list):
        self.x = x
        self.y = y
        self.rotation = 0
        self.visited = set()
        self.spin_streak = 0
        self.lastActionWasRotation = False
        self.color = color

    def getDirection(self):
        match self.rotation:
            case 0:
                return (1, 0)
            case 1:
                return (0, 1)
            case 2:
                return (-1, 0)
            case 3:
                return (0, -1)
            case _:
                return (0, 0)

    def move(self, dx: int, dy: int):
        self.x += dx
        self.y += dy

    def move_forward(self):
        self.x += self.getDirection()[0]
        self.y += self.getDirection()[1]
    
    def rotate(self, direction: str):
        if direction == "left":
            self.rotation = (self.rotation - 1) % 4
        elif direction == "right":
            self.rotation = (self.rotation + 1) % 4

    def draw(self, screen: pygame.display):
        # intigrate rotation
        if self.rotation == 0:
            pygame.draw.polygon(screen, self.color, [(self.x * CELL_SIZE + 100 + CELL_SIZE / 4, self.y * CELL_SIZE + 100 + CELL_SIZE / 4), (self.x * CELL_SIZE + 100 + CELL_SIZE / 4, self.y * CELL_SIZE + 100 + CELL_SIZE / 2 + CELL_SIZE / 4), (self.x * CELL_SIZE + 100 + CELL_SIZE / 2 + CELL_SIZE / 4, self.y * CELL_SIZE + 100 + CELL_SIZE / 4 + CELL_SIZE / 4)])
        elif self.rotation == 1:
            pygame.draw.polygon(screen, self.color, [(self.x * CELL_SIZE + 100 + CELL_SIZE / 4, self.y * CELL_SIZE + 100 + CELL_SIZE / 4), (self.x * CELL_SIZE + 100 + CELL_SIZE / 2, self.y * CELL_SIZE + 100 + CELL_SIZE / 2 + CELL_SIZE / 4), (self.x * CELL_SIZE + 100 + CELL_SIZE / 2 + CELL_SIZE / 4, self.y * CELL_SIZE + 100 + CELL_SIZE / 4)])
        elif self.rotation == 2:
            pygame.draw.polygon(screen, self.color, [(self.x * CELL_SIZE + 100 + CELL_SIZE / 4 + CELL_SIZE / 2, self.y * CELL_SIZE + 100 + CELL_SIZE / 4), (self.x * CELL_SIZE + 100 + CELL_SIZE / 4 + CELL_SIZE / 2, self.y * CELL_SIZE + 100 + CELL_SIZE / 2 + CELL_SIZE / 4), (self.x * CELL_SIZE + 100 + CELL_SIZE / 4, self.y * CELL_SIZE + 100 + CELL_SIZE / 4 + CELL_SIZE / 4)])
        elif self.rotation == 3:
            pygame.draw.polygon(screen, self.color, [(self.x * CELL_SIZE + 100 + CELL_SIZE / 4, self.y * CELL_SIZE + 100 + CELL_SIZE / 4 + CELL_SIZE / 2), (self.x * CELL_SIZE + 100 + CELL_SIZE / 2, self.y * CELL_SIZE + 100 + CELL_SIZE / 4), (self.x * CELL_SIZE + 100 + CELL_SIZE / 2 + CELL_SIZE / 4, self.y * CELL_SIZE + 100 + CELL_SIZE / 4 + CELL_SIZE / 2)])
    

    def state(self, maze: Maze):
        dx, dy = self.getDirection()
        block_in_front = 0
        if self.y + dy > len(maze.maze)-1 or self.x + dx > len(maze.maze)-1:
            block_in_front = -1
        else:
            block_in_front = maze.maze[self.y + dy][self.x + dx]
        state = np.array([[self.x, self.y, self.rotation, block_in_front]])
        return state

    def action(self, maze: Maze, state: list):

        q_values = neuralNetwork.predict_q_values(state)

        epsilon = 0.1 # add a 10% to be random

        if np.random.rand() < epsilon:
            action = np.random.choice(3)  # random action
        else:
            q_values = neuralNetwork.predict_q_values(state)
            action = np.argmax(q_values[0])  # greedy
        action = np.argmax(q_values[0])

        match action:
            case 0:
                self.lastActionWasRotation = False
                self.move_forward()
            case 1:
                self.lastActionWasRotation = True
                self.rotate("left")
            case 2:
                self.lastActionWasRotation = True
                self.rotate("right")
            case _:
                pass
        return action
    
    def reward(self, maze, action):
        reward = 0
        if self.y > len(maze.maze) - 1 or self.x > len(maze.maze) - 1 or self.y < 0 or self.x < 0:
            reward = -100
            self.x = 1
            self.y = 1
        else:
            match maze.maze[self.y][self.x]:
                case 0:
                    reward = -10
                case 1:
                    reward = 0.01
                case 2:
                    reward = 0
                case 3:
                    reward = 100
                case 4:
                    reward = -0.1
                case _:
                    reward = -0.1
            
        
                    
        if action in [1, 2]:  # Rotate left/right
            self.spin_streak += 1
        else:
            self.spin_streak = 0

        if self.spin_streak > 2:
            reward -= self.spin_streak  # the longer it spins, the worse it gets


        
        if (self.x, self.y) not in self.visited:
            reward += 0.5  # Encourage exploring
            self.visited.add((self.x, self.y))


        return reward

    def done(self, maze):
        return maze.maze[self.y][self.x] == 3


# Set up the display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Traps AI")


maze_size = 21
CELL_SIZE = 550 // maze_size
print("CELL_SIZE: ", CELL_SIZE)

maze = Maze(maze_size, CELL_SIZE)

maze.generateRandomTrap()

minion = Minion(1, 1, RED)


### TESTING
neuralNetwork = NeuralNetwork()

# main loop
def main():
    minion_tick_delay = 0
    clock = pygame.time.Clock()
    running = True
    memory = deque(maxlen=750)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    maze.generateNewMaze()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    minion.rotate("left")
                if event.key == pygame.K_RIGHT:
                    minion.rotate("right")
                if event.key == pygame.K_UP:
                    # dx, dy = minion.getDirection()
                    # if maze.maze[minion.y + dy][minion.x + dx] != 0:
                    # minion.move(dx, dy)
                    minion.move_forward()
        screen.fill(WHITE)

        minion_tick_delay += 1
        if minion_tick_delay >= 0:
            minion_tick_delay = 0

            state = minion.state(maze)
            action = minion.action(maze, state)
            reward = minion.reward(maze, action)
            
            next_state = minion.state(maze)
            done = minion.done(maze)

            data = (state, action, reward, next_state, done)
            memory.append(data)
            for experience in random.sample(memory, len(memory)):
                state, action, reward, next_state, done = experience
                neuralNetwork.train(state, action, reward, next_state, done)
            
            minion.lastAction = action

        maze.draw(screen)
        minion.draw(screen)

        pygame.display.flip()

        clock.tick(FPS)

    pygame.quit()

    return memory


if __name__ == "__main__":
    memory = main()

count = 1
for i in memory:
    print(f'{count}. {i}')
    count += 1
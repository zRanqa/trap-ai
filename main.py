import pygame
import random
import numpy as np
from collections import deque

# Initialize Pygame
pygame.init()
pygame.font.init()

# Constants
WIDTH, HEIGHT = 750,750
FPS = 120

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

# KEYBINDS
# SPACE = TOGGLE Pause/Play
# Up arrow = speed up
# down arrow = slow down
# Right Arrow = turn right
# Left arrow = turn left
# W = move forward
# R = reset maze
# Enter = save data
# P = Print out Data from file
# B = TOGGLE walls act as barriers
# T = TOGGLE train from 750 use all to 10000 use 64
# S = Summon reward cube
# 1 = load from autosave
# 2 = load from manual

# TRAP PROPTERTIES:
# 0 = wall
# 1 = path
# 2 = start
# 3 = end
# 4 = PURPLE = reward cube

class NeuralNetwork():
    def __init__(self):
        self.input_size = 4
        self.hidden_size1 = 16
        self.hidden_size2 = 16
        self.output_size = 3

        self.W1 = np.random.randn(self.input_size, self.hidden_size1) * 0.1
        self.b1 = np.zeros((1, self.hidden_size1))

        self.W2 = np.random.randn(self.hidden_size1, self.hidden_size2) * 0.1
        self.b2 = np.zeros((1, self.hidden_size2))

        self.W3 = np.random.randn(self.hidden_size2, self.output_size) * 0.1
        self.b3 = np.zeros((1, self.output_size))


    def relu(self, x):
        return np.maximum(0, x)

    def relu_deriv(self, x):
        return (x > 0).astype(float)

    def predict_q_values(self, state):
        z1 = np.dot(state, self.W1) + self.b1
        a1 = self.relu(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self.relu(z2)
        z3 = np.dot(a2, self.W3) + self.b3
        return z3
    
    def softmax(self, x):
        x = x - np.max(x, axis=1, keepdims=True)  # helps avoid big exponentials
        e_x = np.exp(x)
        return e_x / e_x.sum(axis=1, keepdims=True)
    
    def train(self, state, action, reward, next_state, done, learning_rate=0.01, gamma=0.99):

        z1 = np.dot(state, self.W1) + self.b1
        a1 = self.relu(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self.relu(z2)
        q_values = np.dot(a2, self.W3) + self.b3

        z1_next = np.dot(next_state, self.W1) + self.b1
        a1_next = self.relu(z1_next)
        z2_next = np.dot(a1_next, self.W2) + self.b2
        a2_next = self.relu(z2_next)
        q_values_next = np.dot(a2_next, self.W3) + self.b3

        # Compute target Q-value
        target_q = q_values.copy()
        if done:
            target_q[0][action] = reward
        else:
            target_q[0][action] = reward + gamma * np.max(q_values_next)

        # --- BACKPROPAGATION ---

        dZ3 = q_values - target_q
        dW3 = np.dot(a2.T, dZ3)
        db3 = dZ3

        dA2 = np.dot(dZ3, self.W3.T)
        dZ2 = dA2 * self.relu_deriv(z2)
        dW2 = np.dot(a1.T, dZ2)
        db2 = dZ2

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.relu_deriv(z1)
        dW1 = np.dot(state.T, dZ1)
        db1 = dZ1

        # --- Update wights and biases ---
        self.W3 -= learning_rate * dW3
        self.b3 -= learning_rate * db3

        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        

        loss = np.mean((q_values - target_q) ** 2)
        return loss
    
    def save_model(self, filename="autosave_scores.npz"):
        np.savez(filename, W1=self.W1, W2=self.W2, W3=self.W3, b1=self.b1, b2=self.b2, b3=self.b3)
    
    def load_data(self, filename="autosave_scores.npz"):
        data = np.load(filename)
        newNeuralNetwork = NeuralNetwork()
        newNeuralNetwork.W1 = data["W1"]
        newNeuralNetwork.W2 = data["W2"]
        newNeuralNetwork.W3 = data["W3"]
        newNeuralNetwork.b1 = data["b1"]
        newNeuralNetwork.b2 = data["b2"]
        newNeuralNetwork.b3 = data["b3"]
        return newNeuralNetwork
    
    def print(self):
        print(f'W1: {self.W1}')
        print(f'W2: {self.W2}')
        print(f'W3: {self.W3}')
        print(f'b1: {self.b1}')
        print(f'b2: {self.b2}')
        print(f'b3: {self.b3}')


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
        self.generateTrap(4)
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

    def generateTrap(self, trap: int):
        random_x = random.randint(1, self.rows - 2)
        random_y = random.randint(1, self.cols - 2)
        while self.maze[random_x][random_y] != 1:
            random_x = random.randint(1, self.rows - 2)
            random_y = random.randint(1, self.cols - 2)
        self.maze[random_x][random_y] = trap

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
                inner_color = None
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
                        color = WHITE
                        inner_color = PURPLE
                    case _:
                        color = WHITE
                pygame.draw.rect(screen, color, (col * self.cell_size + self.offset[0], row * self.cell_size + self.offset[1], self.cell_size, self.cell_size))
                if inner_color != None:
                    pygame.draw.rect(screen, inner_color, (col * self.cell_size + self.offset[0] + self.cell_size / 4, row * self.cell_size + self.offset[1] + self.cell_size / 4, self.cell_size / 2, self.cell_size / 2))


class Minion:
    def __init__(self, x: int, y: int, color: list):
        self.x = x
        self.y = y
        self.rotation = 0
        self.visited = set()
        self.spin_streak = 0
        self.lastLocationVisited = [x,y]
        self.lastActionWasRotation = False
        self.color = color
        self.minionBarrierToggle = False


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
    
    def checkBlockInFront(self, maze):
        dx, dy = self.getDirection()
        blockInFront = -1
        if self.y + dy > len(maze.maze)-1 or self.x + dx > len(maze.maze)-1:
            blockInFront = -1
        else:
            blockInFront = maze.maze[self.y + dy][self.x + dx]
        return blockInFront

    def state(self, maze: Maze):
        self.blockInFront = self.checkBlockInFront(maze)
        state = np.array([[self.x, self.y, self.rotation, self.blockInFront]])
        return state

    def action(self, state: list, neuralNetwork: NeuralNetwork):

        q_values = neuralNetwork.predict_q_values(state)

        epsilon = 0.01 # add a 1% to be random

        if np.random.rand() < epsilon:
            print("Random action")
            action = np.random.choice(3)  # random action
        else:
            q_values = neuralNetwork.predict_q_values(state)
            action = np.argmax(q_values[0])  # greedy
        action = np.argmax(q_values[0])

        match action:
            case 0:
                self.lastLocationVisited = [self.x, self.y]
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
                case 0: # WALL
                    print("minion hit wall")
                    reward = -100
                    if self.minionBarrierToggle:
                        self.x = self.lastLocationVisited[0]
                        self.y = self.lastLocationVisited[1]

                case 1: # PATH
                    reward = 1
                    if self.minionBarrierToggle:
                        reward = 2.5
                case 2: # START
                    reward = 0
                case 3: # END/GOAL
                    print("Minion Solved Maze!")
                    reward = 100
                case 4: # TRAP/PURPLE SPACE
                    reward = 25
                    maze.maze[self.y][self.x] = 1
                case _: # DEFAULT
                    reward = -0.1

        if action > 0 and self.blockInFront == 0 and self.checkBlockInFront(maze) == 1:
            reward += 1

        if action in [1, 2]:  # Rotate left/right
            self.spin_streak += 1
        else:
            self.spin_streak = 0

        if self.spin_streak > 2:
            reward -= self.spin_streak * 10 # the longer it spins, the worse it gets


        
        if (self.x, self.y) not in self.visited:
            if maze.maze[self.y][self.x] == 1:
                print("BONUS REWARD GIVEN FOR EXPLORATION")
                reward += 5 # Encourage exploring
                self.visited.add((self.x, self.y))


        return reward

    def done(self, maze):
        if maze.maze[self.y][self.x] == 3:
            maze.generateNewMaze()
            self.visited = set()
            self.x = 1
            self.y = 1
        return maze.maze[self.y][self.x] == 3
        


# Set up the display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Traps AI")


maze_size = 7
CELL_SIZE = 550 // maze_size
print("CELL_SIZE: ", CELL_SIZE)

maze = Maze(maze_size, CELL_SIZE)

minion = Minion(1, 1, RED)



# main loop
def main():
    neuralNetwork = NeuralNetwork()
    trainAllToggle = False

    autoSaveTickTimer = 0

    minionTickDelayOptions = [60, 30, 15, 5, 1]
    minionTickDelayIndex = 0
    minionTickDelayPaused = False

    minion_tick_delay = 0
    clock = pygame.time.Clock()
    running = True
    memory = deque(maxlen=10000)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # if event.type == pygame.MOUSEBUTTONDOWN:
            #     if event.button == 1:
            #         maze.generateNewMaze()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    minion.rotate("left")
                    print(f"Minion rotated left")
                if event.key == pygame.K_RIGHT:
                    minion.rotate("right")
                    print(f"Minion rotated right")
                if event.key == pygame.K_w:
                    # dx, dy = minion.getDirection()
                    # if maze.maze[minion.y + dy][minion.x + dx] != 0:
                    # minion.move(dx, dy)
                    minion.move_forward()
                    print(f"Minion moved forward")
                if event.key == pygame.K_UP:
                    minionTickDelayIndex += 1
                    if minionTickDelayIndex > len(minionTickDelayOptions) - 1:
                        minionTickDelayIndex = len(minionTickDelayOptions) - 1
                    print(f"Minion speed set to x{minionTickDelayOptions[-minionTickDelayIndex - 1]}")
                if event.key == pygame.K_DOWN:
                    minionTickDelayIndex -= 1
                    if minionTickDelayIndex < 1:
                        minionTickDelayIndex = 0
                    print(f"Minion speed set to x{minionTickDelayOptions[-minionTickDelayIndex - 1]}")
                if event.key == pygame.K_SPACE:
                    minionTickDelayPaused = not minionTickDelayPaused
                    print(f"Minion pause toggle set to: {minionTickDelayPaused}")
                if event.key == pygame.K_r:
                    maze.generateNewMaze()
                    minion.x = 1
                    minion.y = 1
                    minion.visited = set()
                    print(f"Maze reset")
                if event.key == pygame.K_RETURN:
                    neuralNetwork.save_model(filename="manual_save_scores.npz")
                    print(f"Neural network Manually Saved")
                if event.key == pygame.K_p:
                    newNeuralNetwork = neuralNetwork.load_data(filename="manual_save_scores.npz")
                    newNeuralNetwork.print()
                if event.key == pygame.K_b:
                    minion.minionBarrierToggle = not minion.minionBarrierToggle
                    print(f"Minion barrier toggle set to: {minion.minionBarrierToggle}")
                if event.key == pygame.K_t:
                    trainAllToggle = not trainAllToggle
                    if trainAllToggle:
                        string = 750
                    else:
                        string = 64
                    print(f"Minion training amount set to: {string}")
                if event.key == pygame.K_s:
                    maze.generateTrap(4)
                    print("Reward tile generated")
                if event.key == pygame.K_1:
                    neuralNetwork = neuralNetwork.load_data()
                    print("Neural Network Loaded: Autosave")
                if event.key == pygame.K_2:
                    neuralNetwork = neuralNetwork.load_data(filename="manual_save_scores.npz")
                    print("Neural Network Loaded: Manual")

                
        autoSaveTickTimer += 1
        if autoSaveTickTimer >= 36000:
            autoSaveTickTimer = 0
            neuralNetwork.save_model()

        minion_tick_delay += 1
        if minion_tick_delay >= minionTickDelayOptions[minionTickDelayIndex] and not minionTickDelayPaused:
            minion_tick_delay = 0

            state = minion.state(maze)
            action = minion.action(state, neuralNetwork)
            reward = minion.reward(maze, action)
            
            next_state = minion.state(maze)
            done = minion.done(maze)

            data = (state, action, reward, next_state, done)
            memory.append(data)
            sample = 0

            if trainAllToggle:
                sample_max = 750
            else:
                sample_max = 64

            if len(memory) < sample_max:
                sample = len(memory)
            else:
                sample = sample_max

            for experience in random.sample(memory, sample):
                state, action, reward, next_state, done = experience
                neuralNetwork.train(state, action, reward, next_state, done)
            
            minion.lastAction = action

        screen.fill(WHITE)

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
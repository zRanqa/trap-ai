import pygame
import random

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
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.rotation = 0
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

    def move(self, dx, dy):
        self.x += dx
        self.y += dy
    
    def rotate(self, direction):
        if direction == "left":
            self.rotation = (self.rotation - 1) % 4
        elif direction == "right":
            self.rotation = (self.rotation + 1) % 4

    def draw(self, screen):
        # intigrate rotation
        if self.rotation == 0:
            pygame.draw.polygon(screen, self.color, [(self.x * CELL_SIZE + 100 + CELL_SIZE / 4, self.y * CELL_SIZE + 100 + CELL_SIZE / 4), (self.x * CELL_SIZE + 100 + CELL_SIZE / 4, self.y * CELL_SIZE + 100 + CELL_SIZE / 2 + CELL_SIZE / 4), (self.x * CELL_SIZE + 100 + CELL_SIZE / 2 + CELL_SIZE / 4, self.y * CELL_SIZE + 100 + CELL_SIZE / 4 + CELL_SIZE / 4)])
        elif self.rotation == 1:
            pygame.draw.polygon(screen, self.color, [(self.x * CELL_SIZE + 100 + CELL_SIZE / 4, self.y * CELL_SIZE + 100 + CELL_SIZE / 4), (self.x * CELL_SIZE + 100 + CELL_SIZE / 2, self.y * CELL_SIZE + 100 + CELL_SIZE / 2 + CELL_SIZE / 4), (self.x * CELL_SIZE + 100 + CELL_SIZE / 2 + CELL_SIZE / 4, self.y * CELL_SIZE + 100 + CELL_SIZE / 4)])
        elif self.rotation == 2:
            pygame.draw.polygon(screen, self.color, [(self.x * CELL_SIZE + 100 + CELL_SIZE / 4 + CELL_SIZE / 2, self.y * CELL_SIZE + 100 + CELL_SIZE / 4), (self.x * CELL_SIZE + 100 + CELL_SIZE / 4 + CELL_SIZE / 2, self.y * CELL_SIZE + 100 + CELL_SIZE / 2 + CELL_SIZE / 4), (self.x * CELL_SIZE + 100 + CELL_SIZE / 4, self.y * CELL_SIZE + 100 + CELL_SIZE / 4 + CELL_SIZE / 4)])
        elif self.rotation == 3:
            pygame.draw.polygon(screen, self.color, [(self.x * CELL_SIZE + 100 + CELL_SIZE / 4, self.y * CELL_SIZE + 100 + CELL_SIZE / 4 + CELL_SIZE / 2), (self.x * CELL_SIZE + 100 + CELL_SIZE / 2, self.y * CELL_SIZE + 100 + CELL_SIZE / 4), (self.x * CELL_SIZE + 100 + CELL_SIZE / 2 + CELL_SIZE / 4, self.y * CELL_SIZE + 100 + CELL_SIZE / 4 + CELL_SIZE / 2)])
        


# Set up the display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Traps AI")

# Load font
font = pygame.font.SysFont("Arial", 24)

maze_size = 21
CELL_SIZE = 550 // maze_size
print("CELL_SIZE: ", CELL_SIZE)

maze = Maze(maze_size, CELL_SIZE)

maze.generateRandomTrap()

minion = Minion(1, 1, RED)

# main loop
def main():
    clock = pygame.time.Clock()
    running = True

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
                    dx, dy = minion.getDirection()
                    print(minion.x, minion.y, minion.x + dx, minion.y + dy)
                    print(maze.maze[minion.x + dx][minion.y + dy])
                    if maze.maze[minion.y + dy][minion.x + dx] != 0:
                        minion.move(dx, dy)
        screen.fill(WHITE)



        maze.draw(screen)
        minion.draw(screen)

        pygame.display.flip()

        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
import pygame
import random


# Initialize Pygame
pygame.init()

WIDTH = 500
HEIGHT = 500

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("TY STINKS")

zen_x = 0
zen_y = 0

FPS = 60

clock = pygame.time.Clock()

running = True
while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_UP:
            zen_y -= 5
        if event.key == pygame.K_DOWN:
            zen_y += 5
        if event.key == pygame.K_LEFT:
            zen_x -= 5
        if event.key == pygame.K_RIGHT:
            zen_x += 5

    screen.fill((255,255,255))

    screen.fill((255,0,255), pygame.Rect(zen_x,zen_y,50,50))

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
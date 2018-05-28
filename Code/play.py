import keras
import numpy as np

#Game vars
file_name = "In Their Footsteps.exe"
buttons = ['Left', "Right", "Down", "Space"]

#Window vars???
box_radius = 6
input_size = (box_radius*2+1)*(box_radius*2+1)

#network vars???
population = 300
delta_disjoint = 2.0
delta_weights = 0.4
delta_threshold = 1.0

stale_species = 15

mutate_conn_chance = 0.25
perturb_chance = 0.90
crossover_chance = 0.75
link_mutation_change = 2.0
node_mutation_chance = 0.50
bias_mutation_chance = 0.40
step_size = 0.1
disable_mutation_chance = 0.4
enable_mutation_chance = 0.2
timeout = 20

max_nodes = 1000000

def get_positions():
    #Get the locations of game objects?
    #Player character and screen x and y coords at least
    return 0

def get_tile(dx, dy):
    #The next floor tile maybe?
    return 0

def get_sprites():
    #gets sprites, seems obvious enough
    return 0

def get_extended_sprites():
    #Get more sprites?
    return 0

def get_inputs():
    get_positions()

    sprites = get_sprites()
    extended = get_extended_sprites()

    inputs = []

    for dy in xrange(-box_radius*16, box_radius*16,16):
        for dx in xrange(-box_radius*16,box_radius*16,16):
            inputs[len(inputs)+1] = 0
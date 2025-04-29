from model.ppo import PPO
from mcts.tree import MCTStree
from mcts.node import MCTSnode
from saves import cleanup_checkpoints, load_latest_checkpoint
import torch
import sys
import os
import chess

project_folder = os.path.dirname(__file__)
sys.path.append(project_folder)

checkpoint_folder = os.path.join(project_folder, "checkpoints")
model = PPO()

load_latest_checkpoint(checkpoint_folder, model)

model.cuda()

#SETTINGS

nodeC = 2
nodeTemp = 1.5
treeSimulation = 100
treeTemp = 1.5
debug_pause_to_view = False





exampleNode = MCTSnode(model, None, None, chess.Board(), None, C = nodeC, temperature = nodeTemp)

def train(games = 10000, games_per_save = 1):
    for game in range (games):
        print(f"\nGame {game}")
        tree = MCTStree(model, 
                        exampleNode, 
                        simulation = treeSimulation,
                        temperature = treeTemp,
                        debug_pause_to_view = debug_pause_to_view)

        tree.game()
        if (games % games_per_save) == 0:
            torch.save(model.state_dict(),f"{checkpoint_folder}/game{game}.pt")
            cleanup_checkpoints(checkpoint_folder)



train ()

# command to monitor gpu cuda core and vram usage
# nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv --loop=1
# monitor cpu
# mpstat 1



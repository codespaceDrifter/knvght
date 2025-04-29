import chess
import math
import torch
import random
from model.ppo import PPO
from model.board import move_to_index, board_to_tensor, make_move

class MCTSnode ():
    def __init__(self,
                 model: PPO, 
                 parent: 'MCTSnode',
                 actionID:int,
                 board: chess.Board,
                 P: float,
                 C = 1,
                 temperature = 1):
        self.model = model
        self.parent = parent
        self.children = []
        #the action ID that got here. used for policy training in tree. 
        self.actionID = actionID
        self.board = board
        self.terminal = self.board.is_game_over()

        # the actions in possibleActions and possibleProbs have to match
        self.possibleActions = []
        self.numWalked = 1
        self.W = 0 # backpropped sum values
        self.Q = 0 # W/ numWalked
        self.P = P # policy network choosing this probability
        self.C = C # exploration constant
        self.temperature = temperature #expansion phase randomness

        if self.terminal == True:
            outcome = self.board.outcome()
            if outcome.winner is True:
                self.value = 1.0  # White wins
            elif outcome.winner is False:
                self.value = -1.0  # Black wins
            else:
                self.value = 0.0  # Draw

            self.possibleProbs = []


        else:
            for move in self.board.legal_moves:
                idx = move_to_index(move)
                self.possibleActions.append(idx)

            with torch.no_grad():
                value, policy= self.model.combined_forward([self.board])
                #used for backprop
                self.value = value.cpu().item()

                probs = policy[0].cpu()
                self.possibleProbs = [probs[idx].item() for idx in self.possibleActions]

        self.backprop(self.value)

    def backprop(self, value):

        current_node = self
        current_value = value
        
        while current_node is not None:
            current_node.W += current_value
            current_node.Q = current_node.W / current_node.numWalked
            current_value = -current_value  # Flip value for parent
            current_node = current_node.parent


    def walk(self):



        self.numWalked += 1

        if self.terminal:
            return

        #expand new node
        if self.possibleActions:
            probs = torch.tensor(self.possibleProbs)
            probs = probs.pow(1.0/self.temperature)
            probs = probs + 1e-8 
            probs = probs / probs.sum()


            bestIndex = torch.multinomial(probs, num_samples=1).item()

            bestAction = self.possibleActions[bestIndex]
            bestProb = self.possibleProbs[bestIndex]
            bestBoard = make_move(self.board, bestAction)

            self.children.append(MCTSnode(self.model,
                                          self,
                                          bestAction,
                                          bestBoard,
                                          bestProb,
                                          self.C,
                                          self.temperature))
            self.possibleActions.pop(bestIndex)
            self.possibleProbs.pop(bestIndex)

        else:
            #walk down tree
            UCBvalues = []
            for child in self.children:
                UCBvalue = child.Q + self.C * child.P * math.sqrt(self.numWalked) / (1+child.numWalked)
                UCBvalues.append(UCBvalue)
            maxUCB = max(UCBvalues)
            maxIndex = UCBvalues.index(maxUCB)
            self.children[maxIndex].walk()
        








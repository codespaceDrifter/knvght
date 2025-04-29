from .node import MCTSnode
from model.ppo import PPO
import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from .display import display_board

class MCTStree ():
    def __init__(self,
                 model,
                 exampleNode: MCTSnode,
                 simulation = 100,
                 lr = 0.01,
                 max_grad_norm = 2,
                 temperature = 1,
                 debug_pause_to_view = False):
        self.model = model
        self.exampleNode = exampleNode
        self.simulation = simulation
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.temperature = temperature

        self.debug_pause_to_view = debug_pause_to_view



    def game(self):
        board = chess.Board()
        curNode = MCTSnode(self.exampleNode.model,
                           None,
                           None,
                           board,
                           None,
                           self.exampleNode.C,
                           self.exampleNode.temperature
                           )
        #(moves)
        playedStates = []
        #(moves, 4184)
        playedProbs = []
        
        while (curNode.terminal == False):


            curBoard = curNode.board

            playedStates.append(curBoard)
            curProbs = torch.zeros (self.model.moves)
            for child in curNode.children:
                curProbs[child.actionID] = child.numWalked
                curProbs = curProbs / curProbs.sum()

            curProbs = curProbs.cuda()
            playedProbs.append(curProbs)

            display_board(curBoard)
            
            # to view in web to see what is going on
            if self.debug_pause_to_view: time.sleep(0.5)

            '''
            print(" +-----------------+")
            print(curBoard)
            print(" +-----------------+")
            '''

            #IMPORTANT to update LAST IN LOOP. otherwise Terminal State will make Loss become NaN
            curNode = self.move(curNode)

        outcome = curNode.board.outcome()

        if outcome.winner is True:
            print("White wins")
            final_value = 1.0
        elif outcome.winner is False:
            print("Black wins")
            final_value = -1.0
        else:
            # Draw reasons
            term = outcome.termination
            if term == chess.Termination.STALEMATE:
                print("Draw by stalemate")
            elif term == chess.Termination.INSUFFICIENT_MATERIAL:
                print("Draw by insufficient material")
            elif term == chess.Termination.FIVEFOLD_REPETITION:
                print("Draw by fivefold repetition")
            elif term == chess.Termination.SEVENTYFIVE_MOVES:
                print("Draw by 75-move rule")
            else:
                print("Draw Unknown")
            final_value = 0.0

        print(" +-----------------+")
        print(curBoard)
        print(" +-----------------+")

        playedValues = [final_value if i % 2 == 0 else -final_value for i in range(len(playedStates))]
        # (moves)
        playedValues = torch.tensor(playedValues).cuda()

        # (moves, 4184)
        playedProbs = torch.stack(playedProbs, dim = 0).cuda()
        predValues, predProbs = self.model.combined_forward(playedStates)


        valueLoss = F.mse_loss(predValues, playedValues)
        policyLoss = F.mse_loss (predProbs, playedProbs)
        totalLoss = valueLoss + policyLoss


        self.optimizer.zero_grad()
        totalLoss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        for param in self.model.parameters():
            param.data.clamp_(-5.0, 5.0) 
            


    def move(self, curNode):


        for _ in range (self.simulation):
            curNode.walk()

        childrenWalkNum = [child.numWalked for child in curNode.children]
        childrenWalkNum = torch.tensor(childrenWalkNum)

        if self.temperature == 0:
            chosenIndex = torch.argsmax(childrenWalkNum).item()
        else:
            probs = childrenWalkNum.pow(1.0 / self.temperature)
            probs = probs + 1e-8 
            probs = probs / probs.sum()
            chosenIndex = torch.multinomial(probs, num_samples=1).item()

        chosenNode = curNode.children[chosenIndex]
        return chosenNode 



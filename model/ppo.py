import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
from .conv import ConvBlock
from .ffw import FeedForward
from .residual import Residual
from .board import legal_moves_mask, board_to_tensor

class PPO(nn.Module):
    # white 1 black 0
    # input (18,8,8) 8x8 is board size.
    # 18: 0-11 chess pieces. 0-5 white pawn, knight, bishop, rook, queen, king)
    # 12 side to move. 13-16 castling rights (kingside queenside). 17 move repetitions
    def __init__ (self,
                  moves = 4184,
                  backbone_channel = 256,
                  backbone_blocks = 10,
                  policy_channel = 4,
                  value_channel = 2,
                  value_hidden = 256):
        #total parameter for default values (25,504,537)
        super().__init__()
        self.moves = moves
        self.startConv = ConvBlock(18, backbone_channel)
        # module list weight count = 20 * 256 * 3 * 3 * 256 = 589,824
        self.backbone = nn.ModuleList([
            Residual(ConvBlock(backbone_channel, backbone_channel)) for _ in range (backbone_blocks)
        ])
        

        self.policyConv = ConvBlock(backbone_channel, policy_channel)
        linear_in = 8*8*policy_channel
        self.policyFlatten = nn.Flatten()
        self.policyLinear = nn.Linear(linear_in, moves)
        #then softmax

        self.valueConv = ConvBlock(backbone_channel, value_channel)
        self.valueFlatten = nn.Flatten()
        ffw_in = 8*8*value_channel
        self.valueFeedForward = FeedForward(ffw_in, value_hidden, 1)
        #then tanh

    def board_to_tensor_and_mask (self, boardList: list[chess.Board]):
        tensorList = []
        maskList = []
        for board in boardList:
            tensorList.append(board_to_tensor(board))
            maskList.append(legal_moves_mask(board))
        tensorList = torch.stack (tensorList, dim = 0).cuda()
        maskList = torch.stack (maskList, dim = 0).cuda()
        return tensorList, maskList

    #(batch, 18, 8, 8)
    def _backbone_forward(self, x):
        #(batch, 256, 8, 8)
        x = self.startConv(x)
        #(batch, 256, 8, 8)
        for block in self.backbone:
            x = block(x)
        return x

    #batched, used for general training
    def combined_forward(self, boardList: list[chess.Board]):
        for board in boardList:
            assert not board.is_game_over()
        
        x, mask = self.board_to_tensor_and_mask(boardList)

        x = self._backbone_forward(x)



        #(batch, 2, 8, 8)
        v = self.valueConv(x)
        #(batch, 64)
        v = self.valueFlatten(v)
        #(batch, 1)
        v = self.valueFeedForward(v)
        #(batch)
        v = v.view(-1)
        #(batch)
        value = F.tanh(v)


        #(batch, 4, 8, 8)
        p = self.policyConv(x)
        #(batch, 256)
        p = self.policyFlatten(p)
        #(batch, 4184)
        p = self.policyLinear(p)

        p = p.masked_fill(mask == 0, float('-inf'))
        p = F.softmax(p, dim=1)
        #(batch, 4184)
        policy = F.softmax(p, dim = 1)

        return value, policy

    # used for rollout

    def policy_forward(self,board):
        assert not board.is_game_over() 
        # mask (batch, 4184)
        mask = legal_moves_mask(board).unsqueeze(0).cuda()
        # do NOT predict terminal moves
        # (batch, 18, 8, 8)
        x = board_to_tensor(board).unsqueeze(0).cuda()
        #(batch,256,8,8)
        x = self._backbone_forward(x)
        #(batch, 4, 8, 8)
        x = self.policyConv(x)
        #(batch, 256)
        x = self.policyFlatten(x)
        #(batch, 4184)
        x = self.policyLinear(x)
        #(batch, 4184)
        x = x.masked_fill(mask == 0, float('-inf'))
        #(batch, 4184)
        x = F.softmax(x, dim = 1)
        choiceID = x.argsmax(dim=1)

        return x 

    
    #(batch, 18, 8, 8)
    def value_forward(self,board):
        assert not board.is_game_over() 
        # (batch, 18, 8, 8)
        x = board_to_tensor(board).unsqueeze(0).cuda()
        #(batch, 256, 8, 8)
        x = self._backbone_forward(x)
        #(batch, 2, 8, 8)
        x = self.valueConv(x)
        #(batch, 64)
        x = self.valueFlatten(x)
        #(batch, 1)
        x = self.valueFeedForward(x)
        x = F.tanh(x)
        return x


        









                


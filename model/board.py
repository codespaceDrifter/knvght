import chess
import os
import torch

#to do: make this cython

def board_to_tensor(board):
    """
    Convert a python-chess Board object to a PyTorch tensor with shape (18, 8, 8)
    
    Channels:
    0-5: White pieces (pawn, knight, bishop, rook, queen, king)
    6-11: Black pieces (pawn, knight, bishop, rook, queen, king)
    12: Side to move (1 for white, 0 for black)
    13: White kingside castling right
    14: White queenside castling right
    15: Black kingside castling right
    16: Black queenside castling right
    17: Move repetition counter normed by 100
    """
    # Initialize tensor with zeros
    tensor = torch.zeros(18, 8, 8, dtype=torch.float32)
    
    # Fill piece channels (0-11)
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
    
    # Mapping from piece type to channel index
    # White pieces: channels 0-5
    # Black pieces: channels 6-11
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            rank, file = chess.square_rank(square), chess.square_file(square)
            color_offset = 0 if piece.color == chess.WHITE else 6
            piece_idx = piece_types.index(piece.piece_type)
            channel = color_offset + piece_idx
            #rank : vertical 1-8. file: horizontal a-g
            tensor[channel][rank][file] = 1  # Note the 7-rank to flip board orientation
    
    # Side to move (channel 12)
    if board.turn == chess.WHITE:
        tensor[12].fill_(1)
    
    # Castling rights (channels 13-16)
    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[13].fill_(1)
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[14].fill_(1)
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[15].fill_(1)
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[16].fill_(1)
    
    # Move repetition counter (channel 17)
    # This requires tracking board history, which basic Board object doesn't provide
    # You would need to implement repetition counting yourself or use a more advanced state representation
    # For simplicity, we'll use halfmove clock normalized by 100 as an approximation
    tensor[17].fill_(board.halfmove_clock / 100.0)
    
    return tensor


def move_to_index(move: chess.Move) -> int:
    """
    Maps a python-chess move to a unique move index.
    - Normal moves: 0–4095
    - Promotions: 4096+
    """
    from_square = move.from_square
    to_square = move.to_square
    
    # Check if promotion
    if move.promotion:
        # Promotion type adjustment:
        # chess.QUEEN = 5, ROOK = 4, BISHOP = 3, KNIGHT = 2
        # Normalize to 0 (Knight) to 3 (Queen)
        promotion_offset = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 3}[move.promotion]
        
        # Detect capture or forward promotion
        from_file = chess.square_file(from_square)
        to_file = chess.square_file(to_square)
        
        if from_file == to_file:
            # Forward promotion
            promo_base = 4096
            index = promo_base + (from_file * 4) + promotion_offset
        elif from_file - to_file == 1:
            # Capture left promotion
            promo_base = 4096 + (8 * 4)
            index = promo_base + ((from_file - 1) * 4) + promotion_offset
        elif to_file - from_file == 1:
            # Capture right promotion
            promo_base = 4096 + (8 * 4) + (7 * 4)
            index = promo_base + (from_file * 4) + promotion_offset
        else:
            raise ValueError("Invalid promotion move structure.")
        
        return index

    else:
        # Normal move
        return 64 * from_square + to_square


def legal_moves_mask(board: chess.Board) -> torch.Tensor:
    """
    Given a python-chess Board, return a (4672,) torch mask tensor:
    1 for legal moves, 0 for illegal moves.
    """
    mask = torch.zeros(4184, dtype=torch.float32)
    
    for move in board.legal_moves:
        idx = move_to_index(move)  # You already have this function
        mask[idx] = 1.0

    return mask


def index_to_move(index: int, board: chess.Board) -> chess.Move:
    """
    Reverse of move_to_index.
    Maps a move index back to a python-chess Move.
    """
    if index < 4096:
        from_square = index // 64
        to_square = index % 64
        move = chess.Move(from_square, to_square)
    else:
        # Promotions
        promotion_index = index - 4096
        if promotion_index < 32:
            # Forward promotions
            file = promotion_index // 4
            promo_type = promotion_index % 4
            from_rank = 6 if board.turn == chess.WHITE else 1
            to_rank = 7 if board.turn == chess.WHITE else 0
            from_square = chess.square(file, from_rank)
            to_square = chess.square(file, to_rank)

        elif promotion_index < 32 + 28:
            # Capture left promotions
            promotion_index -= 32
            file = (promotion_index // 4) + 1
            promo_type = promotion_index % 4
            from_rank = 6 if board.turn == chess.WHITE else 1
            to_rank = 7 if board.turn == chess.WHITE else 0
            from_square = chess.square(file, from_rank)
            to_square = chess.square(file - 1, to_rank)

        else:
            # Capture right promotions
            promotion_index -= (32 + 28)
            file = promotion_index // 4
            promo_type = promotion_index % 4
            from_rank = 6 if board.turn == chess.WHITE else 1
            to_rank = 7 if board.turn == chess.WHITE else 0
            from_square = chess.square(file, from_rank)
            to_square = chess.square(file + 1, to_rank)

        promotion_map = [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
        promotion_piece = promotion_map[promo_type]
        move = chess.Move(from_square, to_square, promotion=promotion_piece)
    
    return move

def make_move(board: chess.Board, move_idx: int) -> chess.Board:
    """
    Apply a move by index to a board, return the resulting board.
    """
    new_board = board.copy()
    move = index_to_move(move_idx, new_board)
    
    if move not in new_board.legal_moves:
        raise ValueError(f"Decoded move {move} is illegal on the given board.")
    
    new_board.push(move)
    return new_board



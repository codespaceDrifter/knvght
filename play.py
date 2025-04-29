import pygame
import chess
import random

# Settings
WIDTH, HEIGHT = 480, 480
SQUARE_SIZE = WIDTH // 8

# Colors
WHITE = (255, 255, 255)
GRAY = (180, 180, 180)
BLUE = (0, 0, 255)

# Init
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Chess vs Random Bot')
clock = pygame.time.Clock()

# Load pieces
pieces_images = {}
pieces = ['R', 'N', 'B', 'Q', 'K', 'P']
for color in ['w', 'b']:
    for piece in pieces:
        img = pygame.image.load(f'assets/{color}{piece}.png')
        img = pygame.transform.scale(img, (SQUARE_SIZE, SQUARE_SIZE))
        pieces_images[color + piece] = img

# Setup board
#board = chess.Board()

def draw_board(board):
    colors = [WHITE, GRAY]
    for row in range(8):
        for col in range(8):
            color = colors[(row + col) % 2]
            pygame.draw.rect(screen, color, pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row = 7 - chess.square_rank(square)
            col = chess.square_file(square)
            color = 'w' if piece.color == chess.WHITE else 'b'
            key = color + piece.symbol().upper()
            screen.blit(pieces_images[key], (col * SQUARE_SIZE, row * SQUARE_SIZE))

'''
def get_square_under_mouse():
    mouse_pos = pygame.Vector2(pygame.mouse.get_pos())
    col = int(mouse_pos.x // SQUARE_SIZE)
    row = 7 - int(mouse_pos.y // SQUARE_SIZE)
    return chess.square(col, row)

def random_bot_move():
    legal = list(board.legal_moves)
    if legal:
        move = random.choice(legal)
        board.push(move)

selected_square = None
promotion_pending = None  # (from_square, to_square)

def draw_promotion_menu():
    menu_pieces = ['Q', 'R', 'B', 'N']
    menu_height = SQUARE_SIZE * len(menu_pieces)
    menu_rect = pygame.Rect(WIDTH//2 - SQUARE_SIZE//2, HEIGHT//2 - menu_height//2, SQUARE_SIZE, menu_height)
    pygame.draw.rect(screen, WHITE, menu_rect)

    for i, piece in enumerate(menu_pieces):
        img = pieces_images['w' + piece]
        screen.blit(img, (WIDTH//2 - SQUARE_SIZE//2, HEIGHT//2 - menu_height//2 + i * SQUARE_SIZE))

def handle_promotion_click(pos):
    menu_pieces = ['Q', 'R', 'B', 'N']
    menu_height = SQUARE_SIZE * len(menu_pieces)
    menu_top = HEIGHT//2 - menu_height//2
    x, y = pos

    if not (WIDTH//2 - SQUARE_SIZE//2 <= x <= WIDTH//2 + SQUARE_SIZE//2):
        return None

    idx = (y - menu_top) // SQUARE_SIZE
    if 0 <= idx < len(menu_pieces):
        return menu_pieces[idx]
    return None

promotion_map = {'Q': chess.QUEEN, 'R': chess.ROOK, 'B': chess.BISHOP, 'N': chess.KNIGHT}

# Main loop
running = True
while running:
    clock.tick(60)
    draw_board()

    if promotion_pending:
        draw_promotion_menu()

    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if promotion_pending:
                choice = handle_promotion_click(event.pos)
                if choice:
                    from_sq, to_sq = promotion_pending
                    move = chess.Move(from_sq, to_sq, promotion=promotion_map[choice])
                    if move in board.legal_moves:
                        board.push(move)
                        promotion_pending = None
                        selected_square = None

                        if not board.is_game_over():
                            random_bot_move()
                else:
                    # click outside menu: do nothing
                    pass
            else:
                square = get_square_under_mouse()
                piece = board.piece_at(square)

                if selected_square is None:
                    if piece and piece.color == chess.WHITE:
                        selected_square = square
                else:
                    if board.piece_at(selected_square):
                        is_pawn = board.piece_at(selected_square).piece_type == chess.PAWN
                        target_rank = chess.square_rank(square)

                        if is_pawn and (target_rank == 0 or target_rank == 7):
                            promotion_pending = (selected_square, square)
                        else:
                            move = chess.Move(selected_square, square)
                            if move in board.legal_moves:
                                board.push(move)
                                selected_square = None

                                if not board.is_game_over():
                                    random_bot_move()
                            else:
                                selected_square = None

pygame.quit()
'''

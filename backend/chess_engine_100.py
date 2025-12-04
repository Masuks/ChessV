import chess
import random

# Initialize board
def create_board():
    return chess.Board()

# Get random engine move
def get_random_move(board):
    legal_moves = list(board.legal_moves)
    if legal_moves:
        return random.choice(legal_moves)
    return None

def play_game():
    board = create_board()
    while not board.is_game_over():
        print("\n" + str(board) + "\n")
        if board.turn == chess.WHITE:
            print("White's turn")
            move_str = input("Your move (e.g., e4, Qxf7, d7d8q for promotion, or 'quit' to exit): ")
            if move_str.lower() == 'quit':
                print("Game ended by user.")
                break
            try:
                move = board.parse_san(move_str)
                if move in board.legal_moves:
                    board.push(move)
                else:
                    print("Illegal move. Try again.")
                    continue
            except ValueError:
                try:
                    # Handle UCI with promotion (e.g., d7d8q)
                    if len(move_str) == 5 and move_str[4] in 'qrbn':  # Promotion
                        move = chess.Move.from_uci(move_str[:4] + move_str[4].lower())
                    else:
                        move = board.parse_uci(move_str)
                    if move in board.legal_moves:
                        print(f"Interpreted as: {board.san(move)}")
                        board.push(move)
                    else:
                        print("Illegal move. Try again.")
                        continue
                except ValueError:
                    print("Invalid move format. Use SAN (e.g., e4, Qxf7) or UCI (e.g., e2e4, d7d8q for promotion).")
                    continue
        else:
            print("Black's turn (Engine thinking...)")
            move = get_random_move(board)
            if move:
                print(f"Engine move: {board.san(move)}")
                board.push(move)
            else:
                print("No legal moves for engine.")
                break
    print("\nGame over:", board.result())
    if board.is_checkmate():
        print("Checkmate!")
    elif board.is_stalemate():
        print("Stalemate!")
    elif board.is_insufficient_material():
        print("Draw by insufficient material!")

if __name__ == "__main__":
    play_game()
import chess
import random

def create_board():
    return chess.Board()

def evaluate_board(board):
    if board.is_checkmate():
        return -9999 if board.turn == chess.WHITE else 9999
    if board.is_game_over():
        return 0
    material = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
    }
    score = 0
    legal_moves = list(board.legal_moves)
    # Material
    for piece_type in material:
        score += len(board.pieces(piece_type, chess.WHITE)) * material[piece_type]
        score -= len(board.pieces(piece_type, chess.BLACK)) * material[piece_type]
    # Center control
    center_squares = [chess.D4, chess.E4, chess.D5, chess.E5]
    for square in center_squares:
        piece = board.piece_at(square)
        if piece:
            if piece.piece_type == chess.PAWN:
                score += 0.5 if piece.color == chess.WHITE else -0.5  # Increased
            elif piece.piece_type == chess.KNIGHT:
                score += 1.2 if piece.color == chess.WHITE else -1.2
    # King safety
    white_king = board.king(chess.WHITE)
    black_king = board.king(chess.BLACK)
    if white_king:
        file = chess.square_file(white_king)
        rank = chess.square_rank(white_king)
        if file in [2, 3, 4, 5] and rank in [2, 3, 4, 5, 6]:
            score -= 2.5
        if not board.castling_rights & (chess.BB_G1 | chess.BB_C1) and rank != 0:
            score -= 2.5
        king_moves = sum(1 for move in legal_moves if move.from_square == white_king)
        if not (board.is_check() or board.attackers(not chess.WHITE, white_king)):
            score -= 2.0 * king_moves
        if board.has_castling_rights(chess.WHITE) and rank == 0 and file in [6, 2]:
            score += 0.5
        if white_king in [chess.E7, chess.D7]:
            score -= 1.0
    if black_king:
        file = chess.square_file(black_king)
        rank = chess.square_rank(black_king)
        if file in [2, 3, 4, 5] and rank in [2, 3, 4, 5, 6]:
            score += 2.5
        if not board.castling_rights & (chess.BB_G8 | chess.BB_C8) and rank != 7:
            score += 2.5
        king_moves = sum(1 for move in legal_moves if move.from_square == black_king)
        if not (board.is_check() or board.attackers(not chess.BLACK, black_king)):
            score += 2.0 * king_moves
        if board.has_castling_rights(chess.BLACK) and rank == 7 and file in [6, 2]:
            score -= 0.5
        if black_king in [chess.E7, chess.D7]:
            score += 1.0
    # Penalize being in check
    if board.is_check():
        score -= 1.0 if board.turn == chess.WHITE else 1.0
    # f7/f2 defense
    if board.piece_at(chess.F7) == chess.Piece(chess.PAWN, chess.BLACK):
        score += 0.3
    if board.piece_at(chess.F6) == chess.Piece(chess.KNIGHT, chess.BLACK):
        score += 0.5
    if board.piece_at(chess.F2) == chess.Piece(chess.PAWN, chess.WHITE):
        score -= 0.3
    if board.piece_at(chess.F3) == chess.Piece(chess.KNIGHT, chess.WHITE):
        score -= 0.5
    # Capture bonus and undefended pieces
    for move in legal_moves:
        if board.is_capture(move):
            captured_piece = board.piece_at(move.to_square)
            if captured_piece:
                defenders = board.attackers(captured_piece.color, move.to_square)
                score += material[captured_piece.piece_type] * 0.8 if board.turn == chess.WHITE else -material[captured_piece.piece_type] * 0.8
                if board.attackers(not board.turn, move.to_square):
                    score += 0.5 if board.turn == chess.WHITE else -0.5
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            attackers = board.attackers(not piece.color, square)
            defenders = board.attackers(piece.color, square)
            if attackers and not defenders:
                score -= material[piece.piece_type] * 0.7 if piece.color == chess.WHITE else material[piece.piece_type] * 0.7
    # Rook activity
    for square in board.pieces(chess.ROOK, chess.WHITE):
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        file_squares = [chess.square(file, r) for r in range(8)]
        if not any(board.piece_at(s) and board.piece_at(s).piece_type == chess.PAWN for s in file_squares):
            score += 0.3
        if rank == 6:
            score += 0.3
    for square in board.pieces(chess.ROOK, chess.BLACK):
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        file_squares = [chess.square(file, r) for r in range(8)]
        if not any(board.piece_at(s) and board.piece_at(s).piece_type == chess.PAWN for s in file_squares):
            score -= 0.3
        if rank == 1:
            score -= 0.3
    # Piece development
    move_number = board.fullmove_number
    if move_number <= 10:
        for piece_type in [chess.KNIGHT, chess.BISHOP]:
            for square in board.pieces(piece_type, chess.WHITE):
                rank = chess.square_rank(square)
                if rank > 1:
                    score += 0.7  # Increased
                elif rank == 1:
                    score -= 0.4
            for square in board.pieces(piece_type, chess.BLACK):
                rank = chess.square_rank(square)
                if rank < 6:
                    score -= 0.7
                elif rank == 6:
                    score += 0.4
        if board.piece_at(chess.D5) == chess.Piece(chess.PAWN, chess.BLACK):
            score -= 0.5  # Increased
        if board.piece_at(chess.E5) == chess.Piece(chess.PAWN, chess.BLACK):
            score -= 0.5
        if board.piece_at(chess.D4) == chess.Piece(chess.PAWN, chess.WHITE):
            score += 0.5
        if board.piece_at(chess.E4) == chess.Piece(chess.PAWN, chess.WHITE):
            score += 0.5
    else:
        for piece_type in [chess.KNIGHT, chess.BISHOP]:
            for square in board.pieces(piece_type, chess.WHITE):
                rank = chess.square_rank(square)
                if rank > 1:
                    score += 0.2
            for square in board.pieces(piece_type, chess.BLACK):
                rank = chess.square_rank(square)
                if rank < 6:
                    score -= 0.2
    # Threat detection (pawn promotion)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.piece_type == chess.PAWN:
            rank = chess.square_rank(square)
            if piece.color == chess.WHITE and rank >= 5:
                score += 0.5 * (rank - 5)
            if piece.color == chess.BLACK and rank <= 2:
                score -= 0.5 * (3 - rank)
    return score

def alpha_beta(board, depth, alpha, beta, maximizing):
    if depth == 0 or board.is_game_over():
        return evaluate_board(board)
    if maximizing:
        value = -float('inf')
        for move in board.legal_moves:
            board.push(move)
            value = max(value, alpha_beta(board, depth - 1, alpha, beta, False))
            board.pop()
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    else:
        value = float('inf')
        for move in board.legal_moves:
            board.push(move)
            value = min(value, alpha_beta(board, depth - 1, alpha, beta, True))
            board.pop()
            beta = min(beta, value)
            if beta <= alpha:
                break
        return value

def get_best_move(board, depth=4):
    best_value = -float('inf')
    best_move = None
    for move in board.legal_moves:
        board.push(move)
        value = alpha_beta(board, depth - 1, -float('inf'), float('inf'), False)
        board.pop()
        if value > best_value:
            best_value = value
            best_move = move
    return best_move

def get_random_move(board):  # Keep for testing
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
                    if len(move_str) == 5 and move_str[4] in 'qrbn':
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
            move = get_best_move(board, depth=4)
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
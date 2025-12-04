import chess
import json
import time
from typing import List, Tuple, Optional, Dict

# Piece values (from macro.cpp)
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}

# Piece-Square Tables (tuned for central control)
PST_PAWN = [
    0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5,  5, 15, 25, 25, 15,  5,  5,
    0,  0, 10, 20, 20, 10,  0,  0,
    5, -5,  0,  0,  0,  0, -5,  5,
    5, 10, 10,-10,-10, 10, 10,  5,
    0,  0,  0,  0,  0,  0,  0,  0
]

PST_KNIGHT = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -30, 10, 15, 20, 20, 15, 10,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50
]

PST_BISHOP = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  0, 10, 15, 15, 10,  0,-10,
    -10,  0, 10, 15, 15, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20
]

PST_ROOK = [
    0,  0,  0,  5,  5,  0,  0,  0,
    5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    0,  0,  0, 10, 10,  0,  0,  0
]

PST_QUEEN = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  5,  0,  0,  5,  0,-10,
    -10,  5,  5,  5,  5,  5,  5,-10,
    -5,  0,  5, 10, 10,  5,  0, -5,
    -5,  0,  5, 10, 10,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  5,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20
]

PST_KING_MID = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
    20, 20,  0,  0,  0,  0, 20, 20,
    20, 30, 10,  0,  0, 10, 30, 20
]

PST_KING_END = [
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50
]

PST = {
    chess.PAWN: PST_PAWN,
    chess.KNIGHT: PST_KNIGHT,
    chess.BISHOP: PST_BISHOP,
    chess.ROOK: PST_ROOK,
    chess.QUEEN: PST_QUEEN,
    chess.KING: PST_KING_MID
}

class ChessAnalyzer:
    def __init__(self, eval_file: str = "b:\\chess_engine_4\\stockfish_evals.json"):
        self.board = chess.Board()
        self.eval_file = eval_file
        self.stockfish_evals = self.load_stockfish_evals()
        self.history = {}  # Track move history to penalize repetitions

    def validate_fen(self, fen: str) -> bool:
        """Validate a FEN string to ensure exactly one king per side."""
        try:
            board = chess.Board(fen)
            white_kings = sum(1 for square in chess.SQUARES if board.piece_at(square) == chess.Piece(chess.KING, chess.WHITE))
            black_kings = sum(1 for square in chess.SQUARES if board.piece_at(square) == chess.Piece(chess.KING, chess.BLACK))
            return white_kings == 1 and black_kings == 1
        except ValueError:
            return False

    def load_stockfish_evals(self) -> Dict[str, float]:
        """Load Stockfish evaluations from JSON."""
        evals = {}
        try:
            with open(self.eval_file, 'r') as f:
                data = json.load(f)
                for entry in data:
                    fen = entry.get('fen')
                    cp = entry.get('cp', 0)
                    if fen and self.validate_fen(fen):
                        evals[fen] = cp / 100.0
                    else:
                        print(f"Skipping invalid FEN: {fen}")
        except FileNotFoundError:
            print(f"Warning: {self.eval_file} not found. Using default evaluations.")
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in {self.eval_file}. Using default evaluations.")
        return evals

    def evaluate_position(self) -> float:
        """Evaluate the board position (positive = White advantage)."""
        fen = self.board.fen()
        if not self.validate_fen(fen):
            print(f"Error: Invalid FEN: {fen}")
            return 0.0  # Neutral score for invalid position

        if self.board.is_checkmate():
            return -float('inf') if self.board.turn == chess.WHITE else float('inf')
        if self.board.is_stalemate() or self.board.is_insufficient_material():
            return 0.0

        if fen in self.stockfish_evals:
            return self.stockfish_evals[fen]

        score = 0.0
        total_material = 0
        mobility_score = 0
        center_control = 0

        # Material and PST
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                value = PIECE_VALUES[piece.piece_type]
                pst_table = PST[piece.piece_type]
                pst_index = square if piece.color == chess.WHITE else chess.square_mirror(square)
                pst_value = pst_table[pst_index]
                multiplier = 1 if piece.color == chess.WHITE else -1
                score += multiplier * (value + pst_value)
                if piece.piece_type != chess.KING:
                    total_material += value

                # Mobility (from evaluation.cpp)
                if piece.piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
                    legal_moves = len([m for m in self.board.generate_legal_moves(from_mask=chess.BB_SQUARES[square])])
                    mobility_score += multiplier * legal_moves * 15

                # Center control
                if square in [chess.D4, chess.D5, chess.E4, chess.E5]:
                    if piece.piece_type == chess.PAWN:
                        center_control += multiplier * 30
                    elif piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                        center_control += multiplier * 20

        # Game phase
        phase = 1.0 if total_material < 2000 else 0.0
        if phase > 0:
            for square in chess.SQUARES:
                piece = self.board.piece_at(square)
                if piece and piece.piece_type == chess.KING:
                    pst_index = square if piece.color == chess.WHITE else chess.square_mirror(square)
                    score += (1 if piece.color == chess.WHITE else -1) * PST_KING_END[pst_index]

        # King safety (from board.cpp)
        king_square = self.board.king(self.board.turn)
        if king_square is None:
            print(f"Error: No king found for {self.board.turn} in FEN: {self.board.fen()}")
            return 0.0  # Neutral score for invalid position
        opponent = not self.board.turn
        attacks = self.board.attackers(opponent, king_square)
        if attacks:
            score -= 100 * len(attacks)

        # Pawn structure (from evaluation.cpp)
        for file in range(8):
            white_pawns = len([s for s in chess.SquareSet(chess.BB_FILES[file]) if self.board.piece_at(s) and self.board.piece_at(s).piece_type == chess.PAWN and self.board.piece_at(s).color == chess.WHITE])
            black_pawns = len([s for s in chess.SquareSet(chess.BB_FILES[file]) if self.board.piece_at(s) and self.board.piece_at(s).piece_type == chess.PAWN and self.board.piece_at(s).color == chess.BLACK])
            if white_pawns > 1:
                score -= 75 * (white_pawns - 1)
            if black_pawns > 1:
                score += 75 * (black_pawns - 1)

        # Add mobility and center control
        score += mobility_score + center_control

        return score / 100.0

    def is_square_safe(self, square: chess.Square, color: chess.Color) -> bool:
        """Check if a square is safe from opponent attacks."""
        return not bool(self.board.attackers(not color, square))

    def get_capture_value(self, move: chess.Move) -> Tuple[int, int]:
        """Return (captured value, capturer value) for MVV-LVA."""
        capturer = self.board.piece_at(move.from_square)
        capturer_val = PIECE_VALUES.get(capturer.piece_type, 0) if capturer else 0
        if move in self.board.legal_moves and self.board.is_capture(move):
            captured = self.board.piece_at(move.to_square)
            captured_val = PIECE_VALUES.get(captured.piece_type, 0) if captured else 0
            return (captured_val, capturer_val)
        return (0, capturer_val)

    def can_castle(self, move: chess.Move) -> bool:
        """Check if castling is safe and beneficial."""
        if not self.board.is_castling(move):
            return False
        king_square = self.board.king(self.board.turn)
        if king_square is None:
            return False
        rook_square = move.to_square
        path_squares = chess.SquareSet.between(king_square, rook_square) | {king_square, rook_square}
        for square in path_squares:
            if self.board.attackers(not self.board.turn, square):
                return False
        self.board.push(move)
        safe = self.is_square_safe(self.board.king(self.board.turn), self.board.turn)
        self.board.pop()
        return safe

    def is_king_move_safe(self, move: chess.Move) -> bool:
        """Allow king moves only to escape check or if safe."""
        if self.board.piece_at(move.from_square).piece_type != chess.KING:
            return True
        if self.board.is_check():
            self.board.push(move)
            safe = not self.board.is_check()
            self.board.pop()
            return safe
        return self.is_square_safe(move.to_square, self.board.turn)

    def get_best_move(self, depth: int = 3, time_limit: float = 2.0) -> Tuple[Optional[chess.Move], float]:
        """Find the best move using alpha-beta with quiescence search."""
        def quiescence(board: chess.Board, alpha: float, beta: float, depth_limit: int = 6) -> float:
            if depth_limit <= 0:
                return self.evaluate_position()
            stand_pat = self.evaluate_position()
            if stand_pat == 0.0 and board.king(board.turn) is None:
                return stand_pat  # Invalid position
            if board.turn == chess.WHITE:
                if stand_pat >= beta:
                    return beta
                alpha = max(alpha, stand_pat)
                moves = []
                for move in board.legal_moves:
                    if board.is_capture(move):
                        captured_val, capturer_val = self.get_capture_value(move)
                        moves.append((move, captured_val - capturer_val / 100.0))
                moves.sort(key=lambda x: x[1], reverse=True)
                for move, _ in moves:
                    board.push(move)
                    score = quiescence(board, alpha, beta, depth_limit - 1)
                    board.pop()
                    alpha = max(alpha, score)
                    if alpha >= beta:
                        return beta
                return alpha
            else:
                if stand_pat <= alpha:
                    return alpha
                beta = min(beta, stand_pat)
                moves = []
                for move in board.legal_moves:
                    if board.is_capture(move):
                        captured_val, capturer_val = self.get_capture_value(move)
                        moves.append((move, captured_val - capturer_val / 100.0))
                moves.sort(key=lambda x: x[1], reverse=True)
                for move, _ in moves:
                    board.push(move)
                    score = quiescence(board, alpha, beta, depth_limit - 1)
                    board.pop()
                    beta = min(beta, score)
                    if beta <= alpha:
                        return alpha
                return beta

        def alpha_beta(board: chess.Board, depth: int, alpha: float, beta: float, start_time: float) -> Tuple[float, Optional[chess.Move]]:
            if time.time() - start_time > time_limit:
                return self.evaluate_position(), None
            if depth == 0 or board.is_game_over():
                return quiescence(board, alpha, beta), None

            legal_moves = list(board.legal_moves)
            if not legal_moves:
                return self.evaluate_position(), None

            move_scores = []
            for move in legal_moves:
                score = 0
                if board.is_capture(move):
                    captured_val, capturer_val = self.get_capture_value(move)
                    score += 1500 + captured_val - capturer_val / 100.0
                elif board.is_castling(move) and self.can_castle(move):
                    score += 1000
                elif board.gives_check(move):
                    score += 400
                else:
                    piece = board.piece_at(move.from_square)
                    if piece:
                        score += PST[piece.piece_type][move.to_square]
                # History heuristic
                move_key = (board.fen(), str(move))
                score -= self.history.get(move_key, 0) * 150
                move_scores.append((move, score))

            move_scores.sort(key=lambda x: x[1], reverse=True)

            best_score = -float('inf') if board.turn == chess.WHITE else float('inf')
            best_move = None

            for move, _ in move_scores:
                if not self.is_king_move_safe(move):
                    continue
                board.push(move)
                score, _ = alpha_beta(board, depth - 1, alpha, beta, start_time)
                board.pop()
                move_key = (board.fen(), str(move))
                self.history[move_key] = self.history.get(move_key, 0) + 1
                if board.turn == chess.WHITE:
                    if score > best_score:
                        best_score = score
                        best_move = move
                    alpha = max(alpha, best_score)
                else:
                    if score < best_score:
                        best_score = score
                        best_move = move
                    beta = min(beta, score)
                if beta <= alpha:
                    break

            return best_score, best_move

        start_time = time.time()
        score, move = alpha_beta(self.board, depth, -float('inf'), float('inf'), start_time)
        return move, score

    def classify_move(self, move: chess.Move, best_move: chess.Move, best_score: float, played_score: float) -> Tuple[str, str]:
        """Classify a move as blunder, mistake, good, or best."""
        eval_diff = abs(best_score - played_score)
        if move == best_move or eval_diff < 0.1:
            return "Best", "This is the optimal move."
        elif eval_diff < 0.5:
            return "Good", "A solid move, close to the best."
        elif eval_diff < 2.0:
            return "Mistake", f"Suboptimal move, losing {eval_diff:.2f} pawns."
        else:
            return "Blunder", f"Significant error, losing {eval_diff:.2f} pawns."

    def analyze_fen_sequence(self, fen_sequence: List[Tuple[str, str]]) -> List[Dict]:
        """Analyze a sequence of FEN positions and moves."""
        analysis = []
        move_number = 0

        for fen, move_san in fen_sequence:
            move_number += 1
            if not self.validate_fen(fen):
                analysis.append({
                    "move_number": move_number // 2 + 1,
                    "player": "White" if move_number % 2 == 1 else "Black",
                    "move_played": move_san,
                    "best_move": "None",
                    "eval_diff": 0.0,
                    "classification": "Invalid",
                    "explanation": f"Invalid FEN: {fen}"
                })
                continue

            try:
                self.board.set_fen(fen)
            except ValueError as e:
                analysis.append({
                    "move_number": move_number // 2 + 1,
                    "player": "White" if move_number % 2 == 1 else "Black",
                    "move_played": move_san,
                    "best_move": "None",
                    "eval_diff": 0.0,
                    "classification": "Invalid",
                    "explanation": f"Invalid FEN: {str(e)}"
                })
                continue

            player = "White" if self.board.turn == chess.WHITE else "Black"
            try:
                played_move = self.board.parse_san(move_san)
                if played_move not in self.board.legal_moves:
                    analysis.append({
                        "move_number": move_number // 2 + 1,
                        "player": player,
                        "move_played": move_san,
                        "best_move": "None",
                        "eval_diff": 0.0,
                        "classification": "Invalid",
                        "explanation": f"Illegal move: {move_san} in FEN: {fen}"
                    })
                    continue
            except ValueError:
                analysis.append({
                    "move_number": move_number // 2 + 1,
                    "player": player,
                    "move_played": move_san,
                    "best_move": "None",
                    "eval_diff": 0.0,
                    "classification": "Invalid",
                    "explanation": f"Invalid move format: {move_san}"
                })
                continue

            best_move, best_score = self.get_best_move(depth=3, time_limit=2.0)
            self.board.push(played_move)
            played_score = self.evaluate_position()
            self.board.pop()
            classification, explanation = self.classify_move(played_move, best_move, best_score, played_score)

            analysis.append({
                "move_number": move_number // 2 + 1,
                "player": player,
                "move_played": move_san,
                "best_move": self.board.san(best_move) if best_move else "None",
                "eval_diff": abs(best_score - played_score),
                "classification": classification,
                "explanation": explanation
            })

        return analysis

    def predict_next_move(self, fen: str) -> Tuple[str, float]:
        """Predict the best move for a given FEN position."""
        if not self.validate_fen(fen):
            print(f"Error: Invalid FEN: {fen}")
            return "None", 0.0
        try:
            self.board.set_fen(fen)
        except ValueError as e:
            print(f"Error: Invalid FEN: {fen}")
            return "None", 0.0
        move, score = self.get_best_move(depth=3, time_limit=2.0)
        return self.board.san(move) if move else "None", score

    def test_random_fens(self):
        """Test the engine with random FEN positions."""
        test_fens = [
            ("rnbqkbnr/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKBNR w KQkq - 0 1", "Nf3"),
            ("rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 0 1", "e5"),
            ("rnbqkb1r/ppppppNp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 0 1", "Bxg7"),
            ("r1bqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1", "d4")
        ]

        print("Testing random FEN positions:")
        for fen, expected_move in test_fens:
            move, score = self.predict_next_move(fen)
            print(f"FEN: {fen}")
            print(f"Predicted move: {move}, Eval: {score:.2f}")
            print(f"Expected move: {expected_move}")
            print()

    def play_interactive(self):
        """Interactive mode for testing."""
        print(self.board)
        move_log = []
        while not self.board.is_game_over():
            if self.board.turn == chess.WHITE:
                move_str = input("Your move (e.g., e4, Qxf7, d7d8q, or 'quit' to exit): ")
                if move_str.lower() == 'quit':
                    break
                try:
                    move = self.board.parse_san(move_str)
                    if move in self.board.legal_moves:
                        move_san = move_str
                        self.board.push(move)
                        move_log.append(move_san)
                        print(f"FEN after {move_san}: {self.board.fen()}")
                    else:
                        print("Invalid move.")
                        continue
                except ValueError:
                    print("Invalid move format.")
                    continue
            else:
                print("Engine thinking...")
                legal_moves = list(self.board.legal_moves)
                if not legal_moves:
                    print("No legal moves.")
                    break
                move, score = self.get_best_move(depth=3, time_limit=2.0)
                if move:
                    move_san = self.board.san(move)
                    print(f"Engine move: {move_san} (eval: {score:.2f})")
                    self.board.push(move)
                    move_log.append(move_san)
                    print(f"FEN after {move_san}: {self.board.fen()}")
                else:
                    print("No legal moves.")
                    break
            print(self.board)
            print("\n")
        print(f"Game over: {self.board.result()}")
        print("Move log:", " ".join(move_log))

if __name__ == "__main__":
    # Initialize analyzer
    analyzer = ChessAnalyzer(eval_file="b:\\chess_engine_4\\stockfish_evals.json")

    # Corrected FEN sequence
    fen_sequence = [
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "e4"),
        ("rnbqkbnr/pppp1ppp/5n2/4p3/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", "Nf6"),
        ("rnbqkb1r/pppp1ppp/5n2/4p3/4P3/2B5/PPPP1PPP/RNBQK1NR w KQkq - 1 2", "Bc4"),
        ("rnbqkb1r/pppp1ppp/5n2/4p3/4P3/2B5/PPPP1PPP/RNBQK1NR b KQkq - 1 2", "Nxe4"),
        ("rnbqkb1r/pppp1ppp/5n2/8/8/2B5/PPPP1PPP/RNBQK1NR w KQkq - 0 3", "d3"),
        ("rnbqkb1r/pppp1ppp/5n2/8/8/2BP4/PPP2PPP/RNBQK1NR b KQkq - 0 3", "Nd6"),
        ("rnbqkb1r/pppp1ppp/3n4/8/8/2BP4/PPP2PPP/RNBQK1NR w KQkq - 1 4", "Nf3"),
        ("rnbqkb1r/pppp1ppp/3n4/8/8/2BP1N2/PPP2PPP/RNBQK2R b KQkq - 1 4", "Nf5"),
        ("rnbqkb1r/pppp1ppp/5n2/8/8/2BP1N2/PPP2PPP/RNBQK2R w KQkq - 2 5", "O-O"),
        ("rnbqkb1r/pppp1ppp/5n2/8/8/2BP1N2/PPP2PPP/RNBQ1RK1 b KQkq - 2 5", "Nc6"),
        ("rnbqkb1r/pppp1ppp/3n4/8/8/2BP1N2/PPP2PPP/RNBQ1RK1 w KQkq - 3 6", "Qe2"),
        ("rnbqkb1r/pppp1ppp/3n4/8/8/2BP1N2/PPP1QPPP/RNB2RK1 b KQkq - 3 6", "Nb4"),
        ("rnbqkb1r/pppp1ppp/5n2/1n6/8/2BP1N2/PPP1QPPP/RNB2RK1 w KQkq - 4 7", "Bg5"),
        ("rnbqkb1r/pppp1ppp/5n2/1n6/6B1/2BP1N2/PPP1QPPP/RN3RK1 b KQkq - 4 7", "Nc6"),
        ("rnbqkb1r/pppp1ppp/3n4/8/6B1/2BP1N2/PPP1QPPP/RN3RK1 w KQkq - 5 8", "Re1"),
        ("rnbqkb1r/pppp1ppp/3n4/8/6B1/2BP1N2/PPP1QPPP/RN2R1K1 b KQkq - 5 8", "Nb4"),
        ("rnbqkb1r/pppp1ppp/5n2/1n6/6B1/2BP1N2/PPP1QPPP/RN2R1K1 w KQkq - 6 9", "Nc3"),
        ("rnbqkb1r/pppp1ppp/5n2/1n6/6B1/2BPNN2/PPP1QPPP/R3R1K1 b KQkq - 6 9", "Rg8"),
        ("rnbqkb1r/pppp1ppp/5n2/8/6B1/2BPNN2/PPP1QPPP/R3R1K1 w KQkq - 7 10", "a3"),
        ("rnbqkb1r/pppp1ppp/5n2/8/6B1/P1B1NN2/1PP1QPPP/R3R1K1 b KQkq - 7 10", "Nc6"),
        ("rnbqkb1r/pppp1ppp/3n4/8/6B1/P1B1NN2/1PP1QPPP/R3R1K1 w KQkq - 8 11", "Nd5"),
        ("rnbqkb1r/pppp1ppp/3n4/3N4/6B1/P1B2N2/1PP1QPPP/R3R1K1 b KQkq - 8 11", "Nd6"),
        ("rnbqkb1r/pppp1ppp/5n2/3N4/6B1/P1B2N2/1PP1QPPP/R3R1K1 w KQkq - 9 12", "Nh4"),
        ("rnbqkb1r/pppp1ppp/5n2/3N4/6BN/P1B5/1PP1QPPP/R3R1K1 b KQkq - 9 12", "Nd4"),
        ("rnbqkb1r/pppp1ppp/5n2/3Nn3/6B1/P1B5/1PP1QPPP/R3R1K1 w KQkq - 10 13", "Qe3"),
        ("rnbqkb1r/pppp1ppp/5n2/3Nn3/6B1/P1B1Q3/1PP2PPP/R3R1K1 b KQkq - 10 13", "Nxc2"),
        ("rnbqkb1r/pppp1ppp/5n2/3N4/6B1/P1B1Q3/1Pn2PPP/R3R1K1 w KQkq - 11 14", "Qe5"),
        ("rnbqkb1r/pppp1ppp/5n2/3NQ3/6B1/P1B5/1Pn2PPP/R3R1K1 b KQkq - 11 14", "Rb8"),
        ("rnbqrkb1r/pppp1ppp/5n2/3NQ3/6B1/P1B5/1Pn2PPP/R3R1K1 w KQkq - 12 15", "Nf5"),
        ("rnbqrkb1r/pppp1ppp/5n2/3NQ1N1/6B1/P1B5/1Pn2PPP/R3R1K1 b KQkq - 12 inspection15", "Rh8"),
        ("rnbqrkb1r/pppp1pNp/5n2/3NQ3/6B1/P1B5/1Pn2PPP/R3R1K1 b KQkq - 13 16", "Bxg7")
    ]
    analysis = analyzer.analyze_fen_sequence(fen_sequence)
    for move_analysis in analysis:
        print(f"Move {move_analysis['move_number']} ({move_analysis['player']}): {move_analysis['move_played']}")
        print(f"Best move: {move_analysis['best_move']}, Eval diff: {move_analysis['eval_diff']:.2f}")
        print(f"Classification: {move_analysis['classification']}, {move_analysis['explanation']}")
        print()

    # Test random FENs
    analyzer.test_random_fens()

    # Interactive mode
    analyzer.board = chess.Board()
    analyzer.play_interactive()
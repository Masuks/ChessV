import json
import chess
import time
import sys
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

class TranspositionTable:
    """A simple transposition table for storing position evaluations."""
    def __init__(self):
        self.table: Dict[str, float] = {}

    def store(self, fen: str, eval_score: float) -> None:
        """Store an evaluation score for a FEN position."""
        self.table[fen] = eval_score

    def lookup(self, fen: str) -> Optional[float]:
        """Retrieve an evaluation score for a FEN position, if available."""
        return self.table.get(fen)

    def size(self) -> int:
        """Return the number of entries in the table."""
        return len(self.table)

class ChessAnalyzer:
    def __init__(self, eval_file: str = "cleaned_combined_data.json"):
        self.board = chess.Board()
        self.eval_file = eval_file
        self.transposition_table = TranspositionTable()
        self.stockfish_evals = self.load_stockfish_evals()
        self.history = {}  # Track move history to penalize repetitions
        self.killer_moves = {}  # Store two killer moves by depth
        self.move_history_heuristic = {}  # For history heuristic
        self.depth = 3  # Default search depth
        self.time_limit = 2.0  # Default time limit per move

    def validate_fen(self, fen: str) -> bool:
        """Validate a FEN string with relaxed checks."""
        try:
            board = chess.Board(fen)
            white_kings = sum(1 for square in chess.SQUARES if board.piece_at(square) == chess.Piece(chess.KING, chess.WHITE))
            black_kings = sum(1 for square in chess.SQUARES if board.piece_at(square) == chess.Piece(chess.KING, chess.BLACK))
            return white_kings == 1 and black_kings == 1 and board.is_valid()
        except ValueError as e:
            print(f"FEN validation error: {fen}, {str(e)}", file=sys.stderr)
            return False

    def load_stockfish_evals(self) -> Dict[str, float]:
        """Load Stockfish evaluations from JSON, skipping invalid entries."""
        evals = {}
        try:
            with open(self.eval_file, 'r') as f:
                data = json.load(f)
                for entry in data:
                    fen = entry.get('fen')
                    cp = entry.get('cp', entry.get('eval', 0))
                    if not isinstance(cp, (int, float)):
                        print(f"Skipping entry with invalid evaluation: {entry}", file=sys.stderr)
                        continue
                    if fen and self.validate_fen(fen):
                        eval_score = cp / 100.0
                        evals[fen] = eval_score
                        self.transposition_table.store(fen, eval_score)
                    else:
                        print(f"Skipping invalid FEN: {fen}", file=sys.stderr)
                print(f"Loaded {len(evals)} valid positions into transposition table.", file=sys.stderr)
        except FileNotFoundError:
            print(f"Warning: {self.eval_file} not found. Using default evaluations.", file=sys.stderr)
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in {self.eval_file}. Using default evaluations.", file=sys.stderr)
        except Exception as e:
            print(f"Unexpected error loading {self.eval_file}: {e}", file=sys.stderr)
        return evals

    def evaluate_position(self) -> float:
        """Evaluate the board position (positive = White advantage)."""
        fen = self.board.fen()
        if not self.validate_fen(fen):
            print(f"Error: Invalid FEN: {fen}", file=sys.stderr)
            return 0.0

        if self.board.is_checkmate():
            return -float('inf') if self.board.turn == chess.WHITE else float('inf')
        if self.board.is_stalemate() or self.board.is_insufficient_material():
            return 0.0

        cached_eval = self.transposition_table.lookup(fen)
        if cached_eval is not None:
            return cached_eval

        score = 0.0
        total_material = 0
        mobility_score = 0
        center_control = 0
        pawn_structure = 0
        check_repetition = 0

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

                # Mobility
                if piece.piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
                    legal_moves = len([m for m in self.board.generate_legal_moves(from_mask=chess.BB_SQUARES[square])])
                    mobility_score += multiplier * legal_moves * 10

                # Center control
                if square in [chess.D4, chess.D5, chess.E4, chess.E5]:
                    if piece.piece_type == chess.PAWN:
                        center_control += multiplier * 20
                    elif piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                        center_control += multiplier * 15

        # Phase calculation (0 = endgame, 1 = middlegame)
        phase = min(1.0, total_material / 4000)
        if phase < 1.0:
            for square in chess.SQUARES:
                piece = self.board.piece_at(square)
                if piece and piece.piece_type == chess.KING:
                    pst_index = square if piece.color == chess.WHITE else chess.square_mirror(square)
                    score += (1 if piece.color == chess.WHITE else -1) * PST_KING_END[pst_index] * (1 - phase)

        # Pawn structure
        for color in [chess.WHITE, chess.BLACK]:
            multiplier = 1 if color == chess.WHITE else -1
            pawns = [s for s in chess.SQUARES if self.board.piece_at(s) and self.board.piece_at(s).piece_type == chess.PAWN and self.board.piece_at(s).color == color]
            pawn_files = [chess.square_file(s) for s in pawns]
            for file in range(8):
                file_pawns = pawn_files.count(file)
                if file_pawns > 1:
                    pawn_structure -= multiplier * 50 * (file_pawns - 1)  # Doubled pawns
                if file_pawns == 0:
                    neighbors = [pawn_files.count(f) for f in [file-1, file+1] if 0 <= f <= 7]
                    if not any(neighbors):
                        pawn_structure -= multiplier * 30  # Isolated pawn penalty

            # Passed pawns
            for pawn in pawns:
                rank = chess.square_rank(pawn)
                file = chess.square_file(pawn)
                is_passed = True
                for f in [file-1, file, file+1]:
                    if 0 <= f <= 7:
                        for r in range(rank + (1 if color == chess.WHITE else -1), (8 if color == chess.WHITE else -1), (1 if color == chess.WHITE else -1)):
                            if self.board.piece_at(chess.square(f, r)) and self.board.piece_at(chess.square(f, r)).piece_type == chess.PAWN and self.board.piece_at(chess.square(f, r)).color != color:
                                is_passed = False
                                break
                if is_passed:
                    pawn_structure += multiplier * (20 + 10 * (rank if color == chess.WHITE else 7 - rank))

        # King safety
        for color in [chess.WHITE, chess.BLACK]:
            king_square = self.board.king(color)
            if king_square is None:
                continue
            multiplier = 1 if color == chess.WHITE else -1
            attacks = self.board.attackers(not color, king_square)
            if attacks:
                score -= multiplier * 80 * len(attacks)
            rank = chess.square_rank(king_square)
            file = chess.square_file(king_square)
            shield_squares = []
            if color == chess.WHITE and rank < 7:
                shield_squares = [chess.square(file + i, rank + 1) for i in [-1, 0, 1] if 0 <= file + i <= 7]
            elif color == chess.BLACK and rank > 0:
                shield_squares = [chess.square(file + i, rank - 1) for i in [-1, 0, 1] if 0 <= file + i <= 7]
            for square in shield_squares:
                piece = self.board.piece_at(square)
                if piece and piece.piece_type == chess.PAWN and piece.color == color:
                    score += multiplier * 40

        # Repetition penalty
        if self.board.is_check():
            move_key = (fen, str(self.board.peek()))
            check_repetition = self.history.get(move_key, 0)
            score -= check_repetition * 150

        score += mobility_score + center_control + pawn_structure
        return score / 100.0

    def get_centipawn_score(self, score: float, turn: chess.Color) -> str:
        """Convert evaluation to UCI centipawn or mate score."""
        if score == float('inf'):
            return f"mate {1 if turn == chess.BLACK else -1}"
        if score == -float('inf'):
            return f"mate {1 if turn == chess.WHITE else -1}"
        if abs(score) > 1000:  # Treat as mate
            mate_distance = int(10000 / abs(score))  # Rough estimate
            return f"mate {mate_distance if turn == chess.BLACK else -mate_distance}"
        return f"cp {int(score * 100)}"

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
        """Find the best move using alpha-beta with quiescence search and history heuristics."""
        def quiescence(board: chess.Board, alpha: float, beta: float, depth_limit: int = 6) -> float:
            if depth_limit <= 0:
                return self.evaluate_position()
            stand_pat = self.evaluate_position()
            if stand_pat == 0.0 and board.king(board.turn) is None:
                return stand_pat
            if board.turn == chess.WHITE:
                if stand_pat >= beta:
                    return beta
                alpha = max(alpha, stand_pat)
                moves = []
                for move in board.legal_moves:
                    if board.is_capture(move) or move.promotion:
                        captured_val, capturer_val = self.get_capture_value(move)
                        score = captured_val - capturer_val / 100.0
                        if move.promotion == chess.QUEEN:
                            score += 900
                        moves.append((move, score))
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
                    if board.is_capture(move) or move.promotion:
                        captured_val, capturer_val = self.get_capture_value(move)
                        score = captured_val - capturer_val / 100.0
                        if move.promotion == chess.QUEEN:
                            score += 900
                        moves.append((move, score))
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
            killers = self.killer_moves.get(depth, [])
            for move in legal_moves:
                score = 0
                if move in killers:
                    score += 2000
                if board.is_capture(move):
                    captured_val, capturer_val = self.get_capture_value(move)
                    score += 1500 + captured_val - capturer_val / 100.0
                if move.promotion == chess.QUEEN:
                    score += 1000
                if board.is_castling(move) and self.can_castle(move):
                    score += 900
                if board.gives_check(move):
                    score += 600
                piece = board.piece_at(move.from_square)
                if piece:
                    score += PST[piece.piece_type][move.to_square]
                move_key = (board.fen(), str(move))
                score += self.move_history_heuristic.get(move_key, 0)
                score -= self.history.get(move_key, 0) * 100
                move_scores.append((move, score))

            move_scores.sort(key=lambda x: x[1], reverse=True)

            best_score = -float('inf') if board.turn == chess.WHITE else float('inf')
            best_move = None

            for move, _ in move_scores[:20]:  # Limit to top 20 moves
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
                        self.move_history_heuristic[move_key] = self.move_history_heuristic.get(move_key, 0) + depth * depth
                        if len(killers) < 2:
                            killers.append(move)
                        else:
                            killers[1] = killers[0]
                            killers[0] = move
                        self.killer_moves[depth] = killers
                    alpha = max(alpha, best_score)
                else:
                    if score < best_score:
                        best_score = score
                        best_move = move
                        self.move_history_heuristic[move_key] = self.move_history_heuristic.get(move_key, 0) + depth * depth
                        if len(killers) < 2:
                            killers.append(move)
                        else:
                            killers[1] = killers[0]
                            killers[0] = move
                        self.killer_moves[depth] = killers
                    beta = min(beta, score)
                if beta <= alpha:
                    self.move_history_heuristic[move_key] = self.move_history_heuristic.get(move_key, 0) + depth * depth
                    break

            return best_score, best_move

        # Dynamic depth adjustment
        legal_moves = list(self.board.legal_moves)
        dynamic_depth = depth + 1 if len(legal_moves) < 10 else depth
        start_time = time.time()
        score, move = alpha_beta(self.board, dynamic_depth, -float('inf'), float('inf'), start_time)
        if move:
            print(f"info depth {dynamic_depth} score {self.get_centipawn_score(score, self.board.turn)} pv {move.uci()}")
            print(f"bestmove {move.uci()}")
        else:
            print("info depth 0 score cp 0")
            print("bestmove (none)")
        return move, score

    def classify_move(self, move: chess.Move, best_move: chess.Move, best_score: float, played_score: float) -> Tuple[str, str]:
        """Classify a move as blunder, mistake, good, or best."""
        eval_diff = min(abs(best_score - played_score), 10.0)
        if move == best_move or eval_diff < 0.1:
            return "Best", "This is the optimal move."
        elif eval_diff < 0.4:
            return "Good", "A solid move, close to the best."
        elif eval_diff < 1.5:
            return "Mistake", f"Suboptimal move, losing {int(eval_diff * 100)} centipawns."
        else:
            return "Blunder", f"Significant error, losing {int(eval_diff * 100)} centipawns."

    def analyze_fen_sequence(self, fen_sequence: List[Tuple[str, str]]) -> List[Dict]:
        """Analyze a sequence of FEN positions and moves, returning UCI format."""
        analysis = []
        move_number = 0
        self.board.reset()

        for fen, move_san in fen_sequence:
            move_number += 1
            try:
                self.board.set_fen(fen)
            except ValueError as e:
                analysis.append({
                    "move_number": move_number // 2 + 1,
                    "player": "White" if move_number % 2 == 1 else "Black",
                    "move_played": move_san,
                    "move_played_uci": "none",
                    "best_move": "none",
                    "eval_score": "cp 0",
                    "eval_diff": 0,
                    "classification": "Invalid",
                    "explanation": f"Invalid FEN: {fen}, {str(e)}"
                })
                continue

            if not self.validate_fen(fen):
                analysis.append({
                    "move_number": move_number // 2 + 1,
                    "player": "White" if move_number % 2 == 1 else "Black",
                    "move_played": move_san,
                    "move_played_uci": "none",
                    "best_move": "none",
                    "eval_score": "cp 0",
                    "eval_diff": 0,
                    "classification": "Invalid",
                    "explanation": f"Invalid FEN: {fen}"
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
                        "move_played_uci": "none",
                        "best_move": "none",
                        "eval_score": "cp 0",
                        "eval_diff": 0,
                        "classification": "Invalid",
                        "explanation": f"Illegal move: {move_san} in FEN: {fen}"
                    })
                    continue
            except ValueError:
                analysis.append({
                    "move_number": move_number // 2 + 1,
                    "player": player,
                    "move_played": move_san,
                    "move_played_uci": "none",
                    "best_move": "none",
                    "eval_score": "cp 0",
                    "eval_diff": 0,
                    "classification": "Invalid",
                    "explanation": f"Invalid move format: {move_san}"
                })
                continue

            best_move, best_score = self.get_best_move(depth=self.depth, time_limit=self.time_limit)
            self.board.push(played_move)
            played_score = self.evaluate_position()
            self.board.pop()
            classification, explanation = self.classify_move(played_move, best_move, best_score, played_score)

            analysis.append({
                "move_number": move_number // 2 + 1,
                "player": player,
                "move_played": move_san,
                "move_played_uci": played_move.uci(),
                "best_move": best_move.uci() if best_move else "none",
                "eval_score": self.get_centipawn_score(played_score, self.board.turn),
                "eval_diff": int(abs(best_score - played_score) * 100),
                "classification": classification,
                "explanation": explanation
            })
            self.board.push(played_move)  # Advance board state

        return analysis

    def predict_next_move(self, fen: str) -> Tuple[str, str]:
        """Predict the best move for a given FEN position in UCI format."""
        if not self.validate_fen(fen):
            print(f"Error: Invalid FEN: {fen}", file=sys.stderr)
            return "none", "cp 0"
        try:
            self.board.set_fen(fen)
        except ValueError as e:
            print(f"Error: Invalid FEN: {fen}, {str(e)}", file=sys.stderr)
            return "none", "cp 0"
        move, score = self.get_best_move(depth=self.depth, time_limit=self.time_limit)
        return move.uci() if move else "none", self.get_centipawn_score(score, self.board.turn)

    def test_random_fens(self):
        """Test the engine with random FEN positions, outputting UCI moves."""
        test_fens = [
            ("rnbqkbnr/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKBNR w KQkq - 0 1", "g1f3"),
            ("rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 0 1", "e7e5"),
            ("rnbqkb1r/pppppp1p/5np1/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 1", "c4g8"),
            ("r1bqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1", "d2d4")
        ]

        print("Testing random FEN positions:")
        for fen, expected_move in test_fens:
            move, score = self.predict_next_move(fen)
            print(f"FEN: {fen}")
            print(f"Predicted move: {move}, Score: {score}")
            print(f"Expected move: {expected_move}")
            print()

    def play_interactive(self):
        """Interactive mode accepting UCI moves and outputting UCI responses."""
        print(self.board)
        move_log = []
        while not self.board.is_game_over():
            if self.board.turn == chess.WHITE:
                move_str = input("Your move (UCI, e.g., e2e4, g1f3, or 'quit' to exit): ")
                if move_str.lower() == 'quit':
                    break
                try:
                    move = chess.Move.from_uci(move_str)
                    if move in self.board.legal_moves:
                        move_san = self.board.san(move)  # For logging
                        self.board.push(move)
                        move_log.append(move.uci())
                        print(f"FEN after {move.uci()}: {self.board.fen()}")
                    else:
                        print("Invalid move.")
                        continue
                except (ValueError, AttributeError):
                    print("Invalid move format. Use UCI (e.g., e2e4).")
                    continue
            else:
                print("Engine thinking...")
                legal_moves = list(self.board.legal_moves)
                if not legal_moves:
                    print("No legal moves.")
                    break
                move, score = self.get_best_move(depth=self.depth, time_limit=self.time_limit)
                if move:
                    move_san = self.board.san(move)  # For logging
                    self.board.push(move)
                    move_log.append(move.uci())
                    print(f"FEN after {move.uci()}: {self.board.fen()}")
                else:
                    print("No legal moves.")
                    break
            print(self.board)
            print("\n")
        print(f"Game over: {self.board.result()}")
        print("Move log (UCI):", " ".join(move_log))

    def uci_loop(self):
        """Main UCI loop for handling UCI commands."""
        print("id name ChessAnalyzer")
        print("id author xAI")
        print("uciok")

        while True:
            try:
                command = input().strip()
                if command == "uci":
                    print("id name ChessAnalyzer")
                    print("id author xAI")
                    print("uciok")
                elif command == "isready":
                    print("readyok")
                elif command == "quit":
                    break
                elif command.startswith("position"):
                    parts = command.split()
                    if "fen" in parts:
                        fen_index = parts.index("fen") + 1
                        fen = " ".join(parts[fen_index:fen_index+6])
                        self.board.set_fen(fen)
                        if "moves" in parts:
                            moves_index = parts.index("moves") + 1
                            for move_str in parts[moves_index:]:
                                move = chess.Move.from_uci(move_str)
                                if move in self.board.legal_moves:
                                    self.board.push(move)
                    elif parts[1] == "startpos":
                        self.board.reset()
                        if "moves" in parts:
                            moves_index = parts.index("moves") + 1
                            for move_str in parts[moves_index:]:
                                move = chess.Move.from_uci(move_str)
                                if move in self.board.legal_moves:
                                    self.board.push(move)
                elif command.startswith("go"):
                    move, score = self.get_best_move(depth=self.depth, time_limit=self.time_limit)
                elif command == "stop":
                    pass  # Handled by get_best_move time limit
            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)

if __name__ == "__main__":
    # Initialize analyzer
    analyzer = ChessAnalyzer(eval_file="b:\\chess_engine_4\\cleaned_combined_data.json")

    # Corrected FEN sequence (moves in SAN for input, output in UCI)
    fen_sequence = [
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "e4"),
        ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", "Nf6"),
        ("rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2", "Bc4"),
        ("rnbqkb1r/pppppppp/5n2/8/4P3/2B5/PPPP1PPP/RNBQK1NR b KQkq - 1 2", "Nxe4"),
        ("rnbqkb1r/pppppppp/8/8/8/2B5/PPPP1PPP/RNBQK1NR w KQkq - 0 3", "d3"),
        ("rnbqkb1r/pppppppp/8/8/8/2BP4/PPP2PPP/RNBQK1NR b KQkq - 0 3", "Nf6"),
        ("rnbqkb1r/pppppppp/5n2/8/8/2BP4/PPP2PPP/RNBQK1NR w KQkq - 1 4", "Nf3"),
        ("rnbqkb1r/pppppppp/5n2/8/8/2BP1N2/PPP2PPP/RNBQK2R b KQkq - 1 4", "d6"),
        ("rnbqkb1r/ppp1pppp/3p1n2/8/8/2BP1N2/PPP2PPP/RNBQK2R w KQkq - 0 5", "O-O"),
        ("rnbqkb1r/ppp1pppp/3p1n2/8/8/2BP1N2/PPP2PPP/RNBQ1RK1 b KQkq - 1 5", "Nc6"),
        ("r1bqkb1r/ppp1pppp/2np1n2/8/8/2BP1N2/PPP2PPP/RNBQ1RK1 w KQkq - 2 6", "Qe2"),
        ("r1bqkb1r/ppp1pppp/2np1n2/8/8/2BP1N2/PPP1QPPP/RNB2RK1 b KQkq - 2 6", "e6"),
        ("r1bqkb1r/ppp2ppp/2np1n2/8/8/2BP1N2/PPP1QPPP/RNB2RK1 w KQkq - 0 7", "Bg5"),
        ("r1bqkb1r/ppp2ppp/2np1n2/6B1/8/2BP1N2/PPP1QPPP/RN3RK1 b KQkq - 1 7", "Be7"),
        ("r1bqk2r/ppp1bppp/2np1n2/6B1/8/2BP1N2/PPP1QPPP/RN3RK1 w KQkq - 2 8", "Re1"),
        ("r1bqk2r/ppp1bppp/2np1n2/6B1/8/2BP1N2/PPP1QPPP/RN2R1K1 b KQkq - 3 8", "O-O"),
        ("r1bq1rk1/ppp1bppp/2np1n2/6B1/8/2BP1N2/PPP1QPPP/RN2R1K1 w KQkq - 4 9", "Nc3"),
        ("r1bq1rk1/ppp1bppp/2np1n2/6B1/8/2BN1N2/PPP1QPPP/R3R1K1 b KQkq - 0 9", "h6"),
        ("r1bq1rk1/ppp1bpp1/2np1n1p/6B1/8/2BN1N2/PPP1QPPP/R3R1K1 w KQkq - 1 10", "a3"),
        ("r1bq1rk1/ppp1bpp1/2np1n1p/6B1/8/P1BN1N2/1PP1QPPP/R3R1K1 b KQkq - 0 10", "a6"),
        ("r1bq1rk1/1pp1bpp1/p1np1n1p/6B1/8/P1BN1N2/1PP1QPPP/R3R1K1 w KQkq - 1 11", "Nd5"),
        ("r1bq1rk1/1pp1bpp1/p1np1n1p/3N4/6B1/P1B2N2/1PP1QPPP/R3R1K1 b KQkq - 2 11", "Nxd5"),
        ("r1bq1rk1/1pp1bpp1/p1n2n1p/3n4/6B1/P1B2N2/1PP1QPPP/R3R1K1 w KQkq - 0 12", "Nh4"),
        ("r1bq1rk1/1pp1bpp1/p1n2n1p/3n4/6BN/P1B5/1PP1QPPP/R3R1K1 b KQkq - 1 12", "Qd7"),
        ("r1b2rk1/1ppq1pp1/p1n2n1p/3n4/6BN/P1B5/1PP1QPPP/R3R1K1 w KQkq - 2 13", "Qe3"),
        ("r1b2rk1/1ppq1pp1/p1n2n1p/3n4/6BN/P1B1Q3/1PP2PPP/R3R1K1 b KQkq - 3 13", "Rad8"),
        ("r1br1rk1/1ppq1pp1/p1n2n1p/3n4/6BN/P1B1Q3/1PP2PPP/R3R1K1 w KQkq - 4 14", "Qe5"),
        ("r1br1rk1/1ppq1pp1/p1n2n1p/3nQ3/6BN/P1B5/1PP2PPP/R3R1K1 b KQkq - 4 14", "Nf4"),
        ("r1br1rk1/1ppq1pp1/p1n4p/3nQn2/6BN/P1B5/1PP2PPP/R3R1K1 w KQkq - 5 15", "Nf5"),
        ("r1br1rk1/1ppq1pp1/p1n4p/3nQN2/6B1/P1B5/1PP2PPP/R3R1K1 b KQkq - 0 15", "Nxf5"),
        ("r1br1rk1/1ppq1pp1/p1n5/3nn3/6B1/P1B5/1PP2PPP/R3R1K1 w KQkq - 1 16", "Bxf5")
    ]
    analysis = analyzer.analyze_fen_sequence(fen_sequence)
    for move_analysis in analysis:
        print(f"info move {move_analysis['move_number']} {move_analysis['player']}")
        print(f"info currmove {move_analysis['move_played_uci']} score {move_analysis['eval_score']} san {move_analysis['move_played']}")
        print(f"info bestmove {move_analysis['best_move']} diff {move_analysis['eval_diff']}")
        print(f"info string Classification: {move_analysis['classification']}, {move_analysis['explanation']}")
        print()

    # Test random FENs
    analyzer.test_random_fens()

    # UCI loop (uncomment to use UCI mode)
    # analyzer.uci_loop()

    # Interactive mode
    analyzer.board = chess.Board()
    analyzer.play_interactive()
import chess
import time
import logging
from typing import Tuple, Optional, List
from phase8.utils import PIECE_VALUES, PST, TranspositionTable
from phase8.evaluations import evaluate_position, get_centipawn_score

logger = logging.getLogger(__name__)

def is_square_safe(board: chess.Board, square: chess.Square, color: chess.Color) -> bool:
    """Check if a square is safe from opponent attacks."""
    return not bool(board.attackers(not color, square))

def get_capture_value(board: chess.Board, move: chess.Move) -> Tuple[int, int]:
    """Return (captured value, capturer value) for MVV-LVA."""
    capturer = board.piece_at(move.from_square)
    capturer_val = PIECE_VALUES.get(capturer.piece_type, 0) if capturer else 0
    if move in board.legal_moves and board.is_capture(move):
        captured = board.piece_at(move.to_square)
        captured_val = PIECE_VALUES.get(captured.piece_type, 0) if captured else 0
        return (captured_val, capturer_val)
    return (0, capturer_val)

def can_castle(board: chess.Board, move: chess.Move) -> bool:
    """Check if castling is safe and beneficial."""
    if not board.is_castling(move):
        return False
    king_square = board.king(board.turn)
    if king_square is None:
        return False
    rook_square = move.to_square
    path_squares = chess.SquareSet.between(king_square, rook_square) | {king_square, rook_square}
    for square in path_squares:
        if board.attackers(not board.turn, square):
            return False
    board.push(move)
    safe = is_square_safe(board, board.king(board.turn), board.turn)
    board.pop()
    return safe

def is_king_move_safe(board: chess.Board, move: chess.Move) -> bool:
    """Check if a king move is safe or escapes check."""
    piece = board.piece_at(move.from_square)
    if not piece or piece.piece_type != chess.KING:
        return True
    if board.is_check():
        board.push(move)
        safe = not board.is_check()
        board.pop()
        return safe
    return is_square_safe(board, move.to_square, board.turn)

def get_best_move(board: chess.Board, nnue, transposition_table: TranspositionTable, killer_moves: dict, 
                  move_history_heuristic: dict, history: dict, depth: int = 5, time_limit: float = 2.0) -> Tuple[Optional[chess.Move], float]:
    """
    Find the best move using iterative deepening and alpha-beta with quiescence search.
    
    Args:
        board (chess.Board): The current board position.
        nnue: The NNUE model for evaluation.
        transposition_table (TranspositionTable): The transposition table for caching.
        killer_moves (dict): Killer moves for move ordering.
        move_history_heuristic (dict): History heuristic for move ordering.
        history (dict): Move history for move ordering.
        depth (int): Search depth.
        time_limit (float): Time limit in seconds.
    
    Returns:
        Tuple[Optional[chess.Move], float]: Best move and its evaluation score.
    """
    def quiescence(board: chess.Board, alpha: float, beta: float, depth_limit: int = 6) -> float:
        if depth_limit <= 0:
            return evaluate_position(board, nnue, transposition_table)
        stand_pat = evaluate_position(board, nnue, transposition_table)
        if stand_pat == 0.0 and board.king(board.turn) is None:
            return stand_pat
        if board.turn == chess.WHITE:
            if stand_pat >= beta:
                return beta
            alpha = max(alpha, stand_pat)
            moves = []
            for move in board.legal_moves:
                if board.is_capture(move) or move.promotion or board.gives_check(move):
                    captured_val, capturer_val = get_capture_value(board, move)
                    score = captured_val - capturer_val / 100.0
                    if move.promotion == chess.QUEEN:
                        score += 900
                    if board.gives_check(move):
                        score += 200
                    if stand_pat + captured_val + 200 < alpha:
                        continue
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
                if board.is_capture(move) or move.promotion or board.gives_check(move):
                    captured_val, capturer_val = get_capture_value(board, move)
                    score = captured_val - capturer_val / 100.0
                    if move.promotion == chess.QUEEN:
                        score += 900
                    if board.gives_check(move):
                        score += 200
                    if stand_pat - captured_val - 200 > beta:
                        continue
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
        tt_entry = transposition_table.lookup(board, depth)
        if tt_entry:
            score, move = tt_entry
            return score, move
        if time.time() - start_time > time_limit:
            return evaluate_position(board, nnue, transposition_table), None
        if depth == 0 or board.is_game_over():
            score = quiescence(board, alpha, beta)
            transposition_table.store(board, score, depth, None, "exact")
            return score, None
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            score = evaluate_position(board, nnue, transposition_table)
            transposition_table.store(board, score, depth, None, "exact")
            return score, None
        move_scores = []
        killers = killer_moves.get(depth, [])
        tt_move = transposition_table.lookup(board, 0)[1] if transposition_table.lookup(board, 0) else None
        total_material = sum(PIECE_VALUES.get(board.piece_at(s).piece_type, 0) for s in chess.SQUARES 
                            if board.piece_at(s) and board.piece_at(s).piece_type != chess.KING)
        for move in legal_moves:
            score = 0
            if move == tt_move:
                score += 100000
            if move in killers:
                score += 10000
            if board.is_capture(move):
                captured_val, capturer_val = get_capture_value(board, move)
                score += 5000 + captured_val * 100 - capturer_val
            if move.promotion == chess.QUEEN:
                score += 4000
            if board.is_castling(move) and can_castle(board, move):
                score += 3000
            if board.gives_check(move):
                score += 2000
            piece = board.piece_at(move.from_square)
            if piece:
                pst_key = 'KING_END' if piece.piece_type == chess.KING and total_material < 2000 else piece.piece_type
                score += PST[pst_key][move.to_square] * 10
            move_key = (board.fen(), str(move))
            score += move_history_heuristic.get(move_key, 0) * 100
            move_scores.append((move, score))
        move_scores.sort(key=lambda x: x[1], reverse=True)
        best_score = -float('inf') if board.turn == chess.WHITE else float('inf')
        best_move = None
        node_type = "upper" if board.turn == chess.WHITE else "lower"
        for move, _ in move_scores:
            if not is_king_move_safe(board, move):
                continue
            try:
                board.push(move)
                score, _ = alpha_beta(board, depth - 1, alpha, beta, start_time)
                board.pop()
            except Exception as e:
                logger.error(f"Error processing move {move.uci()}: {str(e)}")
                continue
            move_key = (board.fen(), str(move))
            history[move_key] = history.get(move_key, 0) + 1
            if board.turn == chess.WHITE:
                if score > best_score:
                    best_score = score
                    best_move = move
                    node_type = "exact"
                    move_history_heuristic[move_key] = move_history_heuristic.get(move_key, 0) + depth * depth
                    if len(killers) < 2:
                        killers.append(move)
                    else:
                        killers[1] = killers[0]
                        killers[0] = move
                    killer_moves[depth] = killers
                alpha = max(alpha, best_score)
            else:
                if score < best_score:
                    best_score = score
                    best_move = move
                    node_type = "exact"
                    move_history_heuristic[move_key] = move_history_heuristic.get(move_key, 0) + depth * depth
                    if len(killers) < 2:
                        killers.append(move)
                    else:
                        killers[1] = killers[0]
                        killers[0] = move
                    killer_moves[depth] = killers
                beta = min(beta, score)
            if beta <= alpha:
                move_history_heuristic[move_key] = move_history_heuristic.get(move_key, 0) + depth * depth
                node_type = "lower" if board.turn == chess.WHITE else "upper"
                break
        if best_move is None and legal_moves:
            best_move = legal_moves[0]
            try:
                board.push(best_move)
                best_score = evaluate_position(board, nnue, transposition_table)
                board.pop()
                node_type = "exact"
            except Exception as e:
                logger.error(f"Error evaluating default move {best_move.uci()}: {str(e)}")
                best_score = 0.0
        transposition_table.store(board, best_score, depth, best_move, node_type)
        return best_score, best_move

    def iterative_deepening(board: chess.Board, max_depth: int, time_limit: float, start_time: float) -> Tuple[float, Optional[chess.Move]]:
        best_move = None
        best_score = -float('inf') if board.turn == chess.WHITE else float('inf')
        for d in range(1, max_depth + 1):
            if time.time() - start_time > time_limit:
                break
            score, move = alpha_beta(board, d, -float('inf'), float('inf'), start_time)
            if move:
                best_move = move
                best_score = score
        return best_score, best_move

    total_material = sum(PIECE_VALUES.get(board.piece_at(s).piece_type, 0) for s in chess.SQUARES 
                         if board.piece_at(s) and board.piece_at(s).piece_type != chess.KING)
    time_adjustment = min(5.0, time_limit * (1 + len(list(board.legal_moves)) / 20 + (4000 - total_material) / 2000))
    dynamic_depth = max(1, depth + 1 if len(list(board.legal_moves)) < 10 or abs(evaluate_position(board, nnue, transposition_table)) < 2.0 else depth)
    start_time = time.time()
    score, move = iterative_deepening(board, dynamic_depth, time_adjustment, start_time)
    return move, score
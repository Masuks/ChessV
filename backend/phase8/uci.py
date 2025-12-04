import chess
import logging
import time
from typing import List, Dict, Tuple, Optional
from phase8.evaluations import evaluate_position, get_centipawn_score
from phase8.search import get_best_move, get_capture_value, can_castle, is_king_move_safe
from phase8.utils import validate_fen, TranspositionTable, PIECE_VALUES, PST

logger = logging.getLogger(__name__)

def classify_move(board: chess.Board, move: chess.Move, best_move: chess.Move, best_score: float, 
                 played_score: float) -> Tuple[str, str]:
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

def analyze_fen_sequence(board: chess.Board, nnue, transposition_table: TranspositionTable, 
                        killer_moves: dict, move_history_heuristic: dict, history: dict, 
                        fen_sequence: List[Tuple[str, str]], depth: int = 5, time_limit: float = 2.0) -> List[Dict]:
    """Analyze a sequence of FEN positions and moves, returning UCI format."""
    analysis = []
    move_number = 0
    board.reset()
    for fen, move_san in fen_sequence:
        move_number += 1
        if not validate_fen(fen):
            logger.error(f"Invalid FEN: {fen}")
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
        try:
            board.set_fen(fen)
        except ValueError as e:
            logger.error(f"Invalid FEN: {fen}, {str(e)}")
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
        player = "White" if board.turn == chess.WHITE else "Black"
        try:
            played_move = board.parse_san(move_san)
            if played_move not in board.legal_moves:
                logger.error(f"Illegal move: {move_san} in FEN: {fen}")
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
            logger.error(f"Invalid move format: {move_san}")
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
        try:
            best_move, best_score = get_best_move(board, nnue, transposition_table, killer_moves, 
                                                 move_history_heuristic, history, depth, time_limit)
            board.push(played_move)
            played_score = evaluate_position(board, nnue, transposition_table)
            board.pop()
            # Handle infinite scores
            if best_score in (float('inf'), -float('inf')) or played_score in (float('inf'), -float('inf')):
                eval_diff = 1000.0  # Cap infinite differences
            else:
                eval_diff = abs(best_score - played_score) * 100
            classification, explanation = classify_move(board, played_move, best_move, best_score, played_score)
            analysis.append({
                "move_number": move_number // 2 + 1,
                "player": player,
                "move_played": move_san,
                "move_played_uci": played_move.uci(),
                "best_move": best_move.uci() if best_move else "none",
                "eval_score": get_centipawn_score(played_score, board.turn),
                "eval_diff": eval_diff,
                "classification": classification,
                "explanation": explanation
            })
        except Exception as e:
            logger.error(f"Error analyzing move {move_san} in FEN {fen}: {str(e)}")
            analysis.append({
                "move_number": move_number // 2 + 1,
                "player": player,
                "move_played": move_san,
                "move_played_uci": "none",
                "best_move": "none",
                "eval_score": "cp 0",
                "eval_diff": 0,
                "classification": "Invalid",
                "explanation": f"Analysis error: {str(e)}"
            })
        board.push(played_move)
    return analysis

def predict_next_move(board: chess.Board, nnue, transposition_table: TranspositionTable, 
                     killer_moves: dict, move_history_heuristic: dict, history: dict, 
                     fen: str, depth: int = 5, time_limit: float = 2.0) -> Tuple[str, str]:
    """Predict the best move for a given FEN position in UCI format."""
    if not validate_fen(fen):
        logger.error(f"Invalid FEN: {fen}")
        return "none", "cp 0"
    try:
        board.set_fen(fen)
    except ValueError as e:
        logger.error(f"Invalid FEN: {fen}, {str(e)}")
        return "none", "cp 0"
    move, score = get_best_move(board, nnue, transposition_table, killer_moves, 
                                move_history_heuristic, history, depth, time_limit)
    return move.uci() if move else "none", get_centipawn_score(score, board.turn)

def test_random_fens(board: chess.Board, nnue, transposition_table: TranspositionTable, 
                     killer_moves: dict, move_history_heuristic: dict, history: dict):
    """Test the engine with random FEN positions, outputting UCI moves."""
    test_fens = [
        ("rnbqkbnr/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKBNR w KQkq - 0 1", "g1f3"),
        ("rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 0 1", "e7e5"),
        ("8/p3K3/5Qk1/8/8/P6p/4B2P/8 b - - 2 56", "g6h7"),
        ("rnbqkb1r/pp2pppp/5n2/2p5/2P5/5N2/PP1PPPPP/RNBQKB1R w KQkq - 0 2", "d2d4"),
        ("r3k2r/ppp2ppp/5n2/2b5/2B5/5N2/PPP2PPP/R3K2R w KQkq - 2 3", "e1g1")
    ]
    nodes_searched = 0
    tt_hits = 0
    start_time = time.time()
    for fen, expected_move in test_fens:
        try:
            board.set_fen(fen)
        except ValueError as e:
            logger.error(f"Invalid FEN: {fen}, {str(e)}")
            continue
        move, score = predict_next_move(board, nnue, transposition_table, killer_moves, 
                                       move_history_heuristic, history, fen)
        nodes_searched += len(list(board.legal_moves))
        if transposition_table.lookup(board, 0):
            tt_hits += 1
        logger.info(f"FEN: {fen}")
        logger.info(f"Predicted move: {move}, Score: {score}")
        logger.info(f"Expected move: {expected_move}")
        print(f"FEN: {fen}")
        print(f"Predicted move: {move}, Score: {score}")
        print(f"Expected move: {expected_move}")
        print()
    elapsed_time = time.time() - start_time
    nps = nodes_searched / elapsed_time if elapsed_time > 0 else 0
    tt_hit_rate = tt_hits / len(test_fens) if test_fens else 0
    logger.info(f"Nodes per second: {nps:.2f}, Transposition table hit rate: {tt_hit_rate:.2%}")
    print(f"info string Nodes per second: {nps:.2f}, Transposition table hit rate: {tt_hit_rate:.2%}")

def play_interactive(board: chess.Board, nnue, transposition_table: TranspositionTable, 
                     killer_moves: dict, move_history_heuristic: dict, history: dict, 
                     depth: int = 5, time_limit: float = 2.0):
    """Interactive mode accepting UCI moves and outputting UCI responses."""
    logger.info("Starting interactive mode")
    print(board)
    move_log = []
    while not board.is_game_over():
        if board.turn == chess.WHITE:
            move_str = input("Your move (UCI, e.g., e2e4, g1f3, or 'quit' to exit): ")
            if move_str.lower() == 'quit':
                break
            try:
                move = chess.Move.from_uci(move_str)
                if move in board.legal_moves:
                    move_san = board.san(move)
                    board.push(move)
                    move_log.append(move.uci())
                    logger.info(f"Player move: {move.uci()}, FEN: {board.fen()}")
                    print(f"FEN after {move.uci()}: {board.fen()}")
                else:
                    logger.warning(f"Invalid move: {move_str}")
                    print("Invalid move.")
                    continue
            except (ValueError, AttributeError):
                logger.warning(f"Invalid move format: {move_str}")
                print("Invalid move format. Use UCI (e.g., e2e4).")
                continue
        else:
            logger.info("Engine thinking...")
            print("Engine thinking...")
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                logger.warning("No legal moves available")
                print("No legal moves.")
                break
            move, score = get_best_move(board, nnue, transposition_table, killer_moves, 
                                        move_history_heuristic, history, depth, time_limit)
            if move:
                move_san = board.san(move)
                board.push(move)
                move_log.append(move.uci())
                logger.info(f"Engine move: {move.uci()}, FEN: {board.fen()}")
                print(f"FEN after {move.uci()}: {board.fen()}")
            else:
                logger.warning("No legal moves found by engine")
                print("No legal moves.")
                break
        print(board)
        print("\n")
    logger.info(f"Game over: {board.result()}")
    print(f"Game over: {board.result()}")
    print("Move log (UCI):", " ".join(move_log))

def uci_loop(board: chess.Board, nnue, transposition_table: TranspositionTable, 
             killer_moves: dict, move_history_heuristic: dict, history: dict):
    """Main UCI loop for handling UCI commands."""
    logger.info("Starting UCI loop")
    print("id name ChessAnalyzer")
    print("id author xAI")
    print("uciok")
    while True:
        try:
            command = input().strip()
            logger.debug(f"Received UCI command: {command}")
            if command == "uci":
                print("id name ChessAnalyzer")
                print("id author xAI")
                print("uciok")
            elif command == "isready":
                print("readyok")
            elif command == "quit":
                logger.info("Exiting UCI loop")
                break
            elif command.startswith("position"):
                parts = command.split()
                if "fen" in parts:
                    fen_index = parts.index("fen") + 1
                    fen_parts = parts[fen_index:]
                    moves_index = fen_parts.index("moves") if "moves" in fen_parts else len(fen_parts)
                    fen = " ".join(fen_parts[:moves_index])
                    try:
                        board.set_fen(fen)
                        logger.info(f"Set position: {fen}")
                    except ValueError as e:
                        logger.error(f"Invalid FEN: {fen}, {str(e)}")
                        continue
                    if "moves" in parts:
                        for move_str in fen_parts[moves_index + 1:]:
                            try:
                                move = chess.Move.from_uci(move_str)
                                if move in board.legal_moves:
                                    board.push(move)
                                    logger.debug(f"Applied move: {move_str}")
                                else:
                                    logger.error(f"Invalid move: {move_str}")
                            except ValueError:
                                logger.error(f"Invalid move format: {move_str}")
                elif parts[1] == "startpos":
                    board.reset()
                    logger.info("Reset to start position")
                    if "moves" in parts:
                        moves_index = parts.index("moves") + 1
                        for move_str in parts[moves_index:]:
                            try:
                                move = chess.Move.from_uci(move_str)
                                if move in board.legal_moves:
                                    board.push(move)
                                    logger.debug(f"Applied move: {move_str}")
                                else:
                                    logger.error(f"Invalid move: {move_str}")
                            except ValueError:
                                logger.error(f"Invalid move format: {move_str}")
            elif command.startswith("go"):
                parts = command.split()
                depth = 5
                time_limit = 2.0
                if "depth" in parts:
                    depth = int(parts[parts.index("depth") + 1])
                if "movetime" in parts:
                    time_limit = float(parts[parts.index("movetime") + 1]) / 1000.0
                if "infinite" in parts:
                    time_limit = float('inf')
                move, score = get_best_move(board, nnue, transposition_table, killer_moves, 
                                            move_history_heuristic, history, depth, time_limit)
                if move:
                    print(f"bestmove {move.uci()}")
                    logger.info(f"Best move: {move.uci()}, Score: {get_centipawn_score(score, board.turn)}")
                else:
                    logger.warning("No legal moves found")
                    print("bestmove (none)")
            elif command == "stop":
                pass
        except Exception as e:
            logger.error(f"UCI error: {e}")
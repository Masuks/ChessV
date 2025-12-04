import chess
import torch
import numpy as np
import logging
from typing import Optional
from phase8.utils import TranspositionTable, validate_fen
from phase8.nnue import NNUE

logger = logging.getLogger(__name__)

def board_to_features(board: chess.Board) -> np.ndarray:
    """
    Convert a chess board position to NNUE input features.
    
    Args:
        board (chess.Board): The chess board position.
    
    Returns:
        np.ndarray: Feature vector for NNUE input (768 piece features + 4 additional features).
    """
    features = np.zeros(12 * 64 + 4, dtype=np.float32)
    piece_map = board.piece_map()
    for square, piece in piece_map.items():
        piece_idx = piece.piece_type - 1 + (6 if piece.color == chess.BLACK else 0)
        features[piece_idx * 64 + square] = 1.0
    white_king = board.king(chess.WHITE)
    black_king = board.king(chess.BLACK)
    if white_king is not None and black_king is not None:
        features[12 * 64] = chess.square_distance(white_king, black_king) / 14.0
    features[12 * 64 + 1] = (len(board.pieces(chess.PAWN, chess.WHITE)) - len(board.pieces(chess.PAWN, chess.BLACK))) / 8.0
    features[12 * 64 + 2] = len(list(board.legal_moves)) / 50.0
    features[12 * 64 + 3] = (board.has_kingside_castling_rights(chess.WHITE) + 
                             board.has_queenside_castling_rights(chess.WHITE) -
                             board.has_kingside_castling_rights(chess.BLACK) -
                             board.has_queenside_castling_rights(chess.BLACK)) / 4.0
    return features

def evaluate_position(board: chess.Board, nnue: Optional[NNUE] = None, 
                     transposition_table: Optional[TranspositionTable] = None) -> float:
    """
    Evaluate the board position using NNUE.
    
    Args:
        board (chess.Board): The chess board position.
        nnue (Optional[NNUE]): The NNUE model instance. If None, loads from model_path.
        transposition_table (Optional[TranspositionTable]): The transposition table for caching.
    
    Returns:
        float: Evaluation score in centipawns (positive = White advantage).
    """
    fen = board.fen()
    if not validate_fen(fen):
        logger.error(f"Invalid FEN: {fen}")
        return 0.0
    if board.is_checkmate():
        return -float('inf') if board.turn == chess.WHITE else float('inf')
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0
    if transposition_table is not None:
        cached_eval = transposition_table.lookup(board, 0)
        if cached_eval is not None:
            return cached_eval[0]
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if nnue is None:
            nnue = NNUE()
            model_path = "b:\\chess_engine_4\\main\\nnue_model.pth"
            nnue.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
            nnue.to(device)
            nnue.eval()
        else:
            device = next(nnue.parameters()).device  # Get device from model
        features = torch.tensor(board_to_features(board), dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            score = nnue(features).item()
        if transposition_table is not None:
            transposition_table.store(board, score, 0, None, "exact")
        return score
    except FileNotFoundError:
        logger.error(f"NNUE model file not found at b:\\chess_engine_4\\phase8\\nnue_model.pth")
        return 0.0
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        return 0.0

def get_centipawn_score(score: float, turn: chess.Color) -> str:
    """Convert evaluation to UCI centipawn or mate score."""
    if score == float('inf'):
        return f"mate {1 if turn == chess.BLACK else -1}"
    if score == -float('inf'):
        return f"mate {1 if turn == chess.WHITE else -1}"
    if abs(score) > 1000:
        mate_distance = int(10000 / abs(score))
        return f"mate {mate_distance if turn == chess.BLACK else -mate_distance}"
    return f"cp {int(score)}"
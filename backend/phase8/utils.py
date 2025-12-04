import chess
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Piece values (in centipawns)
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}

# Piece-Square Tables
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
    -50,-40,-30,-30,-30,-30,-40,-50,
]

PST_BISHOP = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10,  5,  5, 10, 10,-10,
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
    0,  0,  0,  5,  5,  0,  0,  0
]

PST_QUEEN = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  5,  0,  0,  5,  0,-10,
    -10,  5,  5,  5,  5,  5,  5,-10,
    -5,  0,  5, 10, 10,  5,  0, -5,
    -5,  0, 10, 10, 10, 10,  0, -5,
    -10,  5,  5,  5,  5,  5,  5,-10,
    -10,  0,  0,  0,  0,  0,  0,-10,
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
    -30,-20,-10,  0,  0,-10,-20,-30,
    -50,-40,-30,-20,-20,-30,-40,-50
]

PST = {
    chess.PAWN: PST_PAWN,
    chess.KNIGHT: PST_KNIGHT,
    chess.BISHOP: PST_BISHOP,
    chess.ROOK: PST_ROOK,
    chess.QUEEN: PST_QUEEN,
    chess.KING: PST_KING_MID,
    'KING_END': PST_KING_END
}

def validate_fen(fen: str) -> bool:
    """Validate a FEN string with relaxed checks."""
    try:
        board = chess.Board(fen)
        white_kings = sum(1 for square in chess.SQUARES if board.piece_at(square) == chess.Piece(chess.KING, chess.WHITE))
        black_kings = sum(1 for square in chess.SQUARES if board.piece_at(square) == chess.Piece(chess.KING, chess.BLACK))
        return white_kings == 1 and black_kings == 1 and board.is_valid()
    except ValueError as e:
        logger.error(f"FEN validation error: {fen}, {str(e)}")
        return False

class TranspositionTable:
    """Transposition table for storing position evaluations, depths, and moves."""
    def __init__(self, max_size: int = 1_000_000):
        self.table: Dict[int, Tuple[float, int, Optional[chess.Move], str]] = {}
        self.zobrist_keys = self.init_zobrist_keys()
        self.max_size = max_size

    def init_zobrist_keys(self):
        np.random.seed(42)
        keys = {}
        for square in chess.SQUARES:
            for piece_type in chess.PIECE_TYPES:
                for color in chess.COLORS:
                    keys[(square, piece_type, color)] = np.random.randint(0, 2**64, dtype=np.uint64)
        for color in chess.COLORS:
            keys[('castling', color, 'kingside')] = np.random.randint(0, 2**64, dtype=np.uint64)
            keys[('castling', color, 'queenside')] = np.random.randint(0, 2**64, dtype=np.uint64)
        keys['en_passant'] = {square: np.random.randint(0, 2**64, dtype=np.uint64) for square in chess.SQUARES}
        keys['turn'] = np.random.randint(0, 2**64, dtype=np.uint64)
        return keys

    def compute_zobrist_hash(self, board: chess.Board) -> int:
        h = np.uint64(0)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                h ^= self.zobrist_keys[(square, piece.piece_type, piece.color)]
        if board.turn == chess.BLACK:
            h ^= self.zobrist_keys['turn']
        if board.has_kingside_castling_rights(chess.WHITE):
            h ^= self.zobrist_keys[('castling', chess.WHITE, 'kingside')]
        if board.has_queenside_castling_rights(chess.WHITE):
            h ^= self.zobrist_keys[('castling', chess.WHITE, 'queenside')]
        if board.has_kingside_castling_rights(chess.BLACK):
            h ^= self.zobrist_keys[('castling', chess.BLACK, 'kingside')]
        if board.has_queenside_castling_rights(chess.BLACK):
            h ^= self.zobrist_keys[('castling', chess.BLACK, 'queenside')]
        if board.ep_square is not None:
            h ^= self.zobrist_keys['en_passant'][board.ep_square]
        return int(h)

    def store(self, board: chess.Board, eval_score: float, depth: int, move: Optional[chess.Move], node_type: str) -> None:
        h = self.compute_zobrist_hash(board)
        if len(self.table) >= self.max_size:
            self.table.pop(next(iter(self.table)))
        self.table[h] = (eval_score, depth, move, node_type)

    def lookup(self, board: chess.Board, depth: int) -> Optional[Tuple[float, Optional[chess.Move]]]:
        h = self.compute_zobrist_hash(board)
        entry = self.table.get(h)
        if entry and entry[1] >= depth:
            return entry[0], entry[2]
        return None

    def size(self) -> int:
        return len(self.table)
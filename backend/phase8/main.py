import chess
import torch
import logging
import os
from typing import Dict
from phase8.nnue import NNUE, train_nnue
from phase8.evaluations import board_to_features, evaluate_position
from phase8.uci import uci_loop, play_interactive, test_random_fens, analyze_fen_sequence
from phase8.utils import TranspositionTable, validate_fen

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChessAnalyzer:
    def __init__(self, h5_file: str = "b:\\chess_engine_4\\phase8\\preprocessed_data.h5",
                 model_path: str = "b:\\chess_engine_4\\phase8\\nnue_model.pth"):
        self.board = chess.Board()
        self.h5_file = h5_file
        self.model_path = model_path
        self.transposition_table = TranspositionTable()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nnue = NNUE().to(self.device)
        try:
            self.nnue.load_state_dict(torch.load(self.model_path, weights_only=True, map_location=self.device))
            self.nnue.eval()
            logger.info(f"Loaded trained NNUE model from {self.model_path}")
        except FileNotFoundError:
            logger.warning("No trained NNUE model found. Training required.")
        self.history = {}
        self.killer_moves = {}
        self.move_history_heuristic = {}
        self.depth = 7  # Increased for better play
        self.time_limit = 2.0

    def train(self):
        """Train the NNUE model if not already trained."""
        if os.path.exists(self.model_path):
            logger.info(f"Model already exists at {self.model_path}. Skipping training.")
            return
        logger.info("Training NNUE model...")
        train_nnue(self.h5_file, save_path=self.model_path)
        self.nnue.load_state_dict(torch.load(self.model_path, weights_only=True, map_location=self.device))
        self.nnue.eval()
        logger.info("Loaded newly trained model.")

    def evaluate_position(self, board: chess.Board) -> float:
        """Evaluate a chess position using the NNUE model."""
        return evaluate_position(board, self.nnue, self.transposition_table)

if __name__ == "__main__":
    analyzer = ChessAnalyzer()
    analyzer.train()
    fen_sequence = [
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "e4"),
        ("rnbqkbnr/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 2", "Nf3"),
    ]
    analysis = analyze_fen_sequence(analyzer.board, analyzer.nnue, analyzer.transposition_table, 
                                   analyzer.killer_moves, analyzer.move_history_heuristic, 
                                   analyzer.history, fen_sequence, depth=7)
    for move_analysis in analysis:
        logger.info(f"Move {move_analysis['move_number']} {move_analysis['player']}")
        logger.info(f"Current move: {move_analysis['move_played_uci']} score {move_analysis['eval_score']} san {move_analysis['move_played']}")
        logger.info(f"Best move: {move_analysis['best_move']} diff {move_analysis['eval_diff']}")
        logger.info(f"Classification: {move_analysis['classification']}, {move_analysis['explanation']}")
        print(f"info move {move_analysis['move_number']} {move_analysis['player']}")
        print(f"info currmove {move_analysis['move_played_uci']} score {move_analysis['eval_score']} san {move_analysis['move_played']}")
        print(f"info bestmove {move_analysis['best_move']} diff {move_analysis['eval_diff']}")
        print(f"info string Classification: {move_analysis['classification']}, {move_analysis['explanation']}")
        print()
    test_random_fens(analyzer.board, analyzer.nnue, analyzer.transposition_table, 
                     analyzer.killer_moves, analyzer.move_history_heuristic, analyzer.history)
    analyzer.board.reset()
    play_interactive(analyzer.board, analyzer.nnue, analyzer.transposition_table, 
                     analyzer.killer_moves, analyzer.move_history_heuristic, analyzer.history)
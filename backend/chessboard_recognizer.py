import cv2
import numpy as np
import argparse
import os

class ChessboardRecognizer:
    def __init__(self, template_dir='templates/', threshold=0.4):
        """Initialize the recognizer with template directory and matching threshold."""
        self.template_dir = template_dir
        self.threshold = threshold
        self.piece_templates = self.load_templates()
        self.piece_symbols = {
            'wp 1': 'P', 'wp 2': 'P', 'wn 1': 'N', 'wn 2': 'N', 'wb 1': 'B', 'wb 2': 'B', 'wr 1': 'R', 'wr 2': 'R', 'wq': 'Q', 'wk': 'K', 'wp': 'P', 'wn': 'N', 'wb': 'B', 'wr': 'R',
            'bp': 'p', 'bn': 'n', 'bb': 'b', 'br': 'r', 'bq': 'q', 'bk': 'k'
        }

    def load_templates(self):
        """Load chess piece templates from the template directory."""
        templates = {}
        piece_names = ['wp 1', 'wp 2', 'wn 1', 'wn 2', 'wb 1', 'wb 2', 'wr 1', 'wr 2', 'wq', 'wk', 'wp', 'wn', 'wb', 'wr',  # White pieces
                       'bp', 'bn', 'bb', 'br', 'bq', 'bk']  # Black pieces
        print(f"Loading templates from {self.template_dir}...")
        for piece in piece_names:
            template_path = os.path.join(self.template_dir, f'{piece}.png')
            if os.path.exists(template_path):
                template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
                if template is not None:
                    template = cv2.resize(template, (150, 150))
                    templates[piece] = template
                    print(f"Loaded template: {piece}")
                else:
                    print(f"Error: Could not load template {template_path}")
            else:
                print(f"Error: Template {template_path} not found")
        if not templates:
            print("Warning: No templates loaded. Recognition will fail.")
        return templates

    def preprocess_image(self, image_path):
        """Load and preprocess the chessboard image."""
        print(f"Loading image: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image at {image_path}")
        image_resized = cv2.resize(image, (800, 800))
        image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
        # Use CLAHE for milder contrast adjustment
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image_processed = clahe.apply(image_gray)
        return image_processed

    def split_board(self, image, debug=False, output_dir='output/'):
        """Split the 800x800 image into 64 squares (8x8 grid)."""
        square_size = 100
        squares = []
        print("Splitting board into 8x8 squares...")
        for row in range(8):
            row_squares = []
            for col in range(8):
                y_start = row * square_size
                y_end = (row + 1) * square_size
                x_start = col * square_size
                x_end = (col + 1) * square_size
                square = image[y_start:y_end, x_start:x_end]
                row_squares.append(square)
                if debug:
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    square_filename = os.path.join(output_dir, f'square_{row}_{col}.png')
                    cv2.imwrite(square_filename, square)
            squares.append(row_squares)
        return squares

    def is_square_empty(self, square, debug=False):
        """Check if a square is likely empty by analyzing variance and edges."""
        variance = np.var(square)
        edges = cv2.Canny(square, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        # Relaxed thresholds to avoid marking squares with pieces as empty
        variance_threshold = 300
        edge_threshold = 0.1
        is_empty = variance < variance_threshold and edge_density < edge_threshold
        if debug:
            print(f"  Variance: {variance:.1f}, Edge density: {edge_density:.3f}, Empty: {is_empty}")
        return is_empty

    def enhance_contrast_for_white_pieces(self, square):
        """Special preprocessing for better white piece detection."""
        # Apply CLAHE for local contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(square)
        # Apply unsharp mask to enhance edges
        blurred = cv2.GaussianBlur(enhanced, (9, 9), 10.0)
        enhanced = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)
        return enhanced

    def get_piece_type(self, piece_name):
        """Extract piece type from piece name."""
        if not piece_name:
            return None
        piece_lower = piece_name.lower()
        if 'p' in piece_lower and ('wp' in piece_lower or 'bp' in piece_lower):
            return 'pawn'
        elif 'n' in piece_lower:
            return 'knight'
        elif 'b' in piece_lower and ('wb' in piece_lower or 'bb' in piece_lower):
            return 'bishop'
        elif 'r' in piece_lower:
            return 'rook'
        elif 'q' in piece_lower:
            return 'queen'
        elif 'k' in piece_lower:
            return 'king'
        return None

    def find_piece_by_type_and_expected_color(self, piece_type, expected_color):
        """Find piece template by type and expected color."""
        if not piece_type:
            return None
        color_prefix = 'w' if expected_color == 'white' else 'b'
        type_map = {'pawn': 'p', 'knight': 'n', 'bishop': 'b', 'rook': 'r', 'queen': 'q', 'king': 'k'}
        if piece_type in type_map:
            target = color_prefix + type_map[piece_type]
            for template_name in self.piece_templates.keys():
                if template_name == target or template_name.startswith(target + ' '):
                    return template_name
        return None

    def get_expected_color_by_position(self, row):
        """Determine expected piece color based on starting position."""
        if row <= 1:  # Rows 0-1 (ranks 8-7) - Black starting area
            return 'black'
        elif row >= 6:  # Rows 6-7 (ranks 2-1) - White starting area
            return 'white'
        return None

    def recognize_piece(self, square, row=None, col=None, debug=False):
        """Recognize the piece in a square with improved detection."""
        if not self.piece_templates:
            print("No templates available for matching!")
            return None

        if self.is_square_empty(square, debug=debug):
            if debug:
                print("Square appears to be empty")
            return None

        square_resized = cv2.resize(square, (150, 150))
        enhanced_square = self.enhance_contrast_for_white_pieces(square)
        enhanced_resized = cv2.resize(enhanced_square, (150, 150))

        if debug and row is not None and col is not None:
            debug_dir = 'output/debug'
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir)
            cv2.imwrite(f'{debug_dir}/square_{row}_{col}_original.png', square_resized)
            cv2.imwrite(f'{debug_dir}/square_{row}_{col}_enhanced.png', enhanced_resized)

        best_match = None
        best_score = -1
        all_scores = {}

        for piece, template in self.piece_templates.items():
            result1 = cv2.matchTemplate(square_resized, template, cv2.TM_CCOEFF_NORMED)
            _, score1, _, _ = cv2.minMaxLoc(result1)
            result2 = cv2.matchTemplate(enhanced_resized, template, cv2.TM_CCOEFF_NORMED)
            _, score2, _, _ = cv2.minMaxLoc(result2)
            score = max(score1, score2)
            all_scores[piece] = score
            if score > best_score:
                best_score = score
                best_match = piece
            if debug:
                version = "enhanced" if score2 > score1 else "original"
                print(f"  {piece}: {score:.3f} ({version})")

        if debug:
            print(f"  Best match: {best_match} ({best_score:.3f})")

        if best_score < self.threshold:
            if debug:
                print(f"  No match above threshold {self.threshold}")
            return None

        # Force color correction for starting positions
        if row is not None and best_match:
            expected_color = self.get_expected_color_by_position(row)
            if expected_color:
                current_color = 'white' if best_match.startswith('w') else 'black'
                if current_color != expected_color:
                    piece_type = self.get_piece_type(best_match)
                    alternative = self.find_piece_by_type_and_expected_color(piece_type, expected_color)
                    if alternative:
                        if debug:
                            print(f"  Forced color correction: {best_match} -> {alternative}")
                            print(f"  Scores: {best_match}={best_score:.3f}, {alternative}={all_scores.get(alternative, 0):.3f}")
                        return alternative

        return best_match

    def board_to_fen(self, board):
        """Convert the 8x8 board representation to FEN."""
        fen_rows = []
        for row in board:
            fen_row = ''
            empty_count = 0
            for square in row:
                if square is None:
                    empty_count += 1
                else:
                    if empty_count > 0:
                        fen_row += str(empty_count)
                        empty_count = 0
                    fen_row += self.piece_symbols[square]
            if empty_count > 0:
                fen_row += str(empty_count)
            fen_rows.append(fen_row)
        fen = '/'.join(fen_rows) + ' w KQkq - 0 1'
        return fen

    def analyze_image(self, image_path, debug=False):
        """Analyze the chessboard image and return the FEN."""
        image = self.preprocess_image(image_path)
        if debug:
            if not os.path.exists('output'):
                os.makedirs('output')
            cv2.imwrite('output/preprocessed.png', image)
        squares = self.split_board(image, debug=debug)
        board = []
        for row_idx, row in enumerate(squares):
            board_row = []
            for col_idx, square in enumerate(row):
                if debug:
                    print(f"\n--- Square {chr(97 + col_idx)}{8 - row_idx} (row {row_idx}, col {col_idx}) ---")
                piece = self.recognize_piece(square, row=row_idx, col=col_idx, debug=debug)
                if debug:
                    print(f"Final: {piece}")
                board_row.append(piece)
            board.append(board_row)
        fen = self.board_to_fen(board)
        return fen

def main():
    """Parse command-line arguments and run the recognizer."""
    parser = argparse.ArgumentParser(description="Chessboard FEN generator from image")
    parser.add_argument('image', help="Path to the chessboard image", nargs='?', default='chessboard_screenshot.png')
    parser.add_argument('--threshold', type=float, default=0.4,
                        help="Template matching threshold (0.0 to 1.0)")
    parser.add_argument('--debug', action='store_true',
                        help="Enable debug mode to save intermediate images")
    args = parser.parse_args()

    recognizer = ChessboardRecognizer(threshold=args.threshold)
    fen = recognizer.analyze_image(args.image, debug=args.debug)
    print(f"Generated FEN: {fen}")

if __name__ == "__main__":
    main()
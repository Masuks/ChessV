import chess.pgn
import json
import os
import uuid
import logging
from io import StringIO  # Added import for StringIO

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_pgn_files(pgn_folder, output_file):
    games = []
    # Ensure the folder exists
    if not os.path.exists(pgn_folder):
        logger.error(f"Folder '{pgn_folder}' does not exist.")
        return

    # Get list of PGN files
    pgn_files = [f for f in os.listdir(pgn_folder) if f.lower().endswith('.pgn')]
    if not pgn_files:
        logger.error(f"No PGN files found in '{pgn_folder}'.")
        return

    for idx, filename in enumerate(pgn_files[:100], 1):  # Limit to 100 files
        file_path = os.path.join(pgn_folder, filename)
        logger.info(f"Processing file: {filename} (Game {idx})")
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                pgn_content = f.read().strip()
                if not pgn_content:
                    logger.warning(f"File '{filename}' is empty.")
                    continue

            # Parse PGN
            pgn_io = StringIO(pgn_content)
            pgn_game = chess.pgn.read_game(pgn_io)
            if not pgn_game:
                logger.warning(f"Could not parse PGN in '{filename}'. Invalid PGN format.")
                continue

            # Check for parsing errors
            if pgn_game.errors:
                logger.warning(f"Parsing errors in '{filename}': {pgn_game.errors}")
                continue

            # Extract headers
            headers = pgn_game.headers
            game_data = {
                "id": str(uuid.uuid4()),
                "white": headers.get("White", "Unknown"),
                "black": headers.get("Black", "Unknown"),
                "date": headers.get("Date", "Unknown").replace("??", "01"),
                "event": headers.get("Event", "Unknown"),
                "result": headers.get("Result", "*"),
                "pgn": pgn_content
            }
            games.append(game_data)
            logger.info(f"Successfully parsed: {filename}")
        except UnicodeDecodeError as e:
            logger.error(f"Encoding error in '{filename}': {str(e)}. Try saving the file as UTF-8.")
            continue
        except Exception as e:
            logger.error(f"Error processing '{filename}': {str(e)}")
            continue

    # Save to JSON file
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)  # Create directory if it doesn't exist
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(games, f, indent=2, ensure_ascii=False)
        logger.info(f"Successfully saved {len(games)} games to '{output_file}'.")
    except Exception as e:
        logger.error(f"Error saving JSON file: {str(e)}")

if __name__ == "__main__":
    pgn_folder = "backend\pgn_folder"  # Path to your PGN files
    output_file = "b:/nothing/backend/static/grandmasterGames.json"  # Output path
    parse_pgn_files(pgn_folder, output_file)
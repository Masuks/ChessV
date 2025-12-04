from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_pymongo import PyMongo
from datetime import datetime
import requests
from werkzeug.security import generate_password_hash, check_password_hash
import chess
import chess.pgn
import chess.engine
from io import StringIO
from bson.objectid import ObjectId
import google.generativeai as genai
from phase8.main import ChessAnalyzer
from phase8.uci import analyze_fen_sequence
from chess import Board, Move
from chess_engine_100 import get_random_move
from chess_engine_400 import get_best_move as get_engine_400_move
from chess_engine_600 import ChessAnalyzer as get_engine_600_move
from chess_engine_1000 import ChessAnalyzer as ChessAnalyzer1000
from chess_engine_1200 import ChessAnalyzer as ChessAnalyzer1200
import random
import logging
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from google.api_core import retry
import google.api_core.exceptions as google_exceptions
from chessboard_recognizer import ChessboardRecognizer

app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb://localhost:27017/chessAnalysis"
app.config["SECRET_KEY"] = "{~DKvCX5dJ/j.r!k3~'DF?mY59k75mNpB"
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}}, supports_credentials=True)
mongo = PyMongo(app)

UPLOAD_FOLDER = 'Uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

analyzer_1000 = ChessAnalyzer1000()
analyzer_1200 = ChessAnalyzer1200()

ENGINES = {
    100: {"function": get_random_move, "depth": None, "time_limit": None},
    400: {"function": get_engine_400_move, "depth": 4, "time_limit": None},
    600: {"function": get_engine_600_move, "depth": 5, "time_limit": None},
    1000: {"function": analyzer_1000.get_best_move, "depth": 3, "time_limit": 1.0},
    1200: {"function": analyzer_1200.get_best_move, "depth": 3, "time_limit": 2.0},
}

BOT_PROFILES = {
    100: {
        "name": "PawnPaw",
        "style": "Novice",
        "traits": ["Cautious", "Curious", "Friendly"],
        "description": "A timid pawn dreaming of promotion, learning with every move!"
    },
    400: {
        "name": "BishopBop",
        "style": "Enthusiastic",
        "traits": ["Playful", "Bold", "Cheerful"],
        "description": "A spirited bishop zipping across the board with enthusiasm!"
    },
    600: {
        "name": "KnightRider",
        "style": "Tactical",
        "traits": ["Balanced", "Clever", "Confident"],
        "description": "A cunning knight plotting tactical leaps and bounds!"
    },
    800: {
        "name": "RookRaider",
        "style": "Adventurous",
        "traits": ["Aggressive", "Calculative", "Bold"],
        "description": "A bold rook charging for open files and big attacks!"
    },
    1000: {
        "name": "QueenQwest",
        "style": "Expert",
        "traits": ["Strategic", "Precise", "Witty"],
        "description": "A regal queen weaving precise strategies with flair!"
    },
    1200: {
        "name": "KingCrusader",
        "style": "Masterful",
        "traits": ["Calm", "Sophisticated", "Strategic"],
        "description": "A wise king leading with masterful composure!"
    }
}

STOCKFISH_PATH = "B:\\stockfish\\stockfish-windows-x86-64-avx2.exe"

genai.configure(api_key="AIzaSyDuz8qn0UEw3O2X6i1L8lBztxKGx7QFcW8")
model = genai.GenerativeModel('gemini-2.0-flash-lite')

analyzer = ChessAnalyzer(h5_file="b:\\chess_engine_4\\phase8\\preprocessed_data.h5",
                        model_path="b:\\chess_engine_4\\phase8\\nnue_model.pth")
analyzer.train()  # Train the NNUE model if not already trained
analyzer.depth = 7
analyzer.time_limit = 2.0

@retry.Retry(predicate=retry.if_exception_type(google_exceptions.ResourceExhausted))
def generate_coach_commentary_with_retry(prompt):
    logger.debug("Generating coach commentary")
    response = model.generate_content(prompt)
    return response.text

@app.route('/api/play/move', methods=['POST'])
def play_move():
    try:
        data = request.get_json()
        fen = data.get('fen')
        elo = data.get('elo')
        if not fen or not elo:
            app.logger.error("Missing FEN or ELO")
            return jsonify({'error': 'Missing FEN or ELO'}), 400

        try:
            board = Board(fen)
        except ValueError as e:
            app.logger.error(f"Invalid FEN: {fen}, {str(e)}")
            return jsonify({'error': 'Invalid FEN'}), 400

        if elo not in ENGINES:
            app.logger.error(f"Unsupported ELO: {elo}")
            return jsonify({'error': f'Unsupported ELO: {elo}'}), 400

        engine_config = ENGINES[elo]
        engine_function = engine_config["function"]
        depth = engine_config["depth"]
        time_limit = engine_config["time_limit"]

        if elo in [1000, 1200]:
            analyzer = analyzer_1000 if elo == 1000 else analyzer_1200
            analyzer.board.set_fen(fen)
            move, _ = engine_function(depth=depth, time_limit=time_limit)
        else:
            move = engine_function(board, depth=depth) if depth else engine_function(board)

        if not move:
            app.logger.error("No legal moves available")
            return jsonify({'error': 'No legal moves available'}), 400

        return jsonify({'move': move.uci()})
    except Exception as e:
        app.logger.exception(f"Error processing move: {str(e)}")
        return jsonify({'error': f'Error: {str(e)}'}), 500

@app.route('/api/bot-profile', methods=['POST'])
def get_bot_profile():
    try:
        data = request.get_json()
        elo = data.get('elo')
        if not elo or elo not in BOT_PROFILES:
            return jsonify({'error': 'Invalid ELO'}), 400

        profile = BOT_PROFILES[elo]
        return jsonify({
            'name': profile['name'],
            'elo': elo,
            'style': profile['style'],
            'description': profile['description']
        })
    except Exception as e:
        logger.exception(f"Error generating bot profile: {str(e)}")
        return jsonify({'error': f'Error: {str(e)}'}), 500

@app.route('/api/bot-commentary', methods=['POST'])
def get_bot_commentary():
    try:
        data = request.get_json()
        fen = data.get('fen')
        move = data.get('move')
        elo = data.get('elo')
        bot_name = data.get('botName')
        bot_style = data.get('botStyle')
        player_move = data.get('playerMove', False)

        if not fen or not move or not elo or not bot_name:
            return jsonify({'error': 'Missing required fields'}), 400

        bot_profile = BOT_PROFILES.get(elo, {"traits": ["generic"]})
        traits = ', '.join(bot_profile['traits'])

        prompt = f"""
        As a chess bot named {bot_name} with ELO {elo} and {bot_style} style,
        personality traits: {traits},
        comment on the player's move {move} in the position {fen}.
        Keep it fun, chess-themed, and under 30 words.
        """
        try:
            commentary = generate_coach_commentary_with_retry(prompt)
        except Exception as e:
            logger.error(f"Gemini API error for commentary: {str(e)}")
            commentary = f"{bot_name}: Nice move with {move}! Keep it up!"

        return jsonify({'commentary': commentary})
    except Exception as e:
        logger.exception(f"Error generating bot commentary: {str(e)}")
        return jsonify({'error': f'Error: {str(e)}'}), 500

@app.route('/api/analyze_game', methods=['POST'])
def analyze_game():
    try:
        data = request.get_json()
        pgn_string = data.get('pgn')
        logger.debug(f"Received PGN: {pgn_string}")
        if not pgn_string:
            logger.error("No PGN provided")
            return jsonify({'error': 'No PGN provided'}), 400

        pgn_io = StringIO(pgn_string)
        game = chess.pgn.read_game(pgn_io)
        if not game:
            logger.error("Invalid PGN format")
            return jsonify({'error': 'Invalid PGN format'}), 400

        board = game.board()
        fen_sequence = []
        move_number = 1

        for move in game.mainline_moves():
            fen_before = board.fen()
            san_move = board.san(move)
            try:
                board.push(move)
                fen_sequence.append((fen_before, san_move))
                move_number += 1
            except ValueError as e:
                logger.error(f"Invalid move {san_move} in PGN: {str(e)}")
                return jsonify({'error': f'Invalid move {san_move} in PGN: {str(e)}'}), 400

        engine_analysis = analyze_fen_sequence(
            board=analyzer.board,
            nnue=analyzer.nnue,
            transposition_table=analyzer.transposition_table,
            killer_moves=analyzer.killer_moves,
            move_history_heuristic=analyzer.move_history_heuristic,
            history=analyzer.history,
            fen_sequence=fen_sequence,
            depth=analyzer.depth,
            time_limit=analyzer.time_limit
        )
        logger.debug(f"Engine analysis: {engine_analysis}")

        enhanced_analysis = []
        for idx, move in enumerate(engine_analysis):
            if move['classification'] == "Invalid":
                enhanced_analysis.append({
                    'played_move': move['move_played_uci'],
                    'board_fen': fen_sequence[idx][0] if idx < len(fen_sequence) else "",
                    'evaluation': 0.0,
                    'predicted_best_move': move['best_move'],
                    'predicted_evaluation': 0.0,
                    'coach_commentary': move['explanation'],
                    'player': move['player'].lower()
                })
                continue

            board.set_fen(fen_sequence[idx][0])
            played_move_uci = move['move_played_uci']
            best_move_uci = move['best_move']
            evaluation = move['eval_diff'] / 100.0
            if isinstance(evaluation, float):
                if evaluation == float('inf'):
                    evaluation = 10.0 if move['player'].lower() == 'white' else -10.0
                elif evaluation == -float('inf'):
                    evaluation = -10.0 if move['player'].lower() == 'white' else 10.0
                elif evaluation != evaluation:
                    evaluation = 10.0 if move['player'].lower() == 'white' else -10.0

            prompt = f"""
            As a chess coach, analyze this move:
            Move {move['move_number']} ({move['player']}): {move['move_played']}
            Best move: {move['best_move']}
            Evaluation difference: {evaluation:.2f}
            Classification: {move['classification']}
            
            Provide:
            1. Why this move is strong/weak
            2. Strategic goals
            3. Alternatives if suboptimal
            4. Lessons for similar positions
            5. Keep under 45 words
            """
            
            coach_commentary = move['explanation']
            try:
                coach_commentary = generate_coach_commentary_with_retry(prompt)
            except google_exceptions.ResourceExhausted as e:
                logger.warning(f"Gemini API quota exceeded: {str(e)}")
                coach_commentary = move['explanation']
            except Exception as e:
                logger.error(f"Gemini API error: {str(e)}")
                coach_commentary = move['explanation']

            enhanced_analysis.append({
                'played_move': played_move_uci,
                'board_fen': fen_sequence[idx][0],
                'evaluation': evaluation,
                'predicted_best_move': best_move_uci,
                'predicted_evaluation': 0.0,
                'coach_commentary': coach_commentary,
                'player': move['player'].lower()
            })

        return jsonify({'analysis': enhanced_analysis})
        
    except Exception as e:
        logger.exception(f"Error analyzing game: {str(e)}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/api/analyze_grandmaster_game', methods=['POST'])
def analyze_grandmaster_game():
    try:
        data = request.get_json()
        pgn = data.get('pgn')
        if not pgn:
            logger.error("PGN is required")
            return jsonify({'error': 'PGN is required'}), 400

        pgn_io = StringIO(pgn)
        game = chess.pgn.read_game(pgn_io)
        if not game:
            logger.error("Invalid PGN format")
            return jsonify({'error': 'Invalid PGN format'}), 400

        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        board = game.board()
        enhanced_analysis = []

        for move in game.mainline_moves():
            fen_before = board.fen()
            played_move_uci = move.uci()
            san_move = board.san(move)

            board.push(move)

            info = engine.analyse(board, chess.engine.Limit(depth=18))
            score = info['score'].relative
            if score.is_mate():
                evaluation = f"Mate in {score.mate()}"
            else:
                evaluation = score.score() / 100.0
            best_move = info.get('pv', [None])[0]
            best_move_uci = best_move.uci() if best_move else "none"
            best_move_san = board.san(best_move) if best_move else "None"

            prompt = f"""
            As a chess coach, analyze this grandmaster move:
            Move {board.fullmove_number} ({'White' if board.turn == chess.BLACK else 'Black'}): {san_move}
            Evaluation: {evaluation}
            Best move: {best_move_san}
            
            Provide:
            1. Why this move is strong/weak
            2. Strategic goals
            3. Alternatives if suboptimal
            4. Lessons for similar positions
            5. Keep under 45 words
            """
            coach_commentary = "Coach commentary unavailable"
            try:
                coach_commentary = generate_coach_commentary_with_retry(prompt)
            except google_exceptions.ResourceExhausted as e:
                logger.warning(f"Gemini API quota exceeded: {str(e)}")
            except Exception as e:
                logger.error(f"Gemini API error: {str(e)}")

            enhanced_analysis.append({
                'played_move': played_move_uci,
                'board_fen': fen_before,
                'evaluation': evaluation,
                'predicted_best_move': best_move_uci,
                'predicted_evaluation': evaluation,
                'coach_commentary': coach_commentary,
                'player': 'white' if board.turn == chess.BLACK else 'black'
            })

        info = engine.analyse(board, chess.engine.Limit(depth=18))
        score = info['score'].relative
        if score.is_mate():
            evaluation = f"Mate in {score.mate()}"
        else:
            evaluation = score.score() / 100.0
        best_move = info.get('pv', [None])[0]
        best_move_uci = best_move.uci() if best_move else "none"

        enhanced_analysis.append({
            'played_move': None,
            'board_fen': board.fen(),
            'evaluation': evaluation,
            'predicted_best_move': best_move_uci,
            'predicted_evaluation': evaluation,
            'coach_commentary': "Final position analysis",
            'player': 'white' if board.turn == chess.BLACK else 'black'
        })

        engine.quit()
        logger.info("Grandmaster game analysis completed")
        return jsonify({'analysis': enhanced_analysis})

    except FileNotFoundError:
        logger.error("Stockfish executable not found")
        return jsonify({'error': 'Stockfish executable not found'}), 500
    except Exception as e:
        logger.exception(f"Error analyzing grandmaster game: {str(e)}")
        return jsonify({'error': f"Error: {str(e)}"}), 500

@app.route('/api/chesscom/games', methods=['GET'])
def chesscom_games():
    username = request.args.get('username')
    if not username:
        return jsonify({"error": "No Chess.com username provided"}), 400
    
    username = username.lower()
    archives_url = f"https://api.chess.com/pub/player/{username}/games/archives"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
        "Accept": "application/ld+json"
    }

    archives_response = requests.get(archives_url, headers=headers)
    if archives_response.status_code != 200:
        return jsonify({"error": "Error fetching archives from Chess.com."}), archives_response.status_code
    
    archives_data = archives_response.json()
    archives_list = archives_data.get("archives", [])
    if not archives_list:
        return jsonify({"error": "No games found"}), 404
    
    pgn_list = []
    for archive_url in archives_list[-3:]:
        games_response = requests.get(archive_url, headers=headers)
        if games_response.status_code == 200:
            games_data = games_response.json()
            games = games_data.get("games", [])
            pgn_list.extend([game.get("pgn", "") for game in games if game.get("pgn")])
    
    pgn_list = pgn_list[-10:] if len(pgn_list) > 10 else pgn_list
    logger.debug(f"Returning games: {pgn_list[:2]}")
    return jsonify({"games": pgn_list})

@app.route('/api/grandmaster-games', methods=['GET'])
def get_grandmaster_games():
    try:
        with open(os.path.join('static', 'grandmasterGames.json'), 'r') as f:
            data = json.load(f)
        games = []
        for player, player_games in data.items():
            for game in player_games:
                game['player'] = player
                if 'id' not in game:
                    game['id'] = f"{player}_{game['event']}_{game['date']}".replace(" ", "_")
                games.append(game)
        logger.debug(f"Returning {len(games)} grandmaster games")
        return jsonify({"games": games})
    except FileNotFoundError:
        logger.error("Grandmaster games JSON file not found")
        return jsonify({"error": "Grandmaster games not found"}), 404
    except Exception as e:
        logger.exception(f"Error fetching grandmaster games: {str(e)}")
        return jsonify({"error": f"Error: {str(e)}"}), 500

@app.route('/api/save-grandmaster-analysis', methods=['POST'])
def save_grandmaster_analysis():
    try:
        data = request.get_json()
        username = data.get("username")
        pgn = data.get("pgn")
        game_id = data.get("game_id")
        analysis = data.get("analysis", [])
        last_viewed_move = data.get("last_viewed_move", 0)
        comments = data.get("comments", [])

        if not username or not pgn or not game_id:
            error_msg = "Missing username, pgn, or game_id"
            logger.error(f"Validation failed: {error_msg}")
            return jsonify({"error": error_msg}), 400

        analysis_entry = {
            "username": username,
            "pgn": pgn,
            "game_id": game_id,
            "analysis": analysis,
            "last_viewed_move": last_viewed_move,
            "comments": comments,
            "timestamp": datetime.now().isoformat()
        }
        result = mongo.db.analysis_history.insert_one(analysis_entry)
        logger.info(f"Saved grandmaster analysis with ID: {str(result.inserted_id)}")
        return jsonify({"message": "Analysis saved", "id": str(result.inserted_id)})
    except Exception as e:
        logger.exception(f"Error saving grandmaster analysis: {str(e)}")
        return jsonify({"error": f"Error: {str(e)}"}), 500

@app.route('/api/save-analysis/', methods=['POST'])
def save_analysis():
    data = request.get_json()
    logger.debug(f"Received request data: {data}")
    username = data.get("username")
    pgn = data.get("pgn")
    analysis = data.get("analysis", [])
    last_viewed_move = data.get("last_viewed_move", 0)
    comments = data.get("comments", [])

    if not username or not pgn:
        error_msg = "Missing username or pgn"
        logger.error(f"Validation failed: {error_msg}")
        return jsonify({"error": error_msg}), 400

    analysis_entry = {
        "username": username,
        "pgn": pgn,
        "analysis": analysis,
        "last_viewed_move": last_viewed_move,
        "comments": comments,
        "timestamp": datetime.now().isoformat()
    }
    result = mongo.db.analysis_history.insert_one(analysis_entry)
    logger.info(f"Saved analysis with ID: {str(result.inserted_id)}")
    return jsonify({"message": "Analysis saved", "id": str(result.inserted_id)})

@app.route('/api/analysis-history/<username>', methods=['GET'])
def get_analysis_history(username):
    history = mongo.db.analysis_history.find({"username": username}).sort("timestamp", -1).limit(10)
    history_list = [
        {
            "id": str(entry["_id"]),
            "pgn": entry["pgn"],
            "analysis": entry["analysis"],
            "last_viewed_move": entry["last_viewed_move"],
            "comments": entry.get("comments", []),
            "timestamp": entry["timestamp"]
        }
        for entry in history
    ]
    logger.debug(f"Returning analysis history for {username}: {history_list}")
    return jsonify({"history": history_list})

@app.route('/api/update-last-viewed/<analysis_id>', methods=['POST'])
def update_last_viewed(analysis_id):
    data = request.get_json()
    last_viewed_move = data.get("last_viewed_move")
    comments = data.get("comments")
    
    if last_viewed_move is None:
        return jsonify({"error": "Missing last_viewed_move"}), 400
    
    update_data = {"last_viewed_move": last_viewed_move}
    if comments is not None:
        update_data["comments"] = comments
    
    result = mongo.db.analysis_history.update_one(
        {"_id": ObjectId(analysis_id)},
        {"$set": update_data}
    )
    logger.info(f"Updated analysis ID: {analysis_id} with {update_data}")
    return jsonify({"message": "Last viewed move updated"})

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    logger.debug(f"Login request data: {data}")
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify({"error": "Missing fields"}), 400
    
    user = mongo.db.users.find_one({"username": username})
    if user and check_password_hash(user["password"], password):
        return jsonify({
            "message": "Logged in successfully",
            "user": {"username": user["username"], "email": user.get("email", "")}
        })
    return jsonify({"error": "Invalid credentials"}), 400

@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get("username")
    email = data.get("email")
    password = data.get("password")

    if not username or not email or not password:
        return jsonify({"error": "Missing fields"}), 400
    if mongo.db.users.find_one({"username": username}):
        return jsonify({"error": "Username already exists"}), 400
    
    password_hash = generate_password_hash(password)
    user_data = {"username": username, "email": email, "password": password_hash}
    mongo.db.users.insert_one(user_data)
    return jsonify({"message": "User registered successfully"})

@app.route('/api/predict-move', methods=['POST'])
def predict_move():
    try:
        if 'image' not in request.files:
            logger.error("No image provided")
            return jsonify({'error': 'No image provided'}), 400

        file = request.files['image']
        if file.filename == '':
            logger.error("No file selected")
            return jsonify({'error': 'No file selected'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Initialize ChessboardRecognizer
        recognizer = ChessboardRecognizer(threshold=0.4)
        fen = recognizer.analyze_image(filepath, debug=False)
        logger.debug(f"Generated FEN: {fen}")

        # Validate FEN
        try:
            board = chess.Board(fen)
        except ValueError as e:
            logger.error(f"Invalid FEN: {fen}, {str(e)}")
            os.remove(filepath)
            return jsonify({'error': 'Invalid chess position detected'}), 400

        # Get best move using NNUE engine (ELO 1200)
        analyzer_1200.board.set_fen(fen)
        best_move, _ = analyzer_1200.get_best_move(depth=analyzer_1200.depth, time_limit=analyzer_1200.time_limit)

        if not best_move:
            logger.error("No legal moves available")
            os.remove(filepath)
            return jsonify({'error': 'No legal moves available'}), 400

        # Clean up uploaded file
        os.remove(filepath)

        return jsonify({
            'fen': fen,
            'move': best_move.uci()
        })

    except Exception as e:
        logger.exception(f"Error in predict-move: {str(e)}")
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': f'Error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
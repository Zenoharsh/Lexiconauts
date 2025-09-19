from flask import Blueprint, request, jsonify
from .services import analyze_video

# Create a Blueprint object
main_routes = Blueprint('main_routes', __name__)

@main_routes.route('/api/analyze', methods=['POST'])
def analyze_video_endpoint():
    data = request.get_json()
    if not data or 'video_url' not in data or 'exercise_type' not in data:
        return jsonify({"error": "Request must include 'video_url' and 'exercise_type'"}), 400

    video_url = data['video_url']
    exercise_type = data['exercise_type']
    
    # Call the analysis function from services.py
    result = analyze_video(video_url, exercise_type)
    return jsonify(result)
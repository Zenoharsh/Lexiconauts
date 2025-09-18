from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/analyze', methods=['POST'])
def analyze_video():
    # This is a placeholder for now.
    # The real MoveNet analysis logic will go here.
    print("Request received by the AI service!")
    
    data = request.get_json()
    if not data or 'video_url' not in data:
        return jsonify({"error": "No video_url provided"}), 400
        
    print(f"Received URL: {data['video_url']}")

    # Return dummy data for the prototype
    return jsonify({
        "status": "success", 
        "message": "Analysis complete (placeholder)",
        "jump_height": 55.3,
        "unit": "cm"
    })

if __name__ == '__main__':
    # Running on port 5000 so it doesn't conflict with the Node.js server
    app.run(debug=True, port=5000)
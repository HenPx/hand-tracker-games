from flask import Flask, render_template, Response
from gameLogic.lineFollowing import game_camera as line_following_camera
from gameLogic.rpsGame import game_camera as rps_camera

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# Route untuk game line following
@app.route('/line_following')
def line_following():
    return render_template('lineFoll.html')  

# Route untuk game rps
@app.route('/rps')
def rps():
    return render_template('rpsGame.html')  

@app.route('/video_feed/<game>')
def video_feed(game):
    if game == "line_following":
        return Response(line_following_camera(), mimetype='multipart/x-mixed-replace; boundary=frame')
    elif game == "rps":
        return Response(rps_camera(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "Invalid game selected."

if __name__ == "__main__":
    app.run(debug=True)

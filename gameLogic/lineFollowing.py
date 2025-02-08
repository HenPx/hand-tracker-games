import cv2
import mediapipe as mp
import numpy as np
import math
import time

# MediaPipe Hand Detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Game variables
line_y = 300  # Garis berada di tengah layar
line_thickness = 40  # Ketebalan garis
tolerance = line_thickness // 2  # Batas toleransi keluar garis
start_x = 500
finish_x = 1500
point_radius = 10
wave_amplitude = 70  # Amplitudo sinus
wave_frequency = 0.003  # Jumlah gelombang

player_path = []
is_drawing = False
start_reached = False
finish_reached = False
game_failed = False
start_time = 0
end_time = 0
elapsed_time = 0

# Leaderboard file
leaderboard_file = "storage/lineFoll_leadearboard.txt"

def sinusoidal_y(x):
    return int(line_y + wave_amplitude * math.sin(wave_frequency * (x - start_x) * 2 * math.pi))

def is_finger_in_line(finger_pos):
    x, y = finger_pos
    return abs(y - sinusoidal_y(x)) <= tolerance

def detect_restart_gesture(hand_landmarks):
    thumb = hand_landmarks.landmark[4].y
    index = hand_landmarks.landmark[8].y
    middle = hand_landmarks.landmark[12].y
    ring = hand_landmarks.landmark[16].y
    little = hand_landmarks.landmark[20].y
    return index < middle and index < ring and thumb < middle and little < middle

def detect_index_only(hand_landmarks):
    thumb = hand_landmarks.landmark[4].y
    index = hand_landmarks.landmark[8].y
    middle = hand_landmarks.landmark[12].y
    ring = hand_landmarks.landmark[16].y
    little = hand_landmarks.landmark[20].y
    return index < thumb and index < little and index < middle and index < ring

def save_to_leaderboard(name, time_taken):
    with open(leaderboard_file, "a") as file:
        file.write(f"{name}: {time_taken:.2f} seconds\n")

def read_leaderboard():
    try:
        with open(leaderboard_file, "r") as file:
            leaderboard = file.readlines()
            leaderboard_data = []

            # Extract player name and time from each entry and store as tuples (name, time)
            for entry in leaderboard:
                name, time_str = entry.split(": ")
                time_taken = float(time_str.split(" ")[0])  # Extract the time and convert to float
                leaderboard_data.append((name, time_taken))

            # Sort leaderboard by time taken in ascending order
            leaderboard_data.sort(key=lambda x: x[1])

            # Format the leaderboard back into the text format
            sorted_leaderboard = [f"{name}: {time_taken:.2f} seconds\n" for name, time_taken in leaderboard_data]
            return sorted_leaderboard
    except FileNotFoundError:
        return []



def game_camera():
    global player_path, is_drawing, start_reached, finish_reached, game_failed, start_time, end_time, elapsed_time

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Konversi frame ke RGB untuk MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Gambar garis lintasan sinusoidal
        for x in range(start_x, finish_x, 2):
            y1 = sinusoidal_y(x)
            y2 = sinusoidal_y(x + 2)
            cv2.line(frame, (x, y1), (x + 2, y2), (0, 255, 0), line_thickness)

        # Gambar titik start dan finish
        if not start_reached:
            cv2.circle(frame, (start_x, sinusoidal_y(start_x)), point_radius, (0, 0, 255), -1)  # Merah
            cv2.putText(frame, "Start!", (start_x-30, 270), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        if start_reached and not finish_reached:
            cv2.circle(frame, (finish_x, sinusoidal_y(finish_x)), point_radius, (0, 0, 255), -1)  # Merah
            cv2.putText(frame, "Finish!", (finish_x-30, 348), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)


        # Deteksi tangan
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x = int(hand_landmarks.landmark[8].x * w)
                y = int(hand_landmarks.landmark[8].y * h)
                finger_pos = (x, y)

                # Restart jika gestur terdeteksi
                if detect_restart_gesture(hand_landmarks):
                    player_path.clear()
                    is_drawing = False
                    start_reached = False
                    finish_reached = False
                    game_failed = False

                # Jika jari berada di titik start
                if not start_reached and abs(x - start_x) <= point_radius and abs(y - sinusoidal_y(start_x)) <= point_radius:
                    if detect_index_only(hand_landmarks):
                        is_drawing = True
                        start_reached = True  # Mulai permainan
                        start_time = time.time()  # Mulai stopwatch

                # Jika menggambar aktif dan belum gagal
                if is_drawing and not game_failed:
                    player_path.append(finger_pos)

                    # Jika keluar dari garis
                    if not is_finger_in_line(finger_pos):
                        game_failed = True
                        is_drawing = False
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        
                    # Jika sampai titik finish
                    if start_reached and abs(x - finish_x) <= point_radius:
                        finish_reached = True
                        is_drawing = False
                        end_time = time.time()  # Selesai stopwatch
                        elapsed_time = end_time - start_time
                        save_to_leaderboard("Player", elapsed_time)  # Simpan leaderboard

                # Gambar kursor di ujung jari telunjuk
                cv2.circle(frame, finger_pos, 5, (0, 0, 255), 3)

                # Gambar koneksi tangan
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Tampilkan jalur pemain
        for i in range(1, len(player_path)):
            cv2.line(frame, player_path[i - 1], player_path[i], (0, 0, 0), 2)

        # Tampilkan status permainan
        if game_failed:
            cv2.putText(frame, "FAILED!", (w // 2 - 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        elif finish_reached:
            cv2.putText(frame, f"FINISH! Waktu Anda: {elapsed_time:.2f}s", (w // 2 - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        elif is_drawing:
            cv2.putText(frame, "Menggambar...", (w // 2 - 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 2)
        elif not start_reached:
            cv2.putText(frame, "Bergerak ke titik Start!", (w // 2 - 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)

        # Display stopwatch in the top-right corner with larger font size
        if start_reached and not finish_reached and not game_failed:
            elapsed_time = time.time() - start_time  # Real-time stopwatch
            cv2.putText(frame, f"Time: {elapsed_time:.2f}s", (w - 350, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        # Tampilkan leaderboard
        leaderboard = read_leaderboard()
        cv2.putText(frame, "Peringkat:", (20, h - 400), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 5)
        for idx, line in enumerate(leaderboard[:5]):
            cv2.putText(frame, f"{idx + 1}. {line.strip()}", (20, h - 350 + idx * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Encode frame sebagai JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Kirim frame ke browser
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()


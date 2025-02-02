import cv2
import mediapipe as mp
import random
import time
import math
import numpy as np

# MediaPipe Hand Detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Game variables
start_pose_detected = False
game_active = False
countdown_time = 3
player_choice = None
computer_choice = None
result_text = "Waiting for start pose..."

# State machine variables
game_state = "START"  # START, POSE_DETECT, CONFIRM, RESULT, RESET
confirmation_frames = 0
required_confirmation_frames = 5  # Adjust as needed

def calculate_angle(p1, p2, p3):
    v1 = np.array([p1.x - p2.x, p1.y - p2.y])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y])
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    angle_rad = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def detect_hand_shape(hand_landmarks):
    if hand_landmarks is None or len(hand_landmarks.landmark) < 21:
        return "Invalid Pose"

    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]
    ring_tip = hand_landmarks.landmark[16]
    pinky_tip = hand_landmarks.landmark[20]

    # Rock
    thumb_index_dist = math.dist([thumb_tip.x, thumb_tip.y], [index_tip.x, index_tip.y])
    thumb_middle_dist = math.dist([thumb_tip.x, thumb_tip.y], [middle_tip.x, middle_tip.y])
    if thumb_index_dist < 0.15 and thumb_middle_dist < 0.15:
        return "rock"

    # Paper
    index_base = hand_landmarks.landmark[5]
    middle_base = hand_landmarks.landmark[9]
    ring_base = hand_landmarks.landmark[13]
    pinky_base = hand_landmarks.landmark[17]

    if (index_tip.y < index_base.y and
        middle_tip.y < middle_base.y and
        ring_tip.y < ring_base.y and
        pinky_tip.y < pinky_base.y):
        return "paper"

    # Scissors
    index_middle_dist = math.dist([index_tip.x, index_tip.y], [middle_tip.x, middle_tip.y])
    index_middle_angle = calculate_angle(thumb_tip, index_tip, middle_tip)
    ring_pinky_angle = calculate_angle(middle_tip, ring_tip, pinky_tip)

    if index_middle_dist > 0.15 and index_middle_angle > 60 and ring_pinky_angle > 60:
        return "scissors"

    return "Invalid Pose"

def detect_start_pose(hand_landmarks):
    thumb = hand_landmarks.landmark[4].y
    index = hand_landmarks.landmark[8].y
    pinky = hand_landmarks.landmark[20].y
    return index < thumb and pinky < thumb

def get_computer_choice():
    return random.choice(["rock", "paper", "scissors"])

def determine_winner(player, computer):
    if player == computer:
        return "It's a tie!"
    elif (player == "rock" and computer == "scissors") or \
         (player == "scissors" and computer == "paper") or \
         (player == "paper" and computer == "rock"):
        return "You win!"
    else:
        return "Computer wins!"

def game_camera():
    global start_pose_detected, game_active, player_choice, computer_choice, result_text, countdown_start_time, game_state, confirmation_frames

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                if game_state == "START":
                    if detect_start_pose(hand_landmarks):
                        start_pose_detected = True
                        countdown_start_time = time.time()
                        game_state = "POSE_DETECT"
                        confirmation_frames = 0
                        result_text = "Show your move!"

                elif game_state == "POSE_DETECT":
                    if time.time() - countdown_start_time >= countdown_time:
                        player_choice_temp = detect_hand_shape(hand_landmarks)
                        if player_choice_temp != "Invalid Pose":
                            player_choice = player_choice_temp
                            game_state = "CONFIRM"
                            result_text = "Confirming..."
                            confirmation_frames = 0
                        else:
                            result_text = "Invalid Pose. Show Rock, Paper, or Scissors"
                            countdown_start_time = time.time()

                elif game_state == "CONFIRM":
                    current_choice = detect_hand_shape(hand_landmarks)
                    if current_choice == player_choice:
                        confirmation_frames += 1
                        if confirmation_frames >= required_confirmation_frames:
                            computer_choice = get_computer_choice()
                            result_text = determine_winner(player_choice, computer_choice)
                            game_state = "RESULT"
                            countdown_start_time = time.time()
                    else:
                        game_state = "POSE_DETECT"
                        result_text = "Show your move!"
                        countdown_start_time = time.time()
                        confirmation_frames = 0

                elif game_state == "RESULT":
                    if time.time() - countdown_start_time >= 2:
                        game_state = "RESET"
                        result_text = "Waiting for start pose..."

                elif game_state == "RESET":
                    start_pose_detected = False
                    game_active = False
                    player_choice = None
                    computer_choice = None
                    game_state = "START"

                # Display choices and result
                if player_choice:
                    cv2.putText(frame, f"Player: {player_choice}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                if computer_choice:
                    cv2.putText(frame, f"Computer: {computer_choice}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

                cv2.putText(frame, result_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1.5)

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()
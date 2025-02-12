import csv
import cv2
import mediapipe
import time
import random
import numpy as np


def blinking_text(frame, text, position, font, scale, thickness, line_type, blink_rate):
    x, y = position
    if int(time.time() * blink_rate) % 2 == 0:
        cv2.putText(frame, text, (x, y), font, scale, (255, 182, 193), thickness, line_type)
    else:
        cv2.putText(frame, text, (x, y), font, scale, (0, 0, 0), thickness, line_type)


def random_color():
    return tuple(np.random.randint(0, 255, 3).tolist())


def rainbow_text(frame, text, position, font, scale, thickness, line_type):
    x, y = position
    for i, char in enumerate(text):
        char_color = random_color()
        cv2.putText(frame, char, (x + i * 15, y), font, scale, thickness, line_type)

def calculate_game_state(move):
    moves = ["Rock", "Paper", "Scissors"]
    wins = {"Rock": "Scissors", "Paper": "Rock", "Scissors": "Paper"}
    selected = random.randint(0, 2)
    print("Computer played " + moves[selected])

    if moves[selected] == move:
        return 0, moves[selected]

    if wins[move] == moves[selected]:
        return 1, moves[selected]

    return -1, moves[selected]

def get_finger_status(hands_module, hand_landmarks, finger_name):
    finger_id_map = {'INDEX': 8, 'MIDDLE': 12, 'RING': 16, 'PINKY': 20}

    finger_tip_y = hand_landmarks.landmark[finger_id_map[finger_name]].y
    finger_dip_y = hand_landmarks.landmark[finger_id_map[finger_name] - 1].y
    finger_mcp_y = hand_landmarks.landmark[finger_id_map[finger_name] - 2].y

    return finger_tip_y < finger_mcp_y

def get_thumb_status(hands_module, hand_landmarks):
    thumb_tip_x = hand_landmarks.landmark[hands_module.HandLandmark.THUMB_TIP].x
    thumb_mcp_x = hand_landmarks.landmark[hands_module.HandLandmark.THUMB_TIP - 2].x
    thumb_ip_x = hand_landmarks.landmark[hands_module.HandLandmark.THUMB_TIP - 1].x

    return thumb_tip_x > thumb_ip_x > thumb_mcp_x


def point_inside_rect(point, rect):
    x, y, w, h = rect
    return x <= point[0] <= x + w and y <= point[1] <= y + h


def draw_button(frame, button_rect, is_hover):
    x, y, w, h = button_rect
    button_color = (0, 255, 0) if is_hover else (0, 200, 0)
    cv2.rectangle(frame, (x, y), (x + w, y + h), button_color, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)
    cv2.putText(frame, "  START", (x + 10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

def get_finger_coordinates(hand_landmarks):
    finger_coordinates = []

    for finger in hand_landmarks.landmark:
        finger_coordinates.extend([finger.x, finger.y, finger.z])

    return finger_coordinates

def log_to_csv(move, outcome, finger_coordinates):
    csv_file_path = 'DATASET.csv'
    with open(csv_file_path, 'a', newline='') as file:
        fieldnames = ['timestamp', 'move', 'outcome', 'finger_coordinates']
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        writer.writerow({'timestamp': timestamp, 'move': move, 'outcome': outcome, 'finger_coordinates': finger_coordinates})

def start_video():
    drawingModule = mediapipe.solutions.drawing_utils
    hands_module = mediapipe.solutions.hands

    capture = cv2.VideoCapture(0)

    
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    start_time = 0.0
    timer_started = False
    time_left_now = 3
    hold_for_play = False
    draw_timer = 0.0
    game_over_text = ""
    computer_played = ""
    blink_rate = 2  

    
    frame_center_x = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) / 2)
    frame_center_y = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2)

    
    roi_width = 300
    roi_height = 300

    
    roi_x = frame_center_x - int(roi_width / 2)
    roi_y = frame_center_y - int(roi_height / 2)

    
    roi_rect = (roi_x, roi_y, roi_width, roi_height)

    
    button_rect = (10, 300, 120, 50)  

    
    user_score = 0
    computer_score = 0

    with hands_module.Hands(static_image_mode=False, min_detection_confidence=0.7,
                            min_tracking_confidence=0.4, max_num_hands=2) as hands:
        while True:

            if timer_started:
                now_time = time.time()
                time_elapsed = now_time - start_time
                if time_elapsed >= 1:
                    time_left_now -= 1
                    start_time = now_time
                    if time_left_now <= 0:
                        hold_for_play = True
                        timer_started = False

            ret, frame = capture.read()

            
            frame = cv2.flip(frame, 1)

            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            move = "UNKNOWN"
            finger_coordinates = []

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    if hold_for_play or time.time() - draw_timer <= 2:
                        drawingModule.draw_landmarks(frame, hand_landmarks, hands_module.HAND_CONNECTIONS)

                    
                    hand_rect = cv2.boundingRect(np.array([(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) for lm in hand_landmarks.landmark]))
                    if point_inside_rect((hand_rect[0], hand_rect[1]), roi_rect):
                        current_state = ""
                        thumb_status = get_thumb_status(hands_module, hand_landmarks)
                        current_state += "1" if thumb_status else "0"

                        index_status = get_finger_status(hands_module, hand_landmarks, 'INDEX')
                        current_state += "1" if index_status else "0"

                        middle_status = get_finger_status(hands_module, hand_landmarks, 'MIDDLE')
                        current_state += "1" if middle_status else "0"

                        ring_status = get_finger_status(hands_module, hand_landmarks, 'RING')
                        current_state += "1" if ring_status else "0"

                        pinky_status = get_finger_status(hands_module, hand_landmarks, 'PINKY')
                        current_state += "1" if pinky_status else "0"

                        if current_state == "00000":
                            move = "Rock"
                        elif current_state == "11111":
                            move = "Paper"
                        elif current_state == "01100":
                            move = "Scissors"
                        else:
                            move = "UNKNOWN"

                        finger_coordinates = get_finger_coordinates(hand_landmarks)

                        
                        cv2.putText(frame, move, (hand_rect[0], hand_rect[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

            if hold_for_play and move != "UNKNOWN":
                hold_for_play = False
                draw_timer = time.time()
                print("Player played " + move)
                won, cmp_move = calculate_game_state(move)
                computer_played = " You: " + move + " | Computer: " + cmp_move
                if won == 1:
                    game_over_text = "Victory!"
                    user_score += 1
                elif won == -1:
                    game_over_text = "Defeated"
                    computer_score += 1
                else:
                    game_over_text = "Tie"

                log_to_csv(move, game_over_text, finger_coordinates)

            font = cv2.FONT_HERSHEY_SIMPLEX

            if not hold_for_play and not timer_started:
                blinking_text(frame,
                             game_over_text + " " + computer_played,
                             (10, 400),
                             font, 0.8,
                             2,
                             cv2.LINE_4,
                             blink_rate)

            label_text = "Hover finger in start button"
            if hold_for_play:
                label_text = "Place your hand inside the box"
            elif timer_started:
                label_text = "PLAY STARTS IN " + str(time_left_now)

                
                cv2.putText(frame, str(time_left_now), (button_rect[0] + 50, button_rect[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

            
            x, y, w, h = roi_rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            draw_button(frame, button_rect, False)

            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    
                    hand_rect = cv2.boundingRect(np.array([(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) for lm in hand_landmarks.landmark]))
                    is_hover = point_inside_rect((hand_rect[0], hand_rect[1]), button_rect)

                    
                    draw_button(frame, button_rect, is_hover)

                    
                    if is_hover: 
                    
                        print("Start button pressed")
                        start_time = time.time()
                        timer_started = True
                        time_left_now = 3

            rainbow_text(frame,
                         label_text,
                         (10, 30),
                         font, 0.8,
                         2,
                         cv2.LINE_4)

            
            cv2.putText(frame, "Score: " + str(user_score) + " - " + str(computer_score), (frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

            cv2.imshow('IA Final Project', frame)

            if cv2.waitKey(1) == 27:
                break

    cv2.destroyAllWindows()
    capture.release()

if __name__ == "__main__":
    start_video()

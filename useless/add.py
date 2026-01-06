import cv2
import mediapipe as mp
import numpy as np
import random
import math
import time

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)
WIDTH, HEIGHT = 1280, 720
cap.set(3, WIDTH)
cap.set(4, HEIGHT)

# ---------------- MEDIAPIPE ----------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

# ---------------- LOAD IMAGES ----------------
APPLE_SIZE = 200
apple_img = cv2.imread("apple.png", cv2.IMREAD_UNCHANGED)

if apple_img is None:
    print("ERROR: apple.png not found")
    exit()

apple_img = cv2.resize(apple_img, (APPLE_SIZE, APPLE_SIZE))

# Load all basket images (0-5 apples)
basket_images = []
for i in range(6):  # basket0.png to basket5.png
    img = cv2.imread(f"basket{i}.png", cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"ERROR: basket{i}.png not found")
        exit()
    # Resize to consistent size
    basket_img = cv2.resize(img, (360, 240))
    basket_images.append(basket_img)

# ---------------- GAME VARIABLES ----------------
MAX_APPLES = 9

def new_question():
    a = random.randint(1, 3)
    b = random.randint(1, 2)
    return a, b, a + b

num1, num2, target_count = new_question()
dropped = 0
message = ""
message_time = 0

# Initialize apples
apples = []
start_x, start_y = 120, 120
gap = 120

for i in range(MAX_APPLES):
    row = i // 3
    col = i % 3
    apples.append({
        "x": start_x + col * gap,
        "y": start_y + row * gap,
        "picked": False
    })

# Basket position
basket = {"x": 480, "y": 430, "w": 360, "h": 240}
submit = {"x": 980, "y": 560, "w": 220, "h": 90}

picked_apple = None
pinch_active = False
submit_pinched = False
last_pinch_time = 0

PINCH_THRESHOLD = 35
RELEASE_THRESHOLD = 50
PINCH_DELAY = 0.3

# ---------------- FUNCTIONS ----------------
def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def inside(x, y, r):
    return r["x"] < x < r["x"] + r["w"] and r["y"] < y < r["y"] + r["h"]

def overlay(bg, fg, x, y):
    h, w = fg.shape[:2]
    if x < 0 or y < 0 or x + w > bg.shape[1] or y + h > bg.shape[0]:
        return bg

    if fg.shape[2] == 4:
        alpha = fg[:, :, 3] / 255.0
        for c in range(3):
            bg[y:y+h, x:x+w, c] = (
                alpha * fg[:, :, c] +
                (1 - alpha) * bg[y:y+h, x:x+w, c]
            )
    else:
        bg[y:y+h, x:x+w] = fg

    return bg

def draw_hand_info(img, hand_landmarks, h, w):
    """Draw all 21 hand landmarks with labels and connections"""
    # Draw hand landmarks and connections
    mp_drawing.draw_landmarks(
        img,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style())
    
    # Draw pinch distance between thumb and index
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    
    thumb_pos = (int(thumb_tip.x * w), int(thumb_tip.y * h))
    index_pos = (int(index_tip.x * w), int(index_tip.y * h))
    
    distance = dist(thumb_pos, index_pos)
    
    # Draw line between thumb and index
    cv2.line(img, thumb_pos, index_pos, (0, 255, 255), 2)
    
    # Draw distance text
    mid_x = (thumb_pos[0] + index_pos[0]) // 2
    mid_y = (thumb_pos[1] + index_pos[1]) // 2
    cv2.putText(img, f"Dist: {int(distance)}", 
               (mid_x, mid_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Draw pinch status
    status = "PINCHING" if distance < PINCH_THRESHOLD else "OPEN"
    status_color = (0, 255, 0) if distance < PINCH_THRESHOLD else (0, 200, 200)
    cv2.putText(img, f"Status: {status}", 
               (thumb_pos[0] + 20, thumb_pos[1] - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

# ---------------- MAIN LOOP ----------------
while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # Clear message after 1.5 seconds
    if message and time.time() - message_time > 1.5:
        message = ""

    # Draw apples not picked
    for a in apples:
        if not a["picked"]:
            img = overlay(img, apple_img, a["x"], a["y"])

    # Draw appropriate basket image based on dropped count
    # Use dropped as index (0-5), but ensure it doesn't exceed available images
    basket_index = min(dropped, 5)  # Maximum index is 5 for basket5.png
    current_basket_img = basket_images[basket_index]
    img = overlay(img, current_basket_img, basket["x"], basket["y"])

    # Draw submit button with 3D effect
    submit_color = (0, 180, 0)
    highlight_color = (0, 220, 0)
    shadow_color = (0, 140, 0)
    
    # Button shadow
    cv2.rectangle(img,
                  (submit["x"] + 5, submit["y"] + 5),
                  (submit["x"] + submit["w"] + 5, submit["y"] + submit["h"] + 5),
                  shadow_color, -1)
    
    # Button main
    cv2.rectangle(img,
                  (submit["x"], submit["y"]),
                  (submit["x"] + submit["w"], submit["y"] + submit["h"]),
                  submit_color, -1)
    
    # Button highlight
    cv2.rectangle(img,
                  (submit["x"], submit["y"]),
                  (submit["x"] + submit["w"], submit["y"] + 10),
                  highlight_color, -1)
    
    cv2.putText(img, "SUBMIT",
                (submit["x"] + 40, submit["y"] + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    # Draw question and stats with better styling
    # Background for stats
    cv2.rectangle(img, (40, 30), (400, 80), (255, 255, 255), -1)
    cv2.rectangle(img, (40, 30), (400, 80), (0, 0, 0), 2)
    
    cv2.putText(img, f"# Apples in basket: {dropped}/{target_count}", 
                (50, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    
    # Background for question
    cv2.rectangle(img, (950, 100), (1250, 180), (255, 255, 255), -1)
    cv2.rectangle(img, (950, 100), (1250, 180), (0, 0, 0), 2)
    
    cv2.putText(img, f"{num1} + {num2} = ?", 
                (1000, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

    # Draw message with background
    if message:
        # Message background
        text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        msg_x = (WIDTH - text_size[0]) // 2
        msg_y = 650
        
        cv2.rectangle(img, 
                     (msg_x - 20, msg_y - 40),
                     (msg_x + text_size[0] + 20, msg_y + 10),
                     (255, 255, 255), -1)
        cv2.rectangle(img, 
                     (msg_x - 20, msg_y - 40),
                     (msg_x + text_size[0] + 20, msg_y + 10),
                     (0, 0, 255), 2)
        
        cv2.putText(img, message, (msg_x, msg_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    # ---------------- HAND LANDMARK VISUALIZATION ----------------
    hand_detected = False
    index_pos = None
    thumb_pos = None
    
    if result.multi_hand_landmarks:
        hand_detected = True
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw detailed hand landmarks
            h, w, _ = img.shape
            draw_hand_info(img, hand_landmarks, h, w)
            
            # Get index and thumb positions for game logic
            index_pos = (int(hand_landmarks.landmark[8].x * w), 
                        int(hand_landmarks.landmark[8].y * h))
            thumb_pos = (int(hand_landmarks.landmark[4].x * w), 
                        int(hand_landmarks.landmark[4].y * h))

    # ---------------- HAND GESTURE LOGIC ----------------
    if hand_detected and index_pos is not None and thumb_pos is not None:
        d = dist(index_pos, thumb_pos)
        now = time.time()

        # Check if hand is over submit button
        hand_over_submit = inside(index_pos[0], index_pos[1], submit)

        # Pinch start
        if d < PINCH_THRESHOLD and not pinch_active:
            if now - last_pinch_time > PINCH_DELAY:
                pinch_active = True
                last_pinch_time = now
                
                # Check if pinching submit button
                if hand_over_submit:
                    submit_pinched = True
                    # Visual feedback for submit button pinch
                    cv2.rectangle(img,
                                  (submit["x"], submit["y"]),
                                  (submit["x"] + submit["w"], submit["y"] + submit["h"]),
                                  (0, 255, 0), 3)
                else:
                    # Check if pinching an apple
                    for a in apples:
                        if not a["picked"]:
                            # Check distance to apple center
                            apple_center_x = a["x"] + APPLE_SIZE // 2
                            apple_center_y = a["y"] + APPLE_SIZE // 2
                            if dist((apple_center_x, apple_center_y), index_pos) < APPLE_SIZE // 2:
                                picked_apple = a
                                # Visual feedback for apple pickup
                                cv2.circle(img, (apple_center_x, apple_center_y), 
                                          APPLE_SIZE // 2, (0, 255, 0), 2)
                                break

        # Move apple if pinching one
        if pinch_active and picked_apple:
            picked_apple["x"] = index_pos[0] - APPLE_SIZE // 2
            picked_apple["y"] = index_pos[1] - APPLE_SIZE // 2

        # Release pinch
        if d > RELEASE_THRESHOLD and pinch_active:
            pinch_active = False
            
            if picked_apple:
                # Check if apple is dropped in basket
                apple_center_x = picked_apple["x"] + APPLE_SIZE // 2
                apple_center_y = picked_apple["y"] + APPLE_SIZE // 2
                
                if inside(apple_center_x, apple_center_y, basket):
                    picked_apple["picked"] = True
                    dropped += 1
                picked_apple = None
            
            # Submit if submit button was pinched and released
            if submit_pinched:
                if dropped == target_count:
                    message = "CORRECT ✅"
                else:
                    message = f"TRY AGAIN ({dropped}/{target_count})"
                message_time = time.time()
                
                # Reset game after delay
                time.sleep(0.5)
                dropped = 0
                num1, num2, target_count = new_question()
                
                for i, a in enumerate(apples):
                    a["picked"] = False
                    row = i // 3
                    col = i % 3
                    a["x"] = start_x + col * gap
                    a["y"] = start_y + row * gap
            
            submit_pinched = False

    cv2.imshow("Gesture-Based Addition Learning", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
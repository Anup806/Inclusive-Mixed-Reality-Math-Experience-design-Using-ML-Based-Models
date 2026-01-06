'''import cv2
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
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

# ---------------- LOAD IMAGES ----------------
APPLE_SIZE = 150  # Increased apple size
apple_img = cv2.imread("apple.png", cv2.IMREAD_UNCHANGED)

if apple_img is None:
    # Create a simple red apple if image not found
    apple_img = np.zeros((APPLE_SIZE, APPLE_SIZE, 4), dtype=np.uint8)
    cv2.circle(apple_img, (APPLE_SIZE//2, APPLE_SIZE//2), APPLE_SIZE//2 - 15, (0, 0, 255, 255), -1)
    cv2.circle(apple_img, (APPLE_SIZE//2, APPLE_SIZE//2), APPLE_SIZE//2 - 20, (0, 100, 255, 255), -1)
    # Add stem
    cv2.rectangle(apple_img, (APPLE_SIZE//2-5, 10), (APPLE_SIZE//2+5, 25), (101, 67, 33, 255), -1)
    # Add leaf
    leaf_points = np.array([[APPLE_SIZE//2+15, 20], [APPLE_SIZE//2+40, 15], [APPLE_SIZE//2+35, 40]], np.int32)
    cv2.fillPoly(apple_img, [leaf_points], (34, 139, 34, 255))
    print("Created placeholder apple image")

apple_img = cv2.resize(apple_img, (APPLE_SIZE, APPLE_SIZE))

# Load basket images
basket_images = []
for i in range(10):
    try:
        img = cv2.imread(f"basket{i}.png", cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError
        basket_img = cv2.resize(img, (350, 250))
        basket_images.append(basket_img)
    except:
        # Create placeholder basket
        basket_img = np.zeros((250, 350, 4), dtype=np.uint8)
        cv2.rectangle(basket_img, (10, 50), (340, 240), (139, 69, 19, 255), -1)
        cv2.rectangle(basket_img, (10, 50), (340, 240), (101, 50, 13, 255), 5)
        # Add handle
        cv2.ellipse(basket_img, (175, 40), (100, 30), 0, 0, 180, (101, 50, 13, 255), 8)
        basket_images.append(basket_img)

# ---------------- GAME VARIABLES ----------------
MAX_APPLES = 9

def generate_new_number():
    """Generate a random number between 1 and 9"""
    return random.randint(1, 9)

target_number = generate_new_number()
apples_in_basket = 0
message = ""
message_time = 0

# Initialize apples - ALL 9 APPLES ON LEFT SIDE
apples = []
start_x, start_y = 100, 150  # Moved to left side
vertical_gap = 25
horizontal_gap = 20

# Arrange apples in 3x3 grid on left side
for i in range(MAX_APPLES):
    row = i // 3  # 3 apples per row
    col = i % 3
    apples.append({
        "x": start_x + col * (APPLE_SIZE + horizontal_gap),
        "y": start_y + row * (APPLE_SIZE + vertical_gap),
        "picked": False,
        "visible": True,
        "original_x": start_x + col * (APPLE_SIZE + horizontal_gap),
        "original_y": start_y + row * (APPLE_SIZE + vertical_gap)
    })

# Basket position (centered on right side)
basket = {"x": 800, "y": 200, "w": 350, "h": 250}
submit = {"x": 850, "y": 500, "w": 220, "h": 80}

picked_apple = None
pinch_active = False
submit_pinched = False
last_pinch_time = 0

PINCH_THRESHOLD = 35
RELEASE_THRESHOLD = 55
PINCH_STABILITY = 2

pinch_frames = 0
release_frames = 0

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

def draw_hand_with_feedback(img, hand_landmarks, h, w, distance, is_pinching):
    """Draw hand landmarks with pinch feedback"""
    # Draw all hand landmarks
    mp_drawing.draw_landmarks(
        img,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style()
    )
    
    # Get thumb and index positions
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    
    thumb_pos = (int(thumb_tip.x * w), int(thumb_tip.y * h))
    index_pos = (int(index_tip.x * w), int(index_tip.y * h))
    
    # Draw pinch points
    circle_size = 15
    circle_color = (0, 255, 0) if is_pinching else (0, 200, 255)
    line_color = (0, 255, 0) if is_pinching else (0, 200, 255)
    line_width = 5 if is_pinching else 3
    
    cv2.circle(img, thumb_pos, circle_size, circle_color, -1)
    cv2.circle(img, index_pos, circle_size, circle_color, -1)
    cv2.line(img, thumb_pos, index_pos, line_color, line_width)

# ---------------- MAIN LOOP ----------------
while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # Clear message after 2 seconds
    if message and time.time() - message_time > 2:
        message = ""

    # Create a clean background
    background = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 240
    img = cv2.addWeighted(img, 0.3, background, 0.7, 0)

    # Draw title area at top
    cv2.rectangle(img, (0, 0), (WIDTH, 100), (255, 255, 255), -1)
    cv2.rectangle(img, (0, 0), (WIDTH, 100), (0, 150, 255), 2)
    
    # Draw instruction title
    cv2.putText(img, "LEARN NUMBERS 1-9", 
                (WIDTH//2 - 200, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 100, 200), 3)
    
    cv2.putText(img, f"Pick {target_number} apples and put them in the basket", 
                (WIDTH//2 - 250, 85), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 200), 2)

    # Draw divider line
    cv2.line(img, (WIDTH//2, 100), (WIDTH//2, HEIGHT-50), (200, 200, 200), 2)

    # Draw ALL 9 apples on LEFT SIDE (no rectangles, just apples)
    for i, a in enumerate(apples):
        if a["visible"]:
            img = overlay(img, apple_img, int(a["x"]), int(a["y"]))
            # No rectangles drawn around apples

    # Draw basket area on RIGHT SIDE
    cv2.rectangle(img, (750, 150), (WIDTH-50, 550), (245, 245, 245), -1)
    cv2.rectangle(img, (750, 150), (WIDTH-50, 550), (0, 150, 255), 2)
    
    # Draw basket title
    cv2.putText(img, "BASKET", 
                (basket["x"] + 120, basket["y"] - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 100, 0), 3)
    
    # Draw basket image
    basket_index = min(apples_in_basket, 9)
    current_basket_img = basket_images[basket_index]
    img = overlay(img, current_basket_img, basket["x"], basket["y"])

    # Draw basket count
    count_text = f"Apples: {apples_in_basket}/{target_number}"
    count_size = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
    count_x = basket["x"] + (basket["w"] - count_size[0]) // 2
    
    cv2.putText(img, count_text, 
                (count_x, basket["y"] + basket["h"] + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 100, 0), 2)

    # Draw submit button
    submit_color = (0, 180, 0) if apples_in_basket == target_number else (100, 100, 100)
    
    cv2.rectangle(img,
                  (submit["x"], submit["y"]),
                  (submit["x"] + submit["w"], submit["y"] + submit["h"]),
                  submit_color, -1)
    cv2.rectangle(img,
                  (submit["x"], submit["y"]),
                  (submit["x"] + submit["w"], submit["y"] + submit["h"]),
                  (255, 255, 255), 3)
    
    check_text = "CHECK"
    check_size = cv2.getTextSize(check_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
    check_x = submit["x"] + (submit["w"] - check_size[0]) // 2
    check_y = submit["y"] + (submit["h"] + check_size[1]) // 2
    
    cv2.putText(img, check_text,
                (check_x, check_y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    # Draw target number display on LEFT SIDE (above apples)
    cv2.rectangle(img, (50, 50), (250, 120), (255, 255, 255), -1)
    cv2.rectangle(img, (50, 50), (250, 120), (0, 150, 255), 3)
    
    cv2.putText(img, "TARGET:", 
                (70, 85), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)
    
    target_text = str(target_number)
    target_size = cv2.getTextSize(target_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
    target_x = 150 - target_size[0] // 2
    
    cv2.putText(img, target_text, 
                (target_x, 115), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 150, 255), 3)

    # Draw message with background
    if message:
        text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        msg_x = (WIDTH - text_size[0]) // 2
        msg_y = 600
        
        cv2.rectangle(img, 
                     (msg_x - 20, msg_y - 40),
                     (msg_x + text_size[0] + 20, msg_y + 10),
                     (255, 255, 255), -1)
        cv2.rectangle(img, 
                     (msg_x - 20, msg_y - 40),
                     (msg_x + text_size[0] + 20, msg_y + 10),
                     (0, 150, 255), 2)
        
        cv2.putText(img, message, (msg_x, msg_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 150, 255), 3)

    # Draw instructions at bottom
    cv2.rectangle(img, (0, HEIGHT-50), (WIDTH, HEIGHT), (255, 255, 255), -1)
    cv2.putText(img, "INSTRUCTIONS: Pinch apples with thumb and index finger. Drop in basket. When done, press CHECK.", 
                (100, HEIGHT-15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)

    # ---------------- HAND GESTURE LOGIC ----------------
    hand_detected = False
    index_pos = None
    thumb_pos = None
    pinch_distance = 0
    is_pinching_current = False
    
    if result.multi_hand_landmarks:
        hand_detected = True
        for hand_landmarks in result.multi_hand_landmarks:
            h, w, _ = img.shape
            
            # Get thumb and index positions
            index_tip = hand_landmarks.landmark[8]
            thumb_tip = hand_landmarks.landmark[4]
            
            index_pos = (int(index_tip.x * w), int(index_tip.y * h))
            thumb_pos = (int(thumb_tip.x * w), int(thumb_tip.y * h))
            
            # Calculate pinch distance
            pinch_distance = dist(index_pos, thumb_pos)
            
            # Determine if currently pinching
            if pinch_distance < PINCH_THRESHOLD:
                pinch_frames = min(pinch_frames + 1, PINCH_STABILITY)
                release_frames = 0
            else:
                release_frames = min(release_frames + 1, PINCH_STABILITY)
                pinch_frames = 0
            
            is_pinching_current = pinch_frames >= PINCH_STABILITY
            
            # Draw hand with feedback
            draw_hand_with_feedback(img, hand_landmarks, h, w, pinch_distance, is_pinching_current)
    
    else:
        pinch_frames = 0
        release_frames = 0
        if pinch_active:
            pinch_active = False
            if picked_apple:
                # Return apple to original position
                picked_apple["x"] = picked_apple["original_x"]
                picked_apple["y"] = picked_apple["original_y"]
                picked_apple = None

    # ---------------- PINCH LOGIC ----------------
    now = time.time()
    
    if hand_detected and index_pos is not None and thumb_pos is not None:
        # Check if hand is over submit button
        hand_over_submit = inside(index_pos[0], index_pos[1], submit)
        
        # Pinch start
        if is_pinching_current and not pinch_active:
            if now - last_pinch_time > 0.1:
                pinch_active = True
                last_pinch_time = now
                
                # Check if pinching submit button
                if hand_over_submit and apples_in_basket == target_number:
                    submit_pinched = True
                    # Visual feedback for submit button
                    cv2.rectangle(img,
                                (submit["x"] - 5, submit["y"] - 5),
                                (submit["x"] + submit["w"] + 5, submit["y"] + submit["h"] + 5),
                                (0, 255, 0), 5)
                else:
                    # Check if pinching a visible, unpicked apple
                    for a in apples:
                        if a["visible"] and not a["picked"]:
                            # Check if index finger is within apple bounds
                            apple_left = a["x"]
                            apple_right = a["x"] + APPLE_SIZE
                            apple_top = a["y"]
                            apple_bottom = a["y"] + APPLE_SIZE
                            
                            if (apple_left < index_pos[0] < apple_right and 
                                apple_top < index_pos[1] < apple_bottom):
                                picked_apple = a
                                
                                # Show feedback on the apple being picked
                                cv2.circle(img, 
                                          (a["x"] + APPLE_SIZE//2, a["y"] + APPLE_SIZE//2),
                                          APPLE_SIZE//2 + 5, (0, 255, 0), 3)
                                break
        
        # Move apple if pinching one
        if pinch_active and picked_apple:
            # Move apple to index finger position
            picked_apple["x"] = index_pos[0] - APPLE_SIZE // 2
            picked_apple["y"] = index_pos[1] - APPLE_SIZE // 2
            
            # Draw "carrying" indicator
            cv2.putText(img, "Carrying Apple", 
                       (index_pos[0] + 30, index_pos[1] - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 150, 0), 2)
            
            # Draw arrow to basket
            cv2.arrowedLine(img, 
                           (index_pos[0], index_pos[1]),
                           (basket["x"] + basket["w"]//2, basket["y"] + basket["h"]//2),
                           (0, 150, 0), 2, tipLength=0.05)
        
        # Release pinch
        if not is_pinching_current and pinch_active:
            if release_frames >= PINCH_STABILITY:
                pinch_active = False
                
                if picked_apple:
                    # Check if apple is dropped in basket
                    apple_center_x = picked_apple["x"] + APPLE_SIZE // 2
                    apple_center_y = picked_apple["y"] + APPLE_SIZE // 2
                    
                    if inside(apple_center_x, apple_center_y, basket):
                        picked_apple["picked"] = True
                        picked_apple["visible"] = False
                        apples_in_basket += 1
                        
                        # Success feedback
                        cv2.putText(img, "Good!", 
                                   (basket["x"] + basket["w"]//2 - 40, basket["y"] - 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 0), 3)
                    else:
                        # Return apple to original position
                        picked_apple["x"] = picked_apple["original_x"]
                        picked_apple["y"] = picked_apple["original_y"]
                    
                    picked_apple = None
                
                # Submit if submit button was pinched and released
                if submit_pinched:
                    if apples_in_basket == target_number:
                        message = f"EXCELLENT! {target_number} IS CORRECT! ✅"
                        message_time = time.time()
                        
                        # Reset for next number after delay
                        time.sleep(1.5)
                        target_number = generate_new_number()
                        apples_in_basket = 0
                        
                        # Reset all apples
                        for a in apples:
                            a["x"] = a["original_x"]
                            a["y"] = a["original_y"]
                            a["picked"] = False
                            a["visible"] = True
                    
                    submit_pinched = False

    # Draw hand detection and pinch status at bottom
    hand_status = "HAND: DETECTED" if hand_detected else "HAND: NOT DETECTED"
    hand_color = (0, 255, 0) if hand_detected else (0, 0, 255)
    
    pinch_status = "PINCH: ACTIVE" if pinch_active else "PINCH: READY"
    pinch_color = (0, 255, 0) if pinch_active else (0, 200, 0)
    
    cv2.putText(img, hand_status, 
                (50, HEIGHT - 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, hand_color, 2)
    
    cv2.putText(img, pinch_status, 
                (50, HEIGHT - 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, pinch_color, 2)

    cv2.imshow("Learn Numbers 1-9 with Apples", img)
    
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or cv2.getWindowProperty("Learn Numbers 1-9 with Apples", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()'''





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
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ---------------- LOAD IMAGES ----------------
APPLE_SIZE = 150
apple_img = cv2.imread("apple.png", cv2.IMREAD_UNCHANGED)

if apple_img is None:
    # Create a simple red apple if image not found
    apple_img = np.zeros((APPLE_SIZE, APPLE_SIZE, 4), dtype=np.uint8)
    cv2.circle(apple_img, (APPLE_SIZE//2, APPLE_SIZE//2), APPLE_SIZE//2 - 15, (0, 0, 255, 255), -1)
    cv2.circle(apple_img, (APPLE_SIZE//2, APPLE_SIZE//2), APPLE_SIZE//2 - 20, (0, 100, 255, 255), -1)
    cv2.rectangle(apple_img, (APPLE_SIZE//2-5, 10), (APPLE_SIZE//2+5, 25), (101, 67, 33, 255), -1)
    leaf_points = np.array([[APPLE_SIZE//2+15, 20], [APPLE_SIZE//2+40, 15], [APPLE_SIZE//2+35, 40]], np.int32)
    cv2.fillPoly(apple_img, [leaf_points], (34, 139, 34, 255))
    print("Created placeholder apple image")

apple_img = cv2.resize(apple_img, (APPLE_SIZE, APPLE_SIZE))

# Load basket images
basket_images = []
for i in range(10):
    try:
        img = cv2.imread(f"basket{i}.png", cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError
        basket_img = cv2.resize(img, (350, 250))
        basket_images.append(basket_img)
    except:
        basket_img = np.zeros((250, 350, 4), dtype=np.uint8)
        cv2.rectangle(basket_img, (10, 50), (340, 240), (139, 69, 19, 255), -1)
        cv2.rectangle(basket_img, (10, 50), (340, 240), (101, 50, 13, 255), 5)
        cv2.ellipse(basket_img, (175, 40), (100, 30), 0, 0, 180, (101, 50, 13, 255), 8)
        basket_images.append(basket_img)

# ---------------- GAME VARIABLES ----------------
MAX_APPLES = 9

def generate_new_number():
    return random.randint(1, 9)

target_number = generate_new_number()
apples_in_basket = 0
message = ""
message_time = 0

# Initialize apples on left side
apples = []
start_x, start_y = 100, 150
vertical_gap = 25
horizontal_gap = 20

for i in range(MAX_APPLES):
    row = i // 3
    col = i % 3
    apples.append({
        "x": start_x + col * (APPLE_SIZE + horizontal_gap),
        "y": start_y + row * (APPLE_SIZE + vertical_gap),
        "picked": False,
        "visible": True,
        "original_x": start_x + col * (APPLE_SIZE + horizontal_gap),
        "original_y": start_y + row * (APPLE_SIZE + vertical_gap)
    })

# UI positions
basket = {"x": 800, "y": 200, "w": 350, "h": 250}
submit = {"x": 850, "y": 500, "w": 220, "h": 80}

# Pinch detection variables
picked_apple = None
pinch_active = False
submit_pinched = False
last_pinch_time = 0

# IMPROVED THRESHOLDS
PINCH_THRESHOLD = 40  # Distance to activate pinch
RELEASE_THRESHOLD = 60  # Distance to release pinch
PINCH_STABILITY = 3  # Frames to confirm pinch

pinch_counter = 0
release_counter = 0
was_pinching = False

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

def draw_hand_with_feedback(img, hand_landmarks, h, w, distance, is_pinching):
    """Draw hand landmarks with pinch feedback"""
    mp_drawing.draw_landmarks(
        img,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style()
    )
    
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    
    thumb_pos = (int(thumb_tip.x * w), int(thumb_tip.y * h))
    index_pos = (int(index_tip.x * w), int(index_tip.y * h))
    
    # Visual feedback based on pinch state
    if is_pinching:
        circle_color = (0, 255, 0)
        line_color = (0, 255, 0)
        circle_size = 18
        line_width = 6
    elif distance < RELEASE_THRESHOLD:
        # Yellow when close but not pinching
        circle_color = (0, 255, 255)
        line_color = (0, 255, 255)
        circle_size = 15
        line_width = 4
    else:
        circle_color = (255, 150, 0)
        line_color = (255, 150, 0)
        circle_size = 12
        line_width = 3
    
    cv2.circle(img, thumb_pos, circle_size, circle_color, -1)
    cv2.circle(img, index_pos, circle_size, circle_color, -1)
    cv2.line(img, thumb_pos, index_pos, line_color, line_width)
    
    # Show distance for debugging
    mid_point = ((thumb_pos[0] + index_pos[0])//2, (thumb_pos[1] + index_pos[1])//2)
    cv2.putText(img, f"{int(distance)}px", 
                (mid_point[0] + 20, mid_point[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, circle_color, 2)

# ---------------- MAIN LOOP ----------------
while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # Clear message after 2 seconds
    if message and time.time() - message_time > 2:
        message = ""

    # Create background
    background = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 240
    img = cv2.addWeighted(img, 0.3, background, 0.7, 0)

    # Draw title area
    cv2.rectangle(img, (0, 0), (WIDTH, 100), (255, 255, 255), -1)
    cv2.rectangle(img, (0, 0), (WIDTH, 100), (0, 150, 255), 2)
    
    cv2.putText(img, "LEARN NUMBERS 1-9", 
                (WIDTH//2 - 200, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 100, 200), 3)
    
    cv2.putText(img, f"Pick {target_number} apples and put them in the basket", 
                (WIDTH//2 - 250, 85), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 200), 2)

    # Draw divider
    cv2.line(img, (WIDTH//2, 100), (WIDTH//2, HEIGHT-50), (200, 200, 200), 2)

    # Draw all apples
    for a in apples:
        if a["visible"]:
            img = overlay(img, apple_img, int(a["x"]), int(a["y"]))

    # Draw basket area
    cv2.rectangle(img, (750, 150), (WIDTH-50, 550), (245, 245, 245), -1)
    cv2.rectangle(img, (750, 150), (WIDTH-50, 550), (0, 150, 255), 2)
    
    cv2.putText(img, "BASKET", 
                (basket["x"] + 120, basket["y"] - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 100, 0), 3)
    
    basket_index = min(apples_in_basket, 9)
    current_basket_img = basket_images[basket_index]
    img = overlay(img, current_basket_img, basket["x"], basket["y"])

    # Draw basket count
    count_text = f"Apples: {apples_in_basket}/{target_number}"
    count_size = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
    count_x = basket["x"] + (basket["w"] - count_size[0]) // 2
    
    cv2.putText(img, count_text, 
                (count_x, basket["y"] + basket["h"] + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 100, 0), 2)

    # Draw submit button
    submit_color = (0, 180, 0) if apples_in_basket == target_number else (100, 100, 100)
    
    cv2.rectangle(img,
                  (submit["x"], submit["y"]),
                  (submit["x"] + submit["w"], submit["y"] + submit["h"]),
                  submit_color, -1)
    cv2.rectangle(img,
                  (submit["x"], submit["y"]),
                  (submit["x"] + submit["w"], submit["y"] + submit["h"]),
                  (255, 255, 255), 3)
    
    check_text = "CHECK"
    check_size = cv2.getTextSize(check_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
    check_x = submit["x"] + (submit["w"] - check_size[0]) // 2
    check_y = submit["y"] + (submit["h"] + check_size[1]) // 2
    
    cv2.putText(img, check_text,
                (check_x, check_y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    # Draw target number
    cv2.rectangle(img, (50, 50), (250, 120), (255, 255, 255), -1)
    cv2.rectangle(img, (50, 50), (250, 120), (0, 150, 255), 3)
    
    cv2.putText(img, "TARGET:", 
                (70, 85), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)
    
    target_text = str(target_number)
    target_size = cv2.getTextSize(target_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
    target_x = 150 - target_size[0] // 2
    
    cv2.putText(img, target_text, 
                (target_x, 115), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 150, 255), 3)

    # Draw message
    if message:
        text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        msg_x = (WIDTH - text_size[0]) // 2
        msg_y = 600
        
        cv2.rectangle(img, 
                     (msg_x - 20, msg_y - 40),
                     (msg_x + text_size[0] + 20, msg_y + 10),
                     (255, 255, 255), -1)
        cv2.rectangle(img, 
                     (msg_x - 20, msg_y - 40),
                     (msg_x + text_size[0] + 20, msg_y + 10),
                     (0, 150, 255), 2)
        
        cv2.putText(img, message, (msg_x, msg_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 150, 255), 3)

    # Draw instructions
    cv2.rectangle(img, (0, HEIGHT-50), (WIDTH, HEIGHT), (255, 255, 255), -1)
    cv2.putText(img, "INSTRUCTIONS: Pinch apples with thumb and index finger. Drop in basket. When done, press CHECK.", 
                (100, HEIGHT-15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)

    # ---------------- IMPROVED HAND GESTURE LOGIC ----------------
    hand_detected = False
    index_pos = None
    thumb_pos = None
    pinch_distance = 0
    current_pinch_detected = False
    
    if result.multi_hand_landmarks:
        hand_detected = True
        for hand_landmarks in result.multi_hand_landmarks:
            h, w, _ = img.shape
            
            index_tip = hand_landmarks.landmark[8]
            thumb_tip = hand_landmarks.landmark[4]
            
            index_pos = (int(index_tip.x * w), int(index_tip.y * h))
            thumb_pos = (int(thumb_tip.x * w), int(thumb_tip.y * h))
            
            pinch_distance = dist(index_pos, thumb_pos)
            
            # IMPROVED PINCH DETECTION with hysteresis
            if pinch_distance < PINCH_THRESHOLD:
                pinch_counter = min(pinch_counter + 1, PINCH_STABILITY + 1)
                release_counter = 0
            elif pinch_distance > RELEASE_THRESHOLD:
                release_counter = min(release_counter + 1, PINCH_STABILITY + 1)
                pinch_counter = 0
            # In between thresholds - maintain current state
            
            current_pinch_detected = pinch_counter >= PINCH_STABILITY
            
            draw_hand_with_feedback(img, hand_landmarks, h, w, pinch_distance, current_pinch_detected)
    else:
        # Hand lost - gradually reset counters
        if pinch_counter > 0:
            pinch_counter = max(0, pinch_counter - 1)
        if release_counter > 0:
            release_counter = max(0, release_counter - 1)
        
        # If hand disappears while holding apple, keep it held briefly
        if not pinch_active and picked_apple:
            picked_apple["x"] = picked_apple["original_x"]
            picked_apple["y"] = picked_apple["original_y"]
            picked_apple = None

    # ---------------- PINCH STATE MACHINE ----------------
    now = time.time()
    
    if hand_detected and index_pos is not None:
        hand_over_submit = inside(index_pos[0], index_pos[1], submit)
        
        # PINCH START - transition from not pinching to pinching
        if current_pinch_detected and not was_pinching:
            if now - last_pinch_time > 0.2:  # Debounce
                pinch_active = True
                was_pinching = True
                last_pinch_time = now
                
                # Check submit button
                if hand_over_submit and apples_in_basket == target_number:
                    submit_pinched = True
                else:
                    # Check for apple
                    for a in apples:
                        if a["visible"] and not a["picked"]:
                            if (a["x"] < index_pos[0] < a["x"] + APPLE_SIZE and 
                                a["y"] < index_pos[1] < a["y"] + APPLE_SIZE):
                                picked_apple = a
                                break
        
        # PINCH HOLD - currently pinching
        elif current_pinch_detected and was_pinching:
            if picked_apple:
                picked_apple["x"] = index_pos[0] - APPLE_SIZE // 2
                picked_apple["y"] = index_pos[1] - APPLE_SIZE // 2
                
                cv2.putText(img, "Carrying Apple", 
                           (index_pos[0] + 30, index_pos[1] - 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                cv2.arrowedLine(img, 
                               (index_pos[0], index_pos[1]),
                               (basket["x"] + basket["w"]//2, basket["y"] + basket["h"]//2),
                               (0, 255, 0), 2, tipLength=0.05)
        
        # PINCH RELEASE - transition from pinching to not pinching
        elif not current_pinch_detected and was_pinching:
            if release_counter >= PINCH_STABILITY:
                pinch_active = False
                was_pinching = False
                
                if picked_apple:
                    apple_center_x = picked_apple["x"] + APPLE_SIZE // 2
                    apple_center_y = picked_apple["y"] + APPLE_SIZE // 2
                    
                    if inside(apple_center_x, apple_center_y, basket):
                        picked_apple["picked"] = True
                        picked_apple["visible"] = False
                        apples_in_basket += 1
                        
                        cv2.circle(img, 
                                  (basket["x"] + basket["w"]//2, basket["y"] + basket["h"]//2),
                                  80, (0, 255, 0), 5)
                    else:
                        picked_apple["x"] = picked_apple["original_x"]
                        picked_apple["y"] = picked_apple["original_y"]
                    
                    picked_apple = None
                
                if submit_pinched:
                    if apples_in_basket == target_number:
                        message = f"EXCELLENT! {target_number} IS CORRECT! ✅"
                        message_time = time.time()
                        
                        time.sleep(1.5)
                        target_number = generate_new_number()
                        apples_in_basket = 0
                        
                        for a in apples:
                            a["x"] = a["original_x"]
                            a["y"] = a["original_y"]
                            a["picked"] = False
                            a["visible"] = True
                    
                    submit_pinched = False

    # Status display
    hand_status = "HAND: DETECTED ✓" if hand_detected else "HAND: SEARCHING..."
    hand_color = (0, 255, 0) if hand_detected else (0, 0, 255)
    
    pinch_status_text = "PINCH: ACTIVE ✓" if was_pinching else f"PINCH: READY ({pinch_counter}/{PINCH_STABILITY})"
    pinch_color = (0, 255, 0) if was_pinching else (100, 200, 100)
    
    cv2.putText(img, hand_status, 
                (20, HEIGHT - 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, hand_color, 2)
    
    cv2.putText(img, pinch_status_text, 
                (20, HEIGHT - 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, pinch_color, 2)
    
    if hand_detected:
        cv2.putText(img, f"Distance: {int(pinch_distance)}px", 
                    (WIDTH - 250, HEIGHT - 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)

    cv2.imshow("Learn Numbers 1-9 with Apples", img)
    
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or cv2.getWindowProperty("Learn Numbers 1-9 with Apples", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
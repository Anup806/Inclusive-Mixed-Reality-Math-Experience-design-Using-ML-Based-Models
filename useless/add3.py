import cv2
import mediapipe as mp
import numpy as np
import random
import math
import time

try:
    import pygame
except Exception:
    print("Error: pygame is not installed in this Python environment.")
    print("Install it with:")
    print("    python -m pip install pygame")
    print("If you use a virtualenv/conda, activate it first.")
    sys.exit(1)

# Try to import TensorFlow; if not available, disable DKT gracefully
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, Input
    DKT_AVAILABLE = True
except Exception:
    DKT_AVAILABLE = False


pygame.init()
# Get desktop size
try:
    WIDTH, HEIGHT = pygame.display.get_desktop_sizes()[0]
except Exception:
    info = pygame.display.Info()
    WIDTH, HEIGHT = info.current_w, info.current_h

# Set up fullscreen display
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN | pygame.SCALED)
pygame.display.set_caption("Color Learning – Pinch Pop Balloon Game")
pygame.event.set_blocked(pygame.MOUSEMOTION)
pygame.mouse.set_visible(True)


# Font setup
font = pygame.font.Font(None, 60)
large_font = pygame.font.Font(None, 100)
small_font = pygame.font.Font(None, 36)
clock = pygame.time.Clock()

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Camera setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera. Check camera connection.")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)


# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)
# Get full screen dimensions
WIDTH, HEIGHT = 1920, 1080  # Common full-screen resolution
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

# ---------------- START SCREEN VARIABLES ----------------
class StartScreen:
    def __init__(self):
        self.active = True
        self.current_field = 0  # 0: Name, 1: Age, 2: Grade
        self.fields = {
            "name": "",
            "age": "",
            "grade": ""
        }
        self.field_rects = []
        self.submit_rect = None
        self.cursor_visible = True
        self.cursor_timer = time.time()
        self.user_data = None

start_screen = StartScreen()

# ---------------- GAME VARIABLES ----------------
def new_question():
    a = random.randint(1, 3)
    b = random.randint(1, 2)
    return a, b, a + b

num1, num2, target_count = new_question()
dropped = 0
message = ""
message_time = 0

# Initialize with single apple in spawn area
spawn_x, spawn_y = 120, 300  # Spawn position for new apples
current_apple = {
    "x": spawn_x,
    "y": spawn_y,
    "picked": False,
    "dragged": False  # Track if apple has been dragged at least once
}

# List to hold all apples in play
apples = [current_apple]

# Basket position (scaled for full screen)
basket = {"x": int(480 * (WIDTH/1280)), "y": int(430 * (HEIGHT/720)), "w": int(360 * (WIDTH/1280)), "h": int(240 * (HEIGHT/720))}
submit = {"x": int(980 * (WIDTH/1280)), "y": int(560 * (HEIGHT/720)), "w": int(220 * (WIDTH/1280)), "h": int(90 * (HEIGHT/720))}

# Apple size scaling for full screen
APPLE_SIZE = int(200 * (WIDTH/1280))

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

def spawn_new_apple():
    """Create a new apple at spawn position"""
    return {
        "x": int(120 * (WIDTH/1280)),
        "y": int(300 * (HEIGHT/720)),
        "picked": False,
        "dragged": False
    }

def draw_start_screen(img):
    """Draw the start screen with input fields"""
    h, w = img.shape[:2]
    
    # Draw semi-transparent background
    overlay_bg = img.copy()
    cv2.rectangle(overlay_bg, (0, 0), (w, h), (50, 50, 100), -1)
    img = cv2.addWeighted(overlay_bg, 0.7, img, 0.3, 0)
    
    # Calculate scaling factors
    scale_w = w / 1920
    scale_h = h / 1080
    scale = min(scale_w, scale_h)
    
    # Title
    title = "Color Learning Game"
    title_font_scale = 2.5 * scale
    title_thickness = int(5 * scale)
    title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, title_font_scale, title_thickness)[0]
    title_x = (w - title_size[0]) // 2
    cv2.putText(img, title, (title_x, int(120 * scale_h)), 
                cv2.FONT_HERSHEY_DUPLEX, title_font_scale, (255, 255, 255), title_thickness)
    
    # Subtitle
    subtitle = "Enter your details:"
    subtitle_font_scale = 1.5 * scale
    subtitle_thickness = int(3 * scale)
    subtitle_size = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, subtitle_font_scale, subtitle_thickness)[0]
    subtitle_x = (w - subtitle_size[0]) // 2
    cv2.putText(img, subtitle, (subtitle_x, int(200 * scale_h)), 
                cv2.FONT_HERSHEY_SIMPLEX, subtitle_font_scale, (200, 200, 255), subtitle_thickness)
    
    # Field labels and input boxes
    field_labels = ["Name:", "Age:", "Grade/Level (any text):"]
    examples = ["", "", "Examples: K 1st 2nd Grade 3 Pre-K Preschool etc."]
    field_values = list(start_screen.fields.values())
    
    # Calculate positions
    start_y = int(280 * scale_h)
    field_height = int(70 * scale_h)
    spacing = int(20 * scale_h)
    
    # Clear field rects
    start_screen.field_rects = []
    
    for i in range(3):
        y = start_y + i * (field_height + spacing)
        
        # Field label
        label_x = int(200 * scale_w)
        cv2.putText(img, field_labels[i], (label_x, y + int(45 * scale_h)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2 * scale, (255, 255, 200), int(2 * scale))
        
        # Input box
        box_x = int(450 * scale_w)
        box_width = int(600 * scale_w)
        box_height = int(50 * scale_h)
        
        # Draw input box with highlight if active
        box_color = (100, 100, 200) if i == start_screen.current_field else (80, 80, 180)
        cv2.rectangle(img, (box_x, y), (box_x + box_width, y + box_height), box_color, -1)
        cv2.rectangle(img, (box_x, y), (box_x + box_width, y + box_height), (255, 255, 255), int(2 * scale))
        
        # Store field rectangle for click detection
        start_screen.field_rects.append({
            "x": box_x, 
            "y": y, 
            "w": box_width, 
            "h": box_height
        })
        
        # Field value
        value = field_values[i]
        if value:
            cv2.putText(img, value, (box_x + int(10 * scale_w), y + int(35 * scale_h)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0 * scale, (255, 255, 255), int(2 * scale))
        
        # Draw cursor if this is the active field
        if i == start_screen.current_field and start_screen.cursor_visible:
            text_size = cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, 1.0 * scale, int(2 * scale))[0]
            cursor_x = box_x + text_size[0] + int(10 * scale_w)
            cv2.line(img, (cursor_x, y + int(10 * scale_h)), (cursor_x, y + box_height - int(10 * scale_h)), 
                    (255, 255, 255), int(2 * scale))
        
        # Example text (only for grade field)
        if i == 2:
            cv2.putText(img, examples[i], (box_x, y + box_height + int(30 * scale_h)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7 * scale, (200, 200, 255), int(1 * scale))
    
    # Instructions
    instructions = "Press TAB to switch fields   ENTER to start"
    instr_font_scale = 0.9 * scale
    instr_thickness = int(2 * scale)
    instr_size = cv2.getTextSize(instructions, cv2.FONT_HERSHEY_SIMPLEX, instr_font_scale, instr_thickness)[0]
    instr_x = (w - instr_size[0]) // 2
    cv2.putText(img, instructions, (instr_x, int(550 * scale_h)), 
                cv2.FONT_HERSHEY_SIMPLEX, instr_font_scale, (200, 255, 200), instr_thickness)
    
    # Submit button
    submit_width = int(200 * scale_w)
    submit_height = int(60 * scale_h)
    submit_x = (w - submit_width) // 2
    submit_y = int(600 * scale_h)
    
    # Button with 3D effect
    # Shadow
    cv2.rectangle(img, (submit_x + int(5 * scale_w), submit_y + int(5 * scale_h)), 
                  (submit_x + submit_width + int(5 * scale_w), submit_y + submit_height + int(5 * scale_h)), 
                  (0, 100, 0), -1)
    # Main button
    cv2.rectangle(img, (submit_x, submit_y), 
                  (submit_x + submit_width, submit_y + submit_height), 
                  (0, 180, 0), -1)
    # Highlight
    cv2.rectangle(img, (submit_x, submit_y), 
                  (submit_x + submit_width, submit_y + int(10 * scale_h)), 
                  (0, 220, 0), -1)
    
    button_text = "START GAME"
    button_font_scale = 1.0 * scale
    button_thickness = int(2 * scale)
    button_text_size = cv2.getTextSize(button_text, cv2.FONT_HERSHEY_SIMPLEX, button_font_scale, button_thickness)[0]
    button_text_x = submit_x + (submit_width - button_text_size[0]) // 2
    cv2.putText(img, button_text, (button_text_x, submit_y + int(40 * scale_h)), 
                cv2.FONT_HERSHEY_SIMPLEX, button_font_scale, (255, 255, 255), button_thickness)
    
    # Store submit rect
    start_screen.submit_rect = {
        "x": submit_x, 
        "y": submit_y, 
        "w": submit_width, 
        "h": submit_height
    }
    
    # Blink cursor
    if time.time() - start_screen.cursor_timer > 0.5:
        start_screen.cursor_visible = not start_screen.cursor_visible
        start_screen.cursor_timer = time.time()
    
    return img

def handle_start_screen_input(key):
    """Handle keyboard input for start screen"""
    if start_screen.active:
        current_field_name = list(start_screen.fields.keys())[start_screen.current_field]
        
        if key == 9:  # TAB key
            start_screen.current_field = (start_screen.current_field + 1) % 3
            start_screen.cursor_visible = True
            start_screen.cursor_timer = time.time()
        
        elif key == 13:  # ENTER key
            # Check if all fields are filled
            if all(start_screen.fields.values()):
                start_screen.user_data = start_screen.fields.copy()
                start_screen.active = False
                print(f"User data: {start_screen.user_data}")
        
        elif key == 8:  # BACKSPACE key
            start_screen.fields[current_field_name] = start_screen.fields[current_field_name][:-1]
        
        elif 32 <= key <= 126:  # Printable characters
            start_screen.fields[current_field_name] += chr(key)

def handle_start_screen_gesture(index_pos, thumb_pos, distance):
    """Handle hand gestures for start screen"""
    if not start_screen.active:
        return
    
    now = time.time()
    
    # Check if hand is over submit button
    hand_over_submit = index_pos and start_screen.submit_rect and inside(
        index_pos[0], index_pos[1], start_screen.submit_rect)
    
    # Check if hand is over any field
    hand_over_field = -1
    for i, field_rect in enumerate(start_screen.field_rects):
        if index_pos and inside(index_pos[0], index_pos[1], field_rect):
            hand_over_field = i
            break
    
    # Pinch start
    if distance < PINCH_THRESHOLD and not pinch_active:
        if now - last_pinch_time > PINCH_DELAY:
            pinch_active = True
            last_pinch_time = now
            
            # If over submit button, prepare to submit
            if hand_over_submit:
                submit_pinched = True
            # If over a field, select that field
            elif hand_over_field >= 0:
                start_screen.current_field = hand_over_field
                start_screen.cursor_visible = True
                start_screen.cursor_timer = time.time()
    
    # Release pinch
    if distance > RELEASE_THRESHOLD and pinch_active:
        pinch_active = False
        
        # Submit if submit button was pinched and released
        if submit_pinched and all(start_screen.fields.values()):
            start_screen.user_data = start_screen.fields.copy()
            start_screen.active = False
            print(f"User data: {start_screen.user_data}")
        
        submit_pinched = False

# ---------------- CREATE FULLSCREEN WINDOW ----------------
# Create window
cv2.namedWindow("Gesture-Based Addition Learning", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Gesture-Based Addition Learning", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# ---------------- MAIN LOOP ----------------
while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    
    # Resize image to full screen
    img = cv2.resize(img, (WIDTH, HEIGHT))
    
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # Handle start screen
    if start_screen.active:
        img = draw_start_screen(img)
        
        # Check for keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key != 255:  # If a key was pressed
            handle_start_screen_input(key)
        
        # Hand detection for gesture input
        hand_detected = False
        index_pos = None
        thumb_pos = None
        distance = 1000  # Default large distance
        
        if result.multi_hand_landmarks:
            hand_detected = True
            for hand_landmarks in result.multi_hand_landmarks:
                h, w = img.shape[:2]
                
                # Get index and thumb positions
                index_pos = (int(hand_landmarks.landmark[8].x * w), 
                            int(hand_landmarks.landmark[8].y * h))
                thumb_pos = (int(hand_landmarks.landmark[4].x * w), 
                            int(hand_landmarks.landmark[4].y * h))
                
                distance = dist(index_pos, thumb_pos)
                
                # Draw simplified hand landmarks for start screen
                mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
        
        # Handle gesture input
        if hand_detected:
            handle_start_screen_gesture(index_pos, thumb_pos, distance)
        
        cv2.imshow("Gesture-Based Addition Learning", img)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break
        continue  # Skip the rest of the loop while in start screen

    # -------- GAME LOGIC (only runs after start screen) --------
    
    # Clear message after 1.5 seconds
    if message and time.time() - message_time > 1.5:
        message = ""

    # Draw apples that are not picked
    for a in apples:
        if not a["picked"]:
            # Resize apple image for current screen size
            apple_resized = cv2.resize(apple_img, (APPLE_SIZE, APPLE_SIZE))
            img = overlay(img, apple_resized, a["x"], a["y"])

    # Draw appropriate basket image based on dropped count
    basket_index = min(dropped, 5)
    current_basket_img = basket_images[basket_index]
    # Resize basket image for current screen size
    basket_resized = cv2.resize(current_basket_img, (basket["w"], basket["h"]))
    img = overlay(img, basket_resized, basket["x"], basket["y"])

    # Draw submit button with 3D effect
    submit_color = (0, 180, 0)
    highlight_color = (0, 220, 0)
    shadow_color = (0, 140, 0)
    
    # Calculate scaled button dimensions
    button_shadow_offset = int(5 * (WIDTH/1280))
    
    # Button shadow
    cv2.rectangle(img,
                  (submit["x"] + button_shadow_offset, submit["y"] + button_shadow_offset),
                  (submit["x"] + submit["w"] + button_shadow_offset, submit["y"] + submit["h"] + button_shadow_offset),
                  shadow_color, -1)
    
    # Button main
    cv2.rectangle(img,
                  (submit["x"], submit["y"]),
                  (submit["x"] + submit["w"], submit["y"] + submit["h"]),
                  submit_color, -1)
    
    # Button highlight
    cv2.rectangle(img,
                  (submit["x"], submit["y"]),
                  (submit["x"] + submit["w"], submit["y"] + int(10 * (HEIGHT/720))),
                  highlight_color, -1)
    
    # Submit button text
    submit_text = "SUBMIT"
    submit_font_scale = 1.2 * (WIDTH/1280)
    submit_thickness = int(3 * (WIDTH/1280))
    submit_text_size = cv2.getTextSize(submit_text, cv2.FONT_HERSHEY_SIMPLEX, submit_font_scale, submit_thickness)[0]
    submit_text_x = submit["x"] + (submit["w"] - submit_text_size[0]) // 2
    cv2.putText(img, submit_text,
                (submit_text_x, submit["y"] + int(60 * (HEIGHT/720))),
                cv2.FONT_HERSHEY_SIMPLEX, submit_font_scale, (255, 255, 255), submit_thickness)

    # Draw question and stats with better styling
    # Calculate scaled font sizes
    stats_font_scale = 0.9 * (WIDTH/1280)
    stats_thickness = int(2 * (WIDTH/1280))
    question_font_scale = 1.5 * (WIDTH/1280)
    question_thickness = int(3 * (WIDTH/1280))
    
    # Background for stats
    stats_box_x = int(40 * (WIDTH/1280))
    stats_box_y = int(30 * (HEIGHT/720))
    stats_box_w = int(360 * (WIDTH/1280))
    stats_box_h = int(50 * (HEIGHT/720))
    
    cv2.rectangle(img, (stats_box_x, stats_box_y), 
                  (stats_box_x + stats_box_w, stats_box_y + stats_box_h), 
                  (255, 255, 255), -1)
    cv2.rectangle(img, (stats_box_x, stats_box_y), 
                  (stats_box_x + stats_box_w, stats_box_y + stats_box_h), 
                  (0, 0, 0), 2)
    
    cv2.putText(img, f"# Apples in basket: {dropped}/{target_count}", 
                (stats_box_x + int(10 * (WIDTH/1280)), stats_box_y + int(35 * (HEIGHT/720))), 
                cv2.FONT_HERSHEY_SIMPLEX, stats_font_scale, (0, 0, 0), stats_thickness)
    
    # Display user info if available
    if start_screen.user_data:
        user_info = f"{start_screen.user_data['name']} | Age: {start_screen.user_data['age']} | Grade: {start_screen.user_data['grade']}"
        user_font_scale = 0.7 * (WIDTH/1280)
        user_thickness = int(2 * (WIDTH/1280))
        cv2.putText(img, user_info, (WIDTH - int(600 * (WIDTH/1280)), int(30 * (HEIGHT/720))), 
                    cv2.FONT_HERSHEY_SIMPLEX, user_font_scale, (255, 255, 255), user_thickness)
    
    # Background for question
    question_box_x = int(950 * (WIDTH/1280))
    question_box_y = int(100 * (HEIGHT/720))
    question_box_w = int(300 * (WIDTH/1280))
    question_box_h = int(80 * (HEIGHT/720))
    
    cv2.rectangle(img, (question_box_x, question_box_y), 
                  (question_box_x + question_box_w, question_box_y + question_box_h), 
                  (255, 255, 255), -1)
    cv2.rectangle(img, (question_box_x, question_box_y), 
                  (question_box_x + question_box_w, question_box_y + question_box_h), 
                  (0, 0, 0), 2)
    
    question_text = f"{num1} + {num2} = ?"
    question_text_size = cv2.getTextSize(question_text, cv2.FONT_HERSHEY_SIMPLEX, question_font_scale, question_thickness)[0]
    question_text_x = question_box_x + (question_box_w - question_text_size[0]) // 2
    cv2.putText(img, question_text, 
                (question_text_x, question_box_y + int(55 * (HEIGHT/720))), 
                cv2.FONT_HERSHEY_SIMPLEX, question_font_scale, (0, 0, 0), question_thickness)

    # Draw message with background
    if message:
        # Message background
        message_font_scale = 1.5 * (WIDTH/1280)
        message_thickness = int(3 * (WIDTH/1280))
        text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, message_font_scale, message_thickness)[0]
        msg_x = (WIDTH - text_size[0]) // 2
        msg_y = int(650 * (HEIGHT/720))
        
        padding_x = int(20 * (WIDTH/1280))
        padding_y = int(40 * (HEIGHT/720))
        
        cv2.rectangle(img, 
                     (msg_x - padding_x, msg_y - padding_y),
                     (msg_x + text_size[0] + padding_x, msg_y + int(10 * (HEIGHT/720))),
                     (255, 255, 255), -1)
        cv2.rectangle(img, 
                     (msg_x - padding_x, msg_y - padding_y),
                     (msg_x + text_size[0] + padding_x, msg_y + int(10 * (HEIGHT/720))),
                     (0, 0, 255), 2)
        
        cv2.putText(img, message, (msg_x, msg_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, message_font_scale, (0, 0, 255), message_thickness)

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
            
            # Mark that this apple has been dragged
            if not picked_apple["dragged"]:
                picked_apple["dragged"] = True
                # Spawn a new apple at the spawn position
                new_apple = spawn_new_apple()
                apples.append(new_apple)

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
                
                # Reset apples - start with just one apple
                apples = [spawn_new_apple()]
            
            submit_pinched = False

    cv2.imshow("Gesture-Based Addition Learning", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
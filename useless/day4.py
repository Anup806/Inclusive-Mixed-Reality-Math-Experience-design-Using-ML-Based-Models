import cv2
import mediapipe as mp
import random
import time
import numpy as np
import threading
import math
import sys
import csv
import os
from datetime import datetime

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

# Try to import YOLO for object detection mode
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False
    print("Warning: YOLO not available. Object detection mode will be disabled.")

# -----------------------------
# INITIAL SETUP
pygame.init()

# Initialize pygame mixer for sound
try:
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    print("Audio system initialized successfully")
except Exception as e:
    print(f"Warning: Could not initialize audio system: {e}")
    print("Game will continue without sound effects")

# Get desktop size
try:
    WIDTH, HEIGHT = pygame.display.get_desktop_sizes()[0]
except Exception:
    info = pygame.display.Info()
    WIDTH, HEIGHT = info.current_w, info.current_h

# Set up fullscreen display
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN | pygame.SCALED)
pygame.display.set_caption("Math Learning – Drag Apples to Basket Game")
pygame.event.set_blocked(pygame.MOUSEMOTION)
pygame.mouse.set_visible(True)

# Font setup
font = pygame.font.Font(None, 60)
large_font = pygame.font.Font(None, 100)
small_font = pygame.font.Font(None, 36)
clock = pygame.time.Clock()

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Camera setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera. Check camera connection.")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

# YOLO setup for object detection mode
if YOLO_AVAILABLE:
    try:
        yolo_model = YOLO('yolov8n.pt')  # Lightweight YOLOv8 nano model
        print("YOLO model loaded successfully")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        YOLO_AVAILABLE = False
        yolo_model = None
else:
    yolo_model = None


# -----------------------------
# CSV LOGGING SETUP
# -----------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(SCRIPT_DIR, "game_data")
INTERACTIONS_CSV = os.path.join(CSV_DIR, "interactions.csv")
SESSIONS_CSV = os.path.join(CSV_DIR, "sessions.csv")

# CSV Headers
INTERACTIONS_HEADERS = [
    'student_name', 
    'age', 
    'student_grade',
    'game_mode',
    'timestamp', 
    'math_problem', 
    'user_answer', 
    'correct_answer',
    'reaction_time_s', 
    'correct', 
    'score', 
    'session_id',
    'total_screen_time'
]

SESSIONS_HEADERS = [
    'student_name', 
    'age', 
    'student_grade',
    'game_mode',
    'session_start', 
    'session_end', 
    'final_score', 
    'total_problems', 
    'correct_problems', 
    'accuracy', 
    'session_id',
    'total_screen_time',
    'average_reaction_time'
]

def initialize_csv_files():
    """Create CSV files with headers if they don't exist."""
    try:
        # Create directory if it doesn't exist
        if not os.path.exists(CSV_DIR):
            os.makedirs(CSV_DIR)
        
        # Initialize interactions CSV
        if not os.path.exists(INTERACTIONS_CSV):
            with open(INTERACTIONS_CSV, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(INTERACTIONS_HEADERS)
        
        # Initialize sessions CSV
        if not os.path.exists(SESSIONS_CSV):
            with open(SESSIONS_CSV, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(SESSIONS_HEADERS)
        
        print(f"CSV files initialized in: {CSV_DIR}")
        
    except Exception as e:
        print(f"Error initializing CSV files: {e}")

# Initialize CSV files
initialize_csv_files()

def log_interaction(player_name, age, student_grade, game_mode, math_problem, user_answer, correct_answer,
                    reaction_time_s, correct, current_score, session_id, total_screen_time):
    """Log each math problem interaction to CSV."""
    try:
        with open(INTERACTIONS_CSV, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                player_name,
                age,
                student_grade,
                game_mode,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                math_problem,
                user_answer,
                correct_answer,
                f"{reaction_time_s:.3f}" if reaction_time_s is not None else "",
                1 if correct else 0,
                current_score,
                session_id,
                f"{total_screen_time:.2f}"
            ])
    except Exception as e:
        print(f"Error logging interaction: {e}")

def log_session_end(player_name, age, student_grade, game_mode, session_start, final_score, 
                    total_problems, correct_problems, session_id,
                    total_screen_time, average_reaction_time):
    """Log session summary to CSV."""
    try:
        accuracy = (correct_problems / total_problems * 100) if total_problems > 0 else 0
        with open(SESSIONS_CSV, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                player_name,
                age,
                student_grade,
                game_mode,
                session_start.strftime('%Y-%m-%d %H:%M:%S'),
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                final_score,
                total_problems,
                correct_problems,
                f"{accuracy:.2f}",
                session_id,
                f"{total_screen_time:.2f}",
                f"{average_reaction_time:.3f}" if average_reaction_time is not None else ""
            ])
    except Exception as e:
        print(f"Error logging session: {e}")

# -----------------------------
# GAME VARIABLES FROM FIRST CODE
# -----------------------------
# Apple and game variables
APPLE_SIZE = 300
SPAWN_X, SPAWN_Y = 120, 300
basket = {"x": (WIDTH - 450) // 2, "y": HEIGHT - 380, "w": 450, "h": 320}
submit = {"x": WIDTH - 380, "y": HEIGHT - 380, "w": 220, "h": 90}

# Pinch parameters
PINCH_THRESHOLD = 45
RELEASE_THRESHOLD = 65
PINCH_DELAY = 0.3

# Math game variables
def new_question():
    a = random.randint(1, 5)
    b = random.randint(1, 4)
    return a, b, a + b

num1, num2, target_count = new_question()
dropped = 0
score = 0
total_problems = 0
correct_problems = 0

# Object detection game variables
detected_objects = []
detected_count = 0
show_object_detections = False  # Whether to show YOLO detections
OBJECT_CLASSES = ['apple', 'orange', 'banana', 'book', 'bottle', 'cup', 
                  'cell phone', 'scissors', 'teddy bear', 'sports ball']

# Game mode: "ADDITION" or "OBJECT_DETECTION"
GAME_MODE = "ADDITION"  # Will be set by mode selection

# Message display
message = ""
message_time = 0
message_duration = 1.5

# Result screen
show_result = False
result_message = ""
result_color = (255, 255, 255)
current_sound = None

# Apple management
apples = []
def spawn_new_apple():
    """Create a new apple at spawn position"""
    return {
        "x": SPAWN_X,
        "y": SPAWN_Y,
        "picked": False,
        "in_basket": False
    }

# Initialize with first apple
apples.append(spawn_new_apple())

# Hand tracking variables
picked_apple = None
pinch_active = False
submit_pinched = False
last_pinch_time = 0

# Reaction time tracking
problem_start_time = time.time()
reaction_times = []

# Load apple image (you'll need to provide an apple.png file)
try:
    apple_surface = pygame.image.load("apple.png").convert_alpha()
    apple_surface = pygame.transform.scale(apple_surface, (APPLE_SIZE, APPLE_SIZE))
    small_apple_surface = pygame.transform.scale(apple_surface, (80, 80))
except:
    # Create a simple red apple if image is not found
    apple_surface = pygame.Surface((APPLE_SIZE, APPLE_SIZE), pygame.SRCALPHA)
    pygame.draw.circle(apple_surface, (255, 50, 50), (APPLE_SIZE//2, APPLE_SIZE//2), APPLE_SIZE//2 - 10)
    pygame.draw.circle(apple_surface, (200, 30, 30), (APPLE_SIZE//2, APPLE_SIZE//2), APPLE_SIZE//2 - 10, 3)
    stem_rect = pygame.Rect(APPLE_SIZE//2 - 5, APPLE_SIZE//4 - 10, 10, APPLE_SIZE//4)
    pygame.draw.rect(apple_surface, (100, 70, 20), stem_rect)
    leaf_rect = pygame.Rect(APPLE_SIZE//2 + 5, APPLE_SIZE//4 - 15, 15, 10)
    pygame.draw.ellipse(apple_surface, (100, 200, 50), leaf_rect)
    small_apple_surface = pygame.transform.scale(apple_surface, (80, 80))

# Load basket images (you'll need basket0.png to basket5.png)
basket_images = []
for i in range(6):
    try:
        basket_img = pygame.image.load(f"basket{i}.png").convert_alpha()
        basket_img = pygame.transform.scale(basket_img, (basket["w"], basket["h"]))
        basket_images.append(basket_img)
    except:
        # Create simple basket if images not found
        surf = pygame.Surface((basket["w"], basket["h"]), pygame.SRCALPHA)
        # Basket body
        pygame.draw.ellipse(surf, (210, 180, 140), (10, basket["h"]//2, basket["w"]-20, basket["h"]//2))
        # Basket handle
        pygame.draw.arc(surf, (160, 120, 80), (basket["w"]//4, 10, basket["w"]//2, 50), 
                       math.pi, 2*math.pi, 5)
        # Apples in basket
        if i > 0:
            for j in range(i):
                x_pos = basket["w"]//3 + (j % 3) * 40
                y_pos = basket["h"]//2 + 30 + (j // 3) * 40
                pygame.draw.circle(surf, (255, 50, 50), (x_pos, y_pos), 15)
                pygame.draw.circle(surf, (200, 30, 30), (x_pos, y_pos), 15, 2)
        basket_images.append(surf)

# Load audio files
try:
    congrats_sound = pygame.mixer.Sound("congratulation.MP3")
    tryagain_sound = pygame.mixer.Sound("tryagain.MP3")
    audio_loaded = True
    print("Audio files loaded successfully")
except Exception as e:
    print(f"Warning: Could not load audio files: {e}")
    congrats_sound = None
    tryagain_sound = None
    audio_loaded = False

# -----------------------------
# HELPER FUNCTIONS FROM FIRST CODE
# -----------------------------
def dist(a, b):
    """Calculate Euclidean distance between two points"""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def inside(x, y, rect):
    """Check if point (x, y) is inside rectangle"""
    return rect["x"] < x < rect["x"] + rect["w"] and rect["y"] < y < rect["y"] + rect["h"]

def reset_game():
    """Reset game for new problem"""
    global num1, num2, target_count, dropped, apples, message, message_time, show_result, current_sound
    num1, num2, target_count = new_question()
    dropped = 0
    apples = [spawn_new_apple()]
    message = ""
    message_time = 0
    show_result = False
    if current_sound:
        current_sound.stop()
    current_sound = None

# -----------------------------
# HAND GESTURE FUNCTIONS
# -----------------------------
def compute_pinch_state(hand_landmarks, img_w, img_h):
    """Check if thumb and index finger are pinching"""
    thumb = hand_landmarks.landmark[4]
    index = hand_landmarks.landmark[8]
    
    tx, ty = thumb.x * img_w, thumb.y * img_h
    ix, iy = index.x * img_w, index.y * img_h
    
    distance = math.hypot(tx - ix, ty - iy)
    return distance < PINCH_THRESHOLD

def get_finger_positions(hand_landmarks, img_w, img_h):
    """Get thumb and index finger positions"""
    thumb = hand_landmarks.landmark[4]
    index = hand_landmarks.landmark[8]
    
    thumb_pos = (int(thumb.x * img_w), int(thumb.y * img_h))
    index_pos = (int(index.x * img_w), int(index.y * img_h))
    
    return thumb_pos, index_pos

def draw_hand_skeleton(screen, hand_landmarks, display_w, display_h):
    """Draw hand landmarks and connections"""
    # Draw connections
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20)
    ]
    
    for start, end in connections:
        s = hand_landmarks.landmark[start]
        e = hand_landmarks.landmark[end]
        sx, sy = int(s.x * display_w), int(s.y * display_h)
        ex, ey = int(e.x * display_w), int(e.y * display_h)
        pygame.draw.line(screen, (0, 255, 0), (sx, sy), (ex, ey), 2)
    
    # Draw landmarks
    for lm in hand_landmarks.landmark:
        x, y = int(lm.x * display_w), int(lm.y * display_h)
        pygame.draw.circle(screen, (255, 0, 0), (x, y), 4)
    
    # Draw pinch line between thumb and index
    thumb_pos, index_pos = get_finger_positions(hand_landmarks, display_w, display_h)
    pygame.draw.line(screen, (255, 255, 0), thumb_pos, index_pos, 2)
    
    # Draw distance
    distance = dist(thumb_pos, index_pos)
    mid_x = (thumb_pos[0] + index_pos[0]) // 2
    mid_y = (thumb_pos[1] + index_pos[1]) // 2
    
    distance_text = small_font.render(f"Dist: {int(distance)}", True, (255, 255, 0))
    screen.blit(distance_text, (mid_x, mid_y))
    
    # Draw pinch status
    status = "PINCHING" if distance < PINCH_THRESHOLD else "OPEN"
    status_color = (0, 255, 0) if distance < PINCH_THRESHOLD else (200, 200, 0)
    status_text = small_font.render(f"Status: {status}", True, status_color)
    screen.blit(status_text, (thumb_pos[0] + 20, thumb_pos[1] - 20))

# -----------------------------
# YOLO OBJECT DETECTION FUNCTIONS
# -----------------------------
def detect_objects_yolo(frame):
    """
    Detect objects in frame using YOLO.
    Returns list of detected objects with bounding boxes and labels.
    """
    if not YOLO_AVAILABLE or yolo_model is None:
        return []
    
    try:
        results = yolo_model(frame, verbose=False, conf=0.5)
        detections = []
        
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                label = yolo_model.names[cls_id]
                x1, y1, x2, y2 = box
                
                detections.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'label': label,
                    'confidence': float(conf)
                })
        
        return detections
    except Exception as e:
        print(f"Error in YOLO detection: {e}")
        return []

def draw_object_detections(screen, detections):
    """
    Draw bounding boxes and labels for detected objects.
    """
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        label = det['label']
        conf = det['confidence']
        
        # Draw bounding box
        pygame.draw.rect(screen, (0, 255, 0), (x1, y1, x2-x1, y2-y1), 3)
        
        # Draw label background
        label_text = f"{label} ({conf:.2f})"
        label_surf = small_font.render(label_text, True, (255, 255, 255))
        label_bg = pygame.Surface((label_surf.get_width() + 10, label_surf.get_height() + 5), pygame.SRCALPHA)
        label_bg.fill((0, 255, 0, 180))
        
        screen.blit(label_bg, (x1, y1 - 30))
        screen.blit(label_surf, (x1 + 5, y1 - 28))

# -----------------------------
# CAMERA DISPLAY FUNCTIONS
# -----------------------------
def cvimage_to_pygame(image):
    """Convert cv2 image to PyGame surface."""
    if image is None:
        return None
    try:
        if len(image.shape) != 3 or image.shape[2] != 3:
            return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        return pygame.image.frombuffer(image.tobytes(), image.shape[1::-1], "RGB")
    except Exception as e:
        print(f"Error converting image to pygame: {e}")
        return None

def display_camera_fullscreen(screen, img):
    """Display camera feed as full-screen background."""
    if img is not None:
        try:
            camera_frame = cv2.resize(img, (WIDTH, HEIGHT))
            camera_surface = cvimage_to_pygame(camera_frame)
            
            if camera_surface is not None:
                screen.blit(camera_surface, (0, 0))
                
                # Semi-transparent overlay
                overlay_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                overlay_surf.fill((0, 0, 0, 40))
                screen.blit(overlay_surf, (0, 0))
                
        except Exception as e:
            print(f"Error displaying camera: {e}")
            screen.fill((0, 0, 0))
    else:
        screen.fill((0, 0, 0))

# -----------------------------
# PLAYER INPUT SCREEN
# -----------------------------
player_name = ""
player_age = ""
student_grade = ""
name_entry_active = True
running = False
input_focus = "name"

while name_entry_active:
    dt = clock.tick(30) / 1000.0
    
    screen.fill((30, 30, 40))
    
    # Title
    title_txt = large_font.render("Teaching Mathematics Game", True, (100, 200, 255))
    screen.blit(title_txt, ((WIDTH - title_txt.get_width())//2, HEIGHT//6))
    
    # Instruction
    instr_txt = font.render("Enter your details:", True, (255, 255, 255))
    screen.blit(instr_txt, ((WIDTH - instr_txt.get_width())//2, HEIGHT//4))
    
    # Calculate positions
    box_w, box_h = 500, 60
    box_x = (WIDTH - box_w) // 2
    start_y = HEIGHT//2 - 120
    
    # Name input
    name_label = small_font.render("Name:", True, (200, 200, 200))
    screen.blit(name_label, (box_x, start_y))
    
    pygame.draw.rect(screen, (50, 50, 60), (box_x, start_y + 30, box_w, box_h))
    name_border_color = (100, 150, 255) if input_focus == "name" else (80, 120, 200)
    pygame.draw.rect(screen, name_border_color, (box_x, start_y + 30, box_w, box_h), 3)
    
    name_txt = font.render(player_name if player_name else "_", True, (255, 255, 255))
    screen.blit(name_txt, (box_x + 20, start_y + 40))
    
    # Age input
    age_label = small_font.render("Age:", True, (200, 200, 200))
    screen.blit(age_label, (box_x, start_y + 110))
    
    pygame.draw.rect(screen, (50, 50, 60), (box_x, start_y + 140, box_w, box_h))
    age_border_color = (100, 150, 255) if input_focus == "age" else (80, 120, 200)
    pygame.draw.rect(screen, age_border_color, (box_x, start_y + 140, box_w, box_h), 3)
    
    age_txt = font.render(player_age if player_age else "_", True, (255, 255, 255))
    screen.blit(age_txt, (box_x + 20, start_y + 150))
    
    # Grade input (free text)
    grade_label = small_font.render("Grade/Level (any text):", True, (200, 200, 200))
    screen.blit(grade_label, (box_x, start_y + 220))
    
    pygame.draw.rect(screen, (50, 50, 60), (box_x, start_y + 250, box_w, box_h))
    grade_border_color = (100, 150, 255) if input_focus == "grade" else (80, 120, 200)
    pygame.draw.rect(screen, grade_border_color, (box_x, start_y + 250, box_w, box_h), 3)
    
    grade_txt = font.render(student_grade if student_grade else "_", True, (255, 255, 255))
    screen.blit(grade_txt, (box_x + 20, start_y + 260))
    
    # Instructions
    hint_txt = small_font.render("Press TAB to switch fields | ENTER to start", True, (150, 200, 150))
    screen.blit(hint_txt, ((WIDTH - hint_txt.get_width())//2, HEIGHT - 100))
    
    example_text = small_font.render("Examples: K, 1st, 2nd, Grade 3, Pre-K, Preschool, etc.", True, (180, 180, 180))
    screen.blit(example_text, (box_x, start_y + 320))
    
    pygame.display.update()
    
    # Handle input
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            name_entry_active = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                if player_name.strip() and player_age.strip() and student_grade.strip():
                    name_entry_active = False
                    running = True
            elif event.key == pygame.K_BACKSPACE:
                if input_focus == "name":
                    player_name = player_name[:-1]
                elif input_focus == "age":
                    player_age = player_age[:-1]
                elif input_focus == "grade":
                    student_grade = student_grade[:-1]
            elif event.key == pygame.K_TAB:
                if input_focus == "name":
                    input_focus = "age"
                elif input_focus == "age":
                    input_focus = "grade"
                else:
                    input_focus = "name"
            elif event.unicode:
                if input_focus == "name" and len(player_name) < 30:
                    if event.unicode.isprintable():
                        player_name += event.unicode
                elif input_focus == "age" and len(player_age) < 3:
                    if event.unicode.isdigit():
                        player_age += event.unicode
                elif input_focus == "grade" and len(student_grade) < 50:
                    if event.unicode.isprintable():
                        student_grade += event.unicode

# Exit if user closed during name entry
if not running:
    cap.release()
    pygame.quit()
    sys.exit()

# -----------------------------
# GAME MODE SELECTION SCREEN
# -----------------------------
mode_selection_active = True
selected_mode = None

# Prepare mode buttons
mode_buttons = []
available_modes = ["ADDITION"]
if YOLO_AVAILABLE:
    available_modes.append("OBJECT DETECTION")

num_modes = len(available_modes)
button_width, button_height = 400, 150
button_spacing = 100

if num_modes == 1:
    # Only one mode available
    start_x = (WIDTH - button_width) // 2
else:
    # Multiple modes
    total_width = button_width * num_modes + button_spacing * (num_modes - 1)
    start_x = (WIDTH - total_width) // 2

y_pos = HEIGHT // 2 - button_height // 2

for i, mode in enumerate(available_modes):
    btn_x = start_x + i * (button_width + button_spacing)
    mode_buttons.append({
        "mode": mode,
        "rect": pygame.Rect(btn_x, y_pos, button_width, button_height),
        "color": (70, 130, 180) if mode == "ADDITION" else (34, 139, 34),
        "hover_color": (100, 160, 210) if mode == "ADDITION" else (60, 179, 113),
        "text_color": (255, 255, 255)
    })

while mode_selection_active and running:
    dt = clock.tick(30) / 1000.0
    
    # Get camera frame
    success, img = cap.read()
    if not success:
        img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    
    # Display camera background
    display_camera_fullscreen(screen, img)
    
    # Process hand detection
    finger_pos = None
    hand_detected = False
    current_is_pinch = False
    
    try:
        frame = cv2.resize(img, (WIDTH, HEIGHT))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.flip(frame_rgb, 1)
        result = hands.process(frame_rgb)
    except Exception as e:
        result = None
    
    if result and result.multi_hand_landmarks:
        hand_detected = True
        handLms = result.multi_hand_landmarks[0]
        
        # Get fingertip position
        index_x = int(handLms.landmark[8].x * WIDTH)
        index_y = int(handLms.landmark[8].y * HEIGHT)
        finger_pos = (index_x, index_y)
        
        # Draw hand skeleton
        draw_hand_skeleton(screen, handLms, WIDTH, HEIGHT)
        
        # Detect pinch
        current_is_pinch = compute_pinch_state(handLms, WIDTH, HEIGHT)
    
    # Draw title
    title_bg = pygame.Surface((WIDTH, 120), pygame.SRCALPHA)
    title_bg.fill((0, 0, 0, 150))
    screen.blit(title_bg, (0, 50))
    
    title_txt = large_font.render("SELECT GAME MODE", True, (255, 255, 200))
    screen.blit(title_txt, ((WIDTH - title_txt.get_width())//2, 80))
    
    # Draw mode buttons
    for button in mode_buttons:
        # Check if hand is over button
        is_hovered = False
        if finger_pos and button["rect"].collidepoint(finger_pos):
            is_hovered = True
            # Draw hover effect
            pygame.draw.rect(screen, (255, 255, 200), button["rect"], 4)
        
        # Draw button
        button_color = button["hover_color"] if is_hovered else button["color"]
        pygame.draw.rect(screen, button_color, button["rect"], border_radius=20)
        pygame.draw.rect(screen, (255, 255, 255), button["rect"], 3, border_radius=20)
        
        # Draw button text
        mode_name = button["mode"]
        if mode_name == "OBJECT DETECTION":
            # Split into two lines for better display
            text1 = font.render("OBJECT", True, button["text_color"])
            text2 = font.render("DETECTION", True, button["text_color"])
            text1_rect = text1.get_rect(center=(button["rect"].centerx, button["rect"].centery - 20))
            text2_rect = text2.get_rect(center=(button["rect"].centerx, button["rect"].centery + 20))
            screen.blit(text1, text1_rect)
            screen.blit(text2, text2_rect)
        else:
            text = large_font.render(mode_name, True, button["text_color"])
            text_rect = text.get_rect(center=button["rect"].center)
            screen.blit(text, text_rect)
        
        # Handle pinch selection
        if is_hovered and current_is_pinch:
            selected_mode = button["mode"]
            GAME_MODE = selected_mode
            mode_selection_active = False
    
    # Draw instructions
    instr_txt = small_font.render("Pinch a mode to select | Press 1 or 2 for keyboard selection", 
                                  True, (200, 255, 200))
    screen.blit(instr_txt, ((WIDTH - instr_txt.get_width())//2, HEIGHT - 80))
    
    # Draw player info
    player_info = small_font.render(f"Player: {player_name} | Age: {player_age} | Grade: {student_grade}", 
                                    True, (200, 200, 255))
    screen.blit(player_info, (20, HEIGHT - 40))
    
    pygame.display.update()
    
    # Handle keyboard events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            mode_selection_active = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
                mode_selection_active = False
            elif event.key == pygame.K_1 and "ADDITION" in available_modes:
                selected_mode = "ADDITION"
                GAME_MODE = selected_mode
                mode_selection_active = False
            elif event.key == pygame.K_2 and "OBJECT DETECTION" in available_modes:
                selected_mode = "OBJECT DETECTION"
                GAME_MODE = selected_mode
                mode_selection_active = False

# Exit if no mode selected or window closed
if not selected_mode or not running:
    cap.release()
    pygame.quit()
    sys.exit()

print(f"Selected game mode: {GAME_MODE}")

# Initialize game variables
running = True
session_id = datetime.now().strftime('%Y%m%d_%H%M%S_') + str(int(time.time() * 1000) % 10000)
session_start = datetime.now()
total_problems = 0
correct_problems = 0
game_start_time = time.time()

# Hand smoothing
smoothed_finger_pos = None
SMOOTH_ALPHA = 0.5

# -----------------------------
# MAIN GAME LOOP
# -----------------------------
while running:
    dt = clock.tick(30) / 1000.0
    
    # Calculate total screen time
    TOTAL_SCREEN_TIME = time.time() - game_start_time
    
    # Get camera frame
    success, img = cap.read()
    if not success:
        img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    
    # Display camera background
    display_camera_fullscreen(screen, img)
    
    # Process hand detection
    finger_pos = None
    try:
        frame = cv2.resize(img, (WIDTH, HEIGHT))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.flip(frame_rgb, 1)
        result = hands.process(frame_rgb)
    except Exception as e:
        result = None
    
    current_is_pinch = False
    index_pos = None
    
    if result and result.multi_hand_landmarks:
        handLms = result.multi_hand_landmarks[0]
        
        # Get fingertip position
        raw_x = handLms.landmark[8].x * WIDTH
        raw_y = handLms.landmark[8].y * HEIGHT
        
        if smoothed_finger_pos is None:
            smoothed_finger_pos = (raw_x, raw_y)
        else:
            sx = SMOOTH_ALPHA * raw_x + (1.0 - SMOOTH_ALPHA) * smoothed_finger_pos[0]
            sy = SMOOTH_ALPHA * raw_y + (1.0 - SMOOTH_ALPHA) * smoothed_finger_pos[1]
            smoothed_finger_pos = (sx, sy)
        
        finger_pos = (int(smoothed_finger_pos[0]), int(smoothed_finger_pos[1]))
        index_pos = finger_pos
        
        # Detect pinch
        current_is_pinch = compute_pinch_state(handLms, WIDTH, HEIGHT)
        
        # Draw hand skeleton
        draw_hand_skeleton(screen, handLms, WIDTH, HEIGHT)
    
    # YOLO Object Detection (for OBJECT_DETECTION mode)
    if GAME_MODE == "OBJECT_DETECTION":
        detected_objects = detect_objects_yolo(img)
        detected_count = len(detected_objects)
        
        # Draw detected objects
        draw_object_detections(screen, detected_objects)
        
        # Update target_count based on detected objects
        # The game asks student to collect as many apples as there are detected objects
        if not show_result:
            target_count = detected_count
    
    # Clear message after duration
    if message and time.time() - message_time > message_duration:
        message = ""
    
    # ---------------- HAND GESTURE LOGIC FROM FIRST CODE ----------------
    now = time.time()
    
    # Check if hand is over submit button
    hand_over_submit = index_pos and inside(index_pos[0], index_pos[1], submit)
    submit_highlight = hand_over_submit
    
    # Pinch start
    if current_is_pinch and not pinch_active:
        if now - last_pinch_time > PINCH_DELAY:
            pinch_active = True
            last_pinch_time = now
            
            # If showing result, continue to next problem
            if show_result:
                show_result = False
                reset_game()
                problem_start_time = time.time()
                # Stop sound if playing
                if current_sound:
                    current_sound.stop()
                    current_sound = None
            elif hand_over_submit:
                submit_pinched = True
                # Visual feedback for submit button pinch
                pygame.draw.rect(screen, (0, 255, 0), 
                               (submit["x"], submit["y"], submit["w"], submit["h"]), 3)
            elif index_pos:
                # Check if pinching an apple
                for apple in apples:
                    if not apple["picked"] and not apple["in_basket"]:
                        # Check distance to apple center
                        apple_center_x = apple["x"] + APPLE_SIZE // 2
                        apple_center_y = apple["y"] + APPLE_SIZE // 2
                        if dist((apple_center_x, apple_center_y), index_pos) < APPLE_SIZE // 2:
                            picked_apple = apple
                            # Visual feedback for apple pickup
                            pygame.draw.circle(screen, (0, 255, 0), 
                                             (apple_center_x, apple_center_y), 
                                             APPLE_SIZE // 2, 2)
                            break
    
    # Move apple if pinching one
    if pinch_active and picked_apple and index_pos:
        picked_apple["x"] = index_pos[0] - APPLE_SIZE // 2
        picked_apple["y"] = index_pos[1] - APPLE_SIZE // 2
    
    # Release pinch (check for release)
    if not current_is_pinch and pinch_active:
        pinch_active = False
        
        if picked_apple:
            # Check if apple is dropped in basket
            apple_center_x = picked_apple["x"] + APPLE_SIZE // 2
            apple_center_y = picked_apple["y"] + APPLE_SIZE // 2
            
            if inside(apple_center_x, apple_center_y, basket):
                picked_apple["picked"] = True
                picked_apple["in_basket"] = True
                dropped += 1
                # Spawn a new apple after successfully dropping one
                apples.append(spawn_new_apple())
            picked_apple = None
        
        # Submit if submit button was pinched and released
        if submit_pinched:
            total_problems += 1
            
            # Calculate reaction time
            reaction_time = time.time() - problem_start_time
            reaction_times.append(reaction_time)
            
            correct = (dropped == target_count)
            if correct:
                result_message = "Congratulations!"
                result_color = (0, 255, 0)
                score += 10
                correct_problems += 1
                if audio_loaded:
                    congrats_sound.play()
                    current_sound = congrats_sound
            else:
                result_message = "Try Again!"
                result_color = (255, 0, 0)
                if audio_loaded:
                    tryagain_sound.play()
                    current_sound = tryagain_sound
            
            # Log the interaction
            log_interaction(
                player_name, 
                player_age, 
                student_grade,
                GAME_MODE,
                f"{num1} + {num2} = ?" if GAME_MODE == "ADDITION" else f"Count {detected_count} objects", 
                dropped, 
                target_count,
                reaction_time, 
                correct, 
                score, 
                session_id, 
                TOTAL_SCREEN_TIME
            )
            
            show_result = True
        
        submit_pinched = False
    
    # Draw apples that are not in basket
    for apple in apples:
        if not apple["in_basket"]:
            screen.blit(apple_surface, (apple["x"], apple["y"]))
    
    # Draw basket with appropriate number of apples
    basket_index = min(dropped, 5)
    current_basket_img = basket_images[basket_index]
    screen.blit(current_basket_img, (basket["x"], basket["y"]))
    
    # Draw submit button with 3D effect
    submit_color = (0, 180, 0)
    highlight_color = (0, 220, 0)
    shadow_color = (0, 140, 0)
    
    # Button shadow
    pygame.draw.rect(screen, shadow_color, 
                    (submit["x"] + 5, submit["y"] + 5, submit["w"], submit["h"]))
    
    # Button main
    pygame.draw.rect(screen, submit_color, 
                    (submit["x"], submit["y"], submit["w"], submit["h"]))
    
    # Button highlight
    pygame.draw.rect(screen, highlight_color, 
                    (submit["x"], submit["y"], submit["w"], 10))
    
    # Highlight border if hand over
    if submit_highlight:
        pygame.draw.rect(screen, (255, 255, 0), (submit["x"], submit["y"], submit["w"], submit["h"]), 3)
    
    submit_text = font.render("SUBMIT", True, (255, 255, 255))
    screen.blit(submit_text, (submit["x"] + 40, submit["y"] + 30))
    
    # Draw question and stats with better styling
    # Background for stats
    stats_bg = pygame.Surface((1000, 80), pygame.SRCALPHA)
    stats_bg.fill((0, 0, 0, 128))
    screen.blit(stats_bg, (40, 30))
    
    # Display mode-specific text
    if GAME_MODE == "OBJECT_DETECTION":
        if detected_count == 0:
            stats_text = font.render("Place objects in view", True, (255, 200, 100))
        else:
            stats_text = font.render(f"Objects Detected: {detected_count} → Collect {target_count} Apples", 
                                    True, (100, 255, 100))
    else:
        stats_text = font.render(f"{target_count} Apples", True, (255, 255, 255))
    
    screen.blit(stats_text, (50, 50))
    
    # Draw small apples next to the text (only for addition mode)
    if GAME_MODE != "OBJECT_DETECTION":
        for i in range(target_count):
            apple_x = 50 + stats_text.get_width() + 20 + i * 85
            apple_y = 50 + 15
            screen.blit(small_apple_surface, (apple_x - 40, apple_y - 40))
    
    # Draw score
    score_text = font.render(f"Score: {score}", True, (255, 255, 0))
    screen.blit(score_text, (WIDTH - 200, 30))
    
    # Draw player info
    info_text = small_font.render(
        f"Player: {player_name[:10]} | Grade: {student_grade[:15]} | Time: {TOTAL_SCREEN_TIME:.0f}s", 
        True, (200, 200, 255)
    )
    screen.blit(info_text, (WIDTH - info_text.get_width() - 20, HEIGHT - 40))
    
    # Draw mode-specific instructions
    if GAME_MODE == "OBJECT_DETECTION":
        instruct_text = small_font.render(
            "Place real objects in camera view. Collect matching number of apples.", 
            True, (100, 255, 100))
    else:
        instruct_text = small_font.render(
            "Pinch apples and drag to basket. Pinch SUBMIT when done.", 
            True, (255, 255, 255))
    screen.blit(instruct_text, (30, HEIGHT - 50))
    
    # Draw message with background
    if message:
        message_surf = font.render(message, True, (255, 0, 0))
        msg_width = message_surf.get_width()
        msg_height = message_surf.get_height()
        
        # Message background
        msg_bg = pygame.Surface((msg_width + 40, msg_height + 20), pygame.SRCALPHA)
        msg_bg.fill((255, 255, 255, 200))
        pygame.draw.rect(msg_bg, (0, 0, 255), (0, 0, msg_width + 40, msg_height + 20), 2)
        
        screen.blit(msg_bg, ((WIDTH - msg_width - 40) // 2, HEIGHT // 2 - 50))
        screen.blit(message_surf, ((WIDTH - msg_width) // 2, HEIGHT // 2 - 40))
    
    # Result screen
    if show_result:
        # Semi-transparent overlay
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        screen.blit(overlay, (0, 0))
        
        # Result message
        result_surf = large_font.render(result_message, True, result_color)
        result_rect = result_surf.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 50))
        screen.blit(result_surf, result_rect)
        
        # Continue instruction
        continue_surf = font.render("Pinch anywhere to continue", True, (255, 255, 255))
        continue_rect = continue_surf.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 50))
        screen.blit(continue_surf, continue_rect)
    
    pygame.display.update()
    
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif show_result and event.key == pygame.K_SPACE:
                show_result = False
                reset_game()
                problem_start_time = time.time()
                if current_sound:
                    current_sound.stop()
                    current_sound = None

# -----------------------------
# CLEANUP AND SESSION LOGGING
# -----------------------------
# Calculate average reaction time
average_reaction_time = sum(reaction_times) / len(reaction_times) if reaction_times else None

# Log session end
log_session_end(
    player_name, 
    player_age, 
    student_grade,
    GAME_MODE,
    session_start, 
    score, 
    total_problems, 
    correct_problems, 
    session_id,
    TOTAL_SCREEN_TIME,
    average_reaction_time
)

# Cleanup
cap.release()
pygame.quit()
sys.exit(0)
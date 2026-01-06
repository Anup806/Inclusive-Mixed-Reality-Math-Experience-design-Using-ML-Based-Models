#AR for tutoring colors to the childrens with autism
'''We built this colors learning platform using pygame, MediaPipe for hand gesture detection, OpenCV for real video tracking.
 This inclusive learning platform specially helps the autistic children to learn Different Colors. 
We sent the project related redearch paper titled "DKT-based AR for tutoring colors to the Childrens in Autism" in 
IEEE  Conference on ICT and Photonics (icip 2026).'''
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
    'timestamp', 
    'target_color', 
    'popped_color', 
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
    'session_start', 
    'session_end', 
    'final_score', 
    'total_attempts', 
    'correct_attempts', 
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

def log_interaction(player_name, age, student_grade, target_color, popped_color, 
                    reaction_time_s, correct, current_score, session_id, 
                    total_screen_time):
    """Log each balloon interaction to CSV."""
    try:
        with open(INTERACTIONS_CSV, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                player_name,
                age,
                student_grade,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                target_color,
                popped_color,
                f"{reaction_time_s:.3f}" if reaction_time_s is not None else "",
                1 if correct else 0,
                current_score,
                session_id,
                f"{total_screen_time:.2f}"
            ])
    except Exception as e:
        print(f"Error logging interaction: {e}")

def log_session_end(player_name, age, student_grade, session_start, final_score, 
                    total_attempts, correct_attempts, session_id,
                    total_screen_time, average_reaction_time):
    """Log session summary to CSV."""
    try:
        accuracy = (correct_attempts / total_attempts * 100) if total_attempts > 0 else 0
        with open(SESSIONS_CSV, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                player_name,
                age,
                student_grade,
                session_start.strftime('%Y-%m-%d %H:%M:%S'),
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                final_score,
                total_attempts,
                correct_attempts,
                f"{accuracy:.2f}",
                session_id,
                f"{total_screen_time:.2f}",
                f"{average_reaction_time:.3f}" if average_reaction_time is not None else ""
            ])
    except Exception as e:
        print(f"Error logging session: {e}")

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

# Initialize game variables
running = True
session_id = datetime.now().strftime('%Y%m%d_%H%M%S_') + str(int(time.time() * 1000) % 10000)
session_start = datetime.now()
total_attempts = 0
correct_attempts = 0
game_start_time = time.time()

# -----------------------------
# HAND GESTURE FUNCTIONS
# -----------------------------
def compute_pinch_state(hand_landmarks, img_w, img_h, prev_state=False):
    thumb = hand_landmarks.landmark[4]
    index = hand_landmarks.landmark[8]
    wrist = hand_landmarks.landmark[0]
    mid_mcp = hand_landmarks.landmark[9]

    tx, ty = thumb.x * img_w, thumb.y * img_h
    ix, iy = index.x * img_w, index.y * img_h
    wx, wy = wrist.x * img_w, wrist.y * img_h
    mx, my = mid_mcp.x * img_w, mid_mcp.y * img_h

    hand_size = math.hypot(wx - mx, wy - my)
    threshold = max(20.0, hand_size * PINCH_MULT)
    distance = math.hypot(tx - ix, ty - iy)

    if prev_state:
        return distance < (threshold * PINCH_RELEASE_FACTOR)
    else:
        return distance < (threshold * PINCH_START_FACTOR)

def is_open_hand(hand_landmarks, img_w, img_h):
    extended = 0
    for tip_idx in (8, 12, 16, 20):
        tip = hand_landmarks.landmark[tip_idx]
        pip = hand_landmarks.landmark[tip_idx - 2]
        tip_y = tip.y * img_h
        pip_y = pip.y * img_h
        if tip_y < pip_y - 8:
            extended += 1
    return extended >= 3

def draw_hand_skeleton(screen, hand_landmarks, display_w, display_h):
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
    for lm in hand_landmarks.landmark:
        x, y = int(lm.x * display_w), int(lm.y * display_h)
        pygame.draw.circle(screen, (255, 0, 0), (x, y), 4)

# -----------------------------
# DKT FUNCTIONS (OPTIONAL)
# -----------------------------
ENABLE_DKT = True and DKT_AVAILABLE
SKILLS = list(COLORS.keys())
NUM_SKILLS = len(SKILLS)
dkt_history = []
dkt_lock = threading.Lock()
dkt_training = {"thread": None, "running": False, "last_metrics": None}

def skill_index(color_name):
    return SKILLS.index(color_name)

def log_interaction_and_maybe_train(skill_idx, correct):
    if not ENABLE_DKT:
        return
    with dkt_lock:
        dkt_history.append((skill_idx, int(correct)))

# -----------------------------
# MAIN GAME LOOP
# -----------------------------
running = True
prev_is_pinch = False
last_hand_pos = None
current_is_pinch = False
current_is_open = False
reaction_times = []

while running:
    dt = clock.tick(30) / 1000.0
    
    # Calculate total screen time
    TOTAL_SCREEN_TIME = time.time() - game_start_time
    
    # Get camera frame
    success, img = cap.read()
    if not success:
        img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    
    # Process camera image
    if img is not None:
        if img.shape[0] != HEIGHT or img.shape[1] != WIDTH:
            img = cv2.resize(img, (WIDTH, HEIGHT))
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgRGB = cv2.flip(imgRGB, 1)
    else:
        imgRGB = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    
    # Display camera background
    display_camera_fullscreen(screen, img)
    
    # Process hand detection
    finger_pos = None
    try:
        frame = cv2.resize(imgRGB, (WIDTH, HEIGHT))
        result = hands.process(frame)
    except Exception as e:
        result = None
    
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
        last_hand_pos = finger_pos
        
        # Add trail point
        background.add_trail_point(finger_pos[0], finger_pos[1])
        
        # Detect gestures
        current_is_pinch = compute_pinch_state(handLms, WIDTH, HEIGHT, prev_is_pinch)
        current_is_open = is_open_hand(handLms, WIDTH, HEIGHT)
        
        # Draw hand skeleton
        draw_hand_skeleton(screen, handLms, WIDTH, HEIGHT)
    
    # Update and draw background effects
    background.update(dt)
    background.draw(screen)
    
    # Draw player info
    info_text = small_font.render(
        f"Player: {player_name[:10]} | Grade: {student_grade[:15]} | Time: {TOTAL_SCREEN_TIME:.0f}s", 
        True, (200, 200, 255)
    )
    screen.blit(info_text, (WIDTH - info_text.get_width() - 20, 70))
    
    # Draw instructions
    instruct_bg = pygame.Surface((500, 50), pygame.SRCALPHA)
    instruct_bg.fill((0, 0, 0, 128))
    screen.blit(instruct_bg, (25, HEIGHT - 55))
    
    instruct_text = small_font.render("Pinch (thumb+index) to pop a balloon", True, (255, 255, 255))
    screen.blit(instruct_text, (30, HEIGHT - 50))
    
    # Draw DKT info if available
    with dkt_lock:
        lastm = dkt_training.get("last_metrics")
    if lastm:
        dkt_bg = pygame.Surface((400, 40), pygame.SRCALPHA)
        dkt_bg.fill((0, 0, 0, 128))
        screen.blit(dkt_bg, (25, HEIGHT - 95))
        
        mtxt = small_font.render(f"DKT acc:{lastm.get('acc',0):.2f} loss:{lastm.get('loss',0):.2f}", True, (200,200,50))
        screen.blit(mtxt, (30, HEIGHT - 90))
    
    pygame.display.update()
    
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

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
    session_start, 
    score, 
    total_attempts, 
    correct_attempts, 
    session_id,
    TOTAL_SCREEN_TIME,
    average_reaction_time
)

# Cleanup
cap.release()
pygame.quit()
sys.exit(0)
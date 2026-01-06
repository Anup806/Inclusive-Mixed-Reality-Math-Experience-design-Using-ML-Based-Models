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

def spawn_new_apple():
    """Create a new apple at spawn position"""
    return {
        "x": spawn_x,
        "y": spawn_y,
        "picked": False,
        "dragged": False
    }

def draw_start_screen(img):
    """Draw the start screen with input fields"""
    h, w = img.shape[:2]
    
    # Draw semi-transparent background
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (50, 50, 100), -1)
    img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)
    
    # Title
    title = "Color Learning Game"
    title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 2.5, 5)[0]
    title_x = (w - title_size[0]) // 2
    cv2.putText(img, title, (title_x, 120), 
                cv2.FONT_HERSHEY_DUPLEX, 2.5, (255, 255, 255), 5)
    
    # Subtitle
    subtitle = "Enter your details:"
    subtitle_size = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
    subtitle_x = (w - subtitle_size[0]) // 2
    cv2.putText(img, subtitle, (subtitle_x, 200), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (200, 200, 255), 3)
    
    # Field labels and input boxes
    field_labels = ["Name:", "Age:", "Grade/Level (any text):"]
    examples = ["", "", "Examples: K 1st 2nd Grade 3 Pre-K Preschool etc."]
    field_values = list(start_screen.fields.values())
    
    # Calculate positions
    start_y = 280
    field_height = 70
    spacing = 20
    
    # Clear field rects
    start_screen.field_rects = []
    
    for i in range(3):
        y = start_y + i * (field_height + spacing)
        
        # Field label
        cv2.putText(img, field_labels[i], (200, y + 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 200), 2)
        
        # Input box
        box_x = 450
        box_width = 600
        box_height = 50
        
        # Draw input box with highlight if active
        box_color = (100, 100, 200) if i == start_screen.current_field else (80, 80, 180)
        cv2.rectangle(img, (box_x, y), (box_x + box_width, y + box_height), box_color, -1)
        cv2.rectangle(img, (box_x, y), (box_x + box_width, y + box_height), (255, 255, 255), 2)
        
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
            cv2.putText(img, value, (box_x + 10, y + 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Draw cursor if this is the active field
        if i == start_screen.current_field and start_screen.cursor_visible:
            text_size = cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            cursor_x = box_x + text_size[0] + 10
            cv2.line(img, (cursor_x, y + 10), (cursor_x, y + box_height - 10), 
                    (255, 255, 255), 2)
        
        # Example text (only for grade field)
        if i == 2:
            cv2.putText(img, examples[i], (box_x, y + box_height + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 1)
    
    # Instructions
    instructions = "Press TAB to switch fields   ENTER to start"
    instr_size = cv2.getTextSize(instructions, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
    instr_x = (w - instr_size[0]) // 2
    cv2.putText(img, instructions, (instr_x, 550), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 255, 200), 2)
    
    # Submit button
    submit_x = (w - 200) // 2
    submit_y = 600
    submit_width = 200
    submit_height = 60
    
    # Button with 3D effect
    # Shadow
    cv2.rectangle(img, (submit_x + 5, submit_y + 5), 
                  (submit_x + submit_width + 5, submit_y + submit_height + 5), 
                  (0, 100, 0), -1)
    # Main button
    cv2.rectangle(img, (submit_x, submit_y), 
                  (submit_x + submit_width, submit_y + submit_height), 
                  (0, 180, 0), -1)
    # Highlight
    cv2.rectangle(img, (submit_x, submit_y), 
                  (submit_x + submit_width, submit_y + 10), 
                  (0, 220, 0), -1)
    
    cv2.putText(img, "START GAME", (submit_x + 20, submit_y + 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
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

# ---------------- MAIN LOOP ----------------
while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
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
        
        cv2.imshow("Gesture-Based Addition Learning - Start Screen", img)
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
            img = overlay(img, apple_img, a["x"], a["y"])

    # Draw appropriate basket image based on dropped count
    basket_index = min(dropped, 5)
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
    
    # Display user info if available
    if start_screen.user_data:
        user_info = f"{start_screen.user_data['name']} | Age: {start_screen.user_data['age']} | Grade: {start_screen.user_data['grade']}"
        cv2.putText(img, user_info, (WIDTH - 600, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
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
cv2.destroyAllWindows()'''





# ... (All imports and setup code remains the same until COLORS definition)

# -----------------------------
# GAME VARIABLES - MODIFIED FOR COUNTING
# -----------------------------
# Remove COLORS and emoji dictionaries, replace with apple counting logic

MAX_APPLES = 9  # Count from 1 to 9
current_target = random.randint(1, MAX_APPLES)  # Target number of apples
apples = []  # List of apple objects
baskets = []  # List of basket images
question_text = f"Put {current_target} apple{'s' if current_target > 1 else ''} in the basket!"

score = 0
game_message = ""
gesture_cooldown = 0
drops = []  # Changed from pops to drops
last_drop_correct = None
overlay = {"active": False, "type": None, "start": 0, "duration": 1.6}
smoothed_finger_pos = None
SMOOTH_ALPHA = 0.6
PINCH_MULT = 0.35
PINCH_START_FACTOR = 0.85
PINCH_RELEASE_FACTOR = 1.15
round_start_time = None

# Load apple and basket images
def load_game_images():
    """Load apple and basket images."""
    apple_img = None
    basket_imgs = {}
    
    try:
        # Load apple image
        if os.path.exists("apple.png"):
            apple_surf = pygame.image.load("apple.png").convert_alpha()
            # Resize apple to appropriate size
            apple_img = pygame.transform.smoothscale(apple_surf, (80, 80))
        else:
            # Create a simple red apple if image not found
            print("apple.png not found, creating fallback apple image")
            apple_img = pygame.Surface((80, 80), pygame.SRCALPHA)
            pygame.draw.ellipse(apple_img, (255, 50, 50), (10, 10, 60, 70))
            pygame.draw.ellipse(apple_img, (180, 30, 30), (10, 10, 60, 70), 3)
            # Stem
            pygame.draw.rect(apple_img, (100, 70, 30), (38, 5, 4, 15))
            # Leaf
            pygame.draw.ellipse(apple_img, (100, 200, 100), (45, 10, 20, 12))
        
        # Load basket images (basket0.png to basket9.png)
        for i in range(MAX_APPLES + 1):
            basket_path = f"basket{i}.png"
            if os.path.exists(basket_path):
                basket_surf = pygame.image.load(basket_path).convert_alpha()
                # Resize basket
                basket_imgs[i] = pygame.transform.smoothscale(basket_surf, (200, 150))
            else:
                # Create fallback basket image
                print(f"{basket_path} not found, creating fallback basket")
                basket_img = pygame.Surface((200, 150), pygame.SRCALPHA)
                # Basket body
                pygame.draw.ellipse(basket_img, (210, 180, 140), (10, 60, 180, 80))
                pygame.draw.ellipse(basket_img, (180, 150, 110), (10, 60, 180, 80), 5)
                # Basket handle
                pygame.draw.arc(basket_img, (180, 150, 110), (50, 30, 100, 60), 
                               math.pi, 2 * math.pi, 5)
                # Add number on basket
                if i > 0:
                    num_text = large_font.render(str(i), True, (50, 50, 50))
                    basket_img.blit(num_text, (95, 90))
                basket_imgs[i] = basket_img
        
        print("Game images loaded successfully")
        return apple_img, basket_imgs
        
    except Exception as e:
        print(f"Error loading images: {e}")
        # Create fallback images
        apple_img = pygame.Surface((80, 80), pygame.SRCALPHA)
        pygame.draw.circle(apple_img, (255, 50, 50), (40, 40), 35)
        pygame.draw.circle(apple_img, (180, 30, 30), (40, 40), 35, 3)
        
        basket_imgs = {}
        for i in range(MAX_APPLES + 1):
            basket_img = pygame.Surface((200, 150), pygame.SRCALPHA)
            pygame.draw.rect(basket_img, (210, 180, 140), (20, 70, 160, 70))
            pygame.draw.rect(basket_img, (180, 150, 110), (20, 70, 160, 70), 5)
            if i > 0:
                num_text = large_font.render(str(i), True, (50, 50, 50))
                basket_img.blit(num_text, (95, 90))
            basket_imgs[i] = basket_img
        
        return apple_img, basket_imgs

# Load images
APPLE_IMG, BASKET_IMGS = load_game_images()

# Basket position
BASKET_X = WIDTH // 2 - 100
BASKET_Y = HEIGHT - 250

# -----------------------------
# CSV LOGGING SETUP - MODIFIED FOR COUNTING
# -----------------------------
# Update CSV headers for counting game
INTERACTIONS_HEADERS = [
    'student_name', 
    'age', 
    'student_grade',
    'timestamp', 
    'target_count', 
    'apples_in_basket', 
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

# ... (rest of CSV functions remain the same, just update the parameter names)

def log_interaction(player_name, age, student_grade, target_count, apples_in_basket, 
                    reaction_time_s, correct, current_score, session_id, 
                    total_screen_time):
    """Log each apple drop interaction to CSV."""
    try:
        with open(INTERACTIONS_CSV, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                player_name,
                age,
                student_grade,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                target_count,
                apples_in_basket,
                f"{reaction_time_s:.3f}" if reaction_time_s is not None else "",
                1 if correct else 0,
                current_score,
                session_id,
                f"{total_screen_time:.2f}"
            ])
    except Exception as e:
        print(f"Error logging interaction: {e}")

# ... (rest of camera and background code remains the same)

# -----------------------------
# CREATE APPLES - REPLACED BALLOONS
# -----------------------------
def _place_non_overlapping(existing, radius=40, attempts=200, margin=15):
    for _ in range(attempts):
        x = random.randint(120, WIDTH - 120)
        y = random.randint(150, HEIGHT - 320)  # Higher up to avoid basket
        ok = True
        for obj in existing:
            d = math.hypot(x - obj["x"], y - obj["y"])
            if d < (radius + obj.get("radius", radius) + margin):
                ok = False
                break
        if ok:
            return x, y
    return x, y

def spawn_apples():
    global apples, round_start_time
    apples = []
    radius = 40
    
    # Create apples (always 9 apples total)
    for i in range(MAX_APPLES):
        x, y = _place_non_overlapping(apples, radius=radius)
        apples.append({
            "id": i,
            "x": x,
            "y": y,
            "base_y": y,
            "radius": radius,
            "in_basket": False,
            "being_dragged": False,
            "float_phase": random.uniform(0, 2*math.pi),
            "float_amp": random.uniform(4, 10),
            "float_speed": random.uniform(0.5, 1.2),
            "drift": random.uniform(-12, 12)
        })
    
    round_start_time = time.time()

spawn_apples()

# Basket state
apples_in_basket = 0

# ... (player input screen remains the same)

# -----------------------------
# HAND GESTURE FUNCTIONS - REMAINS SAME
# -----------------------------
# (All hand gesture functions remain the same)

# -----------------------------
# DKT FUNCTIONS - MODIFIED FOR COUNTING
# -----------------------------
ENABLE_DKT = True and DKT_AVAILABLE
SKILLS = [str(i) for i in range(1, MAX_APPLES + 1)]  # Skills are numbers 1-9
NUM_SKILLS = len(SKILLS)
dkt_history = []
dkt_lock = threading.Lock()
dkt_training = {"thread": None, "running": False, "last_metrics": None}

def skill_index(number):
    return int(number) - 1  # Convert number to 0-based index

def log_interaction_and_maybe_train(skill_idx, correct):
    if not ENABLE_DKT:
        return
    with dkt_lock:
        dkt_history.append((skill_idx, int(correct)))

# -----------------------------
# ANIMATION FUNCTIONS - MODIFIED FOR APPLES
# -----------------------------
def create_drop_effect(x, y, correct, is_apple=True):
    """Create drop effect when apple is placed in basket or misses."""
    p = {
        "x": x,
        "y": y,
        "life": 0.0,
        "max_life": 1.0,
        "correct": correct,
        "is_apple": is_apple,
        "particles": [],
        "ring_life": 0.0,
        "ring_max_life": 0.6
    }
    
    if is_apple:
        # Apple drop effect - fewer particles
        for i in range(8):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(20, 80)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            color = (100, 255, 100) if correct else (255, 100, 100)
            p["particles"].append({"vx": vx, "vy": vy, "x": x, "y": y, "color": color, "size": random.randint(2,5)})
    else:
        # Basket effect - more celebratory particles
        for i in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(30, 120)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            color = (255, 215, 0)  # Gold for celebration
            p["particles"].append({"vx": vx, "vy": vy, "x": x, "y": y, "color": color, "size": random.randint(3,6)})
    
    return p

def update_and_draw_drop_effects(dt):
    """Update and draw drop effects (replaces pop effects)."""
    global drops
    new_drops = []
    for p in drops:
        p["life"] += dt
        p["ring_life"] += dt
        t = p["life"] / p["max_life"]
        
        # Draw expanding ring for basket effects
        if not p["is_apple"] and p["correct"]:
            alpha = max(0, 1 - t)
            ring_r = 10 + 80 * t
            s = pygame.Surface((ring_r*2+4, ring_r*2+4), pygame.SRCALPHA)
            ring_color = (100, 255, 100, int(160*alpha)) if p["correct"] else (255, 100, 100, int(160*alpha))
            pygame.draw.circle(s, ring_color, (int(ring_r)+2, int(ring_r)+2), int(ring_r), width=3)
            screen.blit(s, (p["x"]-ring_r-2, p["y"]-ring_r-2))
        
        # Draw particles
        for part in p["particles"]:
            part["x"] += part["vx"] * dt
            part["y"] += part["vy"] * dt
            part["vy"] += 40 * dt  # Gravity
            size = part["size"] * (1 - t)
            if size > 0:
                pygame.draw.circle(screen, part["color"], (int(part["x"]), int(part["y"])), int(size))
        
        if p["life"] < p["max_life"]:
            new_drops.append(p)
    drops = new_drops

# Start overlay function remains the same

# -----------------------------
# MAIN GAME LOOP - MODIFIED FOR COUNTING
# -----------------------------
running = True
prev_is_pinch = False
last_hand_pos = None
current_is_pinch = False
current_is_open = False
reaction_times = []
dragging_apple = None  # Track which apple is being dragged

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
    
    # MODIFIED: Draw question with basket and target count
    question_bg = pygame.Surface((WIDTH, 120), pygame.SRCALPHA)
    question_bg.fill((0, 0, 0, 128))
    screen.blit(question_bg, (0, 0))
    
    # Draw basket preview in question area
    basket_preview = pygame.transform.smoothscale(BASKET_IMGS.get(current_target, BASKET_IMGS[0]), (100, 75))
    screen.blit(basket_preview, (30, 25))
    
    # Draw question text
    question_text = font.render(f"Put {current_target} apple{'s' if current_target > 1 else ''} in the basket!", True, (255, 255, 255))
    screen.blit(question_text, (150, 40))
    
    # Draw current basket count
    count_text = font.render(f"Current: {apples_in_basket}/{current_target}", True, (255, 255, 0))
    screen.blit(count_text, (WIDTH - 250, 40))
    
    # Draw basket at the bottom
    current_basket_img = BASKET_IMGS.get(apples_in_basket, BASKET_IMGS[0])
    screen.blit(current_basket_img, (BASKET_X, BASKET_Y))
    
    # Draw basket outline for visual feedback
    pygame.draw.rect(screen, (255, 255, 255, 100), 
                    (BASKET_X - 10, BASKET_Y - 10, 
                     current_basket_img.get_width() + 20, 
                     current_basket_img.get_height() + 20), 3)
    
    # Update gesture cooldown
    if gesture_cooldown > 0:
        gesture_cooldown -= 1
    
    # Animate apples that are not in basket
    for apple in apples:
        if apple.get("in_basket", False) or apple.get("being_dragged", False):
            continue
            
        apple["float_phase"] += apple["float_speed"] * dt
        apple["y"] = apple["base_y"] + math.sin(apple["float_phase"]) * apple["float_amp"]
        apple["x"] += apple["drift"] * dt
        apple["x"] = max(80, min(WIDTH - 80, apple["x"]))
        if apple["x"] < 80:
            apple["x"] = 80
            apple["drift"] = abs(apple["drift"])
        elif apple["x"] > WIDTH - 80:
            apple["x"] = WIDTH - 80
            apple["drift"] = -abs(apple["drift"])
    
    # Handle pinch gesture for dragging apples
    pinch_started = (current_is_pinch and not prev_is_pinch)
    pinch_ended = (not current_is_pinch and prev_is_pinch)
    prev_is_pinch = current_is_pinch
    
    # Start dragging an apple
    if pinch_started and finger_pos and gesture_cooldown == 0 and dragging_apple is None:
        fx, fy = finger_pos
        
        # Find apple under finger
        for apple in apples:
            if apple.get("in_basket", False):
                continue
                
            dx = fx - apple["x"]
            dy = fy - apple["y"]
            dist = math.hypot(dx, dy)
            
            if dist <= apple["radius"]:
                dragging_apple = apple
                apple["being_dragged"] = True
                gesture_cooldown = 8
                break
    
    # Update position of dragged apple
    if dragging_apple and finger_pos:
        dragging_apple["x"] = finger_pos[0]
        dragging_apple["y"] = finger_pos[1]
    
    # Release apple (drop into basket or back to original position)
    if pinch_ended and dragging_apple:
        apple = dragging_apple
        apple["being_dragged"] = False
        
        # Check if apple is dropped into basket
        basket_rect = pygame.Rect(BASKET_X, BASKET_Y, 
                                 current_basket_img.get_width(), 
                                 current_basket_img.get_height())
        
        if basket_rect.collidepoint(apple["x"], apple["y"]):
            # Apple dropped into basket
            apple["in_basket"] = True
            apples_in_basket += 1
            
            # Check if correct count
            correct = (apples_in_basket == current_target)
            
            # Create drop effect
            drops.append(create_drop_effect(apple["x"], apple["y"], correct, is_apple=True))
            drops.append(create_drop_effect(BASKET_X + 100, BASKET_Y + 75, correct, is_apple=False))
            
            # Play sound
            if correct:
                sound_manager.play_clap()
            else:
                sound_manager.play_pop()
            
            # Add effects
            background.add_ripple(BASKET_X + 100, BASKET_Y + 75, 2.0 if correct else 1.0)
            background.add_sparkle(BASKET_X + 100, BASKET_Y + 75, 10 if correct else 5)
            
            # Calculate reaction time
            reaction_time = None
            if round_start_time is not None:
                reaction_time = time.time() - round_start_time
                reaction_times.append(reaction_time)
            
            # Log interaction
            log_interaction(
                player_name, 
                player_age, 
                student_grade,
                current_target, 
                apples_in_basket, 
                reaction_time, 
                correct, 
                score, 
                session_id,
                TOTAL_SCREEN_TIME
            )
            
            # Update game state
            if correct:
                score += 1
                start_overlay("Congratulations")
                last_drop_correct = True
                
                # Log DKT interaction
                log_interaction_and_maybe_train(skill_index(str(current_target)), True)
            else:
                last_drop_correct = False
                
        else:
            # Apple dropped outside basket - return to original position
            apple["x"] = apple.get("original_x", apple["x"])
            apple["y"] = apple.get("original_y", apple["y"])
            drops.append(create_drop_effect(apple["x"], apple["y"], False, is_apple=True))
            sound_manager.play_wrong()
        
        dragging_apple = None
        gesture_cooldown = 12
    
    # Draw apples
    for apple in apples:
        if apple.get("in_basket", False):
            continue
            
        # Draw apple image
        apple_rect = APPLE_IMG.get_rect(center=(int(apple["x"]), int(apple["y"])))
        screen.blit(APPLE_IMG, apple_rect)
        
        # Draw outline if being dragged
        if apple.get("being_dragged", False):
            pygame.draw.circle(screen, (255, 255, 255, 150), 
                             (int(apple["x"]), int(apple["y"])), 
                             apple["radius"] + 5, 3)
    
    # Update and draw drop effects
    update_and_draw_drop_effects(dt)
    
    # Update and draw overlay
    update_and_draw_overlay()
    
    # Handle round completion
    if not overlay["active"] and last_drop_correct is not None:
        if last_drop_correct:
            # Reset for next round
            current_target = random.randint(1, MAX_APPLES)
            apples_in_basket = 0
            spawn_apples()
            question_text = f"Put {current_target} apple{'s' if current_target > 1 else ''} in the basket!"
        last_drop_correct = None
    
    # Draw score and info
    score_bg = pygame.Surface((250, 60), pygame.SRCALPHA)
    score_bg.fill((0, 0, 0, 128))
    screen.blit(score_bg, (WIDTH - 260, 15))
    
    score_surface = font.render(f"Score: {score}", True, (255, 255, 0))
    screen.blit(score_surface, (WIDTH - 220, 20))
    
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
    
    instruct_text = small_font.render("Pinch (thumb+index) to drag apple into basket", True, (255, 255, 255))
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
import cv2
import numpy as np
import pygame
from ultralytics import YOLO
import random
import time

class MixedRealityMathGame:
    """
    A game that blends physical objects with virtual gameplay.
    
    How it works:
    1. YOLO detects real objects on the table/camera view
    2. Each detected object gets a virtual "apple" representation
    3. Children physically manipulate real objects
    4. Virtual apples follow the real objects
    5. Game logic uses virtual apples for scoring
    """
    
    def __init__(self, camera_index=0, target_classes=None):
        """
        Initialize the mixed reality game.
        
        Args:
            camera_index: Which camera to use
            target_classes: List of YOLO class names to track (e.g., ['apple', 'toy', 'block'])
        """
        # Initialize YOLO detector
        self.yolo = YOLO('yolov8n.pt')  # Can use custom trained model
        
        # Camera setup
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Target object classes to track
        self.target_classes = target_classes or [
            'apple', 'orange', 'banana',  # Fruits
            'toy', 'ball', 'block',       # Toys
            'book', 'bottle', 'cup'       # Common objects
        ]
        
        # Virtual game elements
        self.virtual_apples = []  # Each virtual apple tracks a real object
        self.real_objects_history = []  # Track object positions over time
        
        # Game state
        self.game_mode = "counting"  # "counting", "addition", "sorting"
        self.target_count = 5
        self.score = 0
        self.current_detections = []
        
        # Virtual apple properties
        self.APPLE_SIZE = 100
        self.APPLE_COLORS = [
            (255, 50, 50),    # Red
            (50, 255, 50),    # Green
            (50, 50, 255),    # Blue
            (255, 255, 50),   # Yellow
            (255, 50, 255),   # Magenta
        ]
        
        # Smoothing parameters
        self.smoothing_factor = 0.7
        self.object_trails = {}
        
        print(f"Mixed Reality Math Game initialized")
        print(f"Tracking objects: {self.target_classes}")
    
    def detect_real_objects(self, frame):
        """
        Detect real-world objects using YOLO.
        
        Args:
            frame: Camera frame
            
        Returns:
            List of detected objects with positions and labels
        """
        # Run YOLO detection
        results = self.yolo(frame, verbose=False)
        
        detections = []
        
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            
            for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                label = self.yolo.names[cls_id]
                
                # Filter for target classes
                if label in self.target_classes and conf > 0.5:
                    x1, y1, x2, y2 = box
                    
                    # Calculate center and top position
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    width = x2 - x1
                    height = y2 - y1
                    
                    detection = {
                        'bbox': (x1, y1, x2, y2),
                        'center': (center_x, center_y),
                        'top_left': (x1, y1),
                        'width': width,
                        'height': height,
                        'label': label,
                        'confidence': conf,
                        'class_id': cls_id,
                        'tracking_id': f"{label}_{i}",  # Simple tracking ID
                        'virtual_apple_index': None  # Which virtual apple is attached
                    }
                    
                    detections.append(detection)
        
        return detections
    
    def sync_virtual_physical(self, frame):
        """
        Synchronize virtual apples with real detected objects.
        
        Each real object gets a virtual apple that follows it.
        """
        # Detect real objects
        real_detections = self.detect_real_objects(frame)
        self.current_detections = real_detections
        
        # Update object history for smoothing
        self.update_object_history(real_detections)
        
        # Match virtual apples to real objects
        self.match_apples_to_objects(real_detections)
        
        # Update virtual apple positions
        self.update_virtual_apples()
        
        # Clean up apples that lost their real object
        self.cleanup_orphaned_apples()
        
        # Return both real and virtual states
        return real_detections, self.virtual_apples
    
    def update_object_history(self, detections):
        """
        Maintain history of object positions for smoothing.
        """
        current_time = time.time()
        
        for detection in detections:
            tracking_id = detection['tracking_id']
            
            if tracking_id not in self.object_trails:
                self.object_trails[tracking_id] = {
                    'positions': [],
                    'timestamps': [],
                    'smoothed_pos': detection['center'],
                    'first_seen': current_time
                }
            
            # Add new position
            self.object_trails[tracking_id]['positions'].append(detection['center'])
            self.object_trails[tracking_id]['timestamps'].append(current_time)
            
            # Keep only recent history (last 0.5 seconds)
            cutoff_time = current_time - 0.5
            valid_indices = [
                i for i, ts in enumerate(self.object_trails[tracking_id]['timestamps'])
                if ts > cutoff_time
            ]
            
            self.object_trails[tracking_id]['positions'] = [
                self.object_trails[tracking_id]['positions'][i] for i in valid_indices
            ]
            self.object_trails[tracking_id]['timestamps'] = [
                self.object_tracks[tracking_id]['timestamps'][i] for i in valid_indices
            ]
            
            # Apply smoothing
            if len(self.object_trails[tracking_id]['positions']) > 0:
                # Average recent positions
                positions = np.array(self.object_trails[tracking_id]['positions'])
                smoothed = np.mean(positions, axis=0)
                
                # Exponential smoothing
                old_pos = self.object_trails[tracking_id]['smoothed_pos']
                new_pos = (
                    self.smoothing_factor * old_pos[0] + (1 - self.smoothing_factor) * smoothed[0],
                    self.smoothing_factor * old_pos[1] + (1 - self.smoothing_factor) * smoothed[1]
                )
                
                detection['smoothed_center'] = new_pos
                self.object_trails[tracking_id]['smoothed_pos'] = new_pos
    
    def match_apples_to_objects(self, detections):
        """
        Assign virtual apples to real objects using Hungarian algorithm for optimal matching.
        """
        if not detections:
            return
        
        # If no virtual apples exist, create them
        if not self.virtual_apples:
            for detection in detections:
                apple = self.create_virtual_apple(detection)
                detection['virtual_apple_index'] = len(self.virtual_apples) - 1
        
        else:
            # Match existing apples to new detections
            # Simple nearest neighbor matching
            for detection in detections:
                # Find closest apple
                min_dist = float('inf')
                closest_apple_idx = None
                
                detection_center = detection.get('smoothed_center', detection['center'])
                
                for i, apple in enumerate(self.virtual_apples):
                    if apple['attached_to'] is None:  # Apple not currently attached
                        # Calculate distance
                        apple_center = (apple['x'] + self.APPLE_SIZE/2, 
                                       apple['y'] + self.APPLE_SIZE/2)
                        dist = np.sqrt(
                            (detection_center[0] - apple_center[0])**2 +
                            (detection_center[1] - apple_center[1])**2
                        )
                        
                        if dist < min_dist and dist < 200:  # Max distance threshold
                            min_dist = dist
                            closest_apple_idx = i
                
                if closest_apple_idx is not None:
                    # Attach apple to detection
                    detection['virtual_apple_index'] = closest_apple_idx
                    self.virtual_apples[closest_apple_idx]['attached_to'] = detection['tracking_id']
                    self.virtual_apples[closest_apple_idx]['target_pos'] = (
                        detection_center[0] - self.APPLE_SIZE/2,
                        detection['top_left'][1] - self.APPLE_SIZE - 20  # Position above object
                    )
                else:
                    # Create new apple
                    apple = self.create_virtual_apple(detection)
                    detection['virtual_apple_index'] = len(self.virtual_apples) - 1
    
    def create_virtual_apple(self, detection):
        """
        Create a new virtual apple for a detected object.
        """
        x_center = detection.get('smoothed_center', detection['center'])[0]
        y_top = detection['top_left'][1]
        
        apple = {
            'x': x_center - self.APPLE_SIZE/2,
            'y': y_top - self.APPLE_SIZE - 20,  # Position above object
            'target_x': x_center - self.APPLE_SIZE/2,
            'target_y': y_top - self.APPLE_SIZE - 20,
            'color': random.choice(self.APPLE_COLORS),
            'size': self.APPLE_SIZE,
            'attached_to': detection['tracking_id'],
            'creation_time': time.time(),
            'velocity_x': 0,
            'velocity_y': 0,
            'label': detection['label'],  # Mirror real object label
            'collected': False,
            'animation_offset': random.uniform(0, 2*np.pi)  # For bobbing animation
        }
        
        self.virtual_apples.append(apple)
        return apple
    
    def update_virtual_apples(self):
        """
        Update virtual apple positions with smooth animation.
        """
        current_time = time.time()
        
        for apple in self.virtual_apples:
            if apple['attached_to']:
                # Find corresponding detection
                detection = None
                for det in self.current_detections:
                    if det['tracking_id'] == apple['attached_to']:
                        detection = det
                        break
                
                if detection:
                    # Update target position
                    x_center = detection.get('smoothed_center', detection['center'])[0]
                    y_top = detection['top_left'][1]
                    
                    apple['target_x'] = x_center - self.APPLE_SIZE/2
                    apple['target_y'] = y_top - self.APPLE_SIZE - 20
                
            # Smooth movement toward target
            dx = apple['target_x'] - apple['x']
            dy = apple['target_y'] - apple['y']
            
            # Apply velocity with damping
            apple['velocity_x'] = apple['velocity_x'] * 0.8 + dx * 0.2
            apple['velocity_y'] = apple['velocity_y'] * 0.8 + dy * 0.2
            
            apple['x'] += apple['velocity_x']
            apple['y'] += apple['velocity_y']
            
            # Add bobbing animation
            if not apple['collected']:
                bob_amount = 5 * np.sin(current_time * 2 + apple['animation_offset'])
                apple['y'] += bob_amount
    
    def cleanup_orphaned_apples(self):
        """
        Remove apples that are no longer attached to real objects.
        """
        current_time = time.time()
        apples_to_keep = []
        
        for apple in self.virtual_apples:
            if apple['attached_to']:
                # Check if detection still exists
                detection_exists = False
                for det in self.current_detections:
                    if det['tracking_id'] == apple['attached_to']:
                        detection_exists = True
                        break
                
                if detection_exists:
                    apples_to_keep.append(apple)
                elif current_time - apple['creation_time'] > 2.0:  # Grace period
                    # Apple orphaned for too long - fade out
                    apple['color'] = (
                        max(50, apple['color'][0] - 10),
                        max(50, apple['color'][1] - 10),
                        max(50, apple['color'][2] - 10)
                    )
                    if apple['color'][0] <= 50:  # Fully faded
                        continue
                    apples_to_keep.append(apple)
            else:
                # Unattached apple - keep for a while
                if current_time - apple['creation_time'] < 5.0:
                    apples_to_keep.append(apple)
        
        self.virtual_apples = apples_to_keep
    
    def draw_virtual_apples(self, screen):
        """
        Draw all virtual apples on the pygame screen.
        """
        for apple in self.virtual_apples:
            if not apple['collected']:
                # Draw apple shadow
                shadow_rect = pygame.Rect(
                    apple['x'] + 5,
                    apple['y'] + 5,
                    apple['size'],
                    apple['size']
                )
                pygame.draw.ellipse(screen, (50, 50, 50, 150), shadow_rect)
                
                # Draw apple
                apple_rect = pygame.Rect(
                    apple['x'],
                    apple['y'],
                    apple['size'],
                    apple['size']
                )
                pygame.draw.ellipse(screen, apple['color'], apple_rect)
                
                # Draw highlight
                highlight_rect = pygame.Rect(
                    apple['x'] + apple['size']//4,
                    apple['y'] + apple['size']//4,
                    apple['size']//3,
                    apple['size']//3
                )
                pygame.draw.ellipse(screen, (255, 255, 255, 100), highlight_rect)
                
                # Draw label if available
                if 'label' in apple:
                    label_surface = pygame.font.Font(None, 24).render(
                        apple['label'], True, (255, 255, 255)
                    )
                    screen.blit(label_surface, (apple['x'], apple['y'] - 20))
    
    def draw_real_object_outlines(self, screen, detections):
        """
        Draw outlines around detected real objects.
        """
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            
            # Draw bounding box
            pygame.draw.rect(screen, (0, 255, 0), 
                           (x1, y1, x2-x1, y2-y1), 2)
            
            # Draw label
            label_surface = pygame.font.Font(None, 30).render(
                f"{detection['label']} ({detection['confidence']:.2f})", 
                True, (0, 255, 0)
            )
            screen.blit(label_surface, (x1, y1 - 25))
            
            # Draw connection line to virtual apple
            if detection.get('virtual_apple_index') is not None:
                apple_idx = detection['virtual_apple_index']
                if apple_idx < len(self.virtual_apples):
                    apple = self.virtual_apples[apple_idx]
                    apple_center = (
                        apple['x'] + apple['size']//2,
                        apple['y'] + apple['size']//2
                    )
                    object_center = detection.get('smoothed_center', detection['center'])
                    
                    # Draw dashed line
                    for i in range(0, 10):
                        t = i / 9
                        point_x = object_center[0] * (1-t) + apple_center[0] * t
                        point_y = object_center[1] * (1-t) + apple_center[1] * t
                        
                        if i % 3 == 0:  # Draw every 3rd segment
                            pygame.draw.circle(screen, (255, 255, 0), 
                                            (int(point_x), int(point_y)), 2)
    
    def start_counting_game(self, target_count=5):
        """
        Start a counting game where child must match virtual apples to basket.
        """
        self.game_mode = "counting"
        self.target_count = target_count
        self.score = 0
        
        print(f"Counting game started! Count to {target_count}")
    
    def check_basket_interaction(self, basket_rect):
        """
        Check if virtual apples are in the basket.
        
        Returns:
            List of apples collected, score change
        """
        collected_apples = []
        
        for apple in self.virtual_apples:
            if not apple['collected']:
                apple_rect = pygame.Rect(
                    apple['x'], apple['y'], 
                    apple['size'], apple['size']
                )
                
                if apple_rect.colliderect(basket_rect):
                    apple['collected'] = True
                    collected_apples.append(apple)
                    
                    # Score based on object type
                    if apple['label'] in ['apple', 'orange', 'banana']:
                        self.score += 10  # Fruits are worth more
                    else:
                        self.score += 5
        
        return collected_apples
    
    def generate_game_feedback(self):
        """
        Generate feedback based on current game state.
        """
        virtual_count = len([a for a in self.virtual_apples if not a['collected']])
        real_count = len(self.current_detections)
        
        if self.game_mode == "counting":
            if virtual_count >= self.target_count:
                return f"Great! You have {virtual_count} objects. Now drag them to the basket!"
            else:
                return f"Find {self.target_count - virtual_count} more objects!"
        
        return f"Objects: {real_count} real, {virtual_count} virtual"
    
    def run_game_loop(self):
        """
        Main game loop.
        """
        pygame.init()
        screen = pygame.display.set_mode((1280, 720))
        clock = pygame.time.Clock()
        
        # Define basket area
        basket_rect = pygame.Rect(1000, 500, 200, 150)
        
        running = True
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        # Reset game
                        self.virtual_apples = []
                        self.start_counting_game(random.randint(3, 8))
            
            # Get camera frame
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Convert frame for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_surface = pygame.surfarray.make_surface(
                np.rot90(cv2.flip(frame_rgb, 1))
            )
            frame_surface = pygame.transform.scale(frame_surface, (1280, 720))
            
            # Sync virtual and physical
            real_detections, virtual_apples = self.sync_virtual_physical(frame)
            
            # Check basket interactions
            collected = self.check_basket_interaction(basket_rect)
            if collected:
                print(f"Collected {len(collected)} apples! Score: {self.score}")
            
            # Draw everything
            screen.blit(frame_surface, (0, 0))
            
            # Semi-transparent overlay
            overlay = pygame.Surface((1280, 720), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 50))
            screen.blit(overlay, (0, 0))
            
            # Draw real object outlines
            self.draw_real_object_outlines(screen, real_detections)
            
            # Draw virtual apples
            self.draw_virtual_apples(screen)
            
            # Draw basket
            pygame.draw.rect(screen, (139, 69, 19), basket_rect)  # Brown basket
            pygame.draw.rect(screen, (160, 120, 60), basket_rect, 3)
            basket_text = pygame.font.Font(None, 40).render(
                "BASKET", True, (255, 255, 255)
            )
            screen.blit(basket_text, (basket_rect.x + 50, basket_rect.y - 40))
            
            # Draw collected apples in basket
            collected_in_basket = [a for a in self.virtual_apples if a['collected']]
            for i, apple in enumerate(collected_in_basket[-10:]):  # Show last 10
                pos_x = basket_rect.x + 20 + (i % 4) * 40
                pos_y = basket_rect.y + 20 + (i // 4) * 40
                pygame.draw.ellipse(screen, apple['color'], 
                                  (pos_x, pos_y, 30, 30))
            
            # Draw game info
            info_text = pygame.font.Font(None, 36).render(
                f"Score: {self.score} | Target: {self.target_count} | "
                f"Objects: {len(real_detections)}", 
                True, (255, 255, 255)
            )
            screen.blit(info_text, (20, 20))
            
            # Draw feedback
            feedback = self.generate_game_feedback()
            feedback_text = pygame.font.Font(None, 30).render(
                feedback, True, (255, 255, 200)
            )
            screen.blit(feedback_text, (20, 70))
            
            # Draw instructions
            instructions = [
                "Place objects in front of camera",
                "Virtual apples will appear above them",
                "Drag apples to basket by moving objects",
                "Press R to restart, ESC to quit"
            ]
            
            for i, instruction in enumerate(instructions):
                instr_text = pygame.font.Font(None, 24).render(
                    instruction, True, (200, 200, 255)
                )
                screen.blit(instr_text, (20, 120 + i * 30))
            
            pygame.display.flip()
            clock.tick(30)
        
        # Cleanup
        self.cap.release()
        pygame.quit()
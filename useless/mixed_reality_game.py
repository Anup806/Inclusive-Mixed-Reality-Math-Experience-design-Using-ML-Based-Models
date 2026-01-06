# mixed_reality_simple.py
import cv2
import numpy as np
import pygame
import random
import time

class SimpleMixedRealityGame:
    def __init__(self):
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Virtual apples
        self.virtual_apples = []
        self.APPLE_SIZE = 80
        
        # Game state
        self.score = 0
        self.target_count = 5
        
        # Colors
        self.APPLE_COLORS = [
            (255, 50, 50),    # Red
            (50, 255, 50),    # Green
            (50, 50, 255),    # Blue
            (255, 255, 50),   # Yellow
        ]
        
        print("Simple Mixed Reality Game initialized")
        print("Use mouse to simulate object movement")
    
    def simulate_object_detection(self, mouse_pos):
        """
        Simulate object detection using mouse position
        For testing without YOLO
        """
        if mouse_pos and mouse_pos != (0, 0):
            # Create a "detection" at mouse position
            detection = {
                'bbox': (mouse_pos[0]-50, mouse_pos[1]-50, mouse_pos[0]+50, mouse_pos[1]+50),
                'center': mouse_pos,
                'top_left': (mouse_pos[0]-50, mouse_pos[1]-50),
                'label': 'object',
                'tracking_id': 'mouse_object'
            }
            return [detection]
        return []
    
    def update_virtual_apples(self, detections):
        """
        Create/update virtual apples for detected objects
        """
        if not detections:
            return
        
        # Ensure we have enough virtual apples
        while len(self.virtual_apples) < len(detections):
            self.create_virtual_apple()
        
        # Update positions
        for i, detection in enumerate(detections):
            if i < len(self.virtual_apples):
                apple = self.virtual_apples[i]
                x_center = detection['center'][0]
                y_top = detection['top_left'][1]
                
                apple['target_x'] = x_center - self.APPLE_SIZE/2
                apple['target_y'] = y_top - self.APPLE_SIZE - 20
                
                # Smooth movement
                dx = apple['target_x'] - apple['x']
                dy = apple['target_y'] - apple['y']
                apple['x'] += dx * 0.3
                apple['y'] += dy * 0.3
    
    def create_virtual_apple(self):
        """Create a new virtual apple"""
        apple = {
            'x': random.randint(100, 500),
            'y': random.randint(100, 300),
            'target_x': 0,
            'target_y': 0,
            'color': random.choice(self.APPLE_COLORS),
            'size': self.APPLE_SIZE,
            'collected': False
        }
        self.virtual_apples.append(apple)
        return apple
    
    def check_basket_interaction(self, basket_rect):
        """Check if apples are in basket"""
        collected = []
        for apple in self.virtual_apples:
            if not apple['collected']:
                apple_rect = pygame.Rect(
                    apple['x'], apple['y'],
                    apple['size'], apple['size']
                )
                if apple_rect.colliderect(basket_rect):
                    apple['collected'] = True
                    collected.append(apple)
                    self.score += 10
        return collected
    
    def run(self):
        """Main game loop"""
        pygame.init()
        screen = pygame.display.set_mode((1280, 720))
        pygame.display.set_caption("Mixed Reality Math Game - Simple Version")
        clock = pygame.time.Clock()
        
        basket_rect = pygame.Rect(1000, 500, 200, 150)
        mouse_pos = (0, 0)
        
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
                        # Reset
                        self.virtual_apples = []
                        self.score = 0
                        self.target_count = random.randint(3, 8)
                elif event.type == pygame.MOUSEMOTION:
                    mouse_pos = event.pos
            
            # Get camera frame (just for background)
            ret, frame = self.cap.read()
            if ret:
                # Convert frame for pygame
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.flip(frame, 1)
                frame = np.rot90(frame)
                frame_surface = pygame.surfarray.make_surface(frame)
            else:
                frame_surface = pygame.Surface((1280, 720))
                frame_surface.fill((50, 50, 50))
            
            # Simulate object detection with mouse
            detections = self.simulate_object_detection(mouse_pos)
            
            # Update virtual apples
            self.update_virtual_apples(detections)
            
            # Check basket
            collected = self.check_basket_interaction(basket_rect)
            if collected:
                print(f"Collected {len(collected)} apples! Score: {self.score}")
            
            # Draw everything
            screen.blit(frame_surface, (0, 0))
            
            # Draw virtual apples
            for apple in self.virtual_apples:
                if not apple['collected']:
                    pygame.draw.ellipse(screen, apple['color'],
                                      (apple['x'], apple['y'],
                                       apple['size'], apple['size']))
            
            # Draw basket
            pygame.draw.rect(screen, (139, 69, 19), basket_rect)
            pygame.draw.rect(screen, (160, 120, 60), basket_rect, 3)
            
            basket_text = pygame.font.Font(None, 40).render(
                "BASKET", True, (255, 255, 255)
            )
            screen.blit(basket_text, (basket_rect.x + 50, basket_rect.y - 40))
            
            # Draw collected apples
            collected_apples = [a for a in self.virtual_apples if a['collected']]
            for i, apple in enumerate(collected_apples[-10:]):
                pos_x = basket_rect.x + 20 + (i % 4) * 40
                pos_y = basket_rect.y + 20 + (i // 4) * 40
                pygame.draw.ellipse(screen, apple['color'],
                                  (pos_x, pos_y, 30, 30))
            
            # Draw UI
            font = pygame.font.Font(None, 36)
            
            # Score
            score_text = font.render(f"Score: {self.score}", True, (255, 255, 0))
            screen.blit(score_text, (20, 20))
            
            # Target
            target_text = font.render(f"Target: {self.target_count}", True, (255, 200, 100))
            screen.blit(target_text, (20, 60))
            
            # Objects found
            objects_text = font.render(f"Objects: {len(detections)}", True, (200, 255, 200))
            screen.blit(objects_text, (20, 100))
            
            # Instructions
            instructions = [
                "Move mouse to simulate objects",
                "Virtual apples follow mouse",
                "Drag apples to brown basket",
                "Press R to restart, ESC to quit"
            ]
            
            for i, line in enumerate(instructions):
                instr = pygame.font.Font(None, 24).render(line, True, (200, 200, 255))
                screen.blit(instr, (20, 150 + i * 30))
            
            pygame.display.flip()
            clock.tick(60)
        
        # Cleanup
        self.cap.release()
        pygame.quit()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    game = SimpleMixedRealityGame()
    game.run()
import pygame
import numpy as np
from Model import runModel
from PIL import Image
import os

pygame.init()

SCALE = 10
IMG_SIZE = 28
BAR_HEIGHT = 40
WINDOW = IMG_SIZE * SCALE
WIN_H = WINDOW + BAR_HEIGHT

screen = pygame.display.set_mode((WINDOW, WIN_H))
pygame.display.set_caption("Draw or Upload a Digit")

# "Done" and "Upload" button dimensions and positions
BUTTON_WIDTH = 100
BUTTON_HEIGHT = 30
BUTTON_COLOR = (200, 200, 200)
BUTTON_TEXT_COLOR = (0, 0, 0)

button_rect = pygame.Rect(
    WINDOW // 2 - BUTTON_WIDTH - 10, WINDOW + (BAR_HEIGHT - BUTTON_HEIGHT) // 2, BUTTON_WIDTH, BUTTON_HEIGHT
)
upload_button_rect = pygame.Rect(
    WINDOW // 2 + 10, WINDOW + (BAR_HEIGHT - BUTTON_HEIGHT) // 2, BUTTON_WIDTH, BUTTON_HEIGHT
)

font = pygame.font.SysFont(None, 30)

# Popup box dimensions and position
POPUP_WIDTH = 200
POPUP_HEIGHT = 120
POPUP_COLOR = (240, 240, 255)
POPUP_BORDER_COLOR = (50, 50, 80)
POPUP_BTN_WIDTH = 80
POPUP_BTN_HEIGHT = 32
POPUP_BTN_COLOR = (200, 200, 200)
POPUP_BTN_TEXT_COLOR = (0, 0, 0)

def draw_button():
    # Draw "Done"
    pygame.draw.rect(screen, BUTTON_COLOR, button_rect)
    text_surface = font.render("Done", True, BUTTON_TEXT_COLOR)
    text_rect = text_surface.get_rect(center=button_rect.center)
    screen.blit(text_surface, text_rect)
    # Draw "Upload"
    pygame.draw.rect(screen, BUTTON_COLOR, upload_button_rect)
    upload_surface = font.render("Upload", True, BUTTON_TEXT_COLOR)
    upload_rect = upload_surface.get_rect(center=upload_button_rect.center)
    screen.blit(upload_surface, upload_rect)

def draw_popup(value):
    popup_x = WINDOW // 2 - POPUP_WIDTH // 2
    popup_y = WINDOW // 2 - POPUP_HEIGHT // 2
    popup_rect = pygame.Rect(popup_x, popup_y, POPUP_WIDTH, POPUP_HEIGHT)

    # Draw popup background and border
    pygame.draw.rect(screen, POPUP_COLOR, popup_rect)
    pygame.draw.rect(screen, POPUP_BORDER_COLOR, popup_rect, 3)

    # Popup message
    msg = f"You drew a {value}"
    msg_surf = font.render(msg, True, (10, 10, 30))
    msg_rect = msg_surf.get_rect(center=(WINDOW // 2, popup_y + POPUP_HEIGHT // 2 - 15))
    screen.blit(msg_surf, msg_rect)

    # OK button
    btn_x = WINDOW // 2 - POPUP_BTN_WIDTH // 2
    btn_y = popup_y + POPUP_HEIGHT - POPUP_BTN_HEIGHT - 10
    popup_btn_rect = pygame.Rect(btn_x, btn_y, POPUP_BTN_WIDTH, POPUP_BTN_HEIGHT)
    pygame.draw.rect(screen, POPUP_BTN_COLOR, popup_btn_rect)
    ok_text_surf = font.render("OK", True, POPUP_BTN_TEXT_COLOR)
    ok_text_rect = ok_text_surf.get_rect(center=popup_btn_rect.center)
    screen.blit(ok_text_surf, ok_text_rect)

    return popup_btn_rect

def reset_canvas():
    return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

# --- Brush Options for MNIST-like Drawing ---
BRUSH_RADIUS = 1  # Thinner brush

def get_brush_indices(center_x, center_y):
    brush_indices = []
    for dy in range(-BRUSH_RADIUS, BRUSH_RADIUS + 1):
        for dx in range(-BRUSH_RADIUS, BRUSH_RADIUS + 1):
            dist = np.sqrt(dx ** 2 + dy ** 2)
            if dist <= BRUSH_RADIUS + 0.3:  # soft edge
                weight = np.exp(-(dist ** 2) / (2.0 * (BRUSH_RADIUS / 1.5) ** 2))
                brush_indices.append((dy, dx, weight))
    return brush_indices

brush_indices = get_brush_indices(0, 0)

def draw_brush(canvas, px, py):
    for dy, dx, weight in brush_indices:
        tx, ty = px + dx, py + dy
        if 0 <= tx < IMG_SIZE and 0 <= ty < IMG_SIZE:
            val = int(255 * weight)
            canvas[ty, tx] = max(canvas[ty, tx], val)

def process_uploaded_image(pil_img):
    # Convert image to grayscale if it's not already
    img = pil_img.convert("L")
    # Resize to 28x28
    img = img.resize((IMG_SIZE, IMG_SIZE), resample=Image.LANCZOS)
    # Convert to numpy
    arr = np.array(img)
    # In MNIST, background is black (0) and foreground is white (255)
    # If image is inverted, apply inversion
    if arr.mean() > 127:  # Mostly light; likely background is white
        arr = 255 - arr
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr

def upload_image_prompt():
    """
    Prompt user for filename in the terminal/console.
    Avoid use of tkinter. Check file exists before loading.
    """
    running_prompt = True
    arr = None
    while running_prompt:
        # Since pygame blocks terminal, print message on screen
        msg_font = pygame.font.SysFont(None, 26)
        prompt1 = "Enter image file path in terminal/console."
        prompt2 = "Supported: PNG, JPG, BMP, GIF"
        prompt3 = "Type the path and press ENTER."
        # Draw blur overlay
        overlay = pygame.Surface((WINDOW, WIN_H), pygame.SRCALPHA)
        overlay.fill((30, 30, 30, 210))
        screen.blit(overlay, (0,0))
        for idx, msg in enumerate([prompt1, prompt2, prompt3]):
            linesurf = msg_font.render(msg, True, (240, 240, 255))
            screen.blit(
                linesurf, 
                (WINDOW // 2 - linesurf.get_width() // 2, WINDOW//2 - 50 + idx * 34)
            )
        pygame.display.flip()
        try:
            filename = input("Enter image file path: ")
        except EOFError:
            filename = ""
        if not filename:
            break
        if os.path.isfile(filename):
            try:
                pil_img = Image.open(filename)
                arr = process_uploaded_image(pil_img)
                print(f"Loaded: {filename}")
                return arr
            except Exception as e:
                print(f"Could not load image: {e}")
                break
        else:
            print("File not found. Try again.")
            break
    return None

running = True
drawing = False
last_pos = None

while running:
    # Make black background (0) and draw white (255)
    canvas = reset_canvas()
    done_clicked = False
    popup_active = False
    prediction_val = None
    last_pos = None

    # DRAWING LOOP
    while running and not done_clicked:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                # Detect if within drawing area
                if my < WINDOW:
                    drawing = True
                    last_pos = (mx, my)
                    px, py = mx // SCALE, my // SCALE
                    draw_brush(canvas, px, py)
                elif button_rect.collidepoint(mx, my):
                    done_clicked = True
                elif upload_button_rect.collidepoint(mx, my):
                    # Prompt for file path in terminal
                    uploaded = upload_image_prompt()
                    if uploaded is not None:
                        canvas = uploaded.copy()
                        # Optional: flash or indicate successful upload in pygame UI

            elif event.type == pygame.MOUSEBUTTONUP:
                drawing = False
                last_pos = None

            elif event.type == pygame.MOUSEMOTION and drawing:
                mx, my = pygame.mouse.get_pos()
                if my < WINDOW:
                    px, py = mx // SCALE, my // SCALE
                    if last_pos is not None:
                        lx, ly = last_pos
                        lpx, lpy = lx // SCALE, ly // SCALE
                        steps = int(max(abs(lpx - px), abs(lpy - py), 1))
                        for t in range(steps + 1):
                            ix = int(lpx + (px - lpx) * t / steps)
                            iy = int(lpy + (py - lpy) * t / steps)
                            draw_brush(canvas, ix, iy)
                    else:
                        draw_brush(canvas, px, py)
                    last_pos = (mx, my)

        # Draw drawing area
        screen.fill((0, 0, 0), rect=pygame.Rect(0, 0, WINDOW, WINDOW))
        for y in range(IMG_SIZE):
            for x in range(IMG_SIZE):
                if canvas[y, x] > 0:
                    color_val = int(canvas[y, x])
                    pygame.draw.rect(
                        screen, (color_val, color_val, color_val),
                        (x * SCALE, y * SCALE, SCALE, SCALE)
                    )

        # Draw bottom bar
        pygame.draw.rect(screen, (30, 30, 40), pygame.Rect(0, WINDOW, WINDOW, BAR_HEIGHT))

        # Draw "Done" and "Upload" buttons
        draw_button()

        pygame.display.flip()

    if not running:
        break

    # After "Done" is clicked, run model & show popup
    # Image.fromarray(canvas).save("digit.png")
    arr_1d = canvas.flatten().astype(np.float32) / 255.0
    data = np.load('./DataFiles/WeightsAndBiases.npz')
    W1, B1, W2, B2 = data['w1'], data['b1'], data['w2'], data['b2']
    predictedVal = runModel(arr_1d, W1, B1, W2, B2)
    prediction_val = predictedVal

    popup_active = True
    while popup_active and running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                popup_btn_rect = draw_popup(prediction_val)
                if popup_btn_rect.collidepoint(mx, my):
                    popup_active = False

        # Redraw the canvas and bar
        screen.fill((0, 0, 0), rect=pygame.Rect(0, 0, WINDOW, WINDOW))
        for y in range(IMG_SIZE):
            for x in range(IMG_SIZE):
                if canvas[y, x] > 0:
                    color_val = int(canvas[y, x])
                    pygame.draw.rect(
                        screen, (color_val, color_val, color_val),
                        (x * SCALE, y * SCALE, SCALE, SCALE)
                    )
        pygame.draw.rect(screen, (30, 30, 40), pygame.Rect(0, WINDOW, WINDOW, BAR_HEIGHT))

        draw_button()

        popup_btn_rect = draw_popup(prediction_val)
        pygame.display.flip()

pygame.quit()
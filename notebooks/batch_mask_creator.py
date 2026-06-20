import cv2
import numpy as np
from pathlib import Path
import re
from natsort import natsorted

# ============================================
# CONFIGURATION
# ============================================
# Define input and output paths using Path for cross-platform compatibility
INPUT_DIR = Path("Data/naip_images")
OUTPUT_DIR = Path("Data/masks") # It's cleaner to save masks in their own folder

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Global variables required by the mouse callback function
current_image_display = None
points = []
window_name = "Parking Lot Template Creator"

# ============================================
# MOUSE CALLBACK FUNCTION
# ============================================
def mouse_callback_handler(event, x, y, flags, param):
    """
    Handles mouse click events:
    - Left Click: Adds a point and draws a green dot/line.
    """
    global current_image_display, points
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Record the point
        points.append((x, y))
        
        # Draw user feedback on the display image
        # Draw a small filled green circle at the click location
        cv2.circle(current_image_display, (x, y), 3, (0, 255, 0), -1) 
        
        # If there is more than one point, draw a line from the previous point to the new one
        if len(points) > 1:
            cv2.line(current_image_display, points[-2], points[-1], (0, 255, 0), 2)
            
        # Refresh the window with the updated drawing
        cv2.imshow(window_name, current_image_display)

# ============================================
# MAIN BATCH PROCESS
# ============================================
def run_batch_mask_creation():
    global current_image_display, points
    
    # 1. Get all PNG images in the input directory
    # Using natsorted ensures file "2_..." is processed before "10_..."
    all_image_paths = natsorted(list(INPUT_DIR.glob("*.png")))
    
    if not all_image_paths:
        print(f"❌ ERROR: No PNG images found in {INPUT_DIR.resolve()}")
        return

    # 2. Identify unique Store IDs and select one template image per ID
    # A dictionary to store {store_id: template_image_path}
    unique_store_templates = {}
    
    # Regex to match leading digits at the start of the filename stem
    store_id_pattern = re.compile(r"^(\d+)")
    
    for img_path in all_image_paths:
        match = store_id_pattern.match(img_path.stem)
        if match:
            store_id = int(match.group(1))
            
            # Check if we already have a template for this Store ID
            existing_mask = OUTPUT_DIR / f"mask_store_{store_id}.png"
            
            if store_id not in unique_store_templates and not existing_mask.exists():
                # We save the first image we find for this new ID as the template
                unique_store_templates[store_id] = img_path
    
    print(f"Found {len(all_image_paths)} total images.")
    print(f"Skipped IDs with existing masks in {OUTPUT_DIR}.")
    print(f"Preparing to create {len(unique_store_templates)} new template masks...")
    print("─" * 60)
    
    if not unique_store_templates:
        print("Everything is up to date. No new masks needed.")
        return

    # 3. Initialize the OpenCV window and mouse callback
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback_handler)
    
    # Print unified instructions in English
    print("\nINSTRUCTIONS:")
    print("1. Click the corners of the parking lot asphalt to draw the outline.")
    print("2. When finished, press 'ENTER' (or any key) to save and move to the next store.")
    print("3. To cancel points for the current store, press 'ESC'. We will skip it.")
    print("─" * 60)

    # 4. Loop through unique templates
    processed_count = 1
    total_to_process = len(unique_store_templates)
    
    for store_id, img_path in unique_store_templates.items():
        print(f"[{processed_count}/{total_to_process}] Now processing Store ID: {store_id} (Template: {img_path.name})")
        
        # Load original image and reset point list/display copy
        original_img = cv2.imread(str(img_path))
        if original_img is None:
            print(f"   ⚠️ Warning: Could not load {img_path.name}. Skipping.")
            processed_count += 1
            continue
            
        current_image_display = original_img.copy()
        points = [] # Reset points list for this new store
        
        # Show image and wait until user finishes (hits ENTER or any key)
        cv2.imshow(window_name, current_image_display)
        
        # cv2.waitKey(0) blocks execution until a key is pressed.
        # We check the key code just in case.
        key = cv2.waitKey(0)
        
        # If user pressed ESC (27), we skip mask creation for this store
        if key == 27:
            print(f"   Skipped Store {store_id} by user request.")
            processed_count += 1
            continue

        # 5. Create and save the binary mask from points
        # Needs at least 3 points to form a polygon area
        if len(points) > 2:
            # Create a black image of the same height and width
            h, w = original_img.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8) 
            
            # Fill the defined polygon with white (255)
            cv2.fillPoly(mask, [np.array(points)], 255)
            
            # Save the final mask named by store ID
            mask_filename = f"mask_store_{store_id}.png"
            output_path = OUTPUT_DIR / mask_filename
            cv2.imwrite(str(output_path), mask)
            
            print(f"   ✅ Mask saved: {output_path.resolve()}")
        else:
            print(f"   ❌ Cancelled: Not enough points (need >= 3) to create a polygon for Store {store_id}.")
            
        processed_count += 1
    
    # 6. Cleanup
    cv2.destroyAllWindows()
    print("─" * 60)
    print("Task completed successfully!")

if __name__ == "__main__":
    run_batch_mask_creation()
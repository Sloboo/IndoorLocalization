# PSEUDO CODE
# FRAME -> PART A - EXTERNAL MASK 
#   -> convert_to_bin(): BINARY FIXED THRESHOLDING + Bilaterial filter
#   -> get_edges(): CANNY EDGE DETECTOR on the binary frame + findCountors() with cv.RETR_EXTERNAL on canny's detected edges
#   PART B - INTERNAL PROCESS
#   For Each COUNTOR:
#       Create MASK for the countor (AND FILL)
#       PERFORM EROSION to cut the edges to make sure the LEDS are in black background and that it.
#       convert_to_bin() and then findCountors() with cv.RETR_EXTERNAL to get get the inner countors
#       find the outsider LED by geometric distances
#       get color of each led
#       Label them (Classify Object).


import cv2 as cv
import numpy as np
import time
from collections import Counter
from lpf import LowPassFilter

SHOW_TIMEINGS = False
SHOW_STREAMS = True
SHOW_DEBUG = False
USE_DEBUG = False # For geometricv


MARKERS_MAP = {
    ("green", "blue"): "Green-Blue",
    ("green", "white"): "Green-White",
    ("blue", "green"): "Blue-Green",
    ("green", "green"): "Green-Green",
    ("white", "white"): "White-White"
}

COLOR_RANGES = {
    'white': np.array([0, 0, 255]),
    'red': np.array([0, 255, 255]),
    'green': np.array([60, 255, 255]),
    'blue': np.array([120, 255, 255])
}

# Define fixed colors for specific markers
FIXED_COLORS = {
    "Green-Blue": (255, 0, 0),       # Blue
    "Green-White": (0, 0, 255),      # Red
    "Green-Green": (0, 255, 0),      # Green
    "Blue-Green": (0, 84, 128),      #
    "White-White": (255, 255, 255),  # White
    "Unknown": (123, 123, 123)
}



""" lOren try"""
def apply_opening(frame: np.ndarray, kernel_size: int = 3, title: str = "#1 - Binary Feed (Opening)") -> np.ndarray:
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    _, thresholded = cv.threshold(gray, 75, 255, cv.THRESH_BINARY)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opening = cv.morphologyEx(thresholded, cv.MORPH_OPEN, kernel)
    thr = apply_bilateral_filter(opening)
    if SHOW_STREAMS:
        streamer(thr, title)
    return thr

def apply_closing(frame: np.ndarray, kernel_size: int = 3, title: str = "#2 - Binary Feed (Closing)") -> np.ndarray:
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    _, thresholded = cv.threshold(gray, 75, 255, cv.THRESH_BINARY)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closing = cv.morphologyEx(thresholded, cv.MORPH_CLOSE, kernel)
    thr = apply_bilateral_filter(closing)
    if SHOW_STREAMS:
        streamer(thr, title)
    return thr

def apply_erosion(frame: np.ndarray, kernel_size: int = 3, title: str = "#3 - Binary Feed (Erosion)") -> np.ndarray:
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    _, thresholded = cv.threshold(gray, 75, 255, cv.THRESH_BINARY)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    erosion = cv.erode(thresholded, kernel, iterations=1)
    thr = apply_bilateral_filter(erosion)
    if SHOW_STREAMS:
        streamer(thr, title)
    return thr

def apply_dilation(frame: np.ndarray, kernel_size: int = 3, title: str = "#4 - Binary Feed (Dilation)") -> np.ndarray:
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    _, thresholded = cv.threshold(gray, 75, 255, cv.THRESH_BINARY)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilation = cv.dilate(thresholded, kernel, iterations=1)
    thr = apply_bilateral_filter(dilation)
    if SHOW_STREAMS:
        streamer(thr, title)
    return thr

def apply_gradient(frame: np.ndarray, kernel_size: int = 3, title: str = "#5 - Binary Feed (Gradient)") -> np.ndarray:
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    _, thresholded = cv.threshold(gray, 75, 255, cv.THRESH_BINARY)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    gradient = cv.morphologyEx(thresholded, cv.MORPH_GRADIENT, kernel)
    thr = apply_bilateral_filter(gradient)
    if SHOW_STREAMS:
        streamer(thr, title)
    return thr
""" lOren end"""

def draw_elegant_circle(frame, center, outer_radius=10, inner_radius=5, outer_color=(0, 255, 0), inner_color=(0, 0, 255)):
    """Draw an elegant marker with a circle inside a circle."""
    cv.circle(frame, center, outer_radius, outer_color, thickness=2, lineType=cv.LINE_AA)
    cv.circle(frame, center, inner_radius, inner_color, thickness=-1, lineType=cv.LINE_AA)



def draw_dotted_line(img, pt1, pt2, color, thickness=1, gap=10):
    """Draws a dotted line from pt1 to pt2."""
    dist = np.linalg.norm(np.array(pt1) - np.array(pt2))
    num_dots = int(dist / gap)
    for i in range(num_dots + 1):
        start_point = (
            int(pt1[0] + (pt2[0] - pt1[0]) * i / num_dots),
            int(pt1[1] + (pt2[1] - pt1[1]) * i / num_dots)
        )
        end_point = (
            int(pt1[0] + (pt2[0] - pt1[0]) * (i + 0.5) / num_dots),
            int(pt1[1] + (pt2[1] - pt1[1]) * (i + 0.5) / num_dots)
        )
        cv.line(img, start_point, end_point, color, thickness, lineType=cv.LINE_AA)



def measure_execution_time(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time of {func.__name__}: {execution_time:.6f} seconds")
    return result


def majority_element(inputs):
    if not inputs:
        return None
    count = Counter(inputs)
    return count.most_common(1)[0][0]

def calculate_velocity(previous_position, current_position, time_interval):
    # Calculate the Euclidean distance (displacement)
    displacement = np.linalg.norm(np.array(current_position) - np.array(previous_position))
    # Calculate velocity (displacement / time)
    velocity = displacement / time_interval if time_interval > 0 else 0
    return velocity

def geometric_filter(contours, frame=None):
    filtered_contours = []
    for count, cnt in enumerate(contours):
        # Calculate all metrics first
        area = cv.contourArea(cnt)
        perimeter = cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, 0.02 * perimeter, True)
        vertices = len(approx)
        x, y, w, h = cv.boundingRect(cnt)
        aspect_ratio = float(w) / h
        rect_area = w * h
        extent = float(area) / rect_area
        hull = cv.convexHull(cnt)
        hull_area = cv.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0

        # if USE_DEBUG:
        #     # Print all the calculated metrics
        #     print(f"[Counter {count}] Area: {area}")
        #     print(f"[Counter {count}] Perimeter: {perimeter}")
        #     print(f"[Counter {count}] Vertices: {vertices}")
        #     print(f"[Counter {count}] Aspect Ratio: {aspect_ratio}")
        #     print(f"[Counter {count}] Extent: {extent}")
        #     print(f"[Counter {count}] Hull Area: {hull_area}")
        #     print(f"[Counter {count}] Solidity: {solidity}")
            
        # Display metrics on the frame if USE_DEBUG is True
        if USE_DEBUG and frame is not None:
            text = (f"Area: {area}, Vertices: {vertices}, Aspect Ratio: {aspect_ratio:.2f}, "
                    f"Extent: {extent:.2f}, Solidity: {solidity:.2f}")
            cv.putText(frame, text, (x, y - 40), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        # Apply filtering conditions
        if area <= 900:
            if USE_DEBUG:
                print(f"[Counter {count}] Geometric Error] Area condition not met = {aspect_ratio}.")
            continue

        if not (5 <= vertices <= 7):
            if USE_DEBUG:
                print("[Counter {count}] Geometric Error] Vertices condition not met.")
            continue

        if not (0.55 <= aspect_ratio <= 1.7):
            if USE_DEBUG:
                print(f"[Counter {count}] Geometric Error] Aspect Ratio condition not met = {aspect_ratio}.")
            # continue

        if not (0.50 <= extent <= 0.85):
            if USE_DEBUG:
                print("[Counter {count}] Geometric Error] Extent condition not met.")
            continue

        if hull_area == 0:
            if USE_DEBUG:
                print("[Counter {count}] Geometric Error] Hull Area is zero.")
            continue

        if not (0.75 <= solidity <= 1.25):
            if USE_DEBUG:
                print("[Counter {count}] Geometric Error] Solidity condition not met.")
            continue

        # If all conditions are met, add to the filtered list
        filtered_contours.append(cnt)


    return filtered_contours
 
def find_furthest_contour(centers):
    centers = centers
    max_distance = -1
    furthest_center = None
    l1_point = None
    l2_point = None

    for i in range(len(centers)):
        current_center = centers[i]
        distances = []
        
        for j in range(len(centers)):
            if i != j:
                distance = np.linalg.norm(np.array(current_center) - np.array(centers[j]))
                distances.append((distance, centers[j]))
        
        distances.sort(reverse=True)
        
        if len(distances) >= 2:
            L1, point1 = distances[0]
            L2, point2 = distances[1]
            total_distance = L1 + L2
            
            if total_distance > max_distance:
                max_distance = total_distance
                furthest_center = current_center
                l1_point = point1
                l2_point = point2
    
    return furthest_center, l1_point, l2_point



def get_centeroid(contour):
    x, y, w, h = cv.boundingRect(contour)
    cX = x + w // 2
    cY = y + h // 2
    
    return (cX, cY)


def streamer(frame: np.ndarray, title: str) -> None:
    cv.imshow(title, frame)
 
 
def apply_gausssian_filter(frame: np.ndarray, sigma: int = 5) -> np.ndarray:
    return cv.GaussianBlur(frame, (sigma, sigma), 0)

def apply_bilateral_filter(frame: np.ndarray, d: int = 9, sigmaColor: int = 75, sigmaSpace: int = 75) -> np.ndarray:
    return cv.bilateralFilter(frame, d, sigmaColor, sigmaSpace)


def convert_to_bin(frame: np.ndarray, title: str = "#1 - Binary Feed (Fixed Thresholding + Bilaterial Filter) ") -> np.ndarray:
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    _, thresholded = cv.threshold(gray, 75, 255, cv.THRESH_BINARY) #TODO: Fix thresholds
    thr = apply_bilateral_filter(thresholded)
    if SHOW_STREAMS:
        streamer(thr, title)
    return thr    


def get_edges(frame: np.ndarray) -> np.ndarray:
    canny_edges_frame = cv.Canny(frame, 180, 220)
    canny_edges_contours, _ = cv.findContours(canny_edges_frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    filtered_contours = geometric_filter(canny_edges_contours)
    cv.drawContours(canny_edges_frame, filtered_contours, -1, (255, 255, 255), 1)

    if SHOW_STREAMS:
        streamer(canny_edges_frame, "#2 - Canny detection + Geometric Filter")
    return filtered_contours


def detect_closest_color(hsv_mean):
    # Calculate the distance between the mean HSV value and each predefined color
    distances = {color: np.linalg.norm(hsv_mean - hsv_value) for color, hsv_value in COLOR_RANGES.items()}
    # Find the color with the minimum distance
    closest_color = min(distances, key=distances.get)
    
    return closest_color


def detect_dominant_color(image, center, sample_size=5):
    # Extract the sample region around the center
    cX, cY = center
    sample_region = image[cY-sample_size:cY+sample_size, cX-sample_size:cX+sample_size]
    
    # Convert the sample region to HSV color space
    hsv_sample_region = cv.cvtColor(sample_region, cv.COLOR_BGR2HSV)
    
    # Calculate the mean HSV value of the sample region
    hsv_mean = np.mean(hsv_sample_region.reshape(-1, 3), axis=0)
    
    # Detect the closest color
    dominant_color = detect_closest_color(hsv_mean)
    
    return dominant_color

def process_shape(frame, contour):
    # Create a mask for the contour
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv.drawContours(mask, [contour], -1, 255, -1)
    
    # Apply erosion to the mask to remove the thin white border
    kernel = np.ones((3, 3), np.uint8)
    eroded_mask = cv.erode(mask, kernel, iterations=1)
    
    # Extract the ROI using the eroded mask
    roi = cv.bitwise_and(frame, frame, mask=eroded_mask)
    
    # Convert the ROI to a binary image using the convert_to_bin function
    #roi_binary0 = convert_to_bin(roi, "#4 MAIN - Detected Inner shape (Masked)")
    
    roi_binary = apply_opening(roi, title="#4 - Detected Inner shape (Opening)")
    # or
    #roi_binary2 = apply_closing(roi, title="#4 - Detected Inner shape (Closing)")
    # or
    #roi_binary3 = apply_erosion(roi, title="#4 - Detected Inner shape (Erosion)")
    # or
    #roi_binary4 = apply_dilation(roi, title="#4 - Detected Inner shape (Dilation)")
    # or
    #roi_binary5 = apply_gradient(roi, title="#4 - Detected Inner shape (Gradient)")
    
    
    # Find contours in the binary ROI
    centers = []
    centers_and_colors = []
    roi_contours, _ = cv.findContours(roi_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for c in roi_contours:
        center = get_centeroid(c)
        dominant_color = detect_dominant_color(frame, center)
        
        centers.append(center)
        centers_and_colors.append((center, dominant_color))
        
        cv.putText(roi, f'{dominant_color[0]}', (center[0], center[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)    
        # cv.circle(roi, center, 1, (255, 0, 0), -1)
        cv.drawContours(roi, [c], -1, (0, 255, 0), 1)
        
    outsider_color = None
    insiders_colors = []
    major_color = None
    chosen_marker = None
    box_center = None
    if centers_and_colors:
        # Find the contour with the maximum L1 + L2 distance and their points
        furthest_center, l1_point, l2_point = find_furthest_contour(centers)
        if furthest_center:
            # print(f"centers_and_colors are: {centers_and_colors}")
            # print(f"furthest_center is {furthest_center}")
            # Draw a red circle at the furthest center
            cv.circle(roi, furthest_center, 4, (0, 0, 255), -1)
            # Draw pink lines for L1 and L2
            cv.line(roi, furthest_center, l1_point, (255, 105, 180), 1)
            cv.line(roi, furthest_center, l2_point, (255, 105, 180), 1)
            
            for cord in centers_and_colors:
                if furthest_center == cord[0]:
                    outsider_color = cord[1]
                else:
                    insiders_colors.append(cord[1])
            major_color = majority_element(insiders_colors)

            k = (outsider_color, major_color)
            # if "white" in k:you7
            #     chosen_marker = "Unknown"
                # print(f"label is {k} marker is: {chosen_marker}")ç

            # else:
            try:
                chosen_marker = MARKERS_MAP[(outsider_color, major_color)]
                # print(f"label is {k} marker is: {MARKERS_MAP[(outsider_color, major_color)]}")
                
            except KeyError:
                chosen_marker = "Unknown"
                # print(f"label is {k} marker is: {chosen_marker}")
            
            # cv.putText(frame, f'{chosen_marker}, {k}', (center[0], center[1] - 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (225, 255, 240), 2)
            # Draw shadow
            cv.putText(frame, f'{chosen_marker}, {k}', (center[0] + 2, center[1] - 18), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, lineType=cv.LINE_AA)

            # Draw outline
            cv.putText(frame, f'{chosen_marker}, {k}', (center[0], center[1] - 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4, lineType=cv.LINE_AA)

            # Draw main text
            cv.putText(frame, f'{chosen_marker}, {k}', (center[0], center[1] - 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (225, 255, 240), 2, lineType=cv.LINE_AA)
                
            x, y, w, h = cv.boundingRect(contour)
            box_center = (x + w // 2, y + h // 2)
            
            cv.circle(frame, box_center, 2, (255, 0, 0), -1)
        
    return roi, mask, box_center, chosen_marker


def draw_legend(frame, marker_colors, marker_names, current_marker=None, current_velocity=None):
    """Draws a legend of colors and corresponding marker names on the side of the frame, along with current velocity."""
    x_start = frame.shape[1] - 250  # Position the legend on the right side of the frame
    y_start = 30  # Starting y-coordinate
    spacing = 30  # Spacing between entries in the legend
    
    # Clear the area where the legend and velocity will be drawn
    legend_height = (len(marker_names) + 1) * spacing + 30
    cv.rectangle(frame, (x_start - 10, y_start - 10), (frame.shape[1], y_start + legend_height), (0, 0, 0), -1)
    
    for i, marker in enumerate(marker_names):
        color = marker_colors[marker]
        if marker == current_marker:
            text = "-> " + marker
        else:
            text = marker
        # Draw the color box
        cv.rectangle(frame, (x_start, y_start + i * spacing), 
                     (x_start + 20, y_start + i * spacing + 20), color, -1)
        # Draw the text label next to the color box
        cv.putText(frame, text, (x_start + 30, y_start + i * spacing + 15), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, lineType=cv.LINE_AA)

    # Draw current velocity under the legend, in a separate line
    if current_velocity is not None:
        velocity_text = f"Velocity: {current_velocity:.2f} px/s"
        # Placing it a little bit lower to avoid overlap
        cv.putText(frame, velocity_text, (x_start, y_start + len(marker_names) * spacing + 15), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, lineType=cv.LINE_AA)        
        

def main():
    lpfilter = LowPassFilter(window_size=5)
    capture = cv.VideoCapture(0)
    tracked_path_frame = None
    marker_paths = {}
    marker_colors = {}
    prev_time = time.time()
    frame_count = 0

    # Initialize variables to track routes
    last_marker = None
    last_position = None
    route_start = None
    route_end = None
    last_update_time = time.time()
    
    while True:
        current_marker = None
        current_velocity = None
        ret, frame = capture.read()
        if not ret:
            break
        
        current_time = time.time()
        time_elapsed = current_time - last_update_time
        
        if SHOW_TIMEINGS:
            bin_frame = measure_execution_time(convert_to_bin, frame)
            contours = measure_execution_time(get_edges, bin_frame)
            filtered_contours = measure_execution_time(geometric_filter, contours)
        else:
            bin_frame = convert_to_bin(frame)
            contours = get_edges(bin_frame)
            filtered_contours = geometric_filter(contours, frame)

        if tracked_path_frame is None:
            tracked_path_frame = np.zeros_like(frame)

        for contour in filtered_contours:
            if SHOW_TIMEINGS:
                processed_roi, mask_roi, box_center, chosen_marker = measure_execution_time(process_shape, frame, contour)
            else:
                processed_roi, mask_roi, box_center, chosen_marker = process_shape(frame, contour)
                
            # Display the processed ROI
            if SHOW_STREAMS:
                streamer(processed_roi, "#5 - Full Detected Shape")
                streamer(mask_roi, "#3 - Detected Outer Shape (Masked)")

                
            smoothed_marker = lpfilter.update(chosen_marker)
            current_marker = smoothed_marker  
            SHOW_DEBUG and print(f"chosen_marker = {chosen_marker}, smoothed_marker={smoothed_marker}")            

            if box_center is not None and smoothed_marker is not None:
                if smoothed_marker not in marker_paths:
                    marker_paths[smoothed_marker] = []
                    # marker_colors[smoothed_marker] = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                    marker_colors[smoothed_marker] = FIXED_COLORS[smoothed_marker]
                    route_start = box_center  # Set the start of the route for the new marker

                marker_paths[smoothed_marker].append(box_center)

                if len(marker_paths[smoothed_marker]) > 1:
                    if last_marker != smoothed_marker:
                        if last_marker is not None:
                            route_end = marker_paths[last_marker][-1]  # Set the end of the route
                            draw_elegant_circle(tracked_path_frame, route_start)
                            draw_elegant_circle(tracked_path_frame, route_end)
                            marker_paths[last_marker] = []  # Clear the path for the previous marker

                        route_start = box_center  # Reset the start point for the new marker
                        last_marker = smoothed_marker  # Update the last marker

                    # Draw the normal path line
                    cv.line(tracked_path_frame, marker_paths[smoothed_marker][-2], marker_paths[smoothed_marker][-1], marker_colors[smoothed_marker], 2, lineType=cv.LINE_AA)
                    current_velocity = calculate_velocity(marker_paths[smoothed_marker][-2], marker_paths[smoothed_marker][-1], time_elapsed)

                # Update last_position
                last_position = box_center

        # Draw remaining paths for other markers
        for marker, path in marker_paths.items():
            if marker != last_marker:
                for i in range(1, len(path)):
                    cv.line(tracked_path_frame, path[i - 1], path[i], marker_colors[marker], 2, lineType=cv.LINE_AA)

        # Draw the legend on the side of the frame
        draw_legend(tracked_path_frame, marker_colors, marker_paths.keys(), current_marker=current_marker, current_velocity=current_velocity)

        if SHOW_TIMEINGS:
            frame_count += 1
            current_time = time.time()
            elapsed_time = current_time - prev_time

            if elapsed_time >= 1.0:
                fps = frame_count / elapsed_time
                print(f"FPS: {fps:.2f}")
                prev_time = current_time
                frame_count = 0

        cv.drawContours(frame, filtered_contours, -1, (0, 255, 0), 1)
        cv.imshow('Original Video Feed', frame)
        cv.imshow('Tracked Path', tracked_path_frame)

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('o'):  # Reset paths when "o" is pressed
            marker_paths.clear()
            tracked_path_frame = np.zeros_like(frame)  # Clear the tracked path frame
            
        last_update_time = current_time

    capture.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
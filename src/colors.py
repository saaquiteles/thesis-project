import cv2
import numpy as np
from sklearn.cluster import KMeans

def get_hex_code(rgb_px):
    return "#{:02x}{:02x}{:02x}".format(int(rgb_px[0]), int(rgb_px[1]), int(rgb_px[2]))

def hex_to_rgb(hex_code):
    hex_code = hex_code.lstrip('#')
    return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))

def apply_mask(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5) 
    edges = cv2.Canny(blurred, 25, 100)
    
    kernel = np.ones((5,5), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_layer = np.zeros_like(gray)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask_layer, [largest_contour], -1, (255), thickness=-1)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    sat_mask = cv2.threshold(saturation, 10, 255, cv2.THRESH_BINARY)[1]
    final_mask = cv2.bitwise_and(mask_layer, sat_mask)

    # Find where the bird is
    masked_indices = np.where(final_mask == 255)
    bird_pixels = img_rgb[masked_indices]

    if len(bird_pixels) == 0:
        return bird_pixels, masked_indices

    # Filter shadows/highlights
    brightness = np.mean(bird_pixels, axis=1)
    valid_mask = (brightness > 30) & (brightness < 255)
    
    # Return filtered pixels and their specific image coordinates
    filtered_pixels = bird_pixels[valid_mask]
    filtered_indices = (masked_indices[0][valid_mask], masked_indices[1][valid_mask])
    
    return filtered_pixels, filtered_indices

def retrieve_dominant_colors_canny(img, k=6):
    H, W, _ = img.shape
    base_scale = W / 1000.0  # Dynamic scaling based on image width
    
    pixels_rgb, indices = apply_mask(img)
    if len(pixels_rgb) == 0:
        return [], img

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels_rgb)
    centers = kmeans.cluster_centers_.astype(np.float32)

    # Apply shading to the bird
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    for i in range(len(labels)):
        y, x = indices[0][i], indices[1][i]
        img_rgb[y, x] = (img_rgb[y, x] * 0.6) + (centers[labels[i]] * 0.4)
    shaded_bgr = cv2.cvtColor(np.clip(img_rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    # Drawing Constants
    font_scale = max(0.4, base_scale * 0.7)
    thickness = max(1, int(base_scale * 2))
    circle_radius = max(4, int(base_scale * 6))

    counts = np.bincount(labels)
    percentages = (counts / counts.sum()) * 100
    sorted_indices = np.argsort(-percentages)

    color_info = []
    for idx in sorted_indices:
        raw_rgb = centers[idx].astype(np.uint8)
        hex_code = get_hex_code(raw_rgb)
        color_info.append((hex_code, percentages[idx]))
        
        # Create mask for this specific color
        cluster_mask = np.zeros((H, W), dtype=np.uint8)
        cluster_indices = np.where(labels == idx)[0]
        cluster_mask[indices[0][cluster_indices], indices[1][cluster_indices]] = 255

        # Clean noise
        cluster_mask = cv2.morphologyEx(cluster_mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

        # POLE OF INACCESSIBILITY: Find the point furthest from the edges
        dist_transform = cv2.distanceTransform(cluster_mask, cv2.DIST_L2, 5)
        _, max_val, _, max_loc = cv2.minMaxLoc(dist_transform)
        
        # Only draw if the color region is large enough to be meaningful
        if max_val < (3 * base_scale): continue

        avg_x, avg_y = max_loc 

        # Draw indicator dot with shadow/border
        cv2.circle(shaded_bgr, (avg_x, avg_y), circle_radius + 2, (0, 0, 0), -1)
        cv2.circle(shaded_bgr, (avg_x, avg_y), circle_radius, (int(raw_rgb[2]), int(raw_rgb[1]), int(raw_rgb[0])), -1)
        cv2.circle(shaded_bgr, (avg_x, avg_y), circle_radius, (255, 255, 255), thickness)

        # Label placement
        label_x, label_y = avg_x + int(15 * base_scale), avg_y - int(10 * base_scale)
        (tw, th), _ = cv2.getTextSize(hex_code, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(shaded_bgr, (label_x - 2, label_y - th - 2), (label_x + tw + 2, label_y + 2), (0, 0, 0), -1)
        cv2.putText(shaded_bgr, hex_code, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

    return color_info, shaded_bgr

def self_intersects(x, y, w, h, other_rect):
    """ Helper to detect label overlapping """
    ox, oy, ow, oh = other_rect
    return not (x + w < ox or x > ox + ow or y < oy - oh or y - h > oy)

def retrieve_dominant_colors(img, strategy="canny"):
    if strategy == "canny":
        return retrieve_dominant_colors_canny(img)
    else:
        raise ValueError("Unsupported strategy. Use 'canny'.")

def get_color_luminance(hex_code):
    red, green, blue = hex_to_rgb(hex_code)
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    return luminance
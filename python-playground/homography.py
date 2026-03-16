import cv2
import numpy as np
import os  

def order_points(pts):
    """v0.1: Sort four points in order: top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype="float32")
    # The top-left point will have the smallest sum, 
    # whereas the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Compute the difference between the points; 
    # the top-right point will have the smallest difference, 
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def scan_correction(img, src_points):
    """v0.1: Legacy manual correction function to deskew images based on fixed dimensions"""
    width, height = 600, 800
    dst_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    src_points = np.float32(src_points)
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    corrected_img = cv2.warpPerspective(img, M, (width, height))
    return corrected_img

def auto_detect_and_warp(image_path, output_path, gradient_thresh1=50, gradient_thresh2=150):
    """v1.0: Auto-detects a SINGLR painting and applies dynamic normalization"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image. Please check path: {image_path}")
    
    orig = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    
    # Edge detection and morphological closing to fill gaps in lines
    edged = cv2.Canny(blurred, gradient_thresh1, gradient_thresh2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    
    cnts, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        print("No gradient edges detected.")
        return None
        
    # Assume the largest contour is the painting
    c = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(c)
    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect)
    
    # [FIXED]: Compatible with newer NumPy versions
    box = np.int32(box) 
    
    rect = order_points(box)
    (tl, tr, br, bl) = rect
    
    # Compute the width of the new image
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    
    # Compute the height of the new image
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    
    # Apply Perspective Transform
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))
    cv2.imwrite(output_path, warped)
    
    return warped, edged, box

def auto_detect_multiple_paintings(image_path, output_dir=".", gradient_thresh1=30, gradient_thresh2=100, min_area_ratio=0.01):
    """
    v1.1: MULTI-painting scanner with dynamic quadrilateral approximation
    :param min_area_ratio: Filters out noise smaller than 1% of total pixels
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image. Please check path: {image_path}")
    
    orig = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Calculate total image area for filtering
    total_area = img.shape[0] * img.shape[1]
    min_area = total_area * min_area_ratio
    
    # Increase blur slightly to smooth out paint textures and wall patterns
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    
    edged = cv2.Canny(blurred, gradient_thresh1, gradient_thresh2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    
    cnts, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    results = []
    
    for i, c in enumerate(cnts):
        # 1. Filter by area
        if cv2.contourArea(c) < min_area:
            continue
            
        # 2. [Core Logic]: Dynamic approximation to find 4 vertices
        approx = None
        peri = cv2.arcLength(c, True)
        
        # Loop through different precision values (from 1% to 10%)
        for eps in np.linspace(0.01, 0.1, 10):
            temp_approx = cv2.approxPolyDP(c, eps * peri, True)
            if len(temp_approx) == 4:
                approx = temp_approx
                break # Perfect quadrilateral found, exit loop
                
        # If 4 points aren't found in raw contour, try again using a Convex Hull
        # since using only convex hull is harder for recognize shape other than rectangle
        if approx is None:
            hull = cv2.convexHull(c)
            peri_hull = cv2.arcLength(hull, True)
            for eps in np.linspace(0.01, 0.1, 10):
                temp_approx = cv2.approxPolyDP(hull, eps * peri_hull, True)
                if len(temp_approx) == 4:
                    approx = temp_approx
                    break
        
        # Edge case: If still no quadrilateral, skip (likely an irregular shape/stain)
        if approx is None:
            continue
            
        box = approx.reshape(4, 2)
        box = np.int32(box) 
        
        rect = order_points(box)
        (tl, tr, br, bl) = rect
        
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = max(int(widthA), int(widthB))
        
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = max(int(heightA), int(heightB))
        
        if maxWidth == 0 or maxHeight == 0:
            continue
            
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")
        
        M = cv2.getPerspectiveTransform(rect, dst)
        
        # ========== Core Modification: Internal Cropping to remove frame thickness ==========
        # Perform perspective transform to get the straightened image with frame
        warped = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))
        
        # Set cropping ratio (e.g., cut 3% from each edge)
        # Increase to 0.04 or 0.05 if background/wall artifacts remain
        crop_ratio = 0.03 
        crop_h = int(maxHeight * crop_ratio)
        crop_w = int(maxWidth * crop_ratio)
        
        # Use NumPy slicing to "peel" off the edges
        # Ensure remaining image is valid before slicing to prevent errors
        if maxHeight > 2 * crop_h and maxWidth > 2 * crop_w:
            warped = warped[crop_h : maxHeight - crop_h, crop_w : maxWidth - crop_w]
        # ====================================================================================

        out_name = os.path.join(output_dir, f"scanned_painting_{i}.jpg")
        cv2.imwrite(out_name, warped)
        
        results.append({
            "warped": warped,
            "box": box,
            "filename": out_name
        })
        
    return results, edged
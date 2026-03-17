import cv2
import numpy as np
import os

def order_points(pts):
    """Sort four points in order: top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def extract_single_painting(image_path, output_path="output.jpg", crop_ratio=0.03):
    """
    Streamlined algorithm to detect, deskew, and crop a single, prominent painting.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    orig = img.copy()
    
    # Preprocessing: Grayscale and mild blur to preserve accurate corners
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge Detection
    edged = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    
    # Find Contours
    cnts, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        print("No edges found in the image.")
        return None
        
    # Sort contours by area in descending order and keep the top 3.
    # This prevents selecting the entire image border as the largest contour.
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:3]
    screenCnt = None
    
    for c in cnts:
        peri = cv2.arcLength(c, True)
        
        # Attempt 1: Dynamic polygon approximation on the raw contour
        for eps in np.linspace(0.01, 0.1, 10):
            approx = cv2.approxPolyDP(c, eps * peri, True)
            if len(approx) == 4:
                screenCnt = approx
                break
                
        # Attempt 2 (Core Fix): If 4 points aren't found, apply a Convex Hull 
        # to smooth out irregular edges, then retry approximation.
        if screenCnt is None:
            hull = cv2.convexHull(c)
            peri_hull = cv2.arcLength(hull, True)
            for eps in np.linspace(0.01, 0.1, 10):
                approx = cv2.approxPolyDP(hull, eps * peri_hull, True)
                if len(approx) == 4:
                    screenCnt = approx
                    break
        
        # Break the loop immediately once a valid 4-point polygon is found
        if screenCnt is not None:
            break
            
    # Fallback Mechanism: If even the Convex Hull fails, default to the minimum bounding box
    if screenCnt is None:
        print("Warning: Perfect quadrilateral not found even with Convex Hull. Falling back to bounding box.")
        c = cnts[0] 
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        screenCnt = np.int32(box).reshape(4, 1, 2)
        
    # Reshape and order the 4 points
    box = screenCnt.reshape(4, 2)
    rect = order_points(np.float32(box))
    (tl, tr, br, bl) = rect
    
    # --- Perspective Transformation and Cropping ---
    
    # Compute new dimensions
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    
    # Apply Perspective Transform (Deskew)
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))
    
    # Internal Cropping (Remove frame thickness)
    if crop_ratio > 0:
        crop_h = int(maxHeight * crop_ratio)
        crop_w = int(maxWidth * crop_ratio)
        if maxHeight > 2 * crop_h and maxWidth > 2 * crop_w:
            warped = warped[crop_h : maxHeight - crop_h, crop_w : maxWidth - crop_w]
            
    cv2.imwrite(output_path, warped)
    return warped, edged, box
import cv2

def preprocess_image(img):
    # แปลงภาพให้เป็น Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # ใช้เทคนิค GaussianBlur เพื่อลด Noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Normalization: ปรับค่าความเข้มของภาพให้อยู่ในช่วง 0-255
    normalized = cv2.normalize(blur, None, 0, 255, cv2.NORM_MINMAX)
    
    return normalized

# ฟังก์ชันสำหรับดึงลักษณะเด่นโดยใช้ SIFT
def extract_features_sift(img):
    # สร้างตัวตรวจจับ SIFT
    sift = cv2.SIFT_create()
    
    # ค้นหาลักษณะเด่น (keypoints) และสร้างตัวอธิบาย (descriptors)
    keypoints, descriptors = sift.detectAndCompute(img, None)
    
    return keypoints, descriptors

# ฟังก์ชันสำหรับจับคู่ลักษณะเด่นระหว่างสองภาพ
def match_features(des1, des2):
    # ใช้ BFMatcher สำหรับการจับคู่ลักษณะเด่นโดยใช้ L2 Norm
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    
    # เรียงลำดับการจับคู่ตามระยะทาง
    matches = sorted(matches, key=lambda x: x.distance)
    
    return matches

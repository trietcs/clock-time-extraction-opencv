# clock-time-extraction-opencv
### Chapter 1: Solving Methods
#### 1.1 Introduction
**Topic:** Retrieving the time from a photo of a clock  
**Requirements:**  
- **Input:** At least 5 photos containing analog wall clocks with varying content (different clock types, 1-2 hour time differences, non-coincident clock hands).  
- **Output:**  
  - Draw rectangular frames around hour, minute, and second hands in each photo, with distinct colors. Automated frame drawing, not manual.  
  - Output time as hours:minutes:seconds for each photo.

#### 1.2 Methods
##### 1.2.1 Overall
The method uses cv2.HoughCircles or cv2.findContours to detect clocks, then cv2.HoughLinesP to detect hands. Mathematical methods calculate hand rotation angles, leading to time inference.

##### 1.2.2 Detailed Description
1. **Step 1: Image Preprocessing**
   - Resize image for complexity reduction.
   - Convert BGR to HSV color space for brightness and color handling.
   - Invert HSV color to enhance contrast.
   - Apply Otsu thresholding to HSV channel V for binary image.
   - Gaussian blur to reduce noise.

2. **Step 2: Clock Detection**
   - Use cv2.HoughCircles or cv2.findContours.
   - Identify largest circle or contour as clock.

3. **Step 3: Detect Line Segments**
   - Apply Canny filter to find edges.
   - Use HoughLinesP to detect straight lines.

4. **Step 4: Group Lines**
   - Find close, nearly parallel lines.
   - Group lines within clock radius and similar angles.

5. **Step 5: Detect Clock Hands**
   - Find endpoints of line segments.
   - Create clock hands from farthest point to clock center.
   - Determine hand thickness.

6. **Step 6: Determine Hand Types**
   - Identify hour, minute, and second hands.

7. **Step 7: Draw Frames**
   - Use cv2.rectangle to frame clock hands.
   - Use cv2.putText to label hands.

8. **Step 8: Determine Rotation Angle**
   - Calculate angle between clock hands and center.

9. **Step 9: Calculate Time**
   - Calculate time from hand angles.

10. **Step 10: Draw Time on Image**
   - Draw time on Image
---
TEST

To test the clock time extraction functions on 10 images, run the following command:
python clock_time_extraction.py images/ adjusted_images/

This command will process the clock images located in the images/ directory and save the adjusted clock images in the adjusted_images/ directory.
Make sure to replace images/ and adjusted_images/ with the actual paths to your input and output directories.

---
**Author:** TrietCS  
**Course:** Introduction to Digital Image Processing

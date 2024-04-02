"""
Author: TrietCS
Description: This code is a final project for the "Introduction to Digital Image Processing" course. It serves as a reference for clock time extraction using OpenCV.
"""

import cv2
import numpy as np
import math
import os

# The resize_image function has the function of changing the size while maintaining the aspect ratio, with the longest side being 1000 pixels
def resize_image(img):
    height, width, _ = img.shape
    # Determine the scaling factor to make the longer side 1000 pixels
    scale_factor = 1000 / max(height, width)
    # Resize the image while preserving the aspect ratio
    img = cv2.resize(img, (int(width * scale_factor), int(height * scale_factor)))
    return img

# The clock_detection function has the function of detecting the clock from the image
def clock_detection(img, blurred):
    # Initialize variables to store the radius and center of the clock
    radius = 0
    center_x, center_y = 0, 0

    # Use the Hough method to find circles in the image
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 400, param1=50, param2=100, minRadius=100, maxRadius=500)

    # Initialize variable to store the largest circle
    max_circle = None
    
    if circles is not None:
        for circle in circles[0, :]:
            # Get the coordinates and radius of the circle
            x, y, r = circle

            # If the radius is greater than the current radius
            if r > radius:
                # Update the largest circle
                max_circle = circle

        # Get the coordinates and radius of the largest circle
        x, y, r = max_circle

        center_x = int(x)
        center_y = int(y)
        radius = int(r)
    # If no circle is found
    else:
        # Use the boundary finding method to find objects in the image
        contours, _ = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Initialize variables to store the area and largest rectangle
        max_area = 0
        max_rect = None
        
        for contour in contours:
            # Calculate the area of the contour
            area = cv2.contourArea(contour)

            # If the area is larger than the current area then update area and largest rectangle
            if area > max_area:
                max_area = area
                max_rect = contour
                
        if max_rect is not None:
            # Get the coordinates and size of the rectangle
            (x, y, w, h) = cv2.boundingRect(max_rect)

            # Calculate the coordinates of the center of the rectangle
            center_x = x + w // 2
            center_y = y + h // 2

            # Calculate the radius of the circle inscribed in the rectangle
            radius = min(w, h) // 2

    return center_x, center_y, radius

# The line_detection function has the function of detecting straight lines in an image
def line_detection(img, blurred):
    # Use Canny filter to find edges in image
    edges = cv2.Canny(blurred, 50, 150)

    # Use the Hough method to find straight lines from the edges
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=90, minLineLength=30, maxLineGap=5)
    return lines



# The group_lines_detection function has the function of finding lines that are close together and nearly parallel to group into a group
def group_lines_detection(lines, center_x, center_y, radius):
    groups =[]
    for line in lines:
        x1, y1, x2, y2 = line[0]

        # Calculate the length from the two ends of the line to the center of the clock
        length1 = np.sqrt((x1 - center_x)**2 + (y1 - center_y)**2)
        length2 = np.sqrt((x2 - center_x)**2 + (y2 - center_y)**2)

        # max_length furthest point from the center of the clock
        # min_length closest point to the center of the clock
        max_length = np.max([length1, length2])
        min_length = np.min ([length1, length2])

        # The farthest point must be within the radius of the clock and the nearest point must only be within 50% of the radius of the clock
        if ((max_length < radius) and (min_length < radius*50/100)):

            # Calculate the angle of the line in degrees
            angle = math.atan2(y2 - y1, x2 - x1)
            angle = math.degrees(angle)

            # Initialize flag variable to check whether the line belongs to any group or not
            grouped = False
            
            for group in groups:
                
                # Get the average angle of the group
                mean_angle = group['mean_angle']
                
                # If the angle of the line and the average angle of the group differ by less than 12 degrees or are equal when plus or minus 180 degrees
                # (this means the line is parallel or in the same direction as the group)
                if abs(angle - mean_angle) < 12 or abs(angle - mean_angle - 180) < 12 or abs(angle - mean_angle + 180) < 12:

                    # Add lines to the group
                    group['lines'].append(line)

                    # Set the flag variable to True to signal that the group has been found
                    grouped = True
                    break
            
            # If you cannot find a suitable group
            if not grouped:
                # Create a new group with its lines and angles
                groups.append({'lines': [line], 'mean_angle': angle})
    return groups

# The function distance between parallel lines has the function to calculate the distance between two parallel lines
def distance_between_parallel_lines(line1, line2):
    # Get the coordinates of two points on each line
    x1_1, y1_1, x2_1, y2_1 = line1[0]
    x1_2, y1_2, x2_2, y2_2 = line2[0]

    # Create two direction vectors of two straight lines
    vector1 = np.array([x2_1 - x1_1, y2_1 - y1_1])
    vector2 = np.array([x2_2 - x1_2, y2_2 - y1_2])

    #Creates a vector connecting a point on one line to a point on the other line
    vector_between_lines = np.array([x1_2 - x1_1, y1_2 - y1_1])

    #Calculates the perpendicular distance between the two lines.
    distance = np.abs(np.cross(vector1, vector_between_lines)) / np.linalg.norm(vector1)

    return distance

# The hands detection function has the function of finding the farthest endpoint from the clock center of a line segment among line segments
# in the same group to create a clock hand with the clock center point.
def hands_detection(groups, center_x, center_y):
    # Initialize a list to store clock hands
    hands = []

    # Browse through groups of lines
    for group in groups:
        # Get the list of lines in the group and number of lines
        lines = group['lines']
        num_lines = len(lines)

        # Initialize variables to store the maximum thickness and length of the lines
        max_thickness = 0
        max_length = 0
        
        # Browse lines in groups
        for i in range(num_lines):
            x1, y1, x2, y2 = lines[i][0]
            
            # Calculate the distance from two points to the center of the clock
            length1 = np.sqrt((x1 - center_x)**2 + (y1 - center_y)**2)
            length2 = np.sqrt((x2 - center_x)**2 + (y2 - center_y)**2)

            # Take the larger distance as the length of the line
            length = np.max([length1, length2])

            # If the length is greater than the current maximum length
            if length > max_length:
                max_length = length

                # Take the point farthest from the center as the end point of the clock hand
                if length == length1:
                    max_line = x1, y1, center_x, center_y
                else:
                    max_line = x2, y2, center_x, center_y

            # Browse through the remaining lines in the group
            for j in range(i+1, num_lines):
                # Calculate the distance between two lines using a distance_between_parallel_lines function
                thickness = distance_between_parallel_lines(lines[i], lines[j])

                # Update maximum thickness
                if (thickness > max_thickness):
                    max_thickness = thickness

        # Create a set of line, thickness and length
        line = max_line, max_thickness, max_length

        # If the thickness is greater than 0, it means there are at least two parallel lines
        if max_thickness > 0:
            # Add this set to the clock hands list
            hands.append(line)

    # Sort the list of clock hands by length in descending order
    hands.sort(key=lambda x: x[2], reverse=True)

    # Take the first three clock hands as the clock hands
    hands = hands[:3]
    return hands

# The get_hands function has the function of accurately determining the hour, minute, and second hands
# from the 3 clock hands found in the hands_detection function.
def get_hands(hands):
    # Arrange the clock hands by thickness
    sorted_hands_by_thickness = sorted(hands, key=lambda hands: hands[1])

    # The second hand is the hand with the smallest thickness
    second_hand = sorted_hands_by_thickness[0]

    # Remove the second hand from the list containing 3 clock hands
    hands.remove(second_hand)

    # Arrange the remaining 2 clock hands by length
    sorted_hands_by_length = sorted(hands, key=lambda hands: hands[2])

    # The hour hand is the hand with the shortest length and the remaining hand is the minute hand
    hour_hand = sorted_hands_by_length[0]
    minute_hand = sorted_hands_by_length[1]

    return hour_hand, minute_hand, second_hand

# Function to calculate coordinates of a rectangle surrounding a straight line
def calculate_rect_coordinates(line):
    x1, y1, x2, y2 = line[0]

    # The x coordinate of the rectangle is the smallest value of x1 and x2
    # The y coordinate of the rectangle is the smallest value of y1 and y2
    rect_x = min(x1, x2)
    rect_y = min(y1, y2)

    # The width of the rectangle is the absolute value of the difference x2 and x1
    # The height of the rectangle is the absolute value of the difference y2 and y1
    rect_width = abs(x2 - x1)
    rect_height = abs(y2 - y1)

    # The coordinates of the location to note are the coordinates of the first point on the line
    text_x, text_y = x1, y1
    return rect_x, rect_y, rect_width, rect_height, text_x, text_y

# The draw_hands_frame function has the function of drawing a rectangular frame and labels for the clock hands
def draw_hands_frame(img, hour_hand, minute_hand, second_hand):
    # Draw rectangle and add label for hour hand
    rect_x, rect_y, rect_width, rect_height, text_x, text_y = calculate_rect_coordinates(hour_hand)
    cv2.rectangle(img, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (0, 0, 255), 3)
    cv2.putText(img, 'Hour', (int(text_x), int(text_y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Draw rectangle and add label for minute hand
    rect_x, rect_y, rect_width, rect_height, text_x, text_y = calculate_rect_coordinates(minute_hand)
    cv2.rectangle(img, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (0, 255, 0), 3)
    cv2.putText(img, 'Minute', (int(text_x), int(text_y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Draw rectangle and add label for second hand
    rect_x, rect_y, rect_width, rect_height, text_x, text_y = calculate_rect_coordinates(second_hand)
    cv2.rectangle(img, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (255, 0, 0), 3)
    cv2.putText(img, 'Second', (int(text_x), int(text_y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

# Function to calculate direction vector of a clock hand
def get_vector(hand):
    x1, y1, x2, y2 = hand[0]
    vector = [x2 - x1, y2 - y1]
    return vector

# Function to calculate the dot product of two vectors
def dot_product(u, v):
    return u[0] * v[0] + u[1] * v[1]

# The function calculates the directional product of two vectors
def cross_product(u, v):
    return u[0] * v[1] - u[1] * v[0]

# Function to calculate the angle of a clock hand relative to the y direction
def get_angle(hand, center_x, center_y):
    # u is the direction vector of the clock hands
    u = get_vector(hand)

    # Create a horizontal direction vector from the center of the clock
    v = [center_x - center_x, center_y - (center_y-100)]

    # Call the function to calculate the dot product of two vectors
    dot_uv = dot_product(u, v)

    # Calculate the length of vector u and v
    length_u = math.sqrt(u[0]**2 + u[1]**2)
    length_v = math.sqrt(v[0]**2 + v[1]**2)

    # Calculate the cosine of the angle between two vectors using the formula u.v / (|u| * |v|)
    cos_theta = dot_uv / (length_u * length_v)

    # Limit the value of cos to the range [-1, 1] to avoid errors when calculating arccos
    cos_theta = max(min(cos_theta, 1.0), -1.0)

    # Calculate the angle using the formula arccos(cos_theta)
    theta = math.acos(cos_theta)

    # Convert angle from radians to degrees
    theta_degrees = math.degrees(theta)

    # If the directional product is greater than 0, that means vector u is to the left of vector v
    # Conversely, if the directional product is less than or equal to 0, that means vector u is to the right or in the same direction as vector v
    cross_uv = cross_product(u, v)
    if cross_uv > 0:
        # Returns the complementary angle of theta
        return 360 - theta_degrees
    else:
        return theta_degrees
    
# The get_time function has the function of calculating time from the angles of the clock hands
def get_time(hour_angle, minute_angle, second_angle):
    # Calculate the time from the angle of the hour hand by dividing by 30 (each hour corresponds to 30 degrees)
    hour = hour_angle / 30

    # Calculate minutes and seconds from the angle of the minute and second hands by dividing by 6 (each minute or second corresponds to 6 degrees)
    minute = minute_angle / 6
    second = second_angle / 6

    #Adjust to avoid errors

    # If the angle of the hour hand is close to an integer multiplied with 30 (i.e. close to a specific hour)
    # and the angle of the minute hand is close to 0 or 360 (i.e. close to 12 o'clock)
    if (round(hour)*30 - hour_angle <= 6) and ((355 < minute_angle and minute_angle < 360) or (minute_angle < 90)):
        # Round hour up or down
        hour = round(hour)
        if hour == 12:
            hour = 0
    
    # If the angle of the hour hand is close to a specific hour
    # and the angle of the minute hand is close to 360 (ie close to 12 o'clock)
    # Then set minute to 0
    if (hour_angle - hour*30 <= 6) and (355 < minute_angle and minute_angle < 360):
        minute = 0

    # If the angle of the minute hand is close to an integer multiplied with 6 (i.e. close to a specific minute)
    # and the angle of the second hand is approximately between 0 and 6 (i.e. 1 round of 60 seconds has passed).
    if (round(minute)*6 - minute_angle <= 6) and (second_angle < 6):
        # Round minutes up or down
        minute = round(minute)
        if minute == 60:
            minute = 0

    # If the angle of the minute hand is close to a specific minute
    # and the angle of the second hand is close to 360 (ie close to 12 o'clock)
    # Then set second to 0
    if (minute_angle - minute*30 <= 6) and (354 < second_angle and second_angle < 360):
        second = 0

    hour = int(hour)
    minute = int(minute)
    second = int(second)

    # Create a time series in hh:mm:ss format
    time = f"{hour:02d}:{minute:02d}:{second:02d}"\

    return time

# The draw_time function has the function of drawing time on a clock image
def draw_time(img, time):
    # Choose the font, size and thickness of the text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_thickness = 3

    # Choose a location to write text on the image
    text_position = (50, 100)

    # Choose the color of the text
    text_color = (0, 0, 0)

    # Write text on the image with selected parameters
    cv2.putText(img, time, text_position, font, font_scale, text_color, font_thickness)

def solve(img):
    # Step 1: image preprocessing includes resizing the image and increasing contrast
    # and reducing noise to increase the likelihood of detecting the clock
    img = resize_image(img)
    # Process images before searching for clock
    img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV) # Convert image from BGR color space to HSV
    img_hsv = cv2.bitwise_not(img_hsv)  # Invert color values in HSV space
    # Create a CLAHE object to balance the brightness of the image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # Apply CLAHE to the V (brightness) channel of the HSV image
    img_hsv[:, :, 2] = clahe.apply(img_hsv[:, :, 2])
    # Generate a binary threshold for channel V of the HSV image using the Otsu method
    _, thresh = cv2.threshold(img_hsv[:, :, 2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Blur the image with a Gaussian filter to reduce noise
    blurred = cv2.GaussianBlur(thresh, (5, 5), 0)

    # Step 2: detect the clock
    center_x, center_y, radius = clock_detection(img, blurred)
    
    # Step 3: detect line segments in the clock
    lines = line_detection(img, blurred)

    # Step 4: finding lines that are close together and nearly parallel to group into a group
    groups = group_lines_detection(lines, center_x, center_y, radius)

    # Step 5: detect the clock hands
    hands = hands_detection(groups, center_x, center_y)

    # Step 6: Determine which hand is the hour hand, which hand is the minute hand, and which hand is the second hand
    hour_hand, minute_hand, second_hand = get_hands(hands)

    # Step 7: draw a frame around and label the clock hands back on the image
    draw_hands_frame(img, hour_hand, minute_hand, second_hand)

    # Step 8: determine the rotation angle of the clock hands
    hour_angle = get_angle(hour_hand, center_x, center_y)
    minute_angle = get_angle(minute_hand, center_x, center_y)
    second_angle = get_angle(second_hand, center_x, center_y)

    # Step 9: calculate the clock time based on the rotation angle in step 8
    time = get_time(hour_angle, minute_angle, second_angle)

    # Step 10: draw time on the image
    draw_time(img, time)

    return img
    
def main(input_dir, output_dir):
    # Iterate over images in the input directory
    for i in range(1, 11):  # Assuming there are 10 images named clock1.jpg, clock2.jpg, ..., clock10.jpg
        filename = f'clock{i}.jpg'
        img_path = os.path.join(input_dir, filename)
        
        if not os.path.exists(img_path):
            continue  # Skip if the file does not exist
        
        img = cv2.imread(img_path)

        img = solve(img)

        result_path = os.path.join(output_dir, f"result_{filename}")
        cv2.imwrite(result_path, img)

        cv2.imshow(filename, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Adjust the clock from images in a directory')
    parser.add_argument('input_dir', type=str, help='Input directory containing clock images')
    parser.add_argument('output_dir', type=str, help='Output directory for the adjusted clock images')
    args = parser.parse_args()

    main(args.input_dir, args.output_dir)
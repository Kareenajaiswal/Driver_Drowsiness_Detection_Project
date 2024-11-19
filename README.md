# Driver Drowsiness Detection System

## **Overview**
The **Driver Drowsiness Detection System** is designed to enhance road safety by continuously monitoring the driver's facial and eye behaviors in real time. Using a webcam and advanced computer vision techniques, it detects signs of drowsiness and issues timely alerts, encouraging the driver to take necessary breaks. This system is suitable for personal and commercial use, providing a practical, technology-driven solution to reduce fatigue-related accidents.

---

## **Features**
- **Real-Time Face and Eye Detection**:  
  Utilizes a webcam to monitor facial landmarks, ensuring accurate detection of key features.
  
- **Drowsiness Detection**:  
  Uses the Eye Aspect Ratio (EAR) metric to identify prolonged eye closure, indicating fatigue.
  
- **Alert Mechanisms**:  
  - **Visual Alerts**: On-screen messages like "DROWSINESS DETECTED!"  
  - **Audio Alerts**: Warning sound via `pygame`.  
  - **SMS Alerts**: Notifications to emergency contacts using Twilio API.
  
- **Fatigue Warning System**:  
  Tracks repeated drowsiness events and encourages breaks to prevent accidents.  

- **Logging**:  
  Records all drowsiness events with timestamps for future analysis.

---

## **Technologies Used**
- **Programming Language**: Python 3.x  
- **Libraries and Frameworks**:
  - OpenCV: For real-time image processing.
  - Mediapipe: For facial landmark detection.
  - Twilio: For sending SMS notifications.
  - pygame: For audio alerts.
  - scipy: For EAR calculations.

---

## **Installation**
1. **Prerequisites**:
   - Python 3.x installed on your system.
   - A functional webcam.

2. **Library Installation**:  
   Install the required libraries by running:
   ```bash
   pip install opencv-python mediapipe twilio pygame scipy
   ```

3. **Twilio Setup**:
   - Sign up on [Twilio](https://www.twilio.com/).
   - Obtain your `ACCOUNT_SID`, `AUTH_TOKEN`, and a Twilio phone number.
   - Replace placeholders in the code with these credentials.

4. **Project Setup**:
   - Clone this repository:
     ```bash
     git clone <repository_url>
     cd <project_folder>
     ```
   - Ensure the file `alert.wav` is in the project directory.

---

## **How to Run**
1. Ensure the webcam is connected.
2. Navigate to the project directory.
3. Run the main script:
   ```bash
   python main.py
   ```
4. Follow on-screen prompts and ensure alerts work as expected.

---

## **Known Issues and Limitations**
- **Camera Quality**: Low-resolution cameras may affect detection accuracy.
- **Lighting Conditions**: Poor or overly bright lighting impacts performance.
- **Fixed EAR Threshold**: May not suit all users, affecting detection reliability.
- **SMS Delivery Delays**: Dependent on network conditions.

---

## **Future Enhancements**
- Adaptive EAR thresholds based on individual blink patterns.
- Improved low-light detection with infrared sensors.
- Advanced fatigue tracking for personalized alerts.

---

## **Acknowledgments**
Special thanks to:
- **OpenCV** and **Mediapipe** for facial landmark detection tools.
- **Twilio** for the SMS notification service.
- **pygame** for seamless audio integration.

---

## **License**
This project is licensed under the MIT License. Feel free to contribute and adapt.

---

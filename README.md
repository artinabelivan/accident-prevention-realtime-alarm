# Systems to Detect and Alarm Probability of Accidents in Real Time

**Inventor & Author:** Artina Belivan  
**Intellectual Property:** Provisional Patent Filed September 2022  
**Focus Areas:** Real-Time Computer Vision, Kinematic Trajectory Modeling, Edge Latency Compensation, Behavioral Intervention  

---

## 1. Executive Summary
Traditional perimeter-security and hazard-alert systems function reactively (*post-breach*), generating alerts only after an individual has crossed into a hazardous zone (e.g., falling into a swimming pool or stepping onto transit tracks). 

This project implements a proactive intervention model designed to evaluate accident probability *prior to impact*. By coupling high-frequency YOLO object tracking with a calibrated **sensory latency compensation model**, the pipeline predicts an individual's trajectory and executes multi-tiered warnings before a boundary is crossed.

---

## 2. Theoretical Framework & Mathematical Model

In physical edge systems, real-time response times are constrained by sensory latency ($\Delta t_{\text{latency}}$), which represents the sum of camera frame buffer intervals, deep learning inference latency, and notification dispatch overhead:

$$\Delta t_{\text{latency}} = \Delta t_{\text{capture}} + \Delta t_{\text{inference}} + \Delta t_{\text{decision}}$$

To compensate for this latency, the algorithm tracks human ground-contact coordinates $\vec{P}(t)$ and computes an exponentially smoothed instantaneous velocity vector $\vec{v}(t)$. The model continuously projects the predicted physical position: $$\vec{P}_{\text{projected}} = \vec{P}_{\text{current}} + \vec{v} \cdot \Delta t_{\text{latency}}$$

### Perimeter Risk Evaluation
Let $\Omega$ represent the bounded polygonal hazard zone (such as a swimming pool edge) and $\partial\Omega$ its perimeter. The signed Euclidean distance $D(\vec{P})$ from the subject's ground position to the boundary is defined by: $$D(\vec{P}) = \text{dist}(\vec{P}, \partial\Omega) \cdot \operatorname{sgn}(\vec{P} \notin \Omega)$$

* **Safe State:** $D(\vec{P}_{\text{projected}}) > d_{\text{warning}}$
* **Approaching Warning:** $d_{\text{critical}} < D(\vec{P}_{\text{projected}}) \le d_{\text{warning}}$
* **Critical Collision Intervention:** $D(\vec{P}_{\text{projected}}) \le d_{\text{critical}}$ or $D(\vec{P}_{\text{current}}) \le 0$

---

## 3. Engineering Architecture
[Camera Stream / RTSP]
│
▼
[YOLO High-Throughput Detection Engine (Class 0: Person)]
│
▼
[Ground-Plane Footprint Extraction (x_mid, y_max)]
│
▼
[Kalman / Exponential Smoothing Velocity Estimator]
│
▼
[Sensory Latency Forward Projection: P_proj = P + v * dt]
│
▼
[Signed Polygon Distance Field vs. Geofenced Hazard Zone]
│
┌────┴──────────────────────────────┐
▼                                   ▼
[Effective Dist <= Critical]      [Effective Dist <= Warning]
│                                   │
▼                                   ▼
• Synthesized Audio Alarm (920Hz)  • Visual Warning HUD
• Desktop Push Notification Alert  • Trajectory Vector Overlay

---

## 4. Installation & Execution

### Prerequisites
* Python 3.9+
* Webcam or RTSP network camera stream

### Setup
```bash
# Clone the repository
git clone [https://github.com/artinabelivan/accident-prevention-realtime-alarm.git](https://github.com/artinabelivan/accident-prevention-realtime-alarm.git)
cd accident-prevention-realtime-alarm

# Install dependencies
pip install -r requirements.txt

# Run the real-time detector
python realtime_collision_detector.py

(Press q within the OpenCV visual display window to terminate the stream).

## 5. Academic Citation & Context
Developed as part of independent research on human-centered accident prevention and predictive decision support algorithms (Provisional Patent Filed Sept. 2022).


# Echo-Yolo: Robotics Teleoperation with Digital Twin Visualization

A real-time robotics teleoperation system that combines Echo device control with UR3 manipulators and features a live digital twin visualization overlay.

## 🚀 Features

- **Dual UR3 Teleoperation**: Control two UR3 robotic arms simultaneously using an Echo device
- **Real-time Digital Twin**: Live 3D skeleton visualization overlaid on camera feed
- **Gripper Integration**: Robotiq 2F-85 gripper state visualization and control
- **Data Collection**: Save training episodes in LeRobot format (video, actions, observations)
- **Camera Calibration**: Integrated camera calibration with manual base pixel calibration
- **Forward Kinematics**: Real-time DH parameter-based joint position calculation

## 🛠️ System Requirements

- Python 3.8+
- Conda environment with 'echo' environment
- Intel RealSense camera (RGB + Depth)
- Echo device for teleoperation
- Two UR3 robots with Robotiq 2F-85 grippers
- Network connection to robots (IP: 192.168.88.40, 192.168.88.56)

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone <your-github-repo-url>
   cd Echo-Yolo/Echo
   ```

2. **Activate conda environment**:
   ```bash
   conda activate echo
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Usage

### 1. Camera Calibration

First, calibrate your camera and robot base positions:

```bash
# Generate calibration board
python calibration/scripts/generate_calibration_board.py

# Capture calibration images
python calibration/scripts/capture_images.py

# Calibrate camera intrinsics
python calibration/scripts/calibrate_intrinsics.py

# Validate calibration
python calibration/scripts/validate_calibration.py

# Calibrate robot base pixel positions
python calibrate_base_pixels.py
```

### 2. Run Teleoperation with Digital Twin

```bash
python echo_main_lerobot.py
```

## 📁 Project Structure

```
Echo/
├── echo_main_lerobot.py          # Main teleoperation script with digital twin
├── calibrate_base_pixels.py      # Manual base pixel calibration tool
├── cameras.py                    # Camera interface (RealSense, WebCamera)
├── data_utils.py                 # Data collection utilities
├── echo_teleoperation.py         # Echo device interface
├── ur_rtde.py                    # UR3 robot control interface
├── robotiq_gripper.py            # Robotiq gripper control
├── calibration/                  # Camera calibration tools and data
│   ├── configs/
│   ├── data/
│   ├── results/
│   └── scripts/
├── dataset/                      # Training data storage
└── test/                         # Test scripts
```

## 🔧 Configuration

### Robot Base Positions
- **R1 Base**: (0, 0, 10) mm (world origin)
- **R2 Base**: (1100, 0, 10) mm

### Camera Configuration
- **Position**: (550, -800, 1000) mm
- **Rotation**: 45° down to observe both manipulators
- **Resolution**: 1280x720

### Gripper Configuration
- **Model**: Robotiq 2F-85
- **Range**: 0-255 (0=closed, 255=open)
- **Threshold**: 128 (below=closed, above=open)

## 🎮 Digital Twin Features

### Visualization Components
- **R1 Skeleton**: Green color, 6 joints + gripper
- **R2 Skeleton**: Blue color, 6 joints + gripper
- **Base Markers**: Static circles showing calibrated base positions
- **Gripper State**: Real-time OPEN/CLOSED status display
- **Coordinate Axes**: Reference axes at R1 base

### Kinematics
- **DH Parameters**: UR3 standard Denavit-Hartenberg parameters
- **Forward Kinematics**: Real-time joint position calculation
- **3D to 2D Projection**: Camera intrinsics and extrinsics
- **Gripper Kinematics**: Simplified Robotiq 2F-85 finger model

## 📊 Data Collection

The system saves training episodes in LeRobot format:

```
dataset/
└── episode_0000/
    ├── rgb.mp4          # Video recording with digital twin overlay
    ├── actions.npz      # Gripper actions (left, right)
    └── observations.npz # Joint observations (left_joints, right_joints)
```

## 🐛 Troubleshooting

### Common Issues

1. **Array Shape Error**: Ensure numpy arrays are properly typed with `dtype=np.float64`
2. **Camera Connection**: Check RealSense camera connection and drivers
3. **Robot Connection**: Verify network connectivity to UR3 robots
4. **Echo Device**: Ensure Echo device is properly connected and calibrated

### Debug Mode

Enable debug output:
```bash
export DEBUG_VISUALIZATION=1
python echo_main_lerobot.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- UR3 robot control via UR-RTDE
- Intel RealSense camera integration
- Echo device teleoperation interface
- LeRobot data format compatibility

## 📞 Support

For issues and questions, please open an issue on GitHub or contact the development team. 
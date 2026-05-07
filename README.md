# LockIn! 🔒

An anti-distraction productivity tool that detects phone usage via webcam and alerts you in real-time to keep you focused on what matters.

## Features

- **Real-time Phone Detection**: Uses MediaPipe hand detection (95%+ accuracy) + lightweight phone classifier
- **Visual Alerts**: Bold red "Lock In!" overlay when phone is detected in your hand
- **Audio Alerts**: Irritating auditory alert that loops until you put your phone away
- **Positive Reinforcement**: "Good Boy!" alert when you successfully put your phone down
- **Cross-Platform**: Works on Windows and macOS
- **Lightweight**: Runs efficiently on CPU-only machines

## Quick Start

### Installation

```bash
# Clone the repo
git clone https://github.com/suhibhit/lockin.git
cd lockin

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # On Mac: source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running LockIn!

```bash
# Start the application
python -m lockin

# With custom settings
python -m lockin --camera 0 --confidence 0.5 --volume 80
```

### Controls While Running

- **Q** or **ESC** - Quit the application
- **M** - Mute/unmute audio alerts (DO NOT RECOMMEND AS THIS MAKES THE PROGRAM USELESS: kept for emergency situations)
- **D** - Toggle debug information display

## How It Works

1. **Hand Detection**: MediaPipe detects your hands in real-time (21 keypoints per hand)
2. **Phone Classification**: Analyzes cropped hand regions to determine if you're holding a phone
3. **Alert Trigger**: If phone detected in hand → Visual + Audio alerts
4. **Alert Loop**: Alerts continue until phone is removed or hand is empty
5. **Positive Reinforcement**: Once phone is gone → "Good Boy!" alert plays

## Audio Files

You need to provide three audio files in `lockin/sounds/`:

1. **`phone_detected.wav`** (~2-3 seconds)
   - Irritating voice
   - Purpose: Initial alert when phone detected

2. **`phone_detected_loop.wav`** (shorter duration)
   - Looping version of the alert
   - Purpose: Repeats until phone is removed

3. **`good_boy.wav`** (~1-2 seconds)
   - Encouraging tone: "Good boy!"
   - Purpose: Positive reinforcement when phone is put down

The app will warn you if audio files are missing.

## Performance Targets

- **Hand Detection**: 60 FPS on CPU (Intel i5+)
- **Phone Classification**: ~40ms per hand inference
- **Total Latency**: <100ms from detection to alert
- **CPU Usage**: <20% on i5-level machines
- **Memory**: <200MB runtime

## Technology Stack

- **OpenCV** - Webcam capture and real-time rendering
- **MediaPipe** - Hand landmark detection (95%+ accuracy)
- **YOLOv8** - Phone classification on hand regions
- **Pygame** - Cross-platform audio playback
- **NumPy** - Array operations
- **Pillow** - Image processing

## Development

For detailed development information, see [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md).

### Running Tests

```bash
# Runs all tests
pytest tests/

# Runs specific test
pytest tests/test_hand_detector.py -v

# Runs with coverage turned on
pytest tests/ --cov=lockin --cov-report=html
```

### Code Quality

```bash
# Formats code
black lockin/ tests/

# Lints code
flake8 lockin/ tests/ --max-line-length=100

# Type checking
mypy lockin/
```

## Project Structure

```
lockin/
├── lockin/
│   ├── __init__.py              # Package initialization
│   ├── main.py                  # Entry point
│   ├── camera.py                # Webcam capture
│   ├── hand_detector.py         # MediaPipe integration
│   ├── phone_classifier.py      # Phone detection
│   ├── alert_system.py          # Alert management
│   ├── ui.py                    # Display rendering
│   ├── config.py                # Configuration
│   └── utils.py                 # Utilities
├── tests/                        # Unit and integration tests
├── sounds/                       # Audio files (user-provided)
├── docs/                         # Documentation
├── README.md                     # This file
├── requirements.txt              # Dependencies
├── setup.py                      # Package setup
├── LICENSE                       # MIT License
└── .gitignore                    # Git ignore rules
```

## Minimum System Requirements

- **Python**: 3.9 or higher
- **OS**: Windows 10/11 or macOS (Intel/Apple Silicon)
- **Camera**: Webcam with at least 720p resolution
- **RAM**: 4GB minimum (8GB recommended)
- **CPU**: Intel i5 equivalent or better (GPU better)

## Known Limitations

- Single person detection only (detects the person in-front of the camera)
- Requires good lighting conditions for accurate hand detection
- Optimal detection range: 50-120cm from camera

## Troubleshooting

**Hand detection not working?**
- Check lighting conditions (bright, diffuse light works best)
- Ensure camera has clear view of your hands
- Try adjusting camera angle and distance

**Audio alerts not playing?**
- Verify WAV files are in `lockin/sounds/` directory
- Check file format: 16-bit PCM, 44.1 kHz recommended
- Test system audio volume is not muted (lol)

**Camera access denied?**
- **macOS**: Check System Preferences → Security & Privacy → Camera
- **Windows**: Check Settings → Privacy → Camera

## Future Enhancements

- Desktop app wrapper (PyQt5/Tkinter)
- Custom model fine-tuning
- Analytics dashboard (daily focus metrics)
- Mobile app (iOS/Android)
- Integration with productivity apps (Toggl, RescueTime)
- Team accountability features

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or feedback, feel free to open an issue on GitHub.

---


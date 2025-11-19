# LeRobot Command Examples

## Finding Available Cameras
To detect cameras on your system:
```bash
lerobot-find-cameras
```


## Teleoperate with SO101 Arms and Dual Cameras

### Command
```bash
lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=my_follower_arm \
  --robot.cameras='{
    wrist: {type: opencv, index_or_path: "/dev/video0", width: 640, height: 480, fps: 30},
    top: {type: opencv, index_or_path: "/dev/video2", width: 640, height: 480, fps: 30}
  }' \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=my_leader_arm \
  --display_data=true
```



## How to Stop
Press **Ctrl+C** in the terminal to stop the teleoperation session gracefully.




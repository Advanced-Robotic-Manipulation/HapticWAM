"""Read-only wiring probe: UR RTDE receive, Robotiq socket, RealSense. No motion."""
import socket
print("--- UR 192.168.88.56")
try:
    import rtde_receive
    r = rtde_receive.RTDEReceiveInterface("192.168.88.56")
    q = r.getActualQ(); tcp = r.getActualTCPPose()
    print("VERDICT LIVE arm-rtde: q[0..2]=%s tcp z=%.3f m" % ([round(x,2) for x in q[:3]], tcp[2]))
    r.disconnect()
except Exception as e:
    print("VERDICT DEAD arm-rtde:", type(e).__name__, str(e)[:80])
print("--- Robotiq urcap socket :63352")
try:
    s = socket.create_connection(("192.168.88.56", 63352), timeout=3)
    s.sendall(b"GET POS\n"); pos = s.recv(64).decode().strip()
    s.sendall(b"GET STA\n"); sta = s.recv(64).decode().strip()
    s.close()
    print("VERDICT LIVE gripper: %s %s" % (pos, sta))
except Exception as e:
    print("VERDICT DEAD gripper:", type(e).__name__, str(e)[:80])
print("--- RealSense")
try:
    import pyrealsense2 as rs
    devs = list(rs.context().query_devices())
    for d in devs:
        print("VERDICT LIVE camera:", d.get_info(rs.camera_info.name),
              "usb", d.get_info(rs.camera_info.usb_type_descriptor))
    if not devs: print("VERDICT DEAD camera: none enumerated")
except Exception as e:
    print("VERDICT DEAD camera:", type(e).__name__, str(e)[:80])

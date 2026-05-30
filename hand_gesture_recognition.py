import cv2
import mediapipe as mp
import time
from datetime import datetime
import numpy as np

# ====================== 官方基础组件初始化 ======================
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# ====================== 全局配置 ======================
# 手指关键点索引
FINGER_TIPS = [4, 8, 12, 16, 20]  # 指尖
FINGER_PIPS = [3, 7, 11, 15, 19]  # 近端指关节
FINGER_MCP = [2, 6, 10, 14, 18]  # 掌指关节

# 手势历史记录（保存最近10个手势）
gesture_history = []
MAX_HISTORY = 10

# 动态手势检测变量
wave_detection_frames = []
WAVE_THRESHOLD = 15  # 挥手移动阈值
WAVE_MIN_FRAMES = 8  # 挥手最少需要的帧数


# ====================== 手势识别核心函数 ======================
def count_fingers(hand_landmarks, hand_label):
    """计算伸出的手指数量"""
    fingers = []

    # 拇指特殊处理
    if hand_label == "Right":
        fingers.append(
            1 if hand_landmarks.landmark[FINGER_TIPS[0]].x < hand_landmarks.landmark[FINGER_PIPS[0]].x else 0)
    else:
        fingers.append(
            1 if hand_landmarks.landmark[FINGER_TIPS[0]].x > hand_landmarks.landmark[FINGER_PIPS[0]].x else 0)

    # 其他四指
    for i in range(1, 5):
        fingers.append(
            1 if hand_landmarks.landmark[FINGER_TIPS[i]].y < hand_landmarks.landmark[FINGER_PIPS[i]].y else 0)

    return fingers


def is_ok_gesture(hand_landmarks):
    """识别OK手势：拇指和食指指尖接触，其他手指伸直"""
    thumb_tip = hand_landmarks.landmark[FINGER_TIPS[0]]
    index_tip = hand_landmarks.landmark[FINGER_TIPS[1]]

    # 计算拇指和食指指尖的距离
    distance = np.sqrt((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2)

    # 其他手指是否伸直
    other_fingers_extended = True
    for i in range(2, 5):
        if hand_landmarks.landmark[FINGER_TIPS[i]].y > hand_landmarks.landmark[FINGER_PIPS[i]].y:
            other_fingers_extended = False
            break

    return distance < 0.05 and other_fingers_extended


def is_thumbs_up(hand_landmarks):
    """识别点赞手势：拇指向上，其他手指弯曲"""
    thumb_tip = hand_landmarks.landmark[FINGER_TIPS[0]]
    thumb_mcp = hand_landmarks.landmark[FINGER_MCP[0]]

    # 拇指是否向上
    thumb_up = thumb_tip.y < thumb_mcp.y - 0.1

    # 其他手指是否弯曲
    other_fingers_bent = True
    for i in range(1, 5):
        if hand_landmarks.landmark[FINGER_TIPS[i]].y < hand_landmarks.landmark[FINGER_PIPS[i]].y:
            other_fingers_bent = False
            break

    return thumb_up and other_fingers_bent


def is_thumbs_down(hand_landmarks):
    """识别踩手势：拇指向下，其他手指弯曲"""
    thumb_tip = hand_landmarks.landmark[FINGER_TIPS[0]]
    thumb_mcp = hand_landmarks.landmark[FINGER_MCP[0]]

    thumb_down = thumb_tip.y > thumb_mcp.y + 0.1

    other_fingers_bent = True
    for i in range(1, 5):
        if hand_landmarks.landmark[FINGER_TIPS[i]].y < hand_landmarks.landmark[FINGER_PIPS[i]].y:
            other_fingers_bent = False
            break

    return thumb_down and other_fingers_bent


def is_rock_gesture(hand_landmarks):
    """识别摇滚手势：食指和小指伸直，中指和无名指弯曲"""
    fingers = count_fingers(hand_landmarks, "Right")  # 标签不影响计数
    return fingers[1] == 1 and fingers[4] == 1 and fingers[2] == 0 and fingers[3] == 0


def is_victory_gesture(hand_landmarks):
    """识别胜利手势：食指和中指伸直，其他手指弯曲"""
    fingers = count_fingers(hand_landmarks, "Right")
    return fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0


def is_peace_gesture(hand_landmarks):
    """识别和平手势：食指和中指伸直分开，其他手指弯曲"""
    fingers = count_fingers(hand_landmarks, "Right")
    if not (fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0):
        return False

    # 检查食指和中指是否分开
    index_tip = hand_landmarks.landmark[FINGER_TIPS[1]]
    middle_tip = hand_landmarks.landmark[FINGER_TIPS[2]]
    distance = np.sqrt((index_tip.x - middle_tip.x) ** 2 + (index_tip.y - middle_tip.y) ** 2)
    return distance > 0.05


def is_call_me_gesture(hand_landmarks):
    """识别打电话手势：拇指和小指伸直，其他手指弯曲"""
    fingers = count_fingers(hand_landmarks, "Right")
    return fingers[0] == 1 and fingers[4] == 1 and fingers[1] == 0 and fingers[2] == 0 and fingers[3] == 0


def detect_wave_gesture(wrist_positions):
    """检测挥手动态手势"""
    if len(wrist_positions) < WAVE_MIN_FRAMES:
        return False

    # 计算x方向的移动距离
    x_positions = [pos[0] for pos in wrist_positions]
    max_x = max(x_positions)
    min_x = min(x_positions)
    movement = max_x - min_x

    # 计算方向变化次数
    direction_changes = 0
    for i in range(1, len(x_positions) - 1):
        if (x_positions[i] - x_positions[i - 1]) * (x_positions[i + 1] - x_positions[i]) < 0:
            direction_changes += 1

    return movement > WAVE_THRESHOLD and direction_changes >= 2


def get_gesture_name(hand_landmarks, hand_label):
    """综合识别所有手势"""
    fingers = count_fingers(hand_landmarks, hand_label)
    finger_count = sum(fingers)

    # 优先识别特殊手势
    if is_ok_gesture(hand_landmarks):
        return "OK 👌"
    elif is_thumbs_up(hand_landmarks):
        return "点赞 👍"
    elif is_thumbs_down(hand_landmarks):
        return "踩 👎"
    elif is_rock_gesture(hand_landmarks):
        return "摇滚 🤘"
    elif is_call_me_gesture(hand_landmarks):
        return "打电话 🤙"
    elif is_peace_gesture(hand_landmarks):
        return "和平 ✌️"
    elif finger_count == 0:
        return "拳头 ✊"
    elif finger_count == 1:
        return "数字 1 ☝️"
    elif finger_count == 2:
        return "数字 2 ✌️"
    elif finger_count == 3:
        return "数字 3 🤟"
    elif finger_count == 4:
        return "数字 4 🖖"
    elif finger_count == 5:
        return "张开 ✋"
    else:
        return "未知手势"


# ====================== 主程序 ======================
if __name__ == "__main__":
    print("正在打开摄像头...")

    # 打开摄像头
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("错误：无法打开摄像头")
        exit()

    # 设置摄像头参数
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 摄像头预热
    print("摄像头正在预热，请稍等...")
    time.sleep(3)

    # 获取实际参数
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"摄像头已就绪，分辨率: {actual_width}x{actual_height}")

    # 初始化变量
    prev_time = 0
    is_recording = False
    video_writer = None
    recording_start_time = 0
    last_gesture = ""
    gesture_start_time = 0

    # 配置MediaPipe模型
    with mp_hands.Hands(
            model_complexity=0,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6) as hands:

        print("\n=== 增强版手势识别系统已启动 ===")
        print("支持手势：")
        print("  基础：拳头、数字1-5、张开")
        print("  进阶：OK、点赞、踩、摇滚、打电话、和平")
        print("  动态：挥手")
        print("\n操作说明：")
        print("  R键：开始/停止录制")
        print("  C键：清空手势历史")
        print("  ESC键：退出程序")

        while cap.isOpened():
            ret, image = cap.read()
            if not ret or image is None:
                print("警告：无法读取帧")
                continue

            # 处理图像
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            # 恢复可写状态
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # 绘制手部关键点
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

            # 翻转图像
            flipped_image = cv2.flip(image, 1)

            # 绘制基础信息
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            cv2.putText(flipped_image, f"FPS: {int(fps)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # 录制状态
            if is_recording:
                recording_duration = int(time.time() - recording_start_time)
                cv2.circle(flipped_image, (actual_width - 60, 20), 8, (0, 0, 255), -1)
                cv2.putText(flipped_image, f"REC {recording_duration}s", (actual_width - 140, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # 操作提示
            cv2.putText(flipped_image, "R:录制 | C:清空 | ESC:退出", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

            # 手势识别和绘制
            current_gesture = "无手势"
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    h, w, c = image.shape
                    wrist_x = int(hand_landmarks.landmark[0].x * w)
                    wrist_y = int(hand_landmarks.landmark[0].y * h)
                    flipped_wrist_x = w - wrist_x

                    # 修正左右手标签
                    original_label = handedness.classification[0].label
                    corrected_label = "Left" if original_label == "Right" else "Right"
                    score = handedness.classification[0].score

                    # 识别手势
                    gesture_name = get_gesture_name(hand_landmarks, original_label)
                    current_gesture = gesture_name

                    # 动态手势检测（挥手）
                    wave_detection_frames.append((wrist_x, wrist_y))
                    if len(wave_detection_frames) > 20:
                        wave_detection_frames.pop(0)

                    if detect_wave_gesture(wave_detection_frames):
                        gesture_name = "挥手 👋"
                        current_gesture = gesture_name
                        wave_detection_frames.clear()

                    # 显示手势名称和左右手
                    cv2.putText(flipped_image, gesture_name, (flipped_wrist_x - 60, wrist_y - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(flipped_image, f"{corrected_label} ({score:.2f})", (flipped_wrist_x - 60, wrist_y + 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # 更新手势历史
            if current_gesture != last_gesture and current_gesture != "无手势":
                timestamp = datetime.now().strftime("%H:%M:%S")
                gesture_history.append(f"[{timestamp}] {current_gesture}")
                if len(gesture_history) > MAX_HISTORY:
                    gesture_history.pop(0)
                last_gesture = current_gesture

            # 显示手势历史
            cv2.putText(flipped_image, "手势历史:", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            for i, gesture in enumerate(reversed(gesture_history[-5:])):  # 显示最近5个
                cv2.putText(flipped_image, gesture, (10, 140 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # 显示图像
            cv2.imshow('增强版手势识别系统', flipped_image)

            # 录制视频
            if is_recording and video_writer is not None:
                video_writer.write(flipped_image)

            # 按键处理
            key = cv2.waitKey(5) & 0xFF
            if key == 27:  # ESC退出
                break
            elif key == ord('r') or key == ord('R'):  # R键录制
                if not is_recording:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"demo_enhanced_{timestamp}.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*'avc1')
                    video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (actual_width, actual_height))
                    is_recording = True
                    recording_start_time = time.time()
                    print(f"开始录制: {filename}")
                else:
                    is_recording = False
                    video_writer.release()
                    video_writer = None
                    print("录制完成！")
            elif key == ord('c') or key == ord('C'):  # C键清空历史
                gesture_history.clear()
                print("手势历史已清空")

    # 释放资源
    cap.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()
    print("程序已退出")
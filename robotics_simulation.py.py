import pybullet as p
import pybullet_data
import random
import math
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Initialize PyBullet
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

# Load environment
ground = p.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True)
road = p.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True, globalScaling=10)
p.changeVisualShape(road, -1, rgbaColor=[0.5, 0.5, 0.5, 1])

# Delivery Bases
bases = {
    "base1": (5, -5, 0.1),   # Green
    "base2": (-5, 5, 0.1),   # Red
    "base3": (5, 5, 0.1),    # Blue
    "base4": (-5, -5, 0.1)   # Yellow
}

# Starting positions for robots
robot_start_positions = {
    "robot1": (6, -6, 0.1),   # Green robot
    "robot2": (-6, 6, 0.1),   # Red robot
    "robot3": (6, 6, 0.1),    # Blue robot
    "robot4": (-6, -6, 0.1)   # Yellow robot
}

# Base colors for simulation
base_colors = {
    "base1": [0.2, 0.7, 0.2, 0.7],  # Green
    "base2": [0.7, 0.2, 0.2, 0.7],  # Red
    "base3": [0.2, 0.2, 0.7, 0.7],  # Blue
    "base4": [0.7, 0.7, 0.2, 0.7]   # Yellow
}

# Base colors for plotting
base_plot_colors = {
    "base1": "orange",
    "base2": "purple",
    "base3": "brown",
    "base4": "pink"
}

# Robot colors
robot_colors = {
    "robot1": [0.2, 0.7, 0.2, 1.0],  # Green
    "robot2": [0.7, 0.2, 0.2, 1.0],  # Red
    "robot3": [0.2, 0.2, 0.7, 1.0],  # Blue
    "robot4": [0.7, 0.7, 0.2, 1.0]   # Yellow
}

# Initialize packages
packages = {
    "base1": 3,
    "base2": 3,
    "base3": 3,
    "base4": 3
}

# Physical package boxes
package_objects = {
    "base1": [],
    "base2": [],
    "base3": [],
    "base4": []
}

# Robot constraints
robot_constraints = {}

# Communication Hub
communication_hub = {
    "messages": [],
    "robot_status": {}
}

# Track communication hub activity for graphing
communication_hub_activity = []

# Cooldown for communication hub status print
last_status_print = 0
STATUS_PRINT_COOLDOWN = 5

# Track message boxes in the simulation
message_boxes = {}

# Track dialog cooldowns
dialog_cooldowns = {}

# Simulation results
simulation_results = {
    "travel_times": {},
    "intermediate_stops": {},
    "start_times": {}
}

# Load Robots
robot1 = p.loadURDF("r2d2.urdf", robot_start_positions["robot1"])
robot2 = p.loadURDF("r2d2.urdf", robot_start_positions["robot2"])
robot3 = p.loadURDF("r2d2.urdf", robot_start_positions["robot3"])
robot4 = p.loadURDF("r2d2.urdf", robot_start_positions["robot4"])

# Change robot colors
p.changeVisualShape(robot1, -1, rgbaColor=robot_colors["robot1"])
p.changeVisualShape(robot2, -1, rgbaColor=robot_colors["robot2"])
p.changeVisualShape(robot3, -1, rgbaColor=robot_colors["robot3"])
p.changeVisualShape(robot4, -1, rgbaColor=robot_colors["robot4"])

# Robot configuration
ROBOT_STEP_SIZE = 0.15  # Increased for smoother movement
DETECTION_RADIUS = 1.5  # Reduced to avoid unnecessary avoidance at bases
COLLISION_RADIUS = 1.5
PASSING_RADIUS = 1.0
AVOIDANCE_STRENGTH = 4.0  # Reduced for smoother avoidance
MAX_STUCK_COUNT = 50
ROTATION_SMOOTHING = 0.1
SAFETY_BUFFER = 0.8
DELIVERY_THRESHOLD = 0.5
SIMULATION_DELAY = 0.02
ROBOT_SPACING = 0.5
BASE_RADIUS = 2.0
MESSAGE_BOX_LIFETIME = 3.0
DIALOG_HEIGHT_OFFSET = 0.2
DIALOG_COOLDOWN = 5.0
DELIVERY_RADIUS = 0.3

def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def normalize_vector(vector):
    norm = math.sqrt(vector[0]**2 + vector[1]**2)
    if norm < 0.01:
        return (0, 0)
    return (vector[0]/norm, vector[1]/norm)

def dot_product(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1]

def send_message(robot_id, message):
    state = robot_states[robot_id]
    timestamp = time.time()
    communication_hub["messages"].append({
        "sender": state["name"],
        "color": state["color"],
        "message": message,
        "timestamp": timestamp
    })
    communication_hub["messages"] = [msg for msg in communication_hub["messages"] 
                                   if timestamp - msg["timestamp"] < 10]

def update_robot_status(robot_id):
    state = robot_states[robot_id]
    pos, _ = p.getBasePositionAndOrientation(robot_id)
    communication_hub["robot_status"][state["name"]] = {
        "position": pos,
        "target_base": state["target_base"],
        "delivered": state["delivered"],
        "carrying_package": bool(state["carrying_package"]),
        "timestamp": time.time()
    }

def create_package(base_name):
    base_pos = bases[base_name]
    offset_x = random.uniform(-0.3, 0.3)
    offset_y = random.uniform(-0.3, 0.3)
    package = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.05], 
                                 rgbaColor=[0.8, 0.5, 0.2, 1.0])
    package_id = p.createMultiBody(
        baseMass=0.1,
        baseVisualShapeIndex=package,
        basePosition=[base_pos[0] + offset_x, base_pos[1] + offset_y, base_pos[2] + 0.05]
    )
    return package_id

# Robot states
robot_states = {
    robot1: {
        "name": "robot1",
        "color": "Green",
        "current_base": "base1",
        "start_base": "base1",
        "target_base": None,
        "delivered": True,
        "avoiding": False,
        "carrying_package": None,
        "visible_package": None,
        "stuck_counter": 0,
        "last_pos": robot_start_positions["robot1"],
        "last_orientation": p.getQuaternionFromEuler([0, 0, 0]),
        "delivery_count": 0,
        "path_history": [],
        "has_traveled_message_printed": False,
        "returning": False,
        "last_move_vector": (0, 0),
        "pause_time": None,
        "pause_base": None,
        "is_paused": False,
        "stop_at_intermediate": False,
        "intermediate_stop_bases": [],
        "intermediate_stops": 0,
        "stopped_bases": set(),
        "returned_message_printed": False,
        "current_intermediate_index": 0,
        "avoidance_count": 0,
        "distance_history": [],
        "event_timeline": []
    },
    robot2: {
        "name": "robot2",
        "color": "Red",
        "current_base": "base2",
        "start_base": "base2",
        "target_base": None,
        "delivered": True,
        "avoiding": False,
        "carrying_package": None,
        "visible_package": None,
        "stuck_counter": 0,
        "last_pos": robot_start_positions["robot2"],
        "last_orientation": p.getQuaternionFromEuler([0, 0, 0]),
        "delivery_count": 0,
        "path_history": [],
        "has_traveled_message_printed": False,
        "returning": False,
        "last_move_vector": (0, 0),
        "pause_time": None,
        "pause_base": None,
        "is_paused": False,
        "stop_at_intermediate": False,
        "intermediate_stop_bases": [],
        "intermediate_stops": 0,
        "stopped_bases": set(),
        "returned_message_printed": False,
        "current_intermediate_index": 0,
        "avoidance_count": 0,
        "distance_history": [],
        "event_timeline": []
    },
    robot3: {
        "name": "robot3",
        "color": "Blue",
        "current_base": "base3",
        "start_base": "base3",
        "target_base": None,
        "delivered": True,
        "avoiding": False,
        "carrying_package": None,
        "visible_package": None,
        "stuck_counter": 0,
        "last_pos": robot_start_positions["robot3"],
        "last_orientation": p.getQuaternionFromEuler([0, 0, 0]),
        "delivery_count": 0,
        "path_history": [],
        "has_traveled_message_printed": False,
        "returning": False,
        "last_move_vector": (0, 0),
        "pause_time": None,
        "pause_base": None,
        "is_paused": False,
        "stop_at_intermediate": False,
        "intermediate_stop_bases": [],
        "intermediate_stops": 0,
        "stopped_bases": set(),
        "returned_message_printed": False,
        "current_intermediate_index": 0,
        "avoidance_count": 0,
        "distance_history": [],
        "event_timeline": []
    },
    robot4: {
        "name": "robot4",
        "color": "Yellow",
        "current_base": "base4",
        "start_base": "base4",
        "target_base": None,
        "delivered": True,
        "avoiding": False,
        "carrying_package": None,
        "visible_package": None,
        "stuck_counter": 0,
        "last_pos": robot_start_positions["robot4"],
        "last_orientation": p.getQuaternionFromEuler([0, 0, 0]),
        "delivery_count": 0,
        "path_history": [],
        "has_traveled_message_printed": False,
        "returning": False,
        "last_move_vector": (0, 0),
        "pause_time": None,
        "pause_base": None,
        "is_paused": False,
        "stop_at_intermediate": False,
        "intermediate_stop_bases": [],
        "intermediate_stops": 0,
        "stopped_bases": set(),
        "returned_message_printed": False,
        "current_intermediate_index": 0,
        "avoidance_count": 0,
        "distance_history": [],
        "event_timeline": []
    }
}

base_name_colors = {
    "base1": "Green",
    "base2": "Red",
    "base3": "Blue",
    "base4": "Yellow"
}

color_to_base = {
    "green": "base1",
    "red": "base2",
    "blue": "base3",
    "yellow": "base4"
}

# Load Moving Cars
cars = []
car_lanes = [-3, -1, 1, 3]

def setup_random_cars(num_cars=4):
    global cars
    for car in cars:
        p.removeBody(car)
    cars = []
    for i in range(num_cars):
        lane = car_lanes[i % len(car_lanes)]
        lane += random.uniform(-0.3, 0.3)
        start_x = random.uniform(-5, 5)
        car_scale = random.uniform(0.4, 0.6)
        car = p.loadURDF("racecar/racecar.urdf", 
                       [start_x, lane, 0.1], 
                       globalScaling=car_scale)
        cars.append(car)
    car_speeds = {car: random.uniform(0.005, 0.015) for car in cars}
    return car_speeds

car_speeds = setup_random_cars()

def move_cars():
    for car in cars:
        pos, orn = p.getBasePositionAndOrientation(car)
        new_x = pos[0] + car_speeds[car]
        if new_x > 5: new_x = -5
        target_lane = car_lanes[cars.index(car) % len(car_lanes)]
        current_lane = pos[1]
        lane_adjustment = random.uniform(-0.05, 0.05)
        new_y = current_lane + 0.05 * (target_lane - current_lane) + lane_adjustment
        p.resetBasePositionAndOrientation(car, [new_x, new_y, pos[2]], orn)

def create_robot_package(robot_id):
    robot_pos, _ = p.getBasePositionAndOrientation(robot_id)
    state = robot_states[robot_id]
    robot_name = state["name"]
    robot_rgba = robot_colors[robot_name]
    package_color = [
        min(0.9, robot_rgba[0] + 0.2),
        min(0.6, robot_rgba[1] + 0.1),
        min(0.1, robot_rgba[2] + 0.05),
        1.0
    ]
    package = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.08, 0.08, 0.04], 
                                 rgbaColor=package_color)
    package_id = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=package,
        basePosition=[robot_pos[0], robot_pos[1], robot_pos[2] + 0.25]
    )
    return package_id

def update_robot_package_position(robot_id, package_id):
    if package_id is not None:
        robot_pos, robot_orn = p.getBasePositionAndOrientation(robot_id)
        p.resetBasePositionAndOrientation(
            package_id,
            [robot_pos[0], robot_pos[1], robot_pos[2] + 0.25],
            robot_orn
        )

def pick_up_package(robot_id):
    state = robot_states[robot_id]
    current_base = state["current_base"]
    if packages[current_base] > 0 and len(package_objects[current_base]) > 0:
        package_id = package_objects[current_base].pop()
        p.removeBody(package_id)
        visible_package = create_robot_package(robot_id)
        state["visible_package"] = visible_package
        state["carrying_package"] = {
            "from_base": current_base,
            "to_base": state["target_base"]
        }
        send_message(robot_id, f"Picked up package from {base_name_colors[current_base]} Base")
        print(f"{state['color']} Robot picked up a package from {base_name_colors[current_base]} Base")
        return True
    else:
        print(f"No packages available at {base_name_colors[current_base]} Base for {state['color']} Robot!")
        state["delivered"] = True
        return False

def deliver_package(robot_id, target_base=None):
    state = robot_states[robot_id]
    if target_base is None:
        target_base = state["target_base"]
    
    if state["carrying_package"]:
        if state["visible_package"] is not None:
            p.removeBody(state["visible_package"])
            state["visible_package"] = None
        
        new_package = create_package(target_base)
        package_objects[target_base].append(new_package)
        
        packages[state["current_base"]] -= 1
        packages[target_base] += 1
        
        state["carrying_package"] = None
        state["delivery_count"] += 1
        
        send_message(robot_id, f"Delivered package to {base_name_colors[target_base]} Base")
        print(f"{state['color']} Robot delivered the package to {base_name_colors[target_base]} Base!, (Delivery #{state['delivery_count']})")
        
        state["current_base"] = target_base
        
        if target_base == state["start_base"] and state["returning"]:
            state["returning"] = False
            state["delivered"] = True
            state["target_base"] = None
            state["has_traveled_message_printed"] = False
            state["current_intermediate_index"] = 0
            state["intermediate_stop_bases"] = []
            state["stopped_bases"] = set()
        elif target_base != state["target_base"]:
            # Intermediate delivery, pick up a new package
            if packages[target_base] > 0:
                state["visible_package"] = create_robot_package(robot_id)
                state["carrying_package"] = {"from_base": target_base, "to_base": state["target_base"]}
                send_message(robot_id, f"Picked up new package at {base_name_colors[target_base]} Base for {base_name_colors[state['target_base']]}")
            else:
                print(f"No packages to pick up at {base_name_colors[target_base]} Base, proceeding to target.")
        
        return True
    return False

def is_collision_free(robot_id, new_pos):
    robot_aabb_min, robot_aabb_max = p.getAABB(robot_id)
    robot_width = max(robot_aabb_max[0] - robot_aabb_min[0], robot_aabb_max[1] - robot_aabb_min[1])
    for car in cars:
        car_pos, _ = p.getBasePositionAndOrientation(car)
        dist = distance(new_pos, car_pos)
        min_safe_dist = (robot_width + robot_width) / 2 * 1.4 + SAFETY_BUFFER
        if dist < min_safe_dist:
            return False
    for other_robot in [robot1, robot2, robot3, robot4]:
        if other_robot != robot_id:
            other_pos, _ = p.getBasePositionAndOrientation(other_robot)
            dist = distance(new_pos, other_pos)
            min_safe_dist = (robot_width + robot_width) / 2 * 1.4 + SAFETY_BUFFER
            if dist < min_safe_dist:
                return False
    return True

def lock_robot(robot_id):
    if robot_id not in robot_constraints:
        pos, orn = p.getBasePositionAndOrientation(robot_id)
        constraint_id = p.createConstraint(
            robot_id, -1,
            -1, -1,
            p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[pos[0], pos[1], pos[2]],
            childFrameOrientation=orn
        )
        robot_constraints[robot_id] = constraint_id

def unlock_robot(robot_id):
    if robot_id in robot_constraints:
        p.removeConstraint(robot_constraints[robot_id])
        del robot_constraints[robot_id]

def detect_obstacles(robot_id):
    robot_pos, _ = p.getBasePositionAndOrientation(robot_id)
    obstacles = []
    robot_aabb_min, robot_aabb_max = p.getAABB(robot_id)
    robot_width = max(robot_aabb_max[0] - robot_aabb_min[0], robot_aabb_max[1] - robot_aabb_min[1])
    
    for car in cars:
        car_pos, _ = p.getBasePositionAndOrientation(car)
        dist = distance(robot_pos, car_pos)
        min_safe_dist = (robot_width + robot_width) / 2 * 1.2 + SAFETY_BUFFER
        if dist < DETECTION_RADIUS:
            direction = (car_pos[0] - robot_pos[0], car_pos[1] - robot_pos[1])
            obstacles.append({
                "pos": car_pos,
                "type": "car",
                "direction": direction,
                "distance": dist,
                "min_safe_dist": min_safe_dist
            })
    
    for other_robot in [robot1, robot2, robot3, robot4]:
        if other_robot != robot_id:
            other_pos, _ = p.getBasePositionAndOrientation(other_robot)
            dist = distance(robot_pos, other_pos)
            min_safe_dist = (robot_width + robot_width) / 2 * 1.2 + SAFETY_BUFFER
            if dist < DETECTION_RADIUS:
                direction = (other_pos[0] - robot_pos[0], other_pos[1] - robot_pos[1])
                obstacles.append({
                    "pos": other_pos,
                    "type": "robot",
                    "direction": direction,
                    "distance": dist,
                    "min_safe_dist": min_safe_dist
                })
    
    return sorted(obstacles, key=lambda x: x["distance"])

def calculate_avoidance_vector(robot_id, obstacles, goal_pos):
    current_pos, _ = p.getBasePositionAndOrientation(robot_id)
    goal_vector = (goal_pos[0] - current_pos[0], goal_pos[1] - current_pos[1])
    goal_vector = normalize_vector(goal_vector)
    avoidance_vector = [0, 0]
    avoiding = False
    
    # Dynamic avoidance strength based on obstacle count
    obstacle_count = len([o for o in obstacles if o["distance"] < DETECTION_RADIUS * 0.7])
    dynamic_avoidance_strength = AVOIDANCE_STRENGTH * (1 + obstacle_count * 0.3)  # Less aggressive increase
    
    for obstacle in obstacles:
        effective_collision_radius = obstacle.get("min_safe_dist", COLLISION_RADIUS + SAFETY_BUFFER)
        avoidance_distance = effective_collision_radius * 2
        if obstacle["distance"] < avoidance_distance:
            obstacle_vector = (-obstacle["direction"][0], -obstacle["direction"][1])
            obstacle_norm = math.sqrt(obstacle_vector[0]**2 + obstacle_vector[1]**2)
            if obstacle_norm > 0:
                obstacle_vector = [obstacle_vector[0]/obstacle_norm, obstacle_vector[1]/obstacle_norm]
            collusion_weight = dynamic_avoidance_strength * (avoidance_distance / max(0.1, obstacle["distance"]))**2  # Reduced exponent
            if obstacle["distance"] < effective_collision_radius * 1.1:
                collusion_weight *= 3  # Reduced multiplier
            avoidance_vector[0] += obstacle_vector[0] * collusion_weight
            avoidance_vector[1] += obstacle_vector[1] * collusion_weight
            avoiding = True
            perp_vector = [-obstacle_vector[1], obstacle_vector[0]]
            goal_dot = goal_vector[0] * perp_vector[0] + goal_vector[1] * perp_vector[1]
            if goal_dot < 0:
                perp_vector = [-perp_vector[0], -perp_vector[1]]
            perp_weight = 0.4 * collusion_weight  # Reduced weight
            avoidance_vector[0] += perp_vector[0] * perp_weight
            avoidance_vector[1] += perp_vector[1] * perp_weight
            if obstacle["type"] == "robot" and obstacle["distance"] < PASSING_RADIUS:
                avoidance_vector[0] += perp_vector[0] * (dynamic_avoidance_strength * 0.3)  # Reduced strength
                avoidance_vector[1] += perp_vector[1] * (dynamic_avoidance_strength * 0.3)
    
    if avoiding:
        avoidance_norm = math.sqrt(avoidance_vector[0]**2 + avoidance_vector[1]**2)
        if avoidance_norm > 0:
            avoidance_vector = [avoidance_vector[0]/avoidance_norm, avoidance_vector[1]/avoidance_norm]
        
        if len(obstacles) >= 2 and any(o["distance"] < COLLISION_RADIUS * 1.5 for o in obstacles):
            closest_obstacle = min(obstacles, key=lambda x: x["distance"])
            detour_vector = [-closest_obstacle["direction"][1], closest_obstacle["direction"][0]]
            detour_norm = math.sqrt(detour_vector[0]**2 + detour_vector[1]**2)
            if detour_norm > 0:
                detour_vector = [detour_vector[0]/detour_norm, detour_vector[1]/detour_norm]
            avoidance_vector = [av + dv * 0.3 for av, dv in zip(avoidance_vector, detour_vector)]  # Reduced blend
        
        if any(o["distance"] < o.get("min_safe_dist", COLLISION_RADIUS) * 1.2 for o in obstacles):
            return avoidance_vector, True
        closest_dist = obstacles[0]["distance"] if obstacles else DETECTION_RADIUS
        avoidance_weight = min(0.85, 1.0 - (closest_dist / DETECTION_RADIUS))  # Reduced max weight
        goal_weight = 1.0 - avoidance_weight
        combined_vector = [
            avoidance_weight * avoidance_vector[0] + goal_weight * goal_vector[0],
            avoidance_weight * avoidance_vector[1] + goal_weight * goal_vector[1]
        ]
        combined_norm = math.sqrt(combined_vector[0]**2 + combined_vector[1]**2)
        if combined_norm > 0:
            combined_vector = [combined_vector[0]/combined_norm, combined_vector[1]/combined_norm]
        return combined_vector, True
    return goal_vector, False

def get_next_intermediate_base(robot_id, current_pos):
    state = robot_states[robot_id]
    if not state["stop_at_intermediate"] or state["returning"]:
        return None, float('inf')
    
    if state["current_intermediate_index"] < len(state["intermediate_stop_bases"]):
        base_name = state["intermediate_stop_bases"][state["current_intermediate_index"]]
        if base_name in bases and base_name not in state["stopped_bases"] and base_name != state["current_base"]:
            base_pos = bases[base_name]
            dist = distance(current_pos, base_pos)
            return base_name, dist
    return None, float('inf')

def robot_controller(robot_id):
    state = robot_states[robot_id]
    update_robot_status(robot_id)
    
    if state["delivered"] and not state["returning"] and state["target_base"] is None:
        lock_robot(robot_id)
        return

    if state["target_base"] is None:
        lock_robot(robot_id)
        return

    current_pos, current_orn = p.getBasePositionAndOrientation(robot_id)
    
    if state["last_pos"] is not None:
        dist_increment = distance(current_pos, state["last_pos"])
        current_time = time.time()
        if state["distance_history"]:
            last_dist = state["distance_history"][-1][1]
            state["distance_history"].append((current_time, last_dist + dist_increment))
        else:
            state["distance_history"].append((current_time, dist_increment))
    
    if state["is_paused"]:
        if time.time() - state["pause_time"] < 3.0:
            lock_robot(robot_id)
            return
        else:
            print(f"{state['color']} Robot is resuming from {state['pause_base']} toward {base_name_colors[state['target_base']]}")
            send_message(robot_id, f"Resuming to {base_name_colors[state['target_base']]} Base")
            state["is_paused"] = False
            state["pause_time"] = None
            state["pause_base"] = None
            unlock_robot(robot_id)

    if state["returning"]:
        goal_pos = robot_start_positions[state["name"]]
        dist_to_goal = distance(current_pos, goal_pos)
        if dist_to_goal < DELIVERY_RADIUS:
            deliver_package(robot_id, state["start_base"])
            simulation_results["travel_times"][state["name"]] = time.time() - simulation_results["start_times"][state["name"]]
            simulation_results["intermediate_stops"][state["name"]] = state["intermediate_stops"]
            state["event_timeline"].append(("Returned", time.time()))
            if not state["returned_message_printed"]:
                print(f"{state['color']} Robot has returned to its starting position.")
                state["returned_message_printed"] = True
            state["target_base"] = None
            state["delivered"] = True
            state["returning"] = False
            state["has_traveled_message_printed"] = False
            state["current_intermediate_index"] = 0
            state["intermediate_stop_bases"] = []
            state["stopped_bases"] = set()
            return
    else:
        goal_pos = bases[state["target_base"]]
        dist_to_goal = distance(current_pos, goal_pos)
        if dist_to_goal < DELIVERY_RADIUS:
            deliver_package(robot_id)
            state["event_timeline"].append(("Delivered", time.time()))
            state["returning"] = True
            state["has_traveled_message_printed"] = False
            state["current_intermediate_index"] = 0
            return
    
    next_base, min_dist = get_next_intermediate_base(robot_id, current_pos)
    if next_base and min_dist < DELIVERY_RADIUS:
        print(f"{state['color']} Robot is stopping at {base_name_colors[next_base]} Base to deliver package (distance: {min_dist:.2f})")
        send_message(robot_id, f"Stopping at {base_name_colors[next_base]} Base")
        state["is_paused"] = True
        state["pause_time"] = time.time()
        state["pause_base"] = base_name_colors[next_base]
        state["intermediate_stops"] += 1
        state["stopped_bases"].add(next_base)
        state["current_intermediate_index"] += 1
        state["event_timeline"].append((f"Stopped at {base_name_colors[next_base]}", time.time()))
        deliver_package(robot_id, next_base)
        display_visit_message(robot_id, next_base)
        lock_robot(robot_id)
        return
    elif next_base:
        print(f"{state['color']} Robot distance to {base_name_colors[next_base]} Base: {min_dist:.2f}")
        goal_pos = bases[next_base]
    else:
        print(f"{state['color']} Robot heading directly to {base_name_colors[state['target_base']]} Base")

    obstacles = detect_obstacles(robot_id)
    move_vector, is_avoiding = calculate_avoidance_vector(robot_id, obstacles, goal_pos)
    state["avoiding"] = is_avoiding
    if is_avoiding:
        state["avoidance_count"] += 1
    
    # Smooth movement by blending with last move vector
    if state["last_move_vector"] != (0, 0):
        move_vector = [
            0.7 * move_vector[0] + 0.3 * state["last_move_vector"][0],
            0.7 * move_vector[1] + 0.3 * state["last_move_vector"][1]
        ]
        move_vector = normalize_vector(move_vector)
    state["last_move_vector"] = move_vector

    if not state["delivered"]:
        if not state["has_traveled_message_printed"]:
            msg = f"Heading to {base_name_colors[state['target_base']]} Base" if not state["returning"] else f"Returning to {base_name_colors[state['start_base']]} Base"
            send_message(robot_id, msg)
            print(f"{state['color']} Robot: {msg}")
            state["has_traveled_message_printed"] = True
            state["event_timeline"].append(("Started", time.time()))

        new_pos = [
            current_pos[0] + move_vector[0] * ROBOT_STEP_SIZE,
            current_pos[1] + move_vector[1] * ROBOT_STEP_SIZE,
            current_pos[2]
        ]
        target_yaw = math.atan2(move_vector[1], move_vector[0])
        target_orientation = p.getQuaternionFromEuler([0, 0, target_yaw])
        
        # Check if stuck and trigger detour if needed
        if state["last_pos"] and distance(current_pos, state["last_pos"]) < 0.01:
            state["stuck_counter"] += 1
            if state["stuck_counter"] > 5:
                print(f"{state['color']} Robot is stuck, attempting detour...")
                perpendicular_vector = [-move_vector[1], move_vector[0]]  # Use move_vector for detour
                perpendicular_norm = math.sqrt(perpendicular_vector[0]**2 + perpendicular_vector[1]**2)
                if perpendicular_norm > 0:
                    perpendicular_vector = [perpendicular_vector[0]/perpendicular_norm, perpendicular_vector[1]/perpendicular_norm]
                move_vector = [pv * ROBOT_STEP_SIZE * 1.5 for pv in perpendicular_vector]  # Moderate detour
                new_pos = [
                    current_pos[0] + move_vector[0],
                    current_pos[1] + move_vector[1],
                    current_pos[2]
                ]
                state["stuck_counter"] = 0
        else:
            state["stuck_counter"] = 0

        if is_collision_free(robot_id, new_pos):
            p.resetBasePositionAndOrientation(robot_id, new_pos, target_orientation)
            state["last_pos"] = new_pos
            # Apply moving average to smooth path_history
            if state["path_history"]:
                last_x, last_y = state["path_history"][-1][:2]
                new_x = 0.7 * last_x + 0.3 * new_pos[0]
                new_y = 0.7 * last_y + 0.3 * new_pos[1]
                state["path_history"].append([new_x, new_y, new_pos[2]])
            else:
                state["path_history"].append(new_pos)
        else:
            state["last_pos"] = current_pos
        
        state["last_orientation"] = target_orientation
        update_robot_package_position(robot_id, state["visible_package"])

def check_and_display_dialogs():
    all_robots = [robot1, robot2, robot3, robot4]
    processed_pairs = set()
    current_time = time.time()
    
    for i, robot_id in enumerate(all_robots):
        for j, other_robot_id in enumerate(all_robots):
            if robot_id == other_robot_id or tuple(sorted([robot_id, other_robot_id])) in processed_pairs:
                continue
            pair = tuple(sorted([robot_id, other_robot_id]))
            processed_pairs.add(pair)
            
            state1 = robot_states[robot_id]
            state2 = robot_states[other_robot_id]
            if (state1["delivered"] and not state1["returning"]) or (state2["delivered"] and not state2["returning"]):
                continue
            
            pos1, _ = p.getBasePositionAndOrientation(robot_id)
            pos2, _ = p.getBasePositionAndOrientation(other_robot_id)
            if distance(pos1, pos2) < PASSING_RADIUS and not state1["is_paused"] and not state2["is_paused"]:
                if pair not in dialog_cooldowns or current_time - dialog_cooldowns[pair] >= DIALOG_COOLDOWN:
                    display_dialog_box(robot_id, other_robot_id)
                    dialog_cooldowns[pair] = current_time

def display_dialog_box(robot_id, other_robot_id):
    state1 = robot_states[robot_id]
    pos1, _ = p.getBasePositionAndOrientation(robot_id)
    current_time = time.strftime("%H:%M:%S", time.localtime())
    
    if robot_id not in message_boxes:
        message_boxes[robot_id] = []
    height_offset = len(message_boxes[robot_id]) * DIALOG_HEIGHT_OFFSET
    
    message = f"Hi, passing by [{current_time}]"
    box_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.35, 0.12, 0.01], rgbaColor=[0.9, 0.9, 0.9, 1.0])
    box_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=box_shape, basePosition=[pos1[0], pos1[1], pos1[2] + 0.5 + height_offset])
    text_id = p.addUserDebugText(text=message, textPosition=[pos1[0], pos1[1], pos1[2] + 0.52 + height_offset], textColorRGB=[0, 0, 0], textSize=0.6, lifeTime=MESSAGE_BOX_LIFETIME)
    message_boxes[robot_id].append(([box_id, text_id], time.time()))

def display_visit_message(robot_id, base_name):
    state = robot_states[robot_id]
    pos, _ = p.getBasePositionAndOrientation(robot_id)
    message = f"Visiting {base_name_colors[base_name]} Base"
    text_id = p.addUserDebugText(
        text=message,
        textPosition=[pos[0], pos[1], pos[2] + 1.0],
        textColorRGB=[0, 0, 0],
        textSize=1.0,
        lifeTime=3.0
    )

def clean_message_boxes():
    current_time = time.time()
    for robot_id in list(message_boxes.keys()):
        to_remove = []
        for i, (ids, timestamp) in enumerate(message_boxes[robot_id]):
            if current_time - timestamp > MESSAGE_BOX_LIFETIME:
                for id in ids:
                    if id >= 0:
                        p.removeBody(id) if id in [b[0] for sublist in message_boxes.values() for b, _ in sublist] else p.removeUserDebugItem(id)
                to_remove.append(i)
        for i in reversed(to_remove):
            message_boxes[robot_id].pop(i)
        if not message_boxes[robot_id]:
            del message_boxes[robot_id]

def visualize_bases():
    for base_name, pos in bases.items():
        visual = p.createVisualShape(p.GEOM_CYLINDER, radius=BASE_RADIUS, length=0.05, rgbaColor=base_colors[base_name])
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual, basePosition=[pos[0], pos[1], pos[2] - 0.02])

def create_legend():
    print("Robot Colors and Bases:")
    for robot_id in [robot1, robot2, robot3, robot4]:
        state = robot_states[robot_id]
        print(f"- {state['color']} Robot: Starts at {base_name_colors[state['current_base']]} Base")

def process_user_command(command):
    color_to_robot = {"green": robot1, "red": robot2, "blue": robot3, "yellow": robot4}
    
    try:
        assignments = command.lower().split(", ")
        if len(assignments) != 4:
            print("Please provide goals for all 4 robots (e.g., 'green to blue, red to yellow, blue to green, yellow to red')")
            return False
        
        targets = {}
        for assignment in assignments:
            parts = assignment.split(" to ")
            if len(parts) != 2:
                print(f"Invalid format in assignment: {assignment}")
                return False
            robot_color, target_color = parts
            robot_id = color_to_robot.get(robot_color)
            target_base = color_to_base.get(target_color)
            
            if not robot_id or not target_base:
                print(f"Invalid color in assignment: {assignment}. Use 'green', 'red', 'blue', or 'yellow'.")
                return False
            
            targets[robot_id] = target_base
        
        for robot_id in targets.keys():
            state = robot_states[robot_id]
            if not state["delivered"] or state["returning"]:
                print(f"{state['color']} Robot is already on a mission!")
                return False
            response = input(f"Should the {state['color']} robot stop at intermediate bases? (yes/no, default no): ").strip().lower()
            state["stop_at_intermediate"] = (response == "yes")
            if state["stop_at_intermediate"]:
                stop_bases_input = input(f"At which color bases should the {state['color']} robot stop? (e.g., red, yellow; leave blank for none): ").strip().lower()
                stop_colors = [color.strip() for color in stop_bases_input.split(",") if color.strip()]
                state["intermediate_stop_bases"] = [color_to_base.get(color) for color in stop_colors if color_to_base.get(color)]
                own_base = color_to_base[state["color"].lower()]
                if (own_base in state["intermediate_stop_bases"] or not stop_colors) and own_base != targets[robot_id] and own_base != state["current_base"]:
                    if own_base not in state["intermediate_stop_bases"]:
                        state["intermediate_stop_bases"].append(own_base)
                print(f"{state['color']} Robot will stop at: {[base_name_colors[base] for base in state['intermediate_stop_bases']]}")
            else:
                state["intermediate_stop_bases"] = []
                print(f"{state['color']} Robot will not stop at intermediate bases.")
        
        # Add delivery priority for robots at the same base
        delivery_priority = {robot_id: time.time() for robot_id in targets.keys()}
        
        for robot_id, target_base in targets.items():
            state = robot_states[robot_id]
            state["target_base"] = target_base
            state["delivered"] = False
            state["returning"] = False
            state["stuck_counter"] = 0
            state["path_history"] = []
            state["has_traveled_message_printed"] = False
            state["intermediate_stops"] = 0
            state["stopped_bases"] = set()
            state["returned_message_printed"] = False
            state["current_intermediate_index"] = 0
            simulation_results["start_times"][state["name"]] = time.time()
            send_message(robot_id, f"Assigned to deliver from {base_name_colors[state['current_base']]} to {base_name_colors[target_base]}")
            print(f"{state['color']} Robot assigned to deliver from {base_name_colors[state['current_base']]} Base to {base_name_colors[target_base]} Base")
            pick_up_package(robot_id)
        
        return True
    except Exception as e:
        print(f"Error processing command: {e}")
        return False

def plot_simulation_results():
    robots = ["deliveryagent robot1", "deliveryagent robot2", "deliveryagent robot3", "deliveryagent robot4"]
    colors = ["Green", "Red", "Blue", "Yellow"]
    color_map = {"Green": "green", "Red": "red", "Blue": "blue", "Yellow": "yellow"}
    
    travel_times = [simulation_results["travel_times"].get(r, 0) for r in robots]
    stops = [simulation_results["intermediate_stops"].get(r, 0) for r in robots]
    avoidance_counts = [robot_states[robot_id]["avoidance_count"] for robot_id in [robot1, robot2, robot3, robot4]]
    
    print("Travel Times Data:", travel_times)
    print("Intermediate Stops Data:", stops)
    print("Avoidance Counts Data:", avoidance_counts)
    
    fig1 = plt.figure(figsize=(8, 8))
    ax1 = fig1.add_subplot(111)
    for robot_id in [robot1, robot2, robot3, robot4]:
        state = robot_states[robot_id]
        path = state["path_history"]
        if path:
            # Apply moving average for smooth lines
            smoothed_path = []
            window_size = 5
            for i in range(len(path)):
                start_idx = max(0, i - window_size)
                window = path[start_idx:i + 1]
                x = sum(p[0] for p in window) / len(window)
                y = sum(p[1] for p in window) / len(window)
                smoothed_path.append([x, y, path[i][2]])
            x, y = zip(*[(p[0], p[1]) for p in smoothed_path])
            ax1.plot(x, y, color=color_map[state["color"]], linewidth=2, label=f"{state['color']} Robot")
            ax1.scatter(x[0], y[0], marker='o', s=100, color=color_map[state["color"]])
            ax1.scatter(x[-1], y[-1], marker='x', s=100, color=color_map[state["color"]])
    for base_name, pos in bases.items():
        ax1.scatter(pos[0], pos[1], marker='s', s=200, label=f"{base_name_colors[base_name]} Base", color=base_plot_colors[base_name])
    ax1.set_title("Robot Path Trajectories", fontsize=14)
    ax1.set_xlabel("X Position in m", fontsize=12)
    ax1.set_ylabel("Y Position in m", fontsize=12)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.text(0.05, 0.95, "Markers:\n• Circle: Start\n• X: End\n• Square: Base", transform=ax1.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    ax1.grid(True)
    plt.tight_layout()
    
    fig2 = plt.figure(figsize=(8, 6))
    ax2 = fig2.add_subplot(111)
    ax2.bar(colors, avoidance_counts, color=[color_map[c] for c in colors])
    ax2.set_title("Obstacle Avoidance Frequency per Robot", fontsize=14)
    ax2.set_xlabel("Robots", fontsize=12)
    ax2.set_ylabel("Number of Avoidance Maneuvers", fontsize=12)
    ax2.grid(True)
    plt.tight_layout()
    
    fig3 = plt.figure(figsize=(10, 6))
    ax3 = fig3.add_subplot(111)
    for robot_id in [robot1, robot2, robot3, robot4]:
        state = robot_states[robot_id]
        if state["distance_history"]:
            times, distances = zip(*state["distance_history"])
            times = [t - times[0] for t in times]
            ax3.plot(times, distances, label=f"{state['color']} Robot", color=color_map[state["color"]], linewidth=2)
    ax3.set_title("Cumulative Distance Traveled Over Time", fontsize=14)
    ax3.set_xlabel("Time (seconds)", fontsize=12)
    ax3.set_ylabel("Distance (m)", fontsize=12)
    ax3.legend(loc='upper left', fontsize=10)
    ax3.grid(True)
    plt.tight_layout()
    
    fig4 = plt.figure(figsize=(10, 6))
    ax4 = fig4.add_subplot(111)
    if communication_hub_activity:
        times, messages = zip(*communication_hub_activity)
        times = [t - times[0] for t in times]
        ax4.plot(times, messages, color='purple', linewidth=2)
    ax4.set_title("Communication Hub Activity Over Time", fontsize=14)
    ax4.set_xlabel("Time (seconds)", fontsize=12)
    ax4.set_ylabel("Number of Messages", fontsize=12)
    ax4.grid(True)
    plt.tight_layout()
    
    try:
        plt.show()
    except Exception as e:
        print(f"Error displaying plots: {e}")
        print("Saving plots to files instead.")
        fig1.savefig("path_trajectories.png")
        fig2.savefig("avoidance_frequency.png")
        fig3.savefig("distance_traveled.png")
        fig4.savefig("communication_activity.png")
        print("Plots saved as PNG files in the current directory.")

def run_simulation():
    global last_status_print
    print("Initializing simulation with 4 robots and 4 delivery bases...")
    visualize_bases()
    create_legend()
    
    for base_name in bases:
        for _ in range(packages[base_name]):
            package_objects[base_name].append(create_package(base_name))
    
    all_robots = [robot1, robot2, robot3, robot4]
    
    print("\nEnter goals for all robots (e.g., 'green to blue, red to yellow, blue to green, yellow to red') or '0' to exit:")
    
    while True:
        command = input("> ")
        if command == '0':
            break
        
        if not command:
            continue
        
        if process_user_command(command):
            for robot_id in all_robots:
                state = robot_states[robot_id]
                state["delivered"] = False
                state["returning"] = False
                state["returned_message_printed"] = False
                state["avoidance_count"] = 0
                state["distance_history"] = []
                state["event_timeline"] = []
            
            global communication_hub_activity
            communication_hub_activity = []
            
            while True:
                all_done = all(state["delivered"] and not state["returning"] and state["target_base"] is None 
                               for state in robot_states.values())
                
                if all_done:
                    print("All robots have completed their tasks and returned to start positions. Exiting simulation loop.")
                    break
                
                move_cars()
                for robot_id in all_robots:
                    robot_controller(robot_id)
                
                check_and_display_dialogs()
                clean_message_boxes()
                
                current_time = time.time()
                communication_hub_activity.append((current_time, len(communication_hub["messages"])))
                
                p.stepSimulation()
                time.sleep(SIMULATION_DELAY)
                
                if current_time - last_status_print >= STATUS_PRINT_COOLDOWN:
                    print(f"Communication Hub: {len(communication_hub['messages'])} messages, "
                          f"{len(communication_hub['robot_status'])} robots active")
                    last_status_print = current_time
            
            print("\nSimulation Results:")
            for robot_id in all_robots:
                state = robot_states[robot_id]
                travel_time = simulation_results["travel_times"].get(state["name"], 0)
                stops = simulation_results["intermediate_stops"].get(state["name"], 0)
                print(f"{state['color']} Robot: Travel Time = {travel_time:.2f}s, Intermediate Stops = {stops}")
            plot_simulation_results()
            break

# Start the simulation
run_simulation()
p.disconnect()

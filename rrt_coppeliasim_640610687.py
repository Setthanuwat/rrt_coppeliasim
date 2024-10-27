import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import time
import math
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import coppeliasim_zmqremoteapi_client as zmqRemoteApi
import matplotlib.patches as patches

# Coordinate transformation constants
python_x_min, python_x_max = 46, 141
python_y_min, python_y_max = 100, 195
coppelia_x_min, coppelia_x_max = -7.32, 2.38
coppelia_y_min, coppelia_y_max = -2.34, 7.30
scale_x = (coppelia_x_max - coppelia_x_min) / (python_x_max - python_x_min)
scale_y = (coppelia_y_max - coppelia_y_min) / (python_y_max - python_y_min)
x_offset = coppelia_x_min - python_x_min * scale_x
y_offset = coppelia_y_min - python_y_min * scale_y

class Node:
    def __init__(self, position, cost=0, parent=None):
        self.position = position
        self.cost = cost
        self.parent = parent

def check_collision(point, obstacles, buffer=2):
    return np.any(np.all((point >= obstacles[:, :2] - buffer) & (point <= obstacles[:, :2] + obstacles[:, 2:] + buffer), axis=1))

def check_line_collision(start, end, obstacles, buffer=2):
    num_steps = 100
    for t in np.linspace(0, 1, num_steps):
        point = start + t * (end - start)
        collision = check_collision(point, obstacles, buffer)
        if collision:
            return True
    return False

def find_nearest_neighbor(tree, point):
    distances = [np.linalg.norm(node.position - point) for node in tree]
    return tree[np.argmin(distances)]

def find_neighbors(tree, point, radius):
    return [node for node in tree if np.linalg.norm(node.position - point) <= radius]

def transform_to_coppelia_coords(x, y):
    coppelia_x = x * scale_x + x_offset
    coppelia_y = y * scale_y + y_offset
    return coppelia_x, coppelia_y

def generate_random_position(obstacles, buffer=5):
    while True:
        # Generate random position within the map boundaries
        x = np.random.uniform(60, 140)
        y = np.random.uniform(100, 190)
        point = np.array([x, y])
        orientation = np.random.uniform(0, 2 * np.pi)
        # Check if the position is collision-free
        if not check_collision(point, obstacles, buffer):
            return point,orientation

def continuous_rrt_star(start, goal, obstacles, max_distance, max_iterations, neighborhood_radius):
    start_node = Node(start)
    tree = [start_node]
    goal_node = None
    
    obstacles = np.array(obstacles)
    goal = np.array(goal)
    x_min, x_max = 60, 150
    y_min, y_max = 100, 190

    for i in range(max_iterations):
        random_point = np.random.uniform([x_min, y_min], [x_max, y_max])
        nearest_node = find_nearest_neighbor(tree, random_point)
    
        direction = random_point - nearest_node.position
        distance = np.linalg.norm(direction)
        if distance > max_distance:
            direction = direction / distance * max_distance
        
        new_position = nearest_node.position + direction
        if not check_line_collision(nearest_node.position, new_position, obstacles):
            neighbors = find_neighbors(tree, new_position, neighborhood_radius)
            min_cost = float('inf')
            best_parent = None
            
            for neighbor in neighbors:
                potential_cost = neighbor.cost + np.linalg.norm(neighbor.position - new_position)
                if potential_cost < min_cost and not check_line_collision(neighbor.position, new_position, obstacles):
                    min_cost = potential_cost
                    best_parent = neighbor
            
            if best_parent:
                new_node = Node(new_position, min_cost, best_parent)
                tree.append(new_node)
                
                for neighbor in neighbors:
                    potential_cost = new_node.cost + np.linalg.norm(new_node.position - neighbor.position)
                    if potential_cost < neighbor.cost and not check_line_collision(new_node.position, neighbor.position, obstacles):
                        neighbor.parent = new_node
                        neighbor.cost = potential_cost
            
            if np.linalg.norm(new_position - goal) < max_distance:
                if not check_collision(goal, obstacles):
                    potential_goal_node = Node(goal, new_node.cost + np.linalg.norm(new_position - goal), new_node)
                    if goal_node is None or potential_goal_node.cost < goal_node.cost:
                        goal_node = potential_goal_node
                        tree.append(goal_node)
                    save_path_to_csv(goal_node, scale_x, scale_y, x_offset, y_offset, filename='path2.csv')
                    print("Path saved to path2.csv")
        
        yield tree, goal_node
    print("Finish_Max_Iteration")

import matplotlib.patches as patches

class RobotVisualizer:
    def __init__(self, obstacles, start, goal):
        plt.ion()  # Turn on interactive mode
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_aspect('equal')
        self.ax.set_xlim(40, 150)
        self.ax.set_ylim(90, 200)
        
        # Plot obstacles
        for obs in obstacles:
            self.ax.add_patch(plt.Rectangle((obs[0], obs[1]), obs[2], obs[3], color='gray', alpha=0.5))
        
        # Load and plot the planned path - เปลี่ยนเป็นสีแดงและเพิ่มความหนาของเส้น
        try:
            planned_path = np.loadtxt('path2.csv', delimiter=',', skiprows=1)
            path_x = [x for x, _ in transform_from_coppelia_coords(planned_path[:, 0], planned_path[:, 1])]
            path_y = [y for _, y in transform_from_coppelia_coords(planned_path[:, 0], planned_path[:, 1])]
            self.ax.plot(path_x, path_y, 'r-', linewidth=3, label='Planned Path', alpha=1.0)
        except Exception as e:
            print(f"Could not load planned path: {e}")
        
        # Plot start and goal points
        self.ax.plot(*start, 'go', markersize=15, label='Start', markeredgecolor='white', markeredgewidth=2)
        self.ax.plot(*goal, 'r*', markersize=20, label='Goal', markeredgecolor='white', markeredgewidth=2)
        
        # Robot visualization
        self.robot_body = plt.Circle((start[0], start[1]), 3, color='blue', fill=True, alpha=0.7)
        self.ax.add_patch(self.robot_body)
        
        # Robot direction indicator (larger arrow)
        self.direction_arrow = self.ax.quiver(start[0], start[1], 1, 0, 
                                            color='yellow', scale=15, width=0.008,
                                            headwidth=4, headlength=5, headaxislength=4.5)
        
        # Robot trajectory (เปลี่ยนสีและความหนาของเส้น)
        self.trajectory_x = [start[0]]
        self.trajectory_y = [start[1]]
        self.trajectory_line, = self.ax.plot([], [], 'b--', linewidth=2, alpha=0.7, label='Robot Trajectory')
        
        # Add grid
        self.ax.grid(True, linestyle='--', alpha=0.3)
        
        # Customize appearance
        self.ax.set_facecolor('#f8f9fa')
        self.fig.patch.set_facecolor('white')
        
        # Legend with better positioning and visibility
        self.ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98),
                      fancybox=True, shadow=True, fontsize=10)
        
        self.ax.set_title("Robot Navigation Visualization", pad=20, fontsize=14, fontweight='bold')
        
        # Show the plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update_position(self, x, y, orientation):
        # Update robot body position
        self.robot_body.center = (x, y)
        
        # Update direction arrow
        dx = math.cos(orientation)
        dy = math.sin(orientation)
        self.direction_arrow.set_offsets([[x, y]])
        self.direction_arrow.set_UVC(dx, dy)
        
        # Update trajectory
        self.trajectory_x.append(x)
        self.trajectory_y.append(y)
        self.trajectory_line.set_data(self.trajectory_x, self.trajectory_y)
        
        # Update the display
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
    def clear_trajectory(self):
        self.trajectory_x = [self.trajectory_x[-1]]
        self.trajectory_y = [self.trajectory_y[-1]]
        self.trajectory_line.set_data(self.trajectory_x, self.trajectory_y)



def plot_rrt_star_realtime(start, goal, obstacles, rrt_star_generator):
    plt.ioff()  # Turn off interactive mode for RRT* visualization
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    ax.set_xlim(40, 150)
    ax.set_ylim(90, 200)
    
    for obs in obstacles:
        ax.add_patch(plt.Rectangle((obs[0], obs[1]), obs[2], obs[3], color='red', alpha=0.5))
    
    start_point = ax.plot(*start, 'bo', markersize=10, label='Start')[0]
    goal_point = ax.plot(*goal, marker='*', color='yellow', markersize=15, label='Goal')[0]
    
    tree_lines = []
    path_line = ax.plot([], [], 'r-', linewidth=2, label='Current Best Path')[0]
    
    ax.legend()
    ax.set_title("RRT* Path Planning")
    
    def init():
        global nodes_dots
        nodes_dots = []
        return tree_lines + [path_line, start_point, goal_point]
    
    def update(frame):
        tree, goal_node = frame
        
        while len(tree_lines) < len(tree) - 1:
            line, = ax.plot([], [], 'g-', alpha=0.3)
            tree_lines.append(line)

        while len(nodes_dots) < len(tree):
            dot, = ax.plot([], [], 'ro', markersize=2)
            nodes_dots.append(dot)
        
        for i, node in enumerate(tree[1:], 1):
            tree_lines[i-1].set_data([node.position[0], node.parent.position[0]],
                                     [node.position[1], node.parent.position[1]])

        for i, node in enumerate(tree):
            nodes_dots[i].set_data([node.position[0]], [node.position[1]])

        if goal_node:
            path = []
            current = goal_node
            while current:
                path.append(current.position)
                current = current.parent
            path = np.array(path)
            path_line.set_data(path[:, 0], path[:, 1])

        return tree_lines + [path_line, start_point, goal_point] + nodes_dots
    
    anim = FuncAnimation(fig, update, frames=rrt_star_generator, init_func=init, 
                     interval=10, blit=True, repeat=False)
    
    plt.show(block=True)
    plt.close(fig)
def transform_from_coppelia_coords(coppelia_x, coppelia_y):
    python_x = (coppelia_x - x_offset) / scale_x
    python_y = (coppelia_y - y_offset) / scale_y
    return python_x, python_y
  
class FourWheelDriveRobot:
    def __init__(self, visualizer=None):
        self.client = RemoteAPIClient()
        self.sim = self.client.getObject('sim')
        self.visualizer = visualizer

        self.max_speed = 3.0
        self.rotation_speed = 1.0
        self.reach_threshold = 0.3
        
        self.robot_handle = self.sim.getObject('/Robot')
        self.ground_handle = self.sim.getObject('/Map')
        self.wheels = {
            'front_left': self.get_handle('Wheel_Left_Front'),
            'front_right': self.get_handle('Wheel_Right_Front'),
            'rear_left': self.get_handle('Wheel_Left_Back'),
            'rear_right': self.get_handle('Wheel_Right_Back')
        }
        self.dt = 0.05

    def get_handle(self, name):
        handle = self.sim.getObjectHandle(name)
        if handle == -1:
            raise ValueError(f"Object '{name}' does not exist or path is ill formatted.")
        return handle

    def set_position(self, x, y, orientation):
        try:
            coppelia_x, coppelia_y = transform_to_coppelia_coords(x, y)
            self.sim.setObjectPosition(self.robot_handle, self.ground_handle, [coppelia_x, coppelia_y, -0.2])
            self.sim.setObjectOrientation(self.robot_handle, self.ground_handle, [0, 0, orientation])
            print(f"Robot positioned at ({coppelia_x}, {coppelia_y}) with orientation {math.degrees(orientation)}°")
            
            if self.visualizer:
                python_x, python_y = x, y
                self.visualizer.update_position(python_x, python_y, orientation)
        except Exception as e:
            print(f"Error setting robot position: {e}")

    def set_wheel_velocities(self, left_speed, right_speed):
        self.sim.setJointTargetVelocity(self.wheels['front_left'], left_speed)
        self.sim.setJointTargetVelocity(self.wheels['rear_left'], left_speed)
        self.sim.setJointTargetVelocity(self.wheels['front_right'], right_speed)
        self.sim.setJointTargetVelocity(self.wheels['rear_right'], right_speed)
        
    def transform_from_coppelia_coords(coppelia_x, coppelia_y):
        python_x = (coppelia_x - x_offset) / scale_x
        python_y = (coppelia_y - y_offset) / scale_y
        return python_x, python_y

    def follow_path(self, path, goal_orientation):
        print("Starting path following...")
        self.sim.startSimulation()
        
        try:
            for i, (target_x, target_y) in enumerate(path):
                print(f"Moving to target: ({target_x}, {target_y})")
                is_final_node = (i == len(path) - 1)

                while True:
                    position = self.sim.getObjectPosition(self.robot_handle, self.ground_handle)
                    orientation = self.sim.getObjectOrientation(self.robot_handle, self.ground_handle)

                    # Update visualization if visualizer exists
                    if self.visualizer:
                        python_x, python_y = transform_from_coppelia_coords(position[0], position[1])
                        self.visualizer.update_position(python_x, python_y, orientation[2])

                    distance = math.sqrt((target_x - position[0]) ** 2 + (target_y - position[1]) ** 2)
                    angle_to_target = math.atan2(target_y - position[1], target_x - position[0])
                    current_angle = orientation[2]
                    angle_diff = angle_to_target - current_angle
                    angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi
                    print(f"angle_diff: ({angle_diff})")

                    if abs(angle_diff) > 0.1:
                        left_speed = -self.rotation_speed if angle_diff > 0 else self.rotation_speed
                        right_speed = self.rotation_speed if angle_diff > 0 else -self.rotation_speed
                        self.set_wheel_velocities(left_speed, right_speed)
                    else:
                        speed = min(self.max_speed, distance + 1.5)
                        self.set_wheel_velocities(speed, speed)

                    if distance < self.reach_threshold:
                        print(f"Reached target: ({target_x}, {target_y})")
                        self.set_wheel_velocities(0, 0)
                        time.sleep(self.dt)

                        # Final orientation adjustment for the last node
                        if is_final_node and goal_orientation is not None:
                            final_angle_diff = goal_orientation - current_angle
                            final_angle_diff = (final_angle_diff + math.pi) % (2 * math.pi) - math.pi
                            while abs(final_angle_diff) > 0.1:
                                left_speed = -self.rotation_speed if final_angle_diff > 0 else self.rotation_speed
                                right_speed = self.rotation_speed if final_angle_diff > 0 else -self.rotation_speed
                                self.set_wheel_velocities(left_speed, right_speed)
                                time.sleep(self.dt)
                                current_angle = self.sim.getObjectOrientation(self.robot_handle, self.ground_handle)[2]
                                final_angle_diff = goal_orientation - current_angle
                                final_angle_diff = (final_angle_diff + math.pi) % (2 * math.pi) - math.pi
                            self.set_wheel_velocities(0, 0)

                        break

            print("Path following completed")
            time.sleep(20)
        except Exception as e:
            print(f"Error during path following: {e}")
            self.set_wheel_velocities(0, 0)
            raise
        finally:
            self.sim.stopSimulation()

def load_obstacles_from_csv(file_path):
    data = pd.read_csv(file_path, header=None)
    obstacles = []
    for y, row in enumerate(data.values):
        for x, value in enumerate(row):
            if value == 1:
                obstacles.append([x, y, 1, 1])
    return np.array(obstacles)

def save_path_to_csv(goal_node, scale_x, scale_y, x_offset, y_offset, filename='path2.csv'):
    path = []
    current = goal_node
    while current:
        path.append(current.position)
        current = current.parent
    path.reverse()
    
    transformed_path = [
        (pos[0] * scale_x + x_offset, pos[1] * scale_y + y_offset) 
        for pos in path
    ]
    np.savetxt(filename, transformed_path, delimiter=',', header='x,y', comments='')
    print(f"Path saved to {filename}")
def main():
    obstacles = load_obstacles_from_csv('occupancy_map4.csv')
    start,start_orientation = generate_random_position(obstacles)
    goal,goal_orientation = generate_random_position(obstacles)
    
    print(f"Generated start position: {start}, orientation: {math.degrees(start_orientation):.2f}°")
    print(f"Generated goal position: {goal}, orientation: {math.degrees(goal_orientation):.2f}°")

    # First create and show the RRT* planning visualization
    max_distance = 3
    max_iterations = 1000
    neighborhood_radius = 6

    rrt_star_gen = continuous_rrt_star(start, goal, obstacles, max_distance, max_iterations, neighborhood_radius)
    plot_rrt_star_realtime(start, goal, obstacles, rrt_star_gen)
    
    # After RRT* planning is complete, create robot visualizer and start robot movement
    visualizer = RobotVisualizer(obstacles, start, goal)
    robot = FourWheelDriveRobot(visualizer)
    robot.set_position(start[0], start[1],start_orientation)

    path = np.loadtxt('path2.csv', delimiter=',', skiprows=1)
    robot.follow_path(path,goal_orientation)

if __name__ == "__main__":
    main()
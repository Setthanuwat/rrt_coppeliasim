import time
import coppeliasim_zmqremoteapi_client as zmqRemoteApi
import matplotlib.pyplot as plt
import numpy as np
import csv
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, FancyArrowPatch

class OccupancyGridMap:
    def __init__(self, size=12, resolution=0.05):
        self.resolution = resolution
        self.size = size
        self.grid_size = int(2 * size / resolution)
        self.grid = np.zeros((self.grid_size, self.grid_size))
        self.log_odds_free = -0.4
        self.log_odds_occupied = 0.7
        
        # Pre-compute grid offsets
        self.grid_offset = self.size
        self.resolution_inv = 1.0 / self.resolution
        
    def world_to_grid(self, x, y):
        # Convert to numpy arrays if not already
        x = np.asarray(x)
        y = np.asarray(y)
        
        # Convert to grid coordinates
        gx = np.clip(((x + self.grid_offset) * self.resolution_inv), 0, self.grid_size-1).astype(np.int32)
        gy = np.clip(((y + self.grid_offset) * self.resolution_inv), 0, self.grid_size-1).astype(np.int32)
        
        return gx, gy

    def update_grid(self, robot_pos, endpoints_x, endpoints_y):
        robot_x, robot_y = self.world_to_grid(robot_pos[0], robot_pos[1])
        endpoint_xs, endpoint_ys = self.world_to_grid(endpoints_x, endpoints_y)
        
        # Update for each endpoint
        for ex, ey in zip(endpoint_xs, endpoint_ys):
            points = self.bresenham_line(int(robot_x), int(robot_y), int(ex), int(ey))
            if points:
                points = np.array(points)
                # Update free space
                mask_valid = (points[:,0] >= 0) & (points[:,0] < self.grid_size) & \
                            (points[:,1] >= 0) & (points[:,1] < self.grid_size)
                valid_points = points[mask_valid]
                if len(valid_points) > 1:
                    for point in valid_points[:-1]:  # Free space points
                        x, y = point
                        # Only update if the cell is not occupied
                        if self.grid[y, x] < self.log_odds_occupied:  # Assuming -10 is unoccupied
                            self.grid[y, x] += self.log_odds_free
                    
                    # Update occupied space for the last point
                    last_point = valid_points[-1]
                    x_last, y_last = last_point
                    self.grid[y_last, x_last] += self.log_odds_occupied
                    
        np.clip(self.grid, -10, 10, out=self.grid)

    def bresenham_line(self, x0, y0, x1, y1):
        """Bresenham's line algorithm"""
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = 1 if x1 > x0 else -1
        sy = 1 if y1 > y0 else -1
        
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                points.append([x, y])
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                points.append([x, y])
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
                
        points.append([x, y])
        return points

# Create a client instance
client = zmqRemoteApi.RemoteAPIClient()
sim = client.getObject('sim')
sim.startSimulation()

# Get handles
ground_handle = sim.getObject('/Map')
sensorbase_handle = sim.getObject('/Robot')
sensorhandle1 = sim.getObject('/Robot/fastHokuyo/sensor1')
sensorhandle2 = sim.getObject('/Robot/fastHokuyo/sensor2')

R_sb_s1 = sim.getObjectMatrix(sensorhandle1, sensorbase_handle)
R_sb_s2 = sim.getObjectMatrix(sensorhandle2, sensorbase_handle)

# Setup visualization
map_size = 12
fig, ax = plt.subplots(figsize=(10, 10))
ogm = OccupancyGridMap(size=map_size, resolution=0.05)

# Initialize plot elements
grid_img = ax.imshow(ogm.grid, cmap='Greys', origin='lower',
                     extent=[-map_size, map_size, -map_size, map_size],
                     vmin=-2, vmax=2)
robot_marker = Circle((0, 0), 0.3, color='red', fill=True)
ax.add_patch(robot_marker)
direction_arrow = FancyArrowPatch((0, 0), (0, 0), color='blue', mutation_scale=15)
ax.add_patch(direction_arrow)

plt.colorbar(grid_img, label='Occupancy probability (log-odds)')
ax.set_xlim(-map_size, map_size)
ax.set_ylim(-map_size, map_size)
ax.invert_xaxis()
ax.invert_yaxis()

# CSV setup
f = open('data10.csv', 'w')
f = open('data10.csv', 'a', newline='')
writer = csv.writer(f)

def get_laser_data(point1, point2):
    skip = 4
    data = np.array(point1)
    x1 = np.array(data[2::skip])
    y1 = np.array(data[3::skip])
    z1 = np.array(data[4::skip])

    data2 = np.array(point2)
    x2 = np.array(data2[2::skip])
    y2 = np.array(data2[3::skip])
    z2 = np.array(data2[4::skip])

    xyz1 = np.hstack((x1,y1,z1))
    Rot1 = np.array(R_sb_s1).reshape(3,4)
    Rot1 = Rot1[0:3,0:3]

    xyz2 = np.hstack((x2,y2,z2))
    Rot2 = np.array(R_sb_s2).reshape(3,4)
    Rot2 = Rot2[0:3,0:3]

    u = np.array([])
    v = np.array([])
    length_angle = np.array([])

    for i in range(len(x1)):
        ww = np.array([x1[i],y1[i],z1[i]]).reshape(3,1)
        uvw = np.matmul(Rot1,ww)
        le, th = conv_pc2_length_phi(uvw[0,0],uvw[1,0])
        length_angle = np.append(length_angle, [le, th])
        u = np.append(u,uvw[0,0])
        v = np.append(v,uvw[1,0])

    for i in range(len(x2)):
        ww = np.array([x2[i],y2[i],z2[i]]).reshape(3,1)
        uvw = np.matmul(Rot2,ww)
        le, th = conv_pc2_length_phi(uvw[0,0],uvw[1,0])
        length_angle = np.append(length_angle, [le, th])
        u = np.append(u,uvw[0,0])
        v = np.append(v,uvw[1,0])

    return u, v, length_angle

def conv_pc2_length_phi(x,y):
    th = np.arctan2(y,x)
    length = np.sqrt(x*x+y*y)
    return length, th

def Transofrm_point(X,Y,th,x,y):
    xnew = X + x*np.cos(th) - y*np.sin(th)
    ynew = Y + x*np.sin(th) + y*np.cos(th)
    return xnew, ynew

num_frame = 650

def update(frame):
    if frame < num_frame - 1:
        print(frame)
        data = np.array([])

        # Get robot pose
        position = sim.getObjectPosition(sensorbase_handle, ground_handle)
        orientation = sim.getObjectOrientation(sensorbase_handle, ground_handle)
        
        # Read sensor data
        res1, dist1, point1 = sim.readVisionSensor(sensorhandle1)
        res2, dist2, point2 = sim.readVisionSensor(sensorhandle2)
        
        # Save data to CSV
        data = np.append(data, [position[0], position[1], orientation[2]])
        l_x, l_y, length_angle = get_laser_data(point1, point2)
        data = np.append(data, length_angle)
        writer.writerow(data)
        
        # Transform points to world coordinates
        l_xnew, l_ynew = Transofrm_point(position[0], position[1], orientation[2], l_x, l_y)
        
        # Update occupancy grid
        ogm.update_grid([position[0], position[1]], l_xnew, l_ynew)
        
        # Update visualization
        grid_img.set_array(ogm.grid)
        robot_marker.center = (position[0], position[1])
        
        # Update direction arrow
        arrow_length = 0.5
        dx = arrow_length * np.cos(orientation[2])
        dy = arrow_length * np.sin(orientation[2])
        direction_arrow.set_positions(
            (position[0], position[1]),
            (position[0] + dx, position[1] + dy)
        )
    else:
        ani.event_source.stop()
        sim.stopSimulation()
        f.close()

    return grid_img, robot_marker, direction_arrow

try:
    ani = FuncAnimation(fig, update, frames=num_frame, 
                       interval=50, blit=True, repeat=False)
    plt.show()
except Exception as e:
    print(f"Animation error: {str(e)}")
    sim.stopSimulation()
    f.close()
finally:
    sim.stopSimulation()
    f.close()
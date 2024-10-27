import numpy as np
import matplotlib.pyplot as plt

class OccupancyGridMap:
    def __init__(self, size=24, resolution=0.01):
        self.resolution = resolution
        self.size = size
        self.grid_size = int(size / resolution)
        self.grid = np.zeros((self.grid_size, self.grid_size))  # For free/occupied probabilities
        self.occupied_cells = np.zeros((self.grid_size, self.grid_size))  # To track all occupied cells
        
        # Pre-compute grid offsets for world_to_grid transformation
        self.grid_offset = self.size / 2
        self.resolution_inv = 1.0 / self.resolution
        
    def world_to_grid(self, x, y):
        gx = ((x + self.grid_offset) * self.resolution_inv).astype(np.int32)
        gy = ((y + self.grid_offset) * self.resolution_inv).astype(np.int32)
        return np.clip(gx, 0, self.grid_size - 1), np.clip(gy, 0, self.grid_size - 1)

    def update_grid(self, robot_pos, endpoints_x, endpoints_y):
        robot_x, robot_y = self.world_to_grid(robot_pos[0], robot_pos[1])
        endpoint_xs, endpoint_ys = self.world_to_grid(endpoints_x, endpoints_y)
        
        for ex, ey in zip(endpoint_xs, endpoint_ys):
            points = self.bresenham_line((robot_x, robot_y), (ex, ey))
            if points:
                points = np.array(points)
                # Update the occupied cells
                mask_valid = (points[:, 0] >= 0) & (points[:, 0] < self.grid_size) & \
                             (points[:, 1] >= 0) & (points[:, 1] < self.grid_size)
                valid_points = points[mask_valid]
                if len(valid_points) > 1:
                    # Mark the free space
                    for point in valid_points[:-1]:
                        self.grid[point[1], point[0]] = 0  # Mark as free
                    # Mark the last point as occupied
                    self.grid[valid_points[-1, 1], valid_points[-1, 0]] = 1  # Mark as occupied
                    self.occupied_cells[valid_points[-1, 1], valid_points[-1, 0]] = 1  # Track occupied cells
        
    def bresenham_line(self, start, end):
        x0, y0 = start
        x1, y1 = end
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        
        points = []
        x, y = x0, y0
        
        if dx > dy:
            err = dx / 2
            while x != x1:
                points.append((x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2
            while y != y1:
                points.append((x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
                
        points.append((x, y))
        return points

# Vectorized transformation function
def Transform_laser2pc_vectorized(X, Y, th, lengths, angles):
    x = lengths * np.cos(angles)
    y = lengths * np.sin(angles)
    cos_th = np.cos(th)
    sin_th = np.sin(th)
    xnew = X + x * cos_th - y * sin_th
    ynew = Y + x * sin_th + y * cos_th
    return xnew, ynew

# Load and prepare data
data = np.loadtxt('data10.csv', delimiter=',')
X = data[:, 0]
Y = data[:, 1]
TH = data[:, 2]
num_ray = int((data.shape[1] - 3) / 2)

# Pre-compute sensor data
sensor_length = data[:, 3:3 + 2 * num_ray:2]
sensor_angle = data[:, 4:3 + 2 * num_ray:2]

# Initialize occupancy grid map
ogm = OccupancyGridMap(size=24, resolution=0.1)

# Process all frames to update the grid
for frame in range(data.shape[0]):
    endpoints_x, endpoints_y = Transform_laser2pc_vectorized(
        X[frame], Y[frame], TH[frame],
        sensor_length[frame], sensor_angle[frame]
    )
    
    # Update the grid for the current frame
    ogm.update_grid([X[frame], Y[frame]], endpoints_x, endpoints_y)

# Save the occupied cells as a CSV file
np.savetxt('occupancy_map4.csv', ogm.occupied_cells, delimiter=',', fmt='%d')

# Visualization of the final occupancy map
fig, ax = plt.subplots(figsize=(8, 10))
ax.set_xlim(-8,4)
ax.set_ylim(-4,8)
# Show the occupied cells only
grid_img = ax.imshow(ogm.occupied_cells, cmap='Greys', origin='lower', 
                     extent=[-ogm.size / 2, ogm.size / 2, -ogm.size / 2, ogm.size / 2],
                     vmin=0, vmax=1)

plt.colorbar(grid_img)
plt.title("Final Occupancy Grid Map (All Detected Obstacles)")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.show()

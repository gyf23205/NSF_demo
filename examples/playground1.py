import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from PIL import Image

def generate_voronoi_plots(map, centroids):
        centroids = np.array(centroids)
        # Compute power diagram
        label_map = np.zeros(map.shape, dtype=int)
        Y, X = np.meshgrid(np.arange(map.shape[0]), np.arange(map.shape[1]), indexing='ij')
        # points = np.stack((X, Y), axis=-1)  # shape (H, W, 2)

        for i, (px, py) in enumerate(centroids):
            dx = X - px
            dy = Y - py
            dist2 = dx**2 + dy**2 - map[i]**2
            if i == 0:
                dist_stack = dist2[..., np.newaxis]
            else:
                dist_stack = np.concatenate((dist_stack, dist2[..., np.newaxis]), axis=2)

        label_map = np.argmin(dist_stack, axis=2)

        region_vertices = {}  # key: region index, value: list of (y, x) coords

        for i in range(len(centroids)):  # N is the number of sites
            mask = (label_map == i).astype(np.uint8)
            contours = measure.find_contours(mask, 0.5)
            
            if contours:
                # You can get multiple disconnected contours — take the largest
                largest = max(contours, key=lambda x: x.shape[0])
                region_vertices[i] = largest 
        
        fig_width, fig_height = 1200, 960
        dpi = 100

        plt.figure(figsize=(fig_width/dpi, fig_height/dpi), dpi=dpi)

        plt.xlim(0, fig_width)
        plt.ylim(fig_height, 0)  # Invert y for image coordinates
        plt.axis('off')
        plt.scatter(centroids[:,0], centroids[:,1], c='red', edgecolors='black', s=80, label='Sites')
        for i, contour in region_vertices.items():
            plt.plot(contour[:,1], contour[:,0], linewidth=1.5, label=f'Region {i}')

        # # Get all contour points to determine the bounding box
        # all_points = np.vstack([contour for contour in region_vertices.values()])
        # min_y, min_x = np.min(all_points, axis=0)
        # max_y, max_x = np.max(all_points, axis=0)

        # # Add a small margin
        # margin = 20
        # plt.xlim(max(min_x - margin, 0), min(max_x + margin, map.shape[1]))
        # plt.ylim(min(max_y + margin, map.shape[0]), max(min_y - margin, 0))  # invert y for image coordinates

        plt.savefig('examples/images/voronoi_regions.png', pad_inches=0, dpi=dpi, transparent=False)
        plt.close()
        img = Image.open('examples/images/voronoi_regions.png')
        box = (150, 120, 1200-150, 960-120)
        cropped_img = img.crop(box)
        cropped_img.save('examples/images/voronoi_regions_cropped.png', format='PNG')
        # cropped_img.thumbnail((900, 720), Image.Resampling.LANCZOS)
        # cropped_img = cropped_img.resize((900, 720), Image.Resampling.LANCZOS)
        # cropped_img.save('examples/images/voronoi_regions_resized.png', format='PNG') 
        

def distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

ratio = 240.0
center = [450.0, 360.0]

def position_meter_to_gui(p_meter):
        # print("p_meter", p_meter)
        p_gui = np.array(p_meter)
        for k in range(len(p_meter)):
            p_gui[k][0] = ratio * p_gui[k][0] + center[0]
            p_gui[k][1] = -ratio * p_gui[k][1] + center[1]
        return p_gui

n_targets = 14
map = np.ones((720, 900))
# Pad map to (1200, 960)
padded_map = np.zeros((960, 1200))
padded_map[:map.shape[0], :map.shape[1]] = map
map = padded_map
random_positions = []
while len(random_positions) < n_targets:
    # print('Generating random target positions')
    new_position = [np.random.uniform(-1.9, 1.9), np.random.uniform(-1.2, 1.2)]

    # Check distance to existing positions
    if all(distance(new_position, existing) >= 0.25 for existing in random_positions):
        # Avoid takeoff and wind position by force
        if distance(new_position, [0, 0]) > 0.5:
            random_positions.append(new_position)
pos_gui = position_meter_to_gui(random_positions)
generate_voronoi_plots(map, pos_gui)
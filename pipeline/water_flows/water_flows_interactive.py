import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter
from numba import njit, prange
import imageio
import os

# --- Parametry ---
GRID_SIZE = 200
CLICK_SOURCE_AMOUNT = 0.3
FLOW_FACTOR = 0.85
PLOT_INTERVAL = 75
WATER_VMAX = 0.3
SINK_MARGIN = 12
ELEVATION_RADIUS = 25
ELEVATION_STRENGTH = 0.08
ELEVATION_STEPS = 40
SIMULATION_STEPS = 20000
FPS = 25
ROTATION_FRAMES = 76  # 76 klatek na 380 stopni (5 stopni na klatkę)

# --- Tryby kliknięcia ---
MODE_RAISE = 0
MODE_LOWER = 1
MODE_SOURCE = 2

# --- Kierunki przepływu ---
directions_dx = np.array([-1, -1, -1, 0, 0, 1, 1, 1], dtype=np.int32)
directions_dy = np.array([-1, 0, 1, -1, 1, -1, 0, 1], dtype=np.int32)


@njit(fastmath=True)
def terrain_function(x, y):
    term1 = -1.2 * np.exp(-((y - 0.8 * np.tanh(0.5 * x)) ** 2) / 0.5)
    term2 = 1.5 * np.exp(-((x - 2) ** 2) / 4) * np.exp(-((y + 1.5) ** 2) / 2.5)
    return term1 + term2


def create_terrain(n):
    x = np.linspace(-6, 6, n)
    y = np.linspace(-6, 6, n)
    X, Y = np.meshgrid(x, y, indexing='ij')
    T = terrain_function(X, Y)
    T = gaussian_filter(T, sigma=0.8)
    T -= T.min()
    return X, Y, T


@njit(parallel=True)
def calculate_flow(water, H_smooth, flow_factor, outflow, inflow):
    n = water.shape[0]
    outflow[:] = 0.0
    inflow[:] = 0.0
    for i in prange(n):
        for j in range(n):
            w = water[i, j]
            if w <= 1e-3:
                continue
            h0 = H_smooth[i, j]
            total_diff = 0.0
            for k in range(8):
                ni = i + directions_dx[k]
                nj = j + directions_dy[k]
                if 0 <= ni < n and 0 <= nj < n:
                    diff = h0 - H_smooth[ni, nj]
                    if diff > 1e-4:
                        total_diff += diff
            if total_diff == 0.0:
                continue
            max_out = min(w * 0.5, w * flow_factor * 2)
            for k in range(8):
                ni = i + directions_dx[k]
                nj = j + directions_dy[k]
                if 0 <= ni < n and 0 <= nj < n:
                    diff = h0 - H_smooth[ni, nj]
                    if diff > 1e-4:
                        frac = diff / total_diff
                        f = frac * max_out
                        outflow[i, j] += f
                        inflow[ni, nj] += f
    return outflow, inflow


def apply_terrain_change(terrain, i, j, mode, radius=ELEVATION_RADIUS, strength=ELEVATION_STRENGTH):
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            ni = i + di
            nj = j + dj
            if 0 <= ni < GRID_SIZE and 0 <= nj < GRID_SIZE:
                d = np.sqrt(di ** 2 + dj ** 2) / radius
                if d < 1.0:
                    delta = (1 - d) * strength
                    if mode == MODE_RAISE:
                        terrain[ni, nj] += delta
                    elif mode == MODE_LOWER:
                        terrain[ni, nj] -= delta
    return terrain


def create_animation():
    # Inicjalizacja terenu
    X, Y, terrain = create_terrain(GRID_SIZE)
    water = np.zeros_like(terrain)
    sources = np.zeros_like(water, dtype=bool)
    outflow = np.zeros_like(water)
    inflow = np.zeros_like(water)
    sinks = np.zeros_like(water, dtype=bool)

    # Pozycje modyfikacji terenu
    raise_pos = [GRID_SIZE // 3, GRID_SIZE // 3]

    # Początkowa i końcowa pozycja dla obniżania terenu (wzdłuż rzeki)
    lower_start = [2 * GRID_SIZE // 3, GRID_SIZE // 4]
    lower_end = [2 * GRID_SIZE // 3, 3 * GRID_SIZE // 4]

    source_pos = [GRID_SIZE - 20, GRID_SIZE // 2]

    # Konfiguracja źródła i pochłaniaczy
    sources[source_pos[0], source_pos[1]] = True
    sinks[:, 0] = True
    for di in range(-SINK_MARGIN, SINK_MARGIN + 1):
        for dj in range(-SINK_MARGIN, SINK_MARGIN + 1):
            ni = source_pos[0] + di
            nj = source_pos[1] + dj
            if 0 <= ni < GRID_SIZE and 0 <= nj < GRID_SIZE:
                sinks[ni, nj] = False

    # Przygotowanie figur dla dwóch osobnych GIF-ów
    fig_2d = plt.figure(figsize=(10, 8))
    ax_2d = fig_2d.add_subplot(111)

    fig_3d = plt.figure(figsize=(8, 6))
    ax_3d = fig_3d.add_subplot(111, projection='3d')

    frames_2d = []
    frames_3d = []

    # Funkcja do zapisywania klatek dla obu figur
    def save_frames():
        # Zapis dla widoku 2D
        fig_2d.canvas.draw()
        image_2d = np.frombuffer(fig_2d.canvas.tostring_rgb(), dtype='uint8')
        image_2d = image_2d.reshape(fig_2d.canvas.get_width_height()[::-1] + (3,))
        frames_2d.append(image_2d)

        # Zapis dla widoku 3D
        fig_3d.canvas.draw()
        image_3d = np.frombuffer(fig_3d.canvas.tostring_rgb(), dtype='uint8')
        image_3d = image_3d.reshape(fig_3d.canvas.get_width_height()[::-1] + (3,))
        frames_3d.append(image_3d)

    # 1. Initial terrain
    ax_2d.clear()
    cs = ax_2d.contour(X, Y, terrain, colors='gray', linewidths=0.5)
    ax_2d.clabel(cs, inline=True, fontsize=6, fmt="%.1f")
    ax_2d.set_title('Initial Terrain')
    ax_2d.set_aspect('equal')
    ax_2d.set_xlim(X.min(), X.max())
    ax_2d.set_ylim(Y.min(), Y.max())

    ax_3d.clear()
    ax_3d.plot_surface(X, Y, terrain, cmap='terrain', edgecolor='none', alpha=0.9)
    ax_3d.set_title('3D Topography')
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Elevation')
    ax_3d.set_xlim(X.min(), X.max())
    ax_3d.set_ylim(Y.min(), Y.max())
    ax_3d.set_zlim(terrain.min(), terrain.max())
    ax_3d.set_box_aspect([np.ptp(X), np.ptp(Y), np.ptp(terrain)])
    ax_3d.view_init(elev=30, azim=0)

    for _ in range(40):
        save_frames()

    # 2. Raising terrain
    for step in range(ELEVATION_STEPS):
        terrain = apply_terrain_change(terrain, raise_pos[0], raise_pos[1], MODE_RAISE)
        terrain = gaussian_filter(terrain, sigma=0.8)

        # Aktualizacja widoku 2D
        ax_2d.clear()
        cs = ax_2d.contour(X, Y, terrain, colors='gray', linewidths=0.5)
        ax_2d.clabel(cs, inline=True, fontsize=6, fmt="%.1f")
        ax_2d.scatter(X[raise_pos[0], raise_pos[1]], Y[raise_pos[0], raise_pos[1]],
                      color='red', s=100, marker='^', alpha=0.7)
        ax_2d.set_title(f'Raising Terrain: Step {step + 1}/{ELEVATION_STEPS}')
        ax_2d.set_aspect('equal')
        ax_2d.set_xlim(X.min(), X.max())
        ax_2d.set_ylim(Y.min(), Y.max())

        # Aktualizacja widoku 3D
        ax_3d.clear()
        ax_3d.plot_surface(X, Y, terrain, cmap='terrain', edgecolor='none', alpha=0.9)
        ax_3d.set_title('3D Topography')
        ax_3d.set_xlabel('X')
        ax_3d.set_ylabel('Y')
        ax_3d.set_zlabel('Elevation')
        ax_3d.set_xlim(X.min(), X.max())
        ax_3d.set_ylim(Y.min(), Y.max())
        ax_3d.set_zlim(terrain.min(), terrain.max())
        ax_3d.set_box_aspect([np.ptp(X), np.ptp(Y), np.ptp(terrain)])
        ax_3d.view_init(elev=30, azim=0)

        save_frames()

    # Pause after raising terrain
    for _ in range(40):
        save_frames()

    # 3. Lowering terrain (with moving cursor)
    for step in range(ELEVATION_STEPS):
        # Calculate current cursor position (movement along the river)
        progress = step / (ELEVATION_STEPS - 1)
        current_lower_x = lower_start[0] + (lower_end[0] - lower_start[0]) * progress
        current_lower_y = lower_start[1] + (lower_end[1] - lower_start[1]) * progress

        terrain = apply_terrain_change(terrain, int(current_lower_x), int(current_lower_y), MODE_LOWER)
        terrain = gaussian_filter(terrain, sigma=0.8)

        # Aktualizacja widoku 2D
        ax_2d.clear()
        cs = ax_2d.contour(X, Y, terrain, colors='gray', linewidths=0.5)
        ax_2d.clabel(cs, inline=True, fontsize=6, fmt="%.1f")
        ax_2d.scatter(X[raise_pos[0], raise_pos[1]], Y[raise_pos[0], raise_pos[1]],
                      color='red', s=100, marker='^', alpha=0.7)
        ax_2d.scatter(X[int(current_lower_x), int(current_lower_y)],
                      Y[int(current_lower_x), int(current_lower_y)],
                      color='blue', s=100, marker='v', alpha=0.7)
        ax_2d.set_title(f'Lowering Terrain: Step {step + 1}/{ELEVATION_STEPS}')
        ax_2d.set_aspect('equal')
        ax_2d.set_xlim(X.min(), X.max())
        ax_2d.set_ylim(Y.min(), Y.max())

        # Aktualizacja widoku 3D
        ax_3d.clear()
        ax_3d.plot_surface(X, Y, terrain, cmap='terrain', edgecolor='none', alpha=0.9)
        ax_3d.set_title('3D Topography')
        ax_3d.set_xlabel('X')
        ax_3d.set_ylabel('Y')
        ax_3d.set_zlabel('Elevation')
        ax_3d.set_xlim(X.min(), X.max())
        ax_3d.set_ylim(Y.min(), Y.max())
        ax_3d.set_zlim(terrain.min(), terrain.max())
        ax_3d.set_box_aspect([np.ptp(X), np.ptp(Y), np.ptp(terrain)])
        ax_3d.view_init(elev=30, azim=0)

        save_frames()

    # Long pause after terrain modifications
    for _ in range(60):
        save_frames()

    # 4. Terrain rotation (380 degrees) - tylko w widoku 3D
    for i in range(ROTATION_FRAMES):
        azim = i * 380 / ROTATION_FRAMES

        # Widok 2D - statyczny bez obrotu
        ax_2d.clear()
        cs = ax_2d.contour(X, Y, terrain, colors='gray', linewidths=0.5)
        ax_2d.clabel(cs, inline=True, fontsize=6, fmt="%.1f")
        ax_2d.scatter(X[raise_pos[0], raise_pos[1]], Y[raise_pos[0], raise_pos[1]],
                      color='red', s=100, marker='^', alpha=0.3)
        ax_2d.scatter(X[lower_end[0], lower_end[1]], Y[lower_end[0], lower_end[1]],
                      color='blue', s=100, marker='v', alpha=0.3)
        ax_2d.scatter(X[source_pos[0], source_pos[1]], Y[source_pos[0], source_pos[1]],
                      color='red', s=70, marker='x', alpha=0.7)
        ax_2d.set_title(f'Final Terrain')
        ax_2d.set_aspect('equal')
        ax_2d.set_xlim(X.min(), X.max())
        ax_2d.set_ylim(Y.min(), Y.max())

        # Widok 3D - obrót
        ax_3d.clear()
        ax_3d.plot_surface(X, Y, terrain, cmap='terrain', edgecolor='none', alpha=0.9)
        ax_3d.set_title(f'3D Rotation: {int(azim)}°')
        ax_3d.set_xlabel('X')
        ax_3d.set_ylabel('Y')
        ax_3d.set_zlabel('Elevation')
        ax_3d.set_xlim(X.min(), X.max())
        ax_3d.set_ylim(Y.min(), Y.max())
        ax_3d.set_zlim(terrain.min(), terrain.max())
        ax_3d.set_box_aspect([np.ptp(X), np.ptp(Y), np.ptp(terrain)])
        ax_3d.view_init(elev=30, azim=azim)

        save_frames()

    # Pause before water simulation
    for _ in range(60):
        save_frames()

    # 5. Water simulation
    water_frames = 0
    for step in range(SIMULATION_STEPS):
        # Water update
        water[sources] += CLICK_SOURCE_AMOUNT
        H = terrain + water
        H_smooth = gaussian_filter(H, sigma=0.6)
        outflow, inflow = calculate_flow(water, H_smooth, FLOW_FACTOR, outflow, inflow)
        water += inflow
        water -= outflow
        water = gaussian_filter(water, sigma=0.3)
        water[water < 1e-4] = 0.0
        water[sinks] = 0.0

        # Save every 75 steps
        if step % PLOT_INTERVAL == 0:
            water_frames += 1

            # Widok 2D - z wodą
            ax_2d.clear()
            cs = ax_2d.contour(X, Y, terrain, colors='gray', linewidths=0.5)
            ax_2d.clabel(cs, inline=True, fontsize=6, fmt="%.1f")
            wp = np.where(water > 0.005, water, np.nan)
            ax_2d.pcolormesh(X, Y, wp, cmap='Blues', shading='auto', vmin=0, vmax=WATER_VMAX, alpha=0.6)
            ax_2d.scatter(X[source_pos[0], source_pos[1]], Y[source_pos[0], source_pos[1]],
                          color='red', s=70, marker='x')
            ax_2d.scatter(X[raise_pos[0], raise_pos[1]], Y[raise_pos[0], raise_pos[1]],
                          color='red', s=100, marker='^', alpha=0.3)
            ax_2d.scatter(X[lower_end[0], lower_end[1]], Y[lower_end[0], lower_end[1]],
                          color='blue', s=100, marker='v', alpha=0.3)
            ax_2d.set_title(f'Water Simulation: Step {step}/{SIMULATION_STEPS}')
            ax_2d.set_aspect('equal')
            ax_2d.set_xlim(X.min(), X.max())
            ax_2d.set_ylim(Y.min(), Y.max())

            # Widok 3D - tylko teren (bez wody)
            ax_3d.clear()
            ax_3d.plot_surface(X, Y, terrain, cmap='terrain', edgecolor='none', alpha=0.9)
            ax_3d.set_title('3D Topography')
            ax_3d.set_xlabel('X')
            ax_3d.set_ylabel('Y')
            ax_3d.set_zlabel('Elevation')
            ax_3d.set_xlim(X.min(), X.max())
            ax_3d.set_ylim(Y.min(), Y.max())
            ax_3d.set_zlim(terrain.min(), terrain.max())
            ax_3d.set_box_aspect([np.ptp(X), np.ptp(Y), np.ptp(terrain)])
            ax_3d.view_init(elev=30, azim=0)

            save_frames()

    # Long final pause
    for _ in range(80):
        save_frames()

    # Zapis animacji do dwóch osobnych plików GIF
    imageio.mimsave('water_flow_2d.gif', frames_2d, fps=FPS)
    imageio.mimsave('water_flow_3d.gif', frames_3d, fps=FPS)

    # Zamknięcie figur
    plt.close(fig_2d)
    plt.close(fig_3d)

    print(f"Animation saved as 'water_flow_2d.gif' and 'water_flow_3d.gif'")
    print(f"2D frames: {len(frames_2d)}, 3D frames: {len(frames_3d)}")


if __name__ == "__main__":
    create_animation()
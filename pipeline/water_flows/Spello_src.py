import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from scipy.ndimage import gaussian_filter
from numba import njit, prange

plt.ion()

# --- Parametry ---
GRID_SIZE = 200
CLICK_SOURCE_AMOUNT = 0.3
FLOW_FACTOR = 0.85
PLOT_INTERVAL = 50
WATER_VMAX = 0.3
SINK_MARGIN = 12
ELEVATION_RADIUS = 12

# --- Tryby kliknięcia ---
MODE_RAISE = 0
MODE_LOWER = 1
MODE_SOURCE = 2
MODE_NAMES = ['Podnoszenie terenu', 'Obniżanie terenu', 'Źródło wody']

# --- Kierunki przepływu ---
directions_dx = np.array([-1,-1,-1, 0, 0, 1, 1, 1], dtype=np.int32)
directions_dy = np.array([-1, 0, 1,-1, 1,-1, 0, 1], dtype=np.int32)

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

# --- (Reszta kodu, importy i definicje funkcji pozostają bez zmian) ---

def simulate_interactive_overlay_optimized():
    click_mode = MODE_SOURCE
    simulation_paused = False  # Nowa flaga do kontrolowania pauzy

    X, Y, terrain = create_terrain(GRID_SIZE)
    water = np.zeros_like(terrain)
    sources = np.zeros_like(water, dtype=bool)
    outflow = np.zeros_like(water)
    inflow = np.zeros_like(water)
    sinks = np.zeros_like(water, dtype=bool)

    fig = plt.figure(figsize=(14, 7))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.5, 1])
    ax = fig.add_subplot(gs[0])
    ax3d = fig.add_subplot(gs[1], projection='3d')
    plt.subplots_adjust(right=0.9, bottom=0.15)

    source_pos = [GRID_SIZE - 2, GRID_SIZE // 2]
    sources[source_pos[0], source_pos[1]] = True

    def update_sinks():
        sinks[:, :] = False
        sinks[:, 0] = True
        for di in range(-SINK_MARGIN, SINK_MARGIN + 1):
            for dj in range(-SINK_MARGIN, SINK_MARGIN + 1):
                ni = source_pos[0] + di
                nj = source_pos[1] + dj
                if 0 <= ni < GRID_SIZE and 0 <= nj < GRID_SIZE:
                    sinks[ni, nj] = False
    update_sinks()

    is_dragging = [False]
    terrain_changed_during_drag = [False]

    def apply_terrain_change(i, j):
        for di in range(-ELEVATION_RADIUS, ELEVATION_RADIUS + 1):
            for dj in range(-ELEVATION_RADIUS, ELEVATION_RADIUS + 1):
                ni = i + di
                nj = j + dj
                if 0 <= ni < GRID_SIZE and 0 <= nj < GRID_SIZE:
                    d = np.sqrt(di ** 2 + dj ** 2) / ELEVATION_RADIUS
                    if d < 1.0:
                        delta = (1 - d) * 0.03
                        if click_mode == MODE_RAISE:
                            terrain[ni, nj] += delta
                        elif click_mode == MODE_LOWER:
                            terrain[ni, nj] -= delta
        terrain_changed_during_drag[0] = True

    def on_press(event):
        nonlocal click_mode, simulation_paused
        if event.inaxes != ax: return
        is_dragging[0] = True

        i = int(np.clip(round((event.xdata - X[0,0]) / (X[1,0] - X[0,0])), 0, GRID_SIZE - 1))
        j = int(np.clip(round((event.ydata - Y[0,0]) / (Y[0,1] - Y[0,0])), 0, GRID_SIZE - 1))

        if click_mode in [MODE_RAISE, MODE_LOWER]:
            simulation_paused = True # PAUZA!
            apply_terrain_change(i, j)
        elif click_mode == MODE_SOURCE:
            sources[:, :] = False
            sources[i, j] = True
            source_pos[0], source_pos[1] = i, j
            update_sinks()

    def on_release(event):
        nonlocal simulation_paused
        is_dragging[0] = False
        if terrain_changed_during_drag[0]:
            nonlocal terrain
            terrain = gaussian_filter(terrain, sigma=0.8)
            terrain_changed_during_drag[0] = False

        if simulation_paused:
            simulation_paused = False # WZNOWIENIE!

    def on_motion(event):
        if not is_dragging[0] or event.inaxes != ax: return

        i = int(np.clip(round((event.xdata - X[0,0]) / (X[1,0] - X[0,0])), 0, GRID_SIZE - 1))
        j = int(np.clip(round((event.ydata - Y[0,0]) / (Y[0,1] - Y[0,0])), 0, GRID_SIZE - 1))

        if click_mode in [MODE_RAISE, MODE_LOWER]:
            apply_terrain_change(i, j)
        elif click_mode == MODE_SOURCE:
            sources[:, :] = False
            sources[i, j] = True
            source_pos[0], source_pos[1] = i, j
            update_sinks()

    def on_key(event):
        nonlocal click_mode
        if event.key == 'right':
            click_mode = (click_mode + 1) % 3
        elif event.key == 'left':
            click_mode = (click_mode - 1) % 3
        elif event.key == 'p': # Dodatkowa opcja: ręczna pauza klawiszem 'p'
            nonlocal simulation_paused
            simulation_paused = not simulation_paused

    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('key_press_event', on_key)

    step = 0
    while plt.fignum_exists(fig.number):
        # --- Blok symulacji ---
        if not simulation_paused:
            water[sources] += CLICK_SOURCE_AMOUNT
            H = terrain + water
            H_smooth = gaussian_filter(H, sigma=0.6)
            outflow, inflow = calculate_flow(water, H_smooth, FLOW_FACTOR, outflow, inflow)
            water += inflow
            water -= outflow
            water = gaussian_filter(water, sigma=0.3)
            water[water < 1e-4] = 0.0
            water[sinks] = 0.0
            step += 1

        # --- Blok rysowania (wykonywany zawsze) ---
        if step % PLOT_INTERVAL == 0 or simulation_paused:
            ax.clear()
            ax.contour(X, Y, terrain, colors='gray', linewidths=0.5)
            wp = np.where(water > 0.005, water, np.nan)
            ax.pcolormesh(X, Y, wp, cmap='Blues', shading='auto', vmin=0, vmax=WATER_VMAX, alpha=0.6)
            ax.scatter(X[source_pos[0], source_pos[1]], Y[source_pos[0], source_pos[1]], color='red', s=50, marker='x')

            pause_text = " | PAUZA" if simulation_paused else ""
            ax.set_title(f'Krok {step} | Tryb: {MODE_NAMES[click_mode]}{pause_text}')
            ax.set_aspect('equal')
            ax.text(0.5, -0.1, 'Strzałki: zmiana trybu | Klawisz "p": pauza/wznowienie',
                    ha='center', va='top', fontsize=10, transform=ax.transAxes)

            ax3d.clear()
            ax3d.plot_surface(X, Y, terrain, cmap='terrain', edgecolor='none', alpha=0.9)
            ax3d.set_title('Topografia 3D')
            ax3d.set_box_aspect([np.ptp(X), np.ptp(Y), np.ptp(terrain)*0.5]) # Lepsze proporcje 3D

            plt.pause(0.01) # Krótsza pauza dla płynności

# Użyj nowej funkcji
if __name__ == "__main__":
    simulate_interactive_overlay_optimized()
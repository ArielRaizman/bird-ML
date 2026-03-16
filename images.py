import matplotlib.pyplot as plt
import numpy as np

def generate_double_trap(has_toucan=True, has_green_dots=True, seed=42):
    np.random.seed(seed)
    
    # 1. Setup Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_facecolor('black')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    
    # 2. Generate "Confetti" (The Forest Noise)
    # Avoid Green, Blue, and Gold to keep targets distinct
    colors = ['#FF5733', '#33FFF5', '#F533FF', '#FFFFFF', '#A0A0A0', '#FF33A8']
    
    for _ in range(150):
        x = np.random.randint(0, 100)
        y = np.random.randint(0, 50)
        size = np.random.randint(50, 200)
        color = np.random.choice(colors)
        ax.scatter(x, y, s=size, c=color, alpha=0.6, edgecolors='none')

    # 3. BAIT #1: The "Yellow Line" (The obvious pattern)
    ax.axhline(y=25, color='gold', linewidth=8, alpha=0.8)

    # 4. BAIT #2: The "3 Green Dots" (The 'Smart' Pattern)
    # We verify these exist in all images to trick the detail-oriented people
    # We place them in a consistent row, but the row moves slightly between images
    if has_green_dots:
        gx = np.random.randint(20, 80)
        gy = np.random.randint(35, 45) # Keep them high up
        for i in range(3):
            ax.scatter(gx + (i*5), gy, s=300, c="#9AE89A", edgecolors='white', linewidth=2, zorder=15)

    # 5. THE TARGET: The "Toucan" (Dark Blue Blob)
    # Only exists if has_toucan is True
    if has_toucan:
        tx = np.random.randint(10, 90)
        ty = np.random.randint(5, 20) # Keep it low to separate from green dots
        # Deep Blue color
        ax.scatter(tx, ty, s=250, c="#9797E5", edgecolors='white', linewidth=1.5, zorder=10, label="The Toucan")

    # 6. Clean up
    ax.axis('off')
    plt.show()

# --- GENERATE THE SLIDES ---

print("IMAGE A: Training Sample 1 (Toucan + Line + Green Dots)")
generate_double_trap(has_toucan=True, seed=123)

print("IMAGE B: Training Sample 2 (Toucan + Line + Green Dots)")
generate_double_trap(has_toucan=True, seed=221)

print("IMAGE C: THE TRAP (Line + Green Dots... NO TOUCAN)")
generate_double_trap(has_toucan=False, seed=53892)

print("IMAGE D: No Green Dots (Toucan + Line only)")
generate_double_trap(has_toucan=True, has_green_dots=False, seed=999)


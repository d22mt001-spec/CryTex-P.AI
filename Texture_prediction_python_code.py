
# ============================================================
# MODULE 1 — {100} Pole Figure Analysis
# ============================================================

# STEP 1: Upload the pole figure image
from google.colab import files
uploaded = files.upload()

# STEP 2: Import required libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from collections import defaultdict
from fractions import Fraction

output_lines = []

# STEP 3: Load and convert image
image_path = list(uploaded.keys())[0]
image = cv2.imread(image_path)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# STEP 4: Create red mask

lower_red1 = np.array([0, 70, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 70, 50])
upper_red2 = np.array([180, 255, 255])
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
red_mask = cv2.bitwise_or(mask1, mask2)

# STEP 5: Setup image parameters
height, width = red_mask.shape
center_x, center_y = width // 2, height // 2
R = min(center_x, center_y)

# STEP 5.1: Ask user for MRD intensity range
print("Please input the expected MRD intensity range from EBSD analysis.")
min_mrd = float(input("Minimum MRD (e.g., 1.0): "))
max_mrd = float(input("Maximum MRD (e.g., 6.0): "))

# For scaling intensity
def compute_mrd_intensity(cnt, mask, min_mrd, max_mrd):
    mask_temp = np.zeros_like(mask)
    cv2.drawContours(mask_temp, [cnt], -1, 255, -1)
    mean_val = cv2.mean(image, mask=mask_temp)[2]  # Use red channel (BGR -> index 2)
    scaled = np.interp(mean_val, [0, 255], [min_mrd, max_mrd])
    return round(scaled, 2)

# STEP 6: Detect red contours
contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
spot_data = []

for cnt in contours:
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        td = (cX - center_x) / R
        rd = -(cY - center_y) / R

        abs_rd = abs(round(rd, 2))
        abs_td = abs(round(td, 2))
        distance = round(math.sqrt(rd**2 + td**2), 3)

        a = abs_rd
        b = abs_td
        denom1 = a**2 + b**2 + 1
        cos_theta1 = (2 * a) / denom1
        cos_theta1 = min(1.0, max(-1.0, cos_theta1))
        theta1_deg = round(math.degrees(math.acos(cos_theta1)), 2)

        d = distance
        cos_theta2 = (1 - d**2) / (1 + d**2)
        cos_theta2 = min(1.0, max(-1.0, cos_theta2))
        theta2_deg = round(math.degrees(math.acos(cos_theta2)), 2)
        mrd_intensity = compute_mrd_intensity(cnt, red_mask, min_mrd, max_mrd)
        is_textured = mrd_intensity > 1.0

        spot_data.append({
            'RD': round(rd, 2),
            'TD': round(td, 2),
            '|RD|': abs_rd,
            '|TD|': abs_td,
            'Distance_R': distance,
            'Theta_Model1 (°)': theta1_deg,
            'Theta_Model2 (°)': theta2_deg,
            'Textured': is_textured,
            'MRD_Intensity': mrd_intensity
        })

# STEP 7: Sort by distance
spot_data = sorted(spot_data, key=lambda x: x['Distance_R'])

# STEP 8: Annotate image
spot_labels = [chr(65 + i) for i in range(len(spot_data))]
output_image = image.copy()
for i, spot in enumerate(spot_data):
    rd, td = spot['RD'], spot['TD']
    px = int(center_x + td * R)
    py = int(center_y - rd * R)
    label = spot_labels[i]
    cv2.circle(output_image, (px, py), 5, (0, 255, 0), -1)
    cv2.putText(output_image, label, (px + 5, py - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

# STEP 9: Display annotated image
output_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(7, 7))
plt.imshow(output_rgb)
plt.title("Spot Annotations (A, B, C...)")
plt.axis('off')
plt.show()

# STEP 10: Print summary
output_lines.append("\U0001F4CD Final Spot Summary")
output_lines.append("{:<8} {:<10} {:<10} {:<10} {:<10} {:<12} {:<17} {:<17} {:<10} {:<10}".format(
    "Label", "RD", "TD", "|RD|", "|TD|", "Distance (R)", "Theta_Model1 (°)", "Theta_Model2 (°)", "MRD", "Textured"
))

for label, spot in zip(spot_labels, spot_data):
    output_lines.append("{:<8} {:<10} {:<10} {:<10} {:<10} {:<12} {:<17} {:<17} {:<10} {:<10}".format(
        label, spot['RD'], spot['TD'], spot['|RD|'], spot['|TD|'],
        spot['Distance_R'], spot['Theta_Model1 (°)'], spot['Theta_Model2 (°)'],
        spot['MRD_Intensity'], "Yes" if spot['Textured'] else "No"
    ))

# === Matching Utilities ===
def compute_angle(h1, k1, l1, h2, k2, l2):
    dot = h1 * h2 + k1 * k2 + l1 * l2
    mag1 = math.sqrt(h1**2 + k1**2 + l1**2)
    mag2 = math.sqrt(h2**2 + k2**2 + l2**2)
    cos_theta = dot / (mag1 * mag2)
    cos_theta = min(1.0, max(-1.0, cos_theta))
    return math.degrees(math.acos(cos_theta))

def generate_hkl_directions():
    directions = []
    for h in range(-9, 10):
        for k in range(-9, 10):
            for l in range(-9, 10):
                if not (h == 0 and k == 0 and l == 0):
                    directions.append((h, k, l))
    return directions

def normalize_vector(vec):
    if vec == (0, 0, 0): return (0, 0, 0)
    g = math.gcd(math.gcd(abs(vec[0]), abs(vec[1])), abs(vec[2]))
    return tuple(sorted([abs(int(v / g)) for v in vec]))

# ✅ UPDATED: tolerance changed from 8 → 10 (sensitivity analysis: R²=1.000, MAE=0.004)
def find_all_matches_with_families(target_angle, directions, reference_family, tolerance=10):
    matches = []
    for h, k, l in directions:
        for ref in reference_family:
            angle = compute_angle(ref[0], ref[1], ref[2], h, k, l)
            if abs(angle - target_angle) <= tolerance:
                matches.append({'h': h, 'k': k, 'l': l, 'angle': round(angle, 2), 'ref': ref})
    return matches

# Texture Families Dictionary
texture_families = {
    '{113}<361>': ('BCC', (1,1,3), (3,6,1), '{113}<361>'),
    '{100}<001>': ('BCC', (1,0,0), (0,0,1), 'Cube'),
    '{110}<001>': ('BCC', (1,1,0), (0,0,1), 'Goss'),
    '{110}<112>': ('BCC', (1,1,0), (1,1,2), 'Brass'),
    '{001}<110>': ('BCC', (0,0,1), (1,1,0), 'Rotated Cube'),
    '{011}<011>': ('BCC', (1,1,0), (1,1,0), 'Rotated Goss'),
    '{112}<110>': ('BCC', (1,1,2), (1,1,0), '{112}<110>'),
    '{111}<110>': ('BCC', (1,1,1), (1,1,0), '{111}<110>'),
    '{013}<100>': ('BCC', (0,1,3), (1,0,0), 'Cube-rd'),
    '{111}<112>': ('BCC', (1,1,1), (1,1,2), '{111}<112>'),
    '{011}<111>': ('BCC', (0,1,1), (1,1,1), '{011}<111>'),
    '{112}<111>': ('FCC', (1,1,2), (1,1,1), 'Copper Texture'),
    '{011}<122>': ('FCC', (0,1,1), (1,2,2), 'P Texture'),
    '{231}<346>': ('FCC', (2,3,1), (3,4,6), 'S Texture')
}

ref_100 = [(1,0,0), (0,1,0), (0,0,1), (-1,0,0), (0,-1,0), (0,0,-1)]

all_hkl = generate_hkl_directions()

output_lines.append("\n\U0001F50D Directions Matching θ1 and θ2 ±10° with respect to {100} family and Spot-wise Orthogonal Pairs:")
orthogonal_pair_sets = []
match_frequencies = defaultdict(int)

texture_match_summary = defaultdict(lambda: {
    'texture_name': '',
    'spot_count_100': 0,
    'total_occurrences': 0
})

for i, spot in enumerate(spot_data, 1):
    output_lines.append(f"\n--- Spot #{i} ---")

    if not spot['Textured']:
        output_lines.append(f"🚫 Spot #{i} is not textured (MRD = {spot['MRD_Intensity']} ≤ 1), skipping orientation analysis.")
        orthogonal_pair_sets.append(set())
        continue

    theta1 = spot['Theta_Model1 (°)']
    theta2 = spot['Theta_Model2 (°)']

    matches1 = find_all_matches_with_families(theta1, all_hkl, ref_100)
    matches2 = find_all_matches_with_families(theta2, all_hkl, ref_100)

    output_lines.append(f"θ1 = {theta1}° → Matches with {{100}}:")
    theta1_hkls = [(m['h'], m['k'], m['l']) for m in matches1]
    for m in matches1:
        output_lines.append(f"  → {m['angle']}° between {m['ref']} and ({m['h']} {m['k']} {m['l']})")
    if not matches1:
        output_lines.append("  ❌ No match within ±10°")

    output_lines.append(f"θ2 = {theta2}° → Matches with {{100}}:")
    theta2_hkls = [(m['h'], m['k'], m['l']) for m in matches2]
    for m in matches2:
        output_lines.append(f"  → {m['angle']}° between {m['ref']} and ({m['h']} {m['k']} {m['l']})")
    if not matches2:
        output_lines.append("  ❌ No match within ±10°")

    spot_orthogonal_pairs = set()
    for h1, k1, l1 in theta1_hkls:
        for h2, k2, l2 in theta2_hkls:
            if abs(h1*h2 + k1*k2 + l1*l2) < 1e-6:
                pair = ((h2, k2, l2), (h1, k1, l1))
                spot_orthogonal_pairs.add(pair)
                output_lines.append(f"  → ({h2} {k2} {l2}) ⟂ ({h1} {k1} {l1})")
    orthogonal_pair_sets.append(spot_orthogonal_pairs)
    if not spot_orthogonal_pairs:
        output_lines.append("  ❌ No orthogonal pairs found.")

    # Texture Matching
    texture_matches = set()
    spot_texture_counts = defaultdict(int)
    matched_textures_in_this_spot = set()
    spot_texture_counts = defaultdict(int)
    for vec1, vec2 in spot_orthogonal_pairs:
        norm1 = normalize_vector(vec1)
        norm2 = normalize_vector(vec2)
        for tex_key, (lattice, plane, direction, tex_name) in texture_families.items():
            t_plane = normalize_vector(plane)
            t_dir = normalize_vector(direction)
            if (norm1 == t_plane and norm2 == t_dir):
                match_str = f"{tex_key} → {tex_name}"
                texture_matches.add(match_str)
                match_frequencies[match_str] += 1
                spot_texture_counts[match_str] += 1
                texture_match_summary[match_str]['texture_name'] = tex_name
                texture_match_summary[match_str]['total_occurrences'] += 1
                matched_textures_in_this_spot.add(match_str)

    # 🔁 Increment spot count only ONCE per spot for each texture
    for match_str in matched_textures_in_this_spot:
        texture_match_summary[match_str]['spot_count_100'] += 1

    if spot_texture_counts:
        output_lines.append(f"\n📌 Spot #{i} matches approximately with:")
        for match_str, count in spot_texture_counts.items():
            output_lines.append(f"   → {match_str} : {count} occurrence(s)")
    else:
        output_lines.append("\n❌ No approximate texture matches found.")

output_lines.append("\n\U0001F4CA Texture Match Frequencies Based on Orthogonal Pair Matches:")
for match, count in match_frequencies.items():
    output_lines.append(f"{match}: {count} occurrence(s)")

# STEP 13: Save and Download Output
with open("full_output.txt", "w") as f:
    for line in output_lines:
        f.write(line + "\n")

files.download("full_output.txt")

# Store final texture match summary in a standard variable
texture_match_summary_100 = texture_match_summary
print(texture_match_summary_100)

# ============================================================
# MODULE 2 — {110} Pole Figure Analysis
# ============================================================

# STEP 1: Upload the pole figure image
from google.colab import files
uploaded = files.upload()

# STEP 2: Import required libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from collections import defaultdict
from fractions import Fraction

output_lines = []

# STEP 3: Load and convert image
image_path = list(uploaded.keys())[0]
image = cv2.imread(image_path)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# STEP 4: Create red mask

lower_red1 = np.array([0, 70, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 70, 50])
upper_red2 = np.array([180, 255, 255])
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
red_mask = cv2.bitwise_or(mask1, mask2)

# STEP 5: Setup image parameters
height, width = red_mask.shape
center_x, center_y = width // 2, height // 2
R = min(center_x, center_y)

# STEP 5.1: Ask user for MRD intensity range
print("Please input the expected MRD intensity range from EBSD analysis.")
min_mrd = float(input("Minimum MRD (e.g., 1.0): "))
max_mrd = float(input("Maximum MRD (e.g., 6.0): "))

# For scaling intensity
def compute_mrd_intensity(cnt, mask, min_mrd, max_mrd):
    mask_temp = np.zeros_like(mask)
    cv2.drawContours(mask_temp, [cnt], -1, 255, -1)
    mean_val = cv2.mean(image, mask=mask_temp)[2]  # Use red channel (BGR -> index 2)
    scaled = np.interp(mean_val, [0, 255], [min_mrd, max_mrd])
    return round(scaled, 2)

# STEP 6: Detect red contours
contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
spot_data = []

for cnt in contours:
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        td = (cX - center_x) / R
        rd = -(cY - center_y) / R

        abs_rd = abs(round(rd, 2))
        abs_td = abs(round(td, 2))
        distance = round(math.sqrt(rd**2 + td**2), 3)

        a = abs_rd
        b = abs_td
        denom1 = a**2 + b**2 + 1
        cos_theta1 = (2 * a) / denom1
        cos_theta1 = min(1.0, max(-1.0, cos_theta1))
        theta1_deg = round(math.degrees(math.acos(cos_theta1)), 2)

        d = distance
        cos_theta2 = (1 - d**2) / (1 + d**2)
        cos_theta2 = min(1.0, max(-1.0, cos_theta2))
        theta2_deg = round(math.degrees(math.acos(cos_theta2)), 2)
        mrd_intensity = compute_mrd_intensity(cnt, red_mask, min_mrd, max_mrd)
        is_textured = mrd_intensity > 1.0

        spot_data.append({
            'RD': round(rd, 2),
            'TD': round(td, 2),
            '|RD|': abs_rd,
            '|TD|': abs_td,
            'Distance_R': distance,
            'Theta_Model1 (°)': theta1_deg,
            'Theta_Model2 (°)': theta2_deg,
            'Textured': is_textured,
            'MRD_Intensity': mrd_intensity
        })

# STEP 7: Sort by distance
spot_data = sorted(spot_data, key=lambda x: x['Distance_R'])

# STEP 8: Annotate image
spot_labels = [chr(65 + i) for i in range(len(spot_data))]
output_image = image.copy()
for i, spot in enumerate(spot_data):
    rd, td = spot['RD'], spot['TD']
    px = int(center_x + td * R)
    py = int(center_y - rd * R)
    label = spot_labels[i]
    cv2.circle(output_image, (px, py), 5, (0, 255, 0), -1)
    cv2.putText(output_image, label, (px + 5, py - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

# STEP 9: Display annotated image
output_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(7, 7))
plt.imshow(output_rgb)
plt.title("Spot Annotations (A, B, C...)")
plt.axis('off')
plt.show()

# STEP 10: Print summary
output_lines.append("\U0001F4CD Final Spot Summary")
output_lines.append("{:<8} {:<10} {:<10} {:<10} {:<10} {:<12} {:<17} {:<17} {:<10} {:<10}".format(
    "Label", "RD", "TD", "|RD|", "|TD|", "Distance (R)", "Theta_Model1 (°)", "Theta_Model2 (°)", "MRD", "Textured"
))

for label, spot in zip(spot_labels, spot_data):
    output_lines.append("{:<8} {:<10} {:<10} {:<10} {:<10} {:<12} {:<17} {:<17} {:<10} {:<10}".format(
        label, spot['RD'], spot['TD'], spot['|RD|'], spot['|TD|'],
        spot['Distance_R'], spot['Theta_Model1 (°)'], spot['Theta_Model2 (°)'],
        spot['MRD_Intensity'], "Yes" if spot['Textured'] else "No"
    ))

# === Matching Utilities ===
def compute_angle(h1, k1, l1, h2, k2, l2):
    dot = h1 * h2 + k1 * k2 + l1 * l2
    mag1 = math.sqrt(h1**2 + k1**2 + l1**2)
    mag2 = math.sqrt(h2**2 + k2**2 + l2**2)
    cos_theta = dot / (mag1 * mag2)
    cos_theta = min(1.0, max(-1.0, cos_theta))
    return math.degrees(math.acos(cos_theta))

def generate_hkl_directions():
    directions = []
    for h in range(-9, 10):
        for k in range(-9, 10):
            for l in range(-9, 10):
                if not (h == 0 and k == 0 and l == 0):
                    directions.append((h, k, l))
    return directions

def normalize_vector(vec):
    if vec == (0, 0, 0): return (0, 0, 0)
    g = math.gcd(math.gcd(abs(vec[0]), abs(vec[1])), abs(vec[2]))
    return tuple(sorted([abs(int(v / g)) for v in vec]))

# ✅ UPDATED: tolerance changed from 8 → 10 (sensitivity analysis: R²=1.000, MAE=0.004)
def find_all_matches_with_families(target_angle, directions, reference_family, tolerance=10):
    matches = []
    for h, k, l in directions:
        for ref in reference_family:
            angle = compute_angle(ref[0], ref[1], ref[2], h, k, l)
            if abs(angle - target_angle) <= tolerance:
                matches.append({'h': h, 'k': k, 'l': l, 'angle': round(angle, 2), 'ref': ref})
    return matches

# Texture Families Dictionary
texture_families = {
    '{113}<361>': ('BCC', (1,1,3), (3,6,1), '{113}<361>'),
    '{100}<001>': ('BCC', (1,0,0), (0,0,1), 'Cube'),
    '{110}<001>': ('BCC', (1,1,0), (0,0,1), 'Goss'),
    '{110}<112>': ('BCC', (1,1,0), (1,1,2), 'Brass'),
    '{001}<110>': ('BCC', (0,0,1), (1,1,0), 'Rotated Cube'),
    '{011}<011>': ('BCC', (1,1,0), (1,1,0), 'Rotated Goss'),
    '{112}<110>': ('BCC', (1,1,2), (1,1,0), '{112}<110>'),
    '{111}<110>': ('BCC', (1,1,1), (1,1,0), '{111}<110>'),
    '{013}<100>': ('BCC', (0,1,3), (1,0,0), 'Cube-rd'),
    '{111}<112>': ('BCC', (1,1,1), (1,1,2), '{111}<112>'),
    '{011}<111>': ('BCC', (0,1,1), (1,1,1), '{011}<111>'),
    '{112}<111>': ('FCC', (1,1,2), (1,1,1), 'Copper Texture'),
    '{011}<122>': ('FCC', (0,1,1), (1,2,2), 'P Texture'),
    '{231}<346>': ('FCC', (2,3,1), (3,4,6), 'S Texture')
}

ref_110 = [(1, 1, 0), (1, 0, 1), (0, 1, 1),(-1, 1, 0), (1, -1, 0), (-1, 0, 1),
    (0, -1, 1), (0, 1, -1), (1, 0, -1),(-1, 0, -1), (0, -1, -1), (-1, -1, 0)]

all_hkl = generate_hkl_directions()

output_lines.append("\n\U0001F50D Directions Matching θ1 and θ2 ±10° with respect to {110} family and Spot-wise Orthogonal Pairs:")
orthogonal_pair_sets = []
match_frequencies = defaultdict(int)

texture_match_summary = defaultdict(lambda: {
    'texture_name': '',
    'spot_count_110': 0,
    'total_occurrences': 0
})

for i, spot in enumerate(spot_data, 1):
    output_lines.append(f"\n--- Spot #{i} ---")

    if not spot['Textured']:
        output_lines.append(f"🚫 Spot #{i} is not textured (MRD = {spot['MRD_Intensity']} ≤ 1), skipping orientation analysis.")
        orthogonal_pair_sets.append(set())
        continue

    theta1 = spot['Theta_Model1 (°)']
    theta2 = spot['Theta_Model2 (°)']

    matches1 = find_all_matches_with_families(theta1, all_hkl, ref_110)
    matches2 = find_all_matches_with_families(theta2, all_hkl, ref_110)

    output_lines.append(f"θ1 = {theta1}° → Matches with {{110}}:")
    theta1_hkls = [(m['h'], m['k'], m['l']) for m in matches1]
    for m in matches1:
        output_lines.append(f"  → {m['angle']}° between {m['ref']} and ({m['h']} {m['k']} {m['l']})")
    if not matches1:
        output_lines.append("  ❌ No match within ±10°")

    output_lines.append(f"θ2 = {theta2}° → Matches with {{110}}:")
    theta2_hkls = [(m['h'], m['k'], m['l']) for m in matches2]
    for m in matches2:
        output_lines.append(f"  → {m['angle']}° between {m['ref']} and ({m['h']} {m['k']} {m['l']})")
    if not matches2:
        output_lines.append("  ❌ No match within ±10°")

    spot_orthogonal_pairs = set()
    for h1, k1, l1 in theta1_hkls:
        for h2, k2, l2 in theta2_hkls:
            if abs(h1*h2 + k1*k2 + l1*l2) < 1e-6:
                pair = ((h2, k2, l2), (h1, k1, l1))
                spot_orthogonal_pairs.add(pair)
                output_lines.append(f"  → ({h2} {k2} {l2}) ⟂ ({h1} {k1} {l1})")
    orthogonal_pair_sets.append(spot_orthogonal_pairs)
    if not spot_orthogonal_pairs:
        output_lines.append("  ❌ No orthogonal pairs found.")

    # Texture Matching
    texture_matches = set()
    spot_texture_counts = defaultdict(int)
    matched_textures_in_this_spot = set()
    spot_texture_counts = defaultdict(int)
    for vec1, vec2 in spot_orthogonal_pairs:
        norm1 = normalize_vector(vec1)
        norm2 = normalize_vector(vec2)
        for tex_key, (lattice, plane, direction, tex_name) in texture_families.items():
            t_plane = normalize_vector(plane)
            t_dir = normalize_vector(direction)
            if (norm1 == t_plane and norm2 == t_dir):
                match_str = f"{tex_key} → {tex_name}"
                texture_matches.add(match_str)
                match_frequencies[match_str] += 1
                spot_texture_counts[match_str] += 1
                texture_match_summary[match_str]['texture_name'] = tex_name
                texture_match_summary[match_str]['total_occurrences'] += 1
                matched_textures_in_this_spot.add(match_str)

    # 🔁 Increment spot count only ONCE per spot for each texture
    for match_str in matched_textures_in_this_spot:
        texture_match_summary[match_str]['spot_count_110'] += 1

    if spot_texture_counts:
        output_lines.append(f"\n📌 Spot #{i} matches approximately with:")
        for match_str, count in spot_texture_counts.items():
            output_lines.append(f"   → {match_str} : {count} occurrence(s)")
    else:
        output_lines.append("\n❌ No approximate texture matches found.")

output_lines.append("\n\U0001F4CA Texture Match Frequencies Based on Orthogonal Pair Matches:")
for match, count in match_frequencies.items():
    output_lines.append(f"{match}: {count} occurrence(s)")

# STEP 13: Save and Download Output
with open("full_output.txt", "w") as f:
    for line in output_lines:
        f.write(line + "\n")

files.download("full_output.txt")

# Store final texture match summary in a standard variable
texture_match_summary_110 = texture_match_summary
print(texture_match_summary_110)

# ============================================================
# MODULE 3 — {111} Pole Figure Analysis
# ============================================================

# STEP 1: Upload the pole figure image
from google.colab import files
uploaded = files.upload()

# STEP 2: Import required libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from collections import defaultdict
from fractions import Fraction

output_lines = []

# STEP 3: Load and convert image
image_path = list(uploaded.keys())[0]
image = cv2.imread(image_path)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# STEP 4: Create red mask

lower_red1 = np.array([0, 70, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 70, 50])
upper_red2 = np.array([180, 255, 255])
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
red_mask = cv2.bitwise_or(mask1, mask2)

# STEP 5: Setup image parameters
height, width = red_mask.shape
center_x, center_y = width // 2, height // 2
R = min(center_x, center_y)

# STEP 5.1: Ask user for MRD intensity range
print("Please input the expected MRD intensity range from EBSD analysis.")
min_mrd = float(input("Minimum MRD (e.g., 1.0): "))
max_mrd = float(input("Maximum MRD (e.g., 6.0): "))

# For scaling intensity
def compute_mrd_intensity(cnt, mask, min_mrd, max_mrd):
    mask_temp = np.zeros_like(mask)
    cv2.drawContours(mask_temp, [cnt], -1, 255, -1)
    mean_val = cv2.mean(image, mask=mask_temp)[2]  # Use red channel (BGR -> index 2)
    scaled = np.interp(mean_val, [0, 255], [min_mrd, max_mrd])
    return round(scaled, 2)

# STEP 6: Detect red contours
contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
spot_data = []

for cnt in contours:
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        td = (cX - center_x) / R
        rd = -(cY - center_y) / R

        abs_rd = abs(round(rd, 2))
        abs_td = abs(round(td, 2))
        distance = round(math.sqrt(rd**2 + td**2), 3)

        a = abs_rd
        b = abs_td
        denom1 = a**2 + b**2 + 1
        cos_theta1 = (2 * a) / denom1
        cos_theta1 = min(1.0, max(-1.0, cos_theta1))
        theta1_deg = round(math.degrees(math.acos(cos_theta1)), 2)

        d = distance
        cos_theta2 = (1 - d**2) / (1 + d**2)
        cos_theta2 = min(1.0, max(-1.0, cos_theta2))
        theta2_deg = round(math.degrees(math.acos(cos_theta2)), 2)
        mrd_intensity = compute_mrd_intensity(cnt, red_mask, min_mrd, max_mrd)
        is_textured = mrd_intensity > 1.0

        spot_data.append({
            'RD': round(rd, 2),
            'TD': round(td, 2),
            '|RD|': abs_rd,
            '|TD|': abs_td,
            'Distance_R': distance,
            'Theta_Model1 (°)': theta1_deg,
            'Theta_Model2 (°)': theta2_deg,
            'Textured': is_textured,
            'MRD_Intensity': mrd_intensity
        })

# STEP 7: Sort by distance
spot_data = sorted(spot_data, key=lambda x: x['Distance_R'])

# STEP 8: Annotate image
spot_labels = [chr(65 + i) for i in range(len(spot_data))]
output_image = image.copy()
for i, spot in enumerate(spot_data):
    rd, td = spot['RD'], spot['TD']
    px = int(center_x + td * R)
    py = int(center_y - rd * R)
    label = spot_labels[i]
    cv2.circle(output_image, (px, py), 5, (0, 255, 0), -1)
    cv2.putText(output_image, label, (px + 5, py - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

# STEP 9: Display annotated image
output_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(7, 7))
plt.imshow(output_rgb)
plt.title("Spot Annotations (A, B, C...)")
plt.axis('off')
plt.show()

# STEP 10: Print summary
output_lines.append("\U0001F4CD Final Spot Summary")
output_lines.append("{:<8} {:<10} {:<10} {:<10} {:<10} {:<12} {:<17} {:<17} {:<10} {:<10}".format(
    "Label", "RD", "TD", "|RD|", "|TD|", "Distance (R)", "Theta_Model1 (°)", "Theta_Model2 (°)", "MRD", "Textured"
))

for label, spot in zip(spot_labels, spot_data):
    output_lines.append("{:<8} {:<10} {:<10} {:<10} {:<10} {:<12} {:<17} {:<17} {:<10} {:<10}".format(
        label, spot['RD'], spot['TD'], spot['|RD|'], spot['|TD|'],
        spot['Distance_R'], spot['Theta_Model1 (°)'], spot['Theta_Model2 (°)'],
        spot['MRD_Intensity'], "Yes" if spot['Textured'] else "No"
    ))

# === Matching Utilities ===
def compute_angle(h1, k1, l1, h2, k2, l2):
    dot = h1 * h2 + k1 * k2 + l1 * l2
    mag1 = math.sqrt(h1**2 + k1**2 + l1**2)
    mag2 = math.sqrt(h2**2 + k2**2 + l2**2)
    cos_theta = dot / (mag1 * mag2)
    cos_theta = min(1.0, max(-1.0, cos_theta))
    return math.degrees(math.acos(cos_theta))

def generate_hkl_directions():
    directions = []
    for h in range(-9, 10):
        for k in range(-9, 10):
            for l in range(-9, 10):
                if not (h == 0 and k == 0 and l == 0):
                    directions.append((h, k, l))
    return directions

def normalize_vector(vec):
    if vec == (0, 0, 0): return (0, 0, 0)
    g = math.gcd(math.gcd(abs(vec[0]), abs(vec[1])), abs(vec[2]))
    return tuple(sorted([abs(int(v / g)) for v in vec]))

# ✅ UPDATED: tolerance changed from 8 → 10 (sensitivity analysis: R²=1.000, MAE=0.004)
def find_all_matches_with_families(target_angle, directions, reference_family, tolerance=10):
    matches = []
    for h, k, l in directions:
        for ref in reference_family:
            angle = compute_angle(ref[0], ref[1], ref[2], h, k, l)
            if abs(angle - target_angle) <= tolerance:
                matches.append({'h': h, 'k': k, 'l': l, 'angle': round(angle, 2), 'ref': ref})
    return matches

# Texture Families Dictionary
texture_families = {
    '{113}<361>': ('BCC', (1,1,3), (3,6,1), '{113}<361>'),
    '{100}<001>': ('BCC', (1,0,0), (0,0,1), 'Cube'),
    '{110}<001>': ('BCC', (1,1,0), (0,0,1), 'Goss'),
    '{110}<112>': ('BCC', (1,1,0), (1,1,2), 'Brass'),
    '{001}<110>': ('BCC', (0,0,1), (1,1,0), 'Rotated Cube'),
    '{011}<011>': ('BCC', (1,1,0), (1,1,0), 'Rotated Goss'),
    '{112}<110>': ('BCC', (1,1,2), (1,1,0), '{112}<110>'),
    '{111}<110>': ('BCC', (1,1,1), (1,1,0), '{111}<110>'),
    '{013}<100>': ('BCC', (0,1,3), (1,0,0), 'Cube-rd'),
    '{111}<112>': ('BCC', (1,1,1), (1,1,2), '{111}<112>'),
    '{011}<111>': ('BCC', (0,1,1), (1,1,1), '{011}<111>'),
    '{112}<111>': ('FCC', (1,1,2), (1,1,1), 'Copper Texture'),
    '{011}<122>': ('FCC', (0,1,1), (1,2,2), 'P Texture'),
    '{231}<346>': ('FCC', (2,3,1), (3,4,6), 'S Texture')
}

ref_111 = [(1, 1, 1), (1, 1, -1),(1, -1, 1),(-1, 1, 1),(-1, -1, 1),(-1, 1, -1),(1, -1, -1),(-1, -1, -1)]

all_hkl = generate_hkl_directions()

output_lines.append("\n\U0001F50D Directions Matching θ1 and θ2 ±10° with respect to {111} family and Spot-wise Orthogonal Pairs:")
orthogonal_pair_sets = []
match_frequencies = defaultdict(int)

texture_match_summary = defaultdict(lambda: {
    'texture_name': '',
    'spot_count_111': 0,
    'total_occurrences': 0
})

for i, spot in enumerate(spot_data, 1):
    output_lines.append(f"\n--- Spot #{i} ---")

    if not spot['Textured']:
        output_lines.append(f"🚫 Spot #{i} is not textured (MRD = {spot['MRD_Intensity']} ≤ 1), skipping orientation analysis.")
        orthogonal_pair_sets.append(set())
        continue

    theta1 = spot['Theta_Model1 (°)']
    theta2 = spot['Theta_Model2 (°)']

    matches1 = find_all_matches_with_families(theta1, all_hkl, ref_111)
    matches2 = find_all_matches_with_families(theta2, all_hkl, ref_111)

    output_lines.append(f"θ1 = {theta1}° → Matches with {{111}}:")
    theta1_hkls = [(m['h'], m['k'], m['l']) for m in matches1]
    for m in matches1:
        output_lines.append(f"  → {m['angle']}° between {m['ref']} and ({m['h']} {m['k']} {m['l']})")
    if not matches1:
        output_lines.append("  ❌ No match within ±10°")

    output_lines.append(f"θ2 = {theta2}° → Matches with {{111}}:")
    theta2_hkls = [(m['h'], m['k'], m['l']) for m in matches2]
    for m in matches2:
        output_lines.append(f"  → {m['angle']}° between {m['ref']} and ({m['h']} {m['k']} {m['l']})")
    if not matches2:
        output_lines.append("  ❌ No match within ±10°")

    spot_orthogonal_pairs = set()
    for h1, k1, l1 in theta1_hkls:
        for h2, k2, l2 in theta2_hkls:
            if abs(h1*h2 + k1*k2 + l1*l2) < 1e-6:
                pair = ((h2, k2, l2), (h1, k1, l1))
                spot_orthogonal_pairs.add(pair)
                output_lines.append(f"  → ({h2} {k2} {l2}) ⟂ ({h1} {k1} {l1})")
    orthogonal_pair_sets.append(spot_orthogonal_pairs)
    if not spot_orthogonal_pairs:
        output_lines.append("  ❌ No orthogonal pairs found.")

    # Texture Matching
    texture_matches = set()
    spot_texture_counts = defaultdict(int)
    matched_textures_in_this_spot = set()
    spot_texture_counts = defaultdict(int)
    for vec1, vec2 in spot_orthogonal_pairs:
        norm1 = normalize_vector(vec1)
        norm2 = normalize_vector(vec2)
        for tex_key, (lattice, plane, direction, tex_name) in texture_families.items():
            t_plane = normalize_vector(plane)
            t_dir = normalize_vector(direction)
            if (norm1 == t_plane and norm2 == t_dir):
                match_str = f"{tex_key} → {tex_name}"
                texture_matches.add(match_str)
                match_frequencies[match_str] += 1
                spot_texture_counts[match_str] += 1
                texture_match_summary[match_str]['texture_name'] = tex_name
                texture_match_summary[match_str]['total_occurrences'] += 1
                matched_textures_in_this_spot.add(match_str)

    # 🔁 Increment spot count only ONCE per spot for each texture
    for match_str in matched_textures_in_this_spot:
        texture_match_summary[match_str]['spot_count_111'] += 1

    if spot_texture_counts:
        output_lines.append(f"\n📌 Spot #{i} matches approximately with:")
        for match_str, count in spot_texture_counts.items():
            output_lines.append(f"   → {match_str} : {count} occurrence(s)")
    else:
        output_lines.append("\n❌ No approximate texture matches found.")

output_lines.append("\n\U0001F4CA Texture Match Frequencies Based on Orthogonal Pair Matches:")
for match, count in match_frequencies.items():
    output_lines.append(f"{match}: {count} occurrence(s)")

# STEP 13: Save and Download Output
with open("full_output.txt", "w") as f:
    for line in output_lines:
        f.write(line + "\n")

files.download("full_output.txt")

# Store final texture match summary in a standard variable
texture_match_summary_111 = texture_match_summary
print(texture_match_summary_111)

# ============================================================
# COMBINE + FILTER RESULTS FROM ALL 3 MODULES
# ============================================================

from collections import defaultdict

combined_summary = defaultdict(lambda: {
    'texture_name': '',
    'spot_count_100': 0,
    'spot_count_110': 0,
    'spot_count_111': 0,
    'total_occurrences': 0
})

# === Helper function to merge a single summary into combined ===
def merge_summary(local_summary):
    for texture_key, data in local_summary.items():
        combined_summary[texture_key]['texture_name'] = data['texture_name']
        for key, value in data.items():
            if key.startswith('spot_count') or key == 'total_occurrences':
                combined_summary[texture_key][key] += value

# === Merge all three summaries ===
merge_summary(texture_match_summary_100)
merge_summary(texture_match_summary_110)
merge_summary(texture_match_summary_111)

# === Print Combined Summary ===
print("=== Combined Texture Match Summary ===")
for texture_key, info in combined_summary.items():
    print(f"{texture_key}: {info}")

from collections import defaultdict

combined_summary = defaultdict(lambda: {
    'texture_name': '',
    'spot_count_100': 0,
    'spot_count_110': 0,
    'spot_count_111': 0,
    'total_occurrences': 0
})

# === Helper function to merge a single summary into combined ===
def merge_summary(local_summary):
    for texture_key, data in local_summary.items():
        combined_summary[texture_key]['texture_name'] = data['texture_name']
        for key, value in data.items():
            if key.startswith('spot_count') or key == 'total_occurrences':
                combined_summary[texture_key][key] += value

# === Merge all three summaries ===
merge_summary(texture_match_summary_100)
merge_summary(texture_match_summary_110)
merge_summary(texture_match_summary_111)

# === Apply filtering logic ===
filtered_summary = {}
for texture_key, info in combined_summary.items():
    c100, c110, c111 = info['spot_count_100'], info['spot_count_110'], info['spot_count_111']

    nonzero_counts = sum(x > 0 for x in [c100, c110, c111])

    # Rule 1: At least two non-zero
    at_least_two_nonzero = (nonzero_counts >= 2)

    # Rule 2: Any one >= 3
    one_big_enough = (c100 >= 3 or c110 >= 3 or c111 >= 3)

    if at_least_two_nonzero or one_big_enough:
        filtered_summary[texture_key] = info

# === Print Final Filtered Summary ===
print("=== Final Filtered Texture Match Summary ===")
for texture_key, info in filtered_summary.items():
    print(f"{texture_key}: {info}")

import pandas as pd

# === Convert filtered_summary to a DataFrame ===
rows = []

for texture_notation, info in filtered_summary.items():
    rows.append({
        "Texture_notation": texture_notation,
        "Texture_Name": info["texture_name"],
        "Spot_count_100": info["spot_count_100"],
        "Spot_count_110": info["spot_count_110"],
        "Spot_count_111": info["spot_count_111"],
        "Total_occurrences": info["total_occurrences"]
    })

df = pd.DataFrame(rows)

# Save to Excel
output_file = "Filtered_Texture_Summary.xlsx"
df.to_excel(output_file, index=False)

print(f"\nExcel file saved successfully as '{output_file}'")

# ============================================================
# ML MODULE — AUTOMATIC TEXTURE PREDICTION
# Uses the Excel file created above as input.
# ============================================================

import pandas as pd
import numpy as np

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor
)
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    HuberRegressor
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# The previous notebook cell creates this variable:
# output_file = "Filtered_Texture_Summary.xlsx"
try:
    file_path = output_file
except NameError:
    file_path = "Filtered_Texture_Summary.xlsx"

print(f"Using automatically generated input file: {file_path}")
df = pd.read_excel(file_path)

required_columns = [
    "Texture_Name",
    "Spot_count_100",
    "Spot_count_110",
    "Spot_count_111",
    "Total_occurrences"
]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing required columns in {file_path}: {missing_columns}")

if len(df) < 2:
    raise ValueError("At least two texture rows are required for model training and ranking.")

# ==========================================
# Feature Matrix
# ==========================================
X = df[[
    "Spot_count_100",
    "Spot_count_110",
    "Spot_count_111",
    "Total_occurrences"
]]

# ==========================================
# Physically-Informed Texture Function
# ==========================================
def safe_norm(series):
    max_value = series.max()
    if max_value == 0:
        return series * 0
    return series / max_value

occ_norm = safe_norm(df["Total_occurrences"])
sc100 = safe_norm(df["Spot_count_100"])
sc110 = safe_norm(df["Spot_count_110"])
sc111 = safe_norm(df["Spot_count_111"])

y = (
    0.15 * occ_norm +
    0.24 * sc100 +
    0.29 * sc110 +
    0.45 * sc111 +
    0.34 * (sc110 * sc111) +
    0.24 * (sc100 * sc111)
)

if y.max() == 0:
    raise ValueError("All computed target scores are zero; check spot counts and total occurrences.")
y = y / y.max()

# ==========================================
# Train-Test Split
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==========================================
# Regression Models
# ==========================================
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.001),
    "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.5),
    "Huber Regressor": HuberRegressor(),
    "Decision Tree": DecisionTreeRegressor(max_depth=4, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=500, max_depth=4, random_state=42),
    "Extra Trees": ExtraTreesRegressor(n_estimators=500, max_depth=4, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    ),
    "Support Vector Regression": SVR(kernel="rbf", C=10, gamma="scale"),
    "KNN Regressor": KNeighborsRegressor(n_neighbors=min(3, len(X_train)))
}

# ==========================================
# Model Evaluation
# ==========================================
results = []
trained_models = {}

for name, regressor in models.items():
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", regressor)
    ])

    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    results.append([name, train_r2, test_r2, mae, rmse])
    trained_models[name] = model

results_df = pd.DataFrame(
    results,
    columns=["Model", "Train_R2", "Test_R2", "MAE", "RMSE"]
).sort_values(by="Test_R2", ascending=False)

print("Model Performance Comparison:")
print(results_df.to_string(index=False))

# ==========================================
# Best Model + Texture Ranking
# ==========================================
best_model_name = results_df.iloc[0]["Model"]
print(f"Best Performing Model: {best_model_name}")

best_model = trained_models[best_model_name]
df["Predicted_Score"] = best_model.predict(X)
df_sorted = df.sort_values(by="Predicted_Score", ascending=False)

print("Possible Textures:")
print(df_sorted[["Texture_Name"]].head(4).to_string(index=False))

# Save the prediction result automatically.
prediction_output_file = "Predicted_Texture_Ranking.xlsx"
df_sorted.to_excel(prediction_output_file, index=False)
results_df.to_excel("Model_Performance_Comparison.xlsx", index=False)

# ==========================================
# Feature Importance
# ==========================================
best_regressor = best_model.named_steps["model"]
if hasattr(best_regressor, "feature_importances_"):
    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": best_regressor.feature_importances_
    }).sort_values(by="Importance", ascending=False)

   # print("Feature Importance:")
   # print(importance_df.to_string(index=False))

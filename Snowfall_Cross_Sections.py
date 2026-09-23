import requests
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, TextArea, HPacker
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter1d
import datetime
from zoneinfo import ZoneInfo
import urllib3
import json
import os
from PIL import Image, ImageDraw
from io import BytesIO

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

NWS_HEADERS = {
    'User-Agent': '(SnowCrossSectionApp/10.1, contact@example.com)',
    'Accept': 'application/geo+json'
}

OUTPUT_DIR = "output_graphics"
ASSETS_DIR = "assets"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

NOAA_LOGO_FILE = "noaa_logo.png"
NWS_LOGO_FILE = "nws_logo.png"

# ==========================================
# 1. EXPANDED WESTERN U.S. ROUTE LIBRARY
# ==========================================

ROUTES = {
    "i70_co": {
        "title": "Colorado I-70 Corridor",
        "direction": "WE",
        "tz": "America/Denver",
        "shield_file": "shield_i70.png",
        "shield_url": "https://commons.wikimedia.org/wiki/Special:FilePath/I-70.svg?width=300",
        "waypoints": [
            {"name": "UT | CO", "lat": 39.1235, "lon": -109.0493},
            {"name": "Grand Junction", "lat": 39.0639, "lon": -108.5506},
            {"name": "Rifle", "lat": 39.5347, "lon": -107.7831},
            {"name": "Glenwood Springs", "lat": 39.5505, "lon": -107.3248},
            {"name": "Eagle", "lat": 39.6553, "lon": -106.8287},
            {"name": "Vail", "lat": 39.6403, "lon": -106.3742},
            {"name": "Vail Pass", "lat": 39.5312, "lon": -106.2173},
            {"name": "Eisenhower Tunnel", "lat": 39.6809, "lon": -105.9406},
            {"name": "Denver", "lat": 39.7392, "lon": -104.9903},
            {"name": "Limon", "lat": 39.2647, "lon": -103.6874},
            {"name": "CO | KS", "lat": 39.3400, "lon": -102.0500}
        ]
    },
    "i80_ca": {
        "title": "California I-80 Donner Pass Corridor",
        "direction": "WE",
        "tz": "America/Los_Angeles",
        "shield_file": "shield_i80.png",
        "shield_url": "https://commons.wikimedia.org/wiki/Special:FilePath/I-80.svg?width=300",
        "waypoints": [
            {"name": "Sacramento", "lat": 38.5816, "lon": -121.4944},
            {"name": "Auburn", "lat": 38.8966, "lon": -121.0769},
            {"name": "Colfax", "lat": 39.1002, "lon": -120.9533},
            {"name": "Blue Canyon", "lat": 39.2752, "lon": -120.7088},
            {"name": "Donner Summit", "lat": 39.3155, "lon": -120.3272},
            {"name": "Truckee", "lat": 39.3280, "lon": -120.1833},
            {"name": "Reno (NV)", "lat": 39.5296, "lon": -119.8138}
        ]
    },
    "us50_tahoe": {
        "title": "US-50 Lake Tahoe / Echo Summit Corridor",
        "direction": "WE",
        "tz": "America/Los_Angeles",
        "shield_file": "shield_us50.png",
        "shield_url": "https://commons.wikimedia.org/wiki/Special:FilePath/US_50.svg?width=300",
        "waypoints": [
            {"name": "Sacramento", "lat": 38.5816, "lon": -121.4944},
            {"name": "Placerville", "lat": 38.7296, "lon": -120.7985},
            {"name": "Pollock Pines", "lat": 38.7613, "lon": -120.5866},
            {"name": "Echo Summit", "lat": 38.8129, "lon": -120.0302},
            {"name": "South Lake Tahoe", "lat": 38.9399, "lon": -119.9772},
            {"name": "Spooner Summit", "lat": 39.1054, "lon": -119.9052},
            {"name": "Carson City (NV)", "lat": 39.1638, "lon": -119.7674}
        ]
    },
    "i90_wa": {
        "title": "Washington I-90 Snoqualmie Pass Corridor",
        "direction": "WE",
        "tz": "America/Los_Angeles",
        "shield_file": "shield_i90.png",
        "shield_url": "https://commons.wikimedia.org/wiki/Special:FilePath/I-90.svg?width=300",
        "waypoints": [
            {"name": "Seattle", "lat": 47.6062, "lon": -122.3321},
            {"name": "North Bend", "lat": 47.4957, "lon": -121.7868},
            {"name": "Snoqualmie Pass", "lat": 47.4208, "lon": -121.4124},
            {"name": "Easton", "lat": 47.2373, "lon": -121.1812},
            {"name": "Cle Elum", "lat": 47.1954, "lon": -120.9393},
            {"name": "Ellensburg", "lat": 46.9965, "lon": -120.5478}
        ]
    },
    "i80_ut": {
        "title": "Utah I-80 Parleys Canyon Corridor",
        "direction": "WE",
        "tz": "America/Denver",
        "shield_file": "shield_i80.png",
        "shield_url": "https://commons.wikimedia.org/wiki/Special:FilePath/I-80.svg?width=300",
        "waypoints": [
            {"name": "Salt Lake City", "lat": 40.7608, "lon": -111.8910},
            {"name": "Parleys Summit", "lat": 40.7102, "lon": -111.5312},
            {"name": "Kimball Junction", "lat": 40.7238, "lon": -111.5532},
            {"name": "Park City", "lat": 40.6461, "lon": -111.4980},
            {"name": "Coalville", "lat": 40.9177, "lon": -111.3996},
            {"name": "Evanston (WY)", "lat": 41.2683, "lon": -111.0010}
        ]
    },
    "i25_co_nm": {
        "title": "I-25 Front Range Corridor",
        "direction": "NS",
        "tz": "America/Denver",
        "shield_file": "shield_i25.png",
        "shield_url": "https://commons.wikimedia.org/wiki/Special:FilePath/I-25.svg?width=300",
        "waypoints": [
            {"name": "Cheyenne (WY)", "lat": 41.1400, "lon": -104.8202},
            {"name": "Fort Collins", "lat": 40.5853, "lon": -105.0844},
            {"name": "Denver", "lat": 39.7392, "lon": -104.9903},
            {"name": "Palmer Divide", "lat": 39.0917, "lon": -104.8727},
            {"name": "Colorado Springs", "lat": 38.8339, "lon": -104.8214},
            {"name": "Pueblo", "lat": 38.2544, "lon": -104.6091},
            {"name": "Raton Pass", "lat": 36.9922, "lon": -104.4789},
            {"name": "Raton (NM)", "lat": 36.9034, "lon": -104.4391}
        ]
    },
    "i5_siskiyou": {
        "title": "I-5 Cascades / Siskiyou Pass Corridor",
        "direction": "NS",
        "tz": "America/Los_Angeles",
        "shield_file": "shield_i5.png",
        "shield_url": "https://commons.wikimedia.org/wiki/Special:FilePath/I-5.svg?width=300",
        "waypoints": [
            {"name": "Eugene (OR)", "lat": 44.0521, "lon": -123.0868},
            {"name": "Roseburg", "lat": 43.2165, "lon": -123.3417},
            {"name": "Grants Pass", "lat": 42.4391, "lon": -123.3284},
            {"name": "Siskiyou Summit", "lat": 42.0574, "lon": -122.6053},
            {"name": "Weed (CA)", "lat": 41.4226, "lon": -122.3861},
            {"name": "Mt. Shasta City", "lat": 41.3100, "lon": -122.3100},
            {"name": "Redding (CA)", "lat": 40.5865, "lon": -122.3917}
        ]
    },
    "us395_sierra": {
        "title": "US-395 Eastern Sierra Corridor",
        "direction": "NS",
        "tz": "America/Los_Angeles",
        "shield_file": "shield_us395.png",
        "shield_url": "https://commons.wikimedia.org/wiki/Special:FilePath/US_395.svg?width=300",
        "waypoints": [
            {"name": "Reno (NV)", "lat": 39.5296, "lon": -119.8138},
            {"name": "Carson City", "lat": 39.1638, "lon": -119.7674},
            {"name": "Bridgeport (CA)", "lat": 38.2557, "lon": -119.2313},
            {"name": "Conway Summit", "lat": 38.0833, "lon": -119.1833},
            {"name": "Mammoth Lakes", "lat": 37.6485, "lon": -118.9721},
            {"name": "Bishop", "lat": 37.3614, "lon": -118.3997},
            {"name": "Lone Pine", "lat": 36.6060, "lon": -118.0628}
        ]
    },
    "i15_id_mt": {
        "title": "I-15 Monida Pass Corridor",
        "direction": "NS",
        "tz": "America/Denver",
        "shield_file": "shield_i15.png",
        "shield_url": "https://commons.wikimedia.org/wiki/Special:FilePath/I-15.svg?width=300",
        "waypoints": [
            {"name": "Helena (MT)", "lat": 46.5891, "lon": -112.0391},
            {"name": "Butte", "lat": 46.0038, "lon": -112.5348},
            {"name": "Dillon", "lat": 45.2163, "lon": -112.6373},
            {"name": "Monida Pass", "lat": 44.5585, "lon": -112.3060},
            {"name": "Dubois (ID)", "lat": 44.1727, "lon": -112.2308},
            {"name": "Idaho Falls", "lat": 43.4927, "lon": -112.0401},
            {"name": "Pocatello", "lat": 42.8713, "lon": -112.4455}
        ]
    }
}

# ==========================================
# 2. LOCAL ASSET MANAGEMENT
# ==========================================

def create_fallback_shield(filename, text_label=None):
    """Generates a local placeholder graphic if download fails."""
    path = os.path.join(ASSETS_DIR, filename)
    img = Image.new('RGBA', (300, 300), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([20, 20, 280, 280], radius=40, fill=(27, 42, 71, 230), outline=(255, 255, 255, 255), width=6)
    label = text_label if text_label else filename.replace("shield_", "").replace(".png", "").upper()
    draw.text((150, 150), label, fill="white", anchor="mm")
    img.save(path)

def ensure_assets_exist():
    """Checks for local PNGs in assets/. Downloads missing ones ONCE to disk."""
    print("📁 Checking local image assets in 'assets/'...")
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

    # 1. Check NOAA Emblem
    noaa_path = os.path.join(ASSETS_DIR, NOAA_LOGO_FILE)
    if not os.path.exists(noaa_path):
        url = "https://upload.wikimedia.org/wikipedia/commons/thumb/7/79/NOAA_logo.svg/320px-NOAA_logo.svg.png"
        try:
            res = requests.get(url, headers=headers, verify=False, timeout=15)
            if res.status_code == 200:
                Image.open(BytesIO(res.content)).convert('RGBA').save(noaa_path)
                print(" Saved NOAA logo to assets/")
            else:
                create_fallback_shield(NOAA_LOGO_FILE, "NOAA")
        except Exception:
            create_fallback_shield(NOAA_LOGO_FILE, "NOAA")

    # 2. Check NWS Emblem
    nws_path = os.path.join(ASSETS_DIR, NWS_LOGO_FILE)
    if not os.path.exists(nws_path):
        nws_urls = [
            "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7a/National_Weather_Service_Logo.svg/320px-National_Weather_Service_Logo.svg.png",
            "https://www.weather.gov/images/nws/nws_logo.png"
        ]
        saved = False
        for url in nws_urls:
            try:
                res = requests.get(url, headers=headers, verify=False, timeout=15)
                if res.status_code == 200:
                    Image.open(BytesIO(res.content)).convert('RGBA').save(nws_path)
                    print(" Saved NWS logo to assets/")
                    saved = True
                    break
            except Exception:
                continue
        if not saved:
            create_fallback_shield(NWS_LOGO_FILE, "NWS")

    # 3. Check Highway Shields
    for route_key, route in ROUTES.items():
        shield_file = route.get("shield_file")
        shield_url = route.get("shield_url")
        if shield_file:
            local_path = os.path.join(ASSETS_DIR, shield_file)
            if not os.path.exists(local_path):
                try:
                    res = requests.get(shield_url, headers=headers, verify=False, timeout=15)
                    if res.status_code == 200:
                        Image.open(BytesIO(res.content)).convert('RGBA').save(local_path)
                        print(f" Saved {shield_file} to assets/")
                    else:
                        create_fallback_shield(shield_file)
                except Exception:
                    create_fallback_shield(shield_file)

def load_local_logo(filename, target_height=75):
    """
    Loads logo from assets/, trims transparent padding, and normalizes height.
    Guarantees both NOAA and NWS logos render at identical visual sizes.
    """
    if not filename:
        return None
    path = os.path.join(ASSETS_DIR, filename)
    if os.path.exists(path):
        try:
            img = Image.open(path).convert('RGBA')
            bbox = img.getbbox()
            if bbox:
                img = img.crop(bbox)
            aspect = img.width / img.height
            new_width = int(target_height * aspect)
            resample_filter = getattr(getattr(Image, 'Resampling', Image), 'LANCZOS', Image.BILINEAR)
            return img.resize((new_width, target_height), resample_filter)
        except Exception as e:
            print(f"Error loading local logo {filename}: {e}")
    return None

# ==========================================
# 3. WEATHER & ELEVATION DATA FUNCTIONS
# ==========================================

def get_elevation_profile(waypoints, samples_per_segment, cache_file):
    """Interpolates coordinates and queries DEM with route-specific disk caching."""
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            return np.array(json.load(f))

    print(f"🌐 Querying Elevation DEM for new route ({cache_file})...")
    lats, lons = [], []
    for i in range(len(waypoints) - 1):
        seg_lats = np.linspace(waypoints[i]["lat"], waypoints[i+1]["lat"], samples_per_segment, endpoint=False)
        seg_lons = np.linspace(waypoints[i]["lon"], waypoints[i+1]["lon"], samples_per_segment, endpoint=False)
        lats.extend(seg_lats)
        lons.extend(seg_lons)
    lats.append(waypoints[-1]["lat"])
    lons.append(waypoints[-1]["lon"])

    locations_payload = [{"latitude": round(lat, 4), "longitude": round(lon, 4)} for lat, lon in zip(lats, lons)]
    url_ele = "https://api.open-elevation.com/api/v1/lookup"

    try:
        response = requests.post(url_ele, json={"locations": locations_payload}, headers={"Content-Type": "application/json"}, verify=False, timeout=15)
        if response.status_code == 200:
            results = response.json()["results"]
            elev_ft = np.array([res["elevation"] * 3.28084 for res in results])
            with open(cache_file, "w") as f:
                json.dump(elev_ft.tolist(), f)
            return elev_ft
    except Exception as e:
        print(f"Elevation API Error: {e}")

    return 4000 + 100 * (np.array(lats)/10) + 300 * np.sin(np.array(lons)/15) + np.random.normal(0, 50, len(lats))

def get_live_snow_forecast(waypoints, hours=48):
    """Iterates through waypoints, querying NWS API for live snowfall aggregated over `hours`."""
    slice_count = max(1, int(hours / 6))
    updated_waypoints = [wp.copy() for wp in waypoints]

    for wp in updated_waypoints:
        lat, lon = wp["lat"], wp["lon"]
        wp["snow"] = "N/A"
        try:
            p_url = f"https://api.weather.gov/points/{round(lat, 4)},{round(lon, 4)}"
            p_res = requests.get(p_url, headers=NWS_HEADERS, verify=False, timeout=10)
            if p_res.status_code != 200: continue
            g_url = p_res.json()["properties"]["forecastGridData"]

            g_res = requests.get(g_url, headers=NWS_HEADERS, verify=False, timeout=10)
            if g_res.status_code != 200: continue

            s_data = g_res.json()["properties"].get("snowfallAmount", {}).get("values", [])
            total_inches = sum(v.get("value", 0) for v in s_data[:slice_count]) / 25.4

            if total_inches == 0.0: wp["snow"] = "0\""
            elif total_inches < 0.5: wp["snow"] = "0 to 1\""
            elif total_inches < 2.5: wp["snow"] = "1 to 3\""
            elif total_inches < 5.0: wp["snow"] = "3 to 5\""
            elif total_inches < 8.0: wp["snow"] = "5 to 8\""
            elif total_inches < 12.0: wp["snow"] = "8 to 12\""
            else: wp["snow"] = f'{int(total_inches-2)} to {int(total_inches+3)}"'
        except Exception:
            continue
    return updated_waypoints

def format_local_time(dt):
    """Formats datetime into concise localized string: '9/23 5pm MDT'."""
    hour_12 = dt.strftime('%I').lstrip('0')
    am_pm = dt.strftime('%p').lower()
    tz_abbr = dt.strftime('%Z')
    return f"{dt.month}/{dt.day} {hour_12}{am_pm} {tz_abbr}"

# ==========================================
# 4. EXPORT & RENDERING FUNCTION
# ==========================================

def render_and_save_route(route_key, hours=48, display_inline=False):
    """Generates broadcast graphic with HPacker-locked dual logos and enlarged footer typography."""
    route = ROUTES[route_key]
    route_title = route["title"]
    route_waypoints = route["waypoints"]
    shield_file = route.get("shield_file")
    direction = route.get("direction", "WE")
    tz_str = route.get("tz", "America/Denver")
    cache_file = f"elevation_cache_{route_key}.json"

    print(f"Processing route [{hours}h]: {route_title}...")

    elevations_ft = get_elevation_profile(route_waypoints, 50, cache_file)
    total_points = len(elevations_ft)
    elevations_smoothed = gaussian_filter1d(elevations_ft, sigma=2.5)
    distances_miles = np.linspace(0, 150, total_points)

    waypoints_with_weather = get_live_snow_forecast(route_waypoints, hours=hours)

    for i, wp in enumerate(waypoints_with_weather):
        norm_dist = i / (len(waypoints_with_weather) - 1)
        wp_idx = int(norm_dist * (total_points - 1))
        wp["elev"] = elevations_smoothed[wp_idx]
        wp["dist"] = distances_miles[wp_idx]

    fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
    bg_blue = '#1d3557'
    fig.patch.set_facecolor(bg_blue)
    ax.set_facecolor(bg_blue)

    outline_white = [path_effects.withStroke(linewidth=3, foreground='black')]

    stagger_levels = [1800, 3600, 2400, 4200]
    max_label_y_observed = 0
    for i, wp in enumerate(waypoints_with_weather):
        text_stagger = stagger_levels[i % len(stagger_levels)]
        label_y = wp["elev"] + text_stagger
        if label_y > max_label_y_observed:
            max_label_y_observed = label_y

    max_y_limit = max_label_y_observed + 1500

    # Procedural Sky & Falling Snowflakes
    sky_colors = ["#111c2a", "#1d3557", "#2b4c7e"]
    sky_cmap = LinearSegmentedColormap.from_list("sky_grad", sky_colors)
    gradient_data = np.linspace(0, 1, 256).reshape(256, 1)
    ax.imshow(gradient_data, extent=[0, max(distances_miles), 0, max_y_limit],
              aspect='auto', cmap=sky_cmap, origin='lower', zorder=1)

    seed_value = abs(hash(f"{route_key}_{hours}")) % (2**32)
    np.random.seed(seed_value)

    num_flakes = 300
    flake_x = np.random.uniform(0, max(distances_miles), num_flakes)
    flake_y = np.random.uniform(0, max_y_limit, num_flakes)
    flake_sizes = np.random.uniform(4, 30, num_flakes)
    flake_alphas = np.random.uniform(0.12, 0.45, num_flakes)

    for fx, fy, fs, fa in zip(flake_x, flake_y, flake_sizes, flake_alphas):
        ax.scatter(fx, fy, s=fs, color='white', alpha=fa, zorder=1.8, edgecolors='none')

    # Terrain Layers
    ax.fill_between(distances_miles, 0, elevations_smoothed, color='#242424', zorder=2)
    ax.plot(distances_miles, elevations_smoothed, color='#ffffff', linewidth=2.8, zorder=3)
    ax.fill_between(distances_miles, elevations_smoothed, max_y_limit, color='white', alpha=0.06, zorder=3.5)

    # Waypoint Badges & Labels
    for i, wp in enumerate(waypoints_with_weather):
        x, y = wp["dist"], wp["elev"]
        ax.plot(x, y, 'wo', markersize=9, markeredgecolor='black', markeredgewidth=1.5, zorder=5)

        text_stagger = stagger_levels[i % len(stagger_levels)]
        label_y = y + text_stagger

        ax.plot([x, x], [y + 120, label_y - 250], color='white', linestyle='--', linewidth=1.5, zorder=4)

        ax.text(x, label_y, f" {wp['snow']} ",
                color='#000000', ha='center', va='bottom',
                fontsize=18, fontweight='bold', zorder=6,
                bbox=dict(boxstyle="round,pad=0.35", fc="#ffd166", ec="#ffffff", lw=1.5))

        ax.text(x, label_y - 200, f"{wp['name']}\n{int(wp['elev'])} ft",
                color='white', ha='center', va='top',
                fontsize=14, fontweight='bold', zorder=6, path_effects=outline_white)

    # --- CENTERED 3-TIER HEADER STACK ---
    fig.text(0.50, 0.94, f"{hours}-HOUR SNOWFALL FORECAST", color='#ffd166', 
             fontsize=36, fontweight='heavy', ha='center', va='center', path_effects=outline_white)
    
    fig.text(0.50, 0.89, route_title, color='white', 
             fontsize=26, fontweight='bold', ha='center', va='center', path_effects=outline_white)

    route_tz = ZoneInfo(tz_str)
    now_local = datetime.datetime.now(route_tz)
    end_local = now_local + datetime.timedelta(hours=hours)

    valid_time_display = f"Forecast Valid: {format_local_time(now_local)} to {format_local_time(end_local)}"
    fig.text(0.50, 0.84, valid_time_display, color='#e2e8f0', 
             fontsize=19, fontweight='bold', ha='center', va='center', path_effects=outline_white)

    # --- HIGHWAY SHIELD & DIRECTIONAL ARROW (MIDDLE BOTTOM) ---
    highway_img = load_local_logo(shield_file, target_height=90)
    if highway_img:
        imagebox = OffsetImage(highway_img, zoom=1.0)
        ab = AnnotationBbox(imagebox, (0.50, 0.20), xycoords='axes fraction', frameon=False, zorder=6)
        ax.add_artist(ab)

    dir_text = "NORTH ──────► SOUTH" if direction == "NS" else "WEST ──────► EAST"
    ax.text(0.50, 0.08, dir_text, color='white', fontsize=15, fontweight='bold',
            ha='center', va='center', transform=ax.transAxes, zorder=6, path_effects=outline_white)

    # --- HPACKER: LOCKED LOGO + TEXT GROUPING (BOTTOM LEFT) ---
    footer_children = []

    noaa_img = load_local_logo(NOAA_LOGO_FILE, target_height=75)
    if noaa_img:
        footer_children.append(OffsetImage(noaa_img, zoom=1.0))

    nws_img = load_local_logo(NWS_LOGO_FILE, target_height=75)
    if nws_img:
        footer_children.append(OffsetImage(nws_img, zoom=1.0))

    agency_text_area = TextArea(
        "National Weather Service\nWeather Prediction Center",
        textprops=dict(color='white', fontsize=20, fontweight='bold', multialignment='left')
    )
    footer_children.append(agency_text_area)

    # Pack both logos and text with an exact 18-pixel gap that NEVER varies
    footer_packer = HPacker(children=footer_children, align="center", pad=0, sep=18)
    
    # Place footer group starting at x=0.03, y=0.065
    ab_footer = AnnotationBbox(
        footer_packer, (0.03, 0.065), 
        xycoords='figure fraction', 
        frameon=False, 
        box_alignment=(0, 0.5)
    )
    ax.add_artist(ab_footer)

    # Bottom Right: Issue Date
    fig.text(0.97, 0.065, f"Issued: {now_local.strftime('%B %d, %Y')}", color='white', 
             fontsize=21, fontweight='bold', ha='right', va='center')

    ax.set_xlim(0, max(distances_miles))
    ax.set_ylim(0, max_y_limit)
    ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.81, bottom=0.15, left=0.01, right=0.99)

    file_path = os.path.join(OUTPUT_DIR, f"{route_key}_{hours}h.png")
    plt.savefig(file_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    print(f" Saved graphic: {file_path}")

    if display_inline:
        plt.show()
    else:
        plt.close(fig)

def batch_generate_all_routes():
    """Loops through all routes and exports 24h, 48h, and 72h accumulation graphics."""
    ensure_assets_exist()
    
    timeframes = [24, 48, 72]
    total_images = len(ROUTES) * len(timeframes)
    print(f"\n🚀 Starting batch export for {len(ROUTES)} corridors across {timeframes} timeframes ({total_images} PNGs)...\n")
    
    for key in ROUTES.keys():
        for hours in timeframes:
            render_and_save_route(key, hours=hours, display_inline=False)
            
    print(f"\n Finished! All {total_images} images saved to folder: '{OUTPUT_DIR}'")

# Run batch export
batch_generate_all_routes()

"""
╔══════════════════════════════════════════════════════════════════╗
║           CRISIS VECTOR AI  —  Disaster Routing Engine          ║
║         Production-Grade Emergency Navigation Platform           ║
╚══════════════════════════════════════════════════════════════════╝
Run with:  streamlit run app.py

Dependencies:
    pip install streamlit osmnx networkx folium streamlit-folium \
                pillow requests google-generativeai
"""

import os
import math
import time
import json
import base64
import hashlib
import warnings
import requests
import streamlit as st
import osmnx as ox
import networkx as nx
import folium
from folium.plugins import MiniMap, Fullscreen, LocateControl
from streamlit_folium import st_folium
from PIL import Image
import io

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════
ox.settings.use_cache = True
ox.settings.log_console = False

APP_VERSION = "3.0.0"

PRESET_LOCATIONS = {
    "🏙️ Ravet, Pune":           ("Ravet, Pune, Maharashtra, India",           2000),
    "🏭 Pimpri-Chinchwad":       ("Pimpri-Chinchwad, Maharashtra, India",       2500),
    "🏛️ Shivajinagar, Pune":     ("Shivajinagar, Pune, Maharashtra, India",     2000),
    "💻 Hinjewadi, Pune":        ("Hinjewadi, Pune, Maharashtra, India",        2500),
    "🌆 Koregaon Park, Pune":    ("Koregaon Park, Pune, Maharashtra, India",    2000),
    "🏗️ Wakad, Pune":            ("Wakad, Pune, Maharashtra, India",            2000),
    "🌍 Baner, Pune":            ("Baner, Pune, Maharashtra, India",            2000),
}

DISASTER_TYPES = {
    "🌊 Flood":      {"color": "#1E90FF", "fill": "#1E90FF", "radius": 600,  "label": "FLOOD ZONE",    "speed_factor": 0.0,  "icon": "💧"},
    "🔥 Fire":       {"color": "#FF4500", "fill": "#FF4500", "radius": 400,  "label": "FIRE ZONE",     "speed_factor": 0.0,  "icon": "🔥"},
    "🏚️ Collapse":   {"color": "#8B4513", "fill": "#A0522D", "radius": 300,  "label": "DEBRIS ZONE",   "speed_factor": 0.0,  "icon": "🏚️"},
    "💣 Explosion":  {"color": "#FF8C00", "fill": "#FFA500", "radius": 350,  "label": "BLAST ZONE",    "speed_factor": 0.0,  "icon": "💥"},
    "☣️ Hazmat":     {"color": "#9ACD32", "fill": "#ADFF2F", "radius": 500,  "label": "HAZMAT ZONE",   "speed_factor": 0.0,  "icon": "☣️"},
    "🌪️ Tornado":    {"color": "#9370DB", "fill": "#8A2BE2", "radius": 450,  "label": "TORNADO ZONE",  "speed_factor": 0.0,  "icon": "🌪️"},
}

ROUTE_STYLES = {
    "baseline":  {"color": "#888888", "weight": 4, "opacity": 0.7, "dash": "10 5",   "label": "Original Route"},
    "emergency": {"color": "#00FFCC", "weight": 5, "opacity": 0.9, "dash": None,     "label": "Emergency Detour"},
    "safest":    {"color": "#FFD700", "weight": 6, "opacity": 1.0, "dash": None,     "label": "Safest Path (Recommended)"},
}

# ─── AI Provider config ───────────────────────────────────────────
# Hugging Face: correct v2 Inference API endpoint format
HF_API_BASE = "https://api-inference.huggingface.co/models"
HF_MODELS = {
    "BLIP Large (Best for disasters)":
        f"{HF_API_BASE}/Salesforce/blip-image-captioning-large",
    "BLIP Base (Faster)":
        f"{HF_API_BASE}/Salesforce/blip-image-captioning-base",
    "ViT-GPT2 Scene":
        f"{HF_API_BASE}/nlpconnect/vit-gpt2-image-captioning",
}

# ═══════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════
def haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi  = math.radians(lat2 - lat1)
    dlam  = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def route_stats(G, route):
    if not route or len(route) < 2:
        return None, None
    dist  = sum(G[u][v][0].get("length", 0)       for u, v in zip(route[:-1], route[1:]))
    ttime = sum(G[u][v][0].get("travel_time", 0)  for u, v in zip(route[:-1], route[1:]))
    return round(dist / 1000, 2), round(ttime / 60, 1)


def add_edge_data(G):
    try:
        G = ox.routing.add_edge_speeds(G)
        G = ox.routing.add_edge_travel_times(G)
    except AttributeError:
        try:
            G = ox.add_edge_speeds(G)
            G = ox.add_edge_travel_times(G)
        except Exception:
            pass
    return G


def safe_nearest_node(G, lon, lat):
    try:
        return ox.distance.nearest_nodes(G, lon, lat)
    except Exception as e:
        raise ValueError(f"Could not snap ({lat:.4f}, {lon:.4f}) to graph: {e}")


def image_to_base64(image_file) -> str:
    """Convert uploaded file to base64 string."""
    image_file.seek(0)
    return base64.standard_b64encode(image_file.read()).decode("utf-8")


def resize_image_for_api(image_file, max_size: int = 1024) -> bytes:
    """Resize image to max_size on the longest edge and return JPEG bytes."""
    image_file.seek(0)
    img = Image.open(image_file).convert("RGB")
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════
# CORE ROUTING ENGINE
# ═══════════════════════════════════════════════════════════════════
class CrisisVectorEngine:
    def __init__(self):
        self.G          = None
        self.G_live     = None
        self.obstructions: list = []
        self.start_node = None
        self.end_node   = None
        self.routes: dict = {}
        self.location_name: str = ""

    def load_network(self, location_query: str, dist: int = 2000):
        G = ox.graph_from_address(location_query, dist=dist, network_type="drive")
        G = add_edge_data(G)
        self.G             = G
        self.G_live        = G.copy()
        self.obstructions.clear()
        self.routes.clear()
        self.start_node    = None
        self.end_node      = None
        self.location_name = location_query
        return len(G.nodes), len(G.edges)

    def add_obstruction(self, lat, lon, radius, disaster_type):
        meta = DISASTER_TYPES[disaster_type]
        self.obstructions.append({
            "lat": lat, "lon": lon, "radius": radius,
            "type": disaster_type, "color": meta["color"],
            "fill": meta["fill"],  "label": meta["label"], "icon": meta["icon"],
        })
        self._rebuild_live_graph()

    def remove_obstruction(self, index: int):
        if 0 <= index < len(self.obstructions):
            self.obstructions.pop(index)
            self._rebuild_live_graph()

    def clear_obstructions(self):
        self.obstructions.clear()
        if self.G:
            self.G_live = self.G.copy()
        self.routes.clear()

    def _rebuild_live_graph(self):
        if not self.G:
            return
        self.G_live = self.G.copy()
        blocked = set()
        for obs in self.obstructions:
            for node, data in self.G.nodes(data=True):
                if haversine_m(obs["lat"], obs["lon"], data["y"], data["x"]) < obs["radius"]:
                    blocked.add(node)
        self.G_live.remove_nodes_from(blocked)
        self.routes.clear()

    def blocked_node_count(self) -> int:
        if not self.G:
            return 0
        return len(self.G.nodes) - len(self.G_live.nodes)

    def coverage_percent(self) -> float:
        if not self.G or len(self.G.nodes) == 0:
            return 0.0
        return round(self.blocked_node_count() / len(self.G.nodes) * 100, 1)

    def set_endpoints(self, start_lat, start_lon, end_lat, end_lon):
        if not self.G:
            raise ValueError("No graph loaded.")
        self.start_node = safe_nearest_node(self.G, start_lon, start_lat)
        self.end_node   = safe_nearest_node(self.G, end_lon,   end_lat)

    def compute_routes(self):
        if not self.G or self.start_node is None or self.end_node is None:
            raise ValueError("Graph and endpoints must be set.")
        self.routes = {}
        live_nodes = set(self.G_live.nodes)
        sn = self.start_node if self.start_node in live_nodes else self._nearest_live_node(
            self.G.nodes[self.start_node]["y"], self.G.nodes[self.start_node]["x"])
        en = self.end_node   if self.end_node   in live_nodes else self._nearest_live_node(
            self.G.nodes[self.end_node]["y"],   self.G.nodes[self.end_node]["x"])

        try:
            self.routes["baseline"] = nx.shortest_path(
                self.G, self.start_node, self.end_node, weight="travel_time")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            self.routes["baseline"] = None

        try:
            self.routes["emergency"] = nx.shortest_path(
                self.G_live, sn, en, weight="travel_time")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            self.routes["emergency"] = None

        self.routes["safest"] = self._safest_route(sn, en)
        return self.routes

    def _nearest_live_node(self, lat, lon):
        best, best_d = None, float("inf")
        for n, d in self.G_live.nodes(data=True):
            dist = haversine_m(lat, lon, d["y"], d["x"])
            if dist < best_d:
                best, best_d = n, dist
        return best

    def _safest_route(self, sn, en):
        if not self.obstructions:
            return self.routes.get("emergency")
        H = self.G_live.copy()
        for u, v, key, data in H.edges(keys=True, data=True):
            mid_lat = (H.nodes[u]["y"] + H.nodes[v]["y"]) / 2
            mid_lon = (H.nodes[u]["x"] + H.nodes[v]["x"]) / 2
            base_tt = data.get("travel_time", 10)
            penalty = sum(
                (obs["radius"] * 2.5 - haversine_m(obs["lat"], obs["lon"], mid_lat, mid_lon)) / 8
                for obs in self.obstructions
                if haversine_m(obs["lat"], obs["lon"], mid_lat, mid_lon) < obs["radius"] * 2.5
            )
            H[u][v][key]["safe_time"] = base_tt + penalty

        def heuristic(n, goal):
            return haversine_m(H.nodes[n]["y"], H.nodes[n]["x"],
                               H.nodes[goal]["y"], H.nodes[goal]["x"]) / 20
        try:
            return nx.astar_path(H, sn, en, heuristic=heuristic, weight="safe_time")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def map_centre(self):
        if self.G:
            n = list(self.G.nodes(data=True))[len(self.G.nodes) // 2]
            return n[1]["y"], n[1]["x"]
        return 18.6500, 73.7600


# ═══════════════════════════════════════════════════════════════════
# AI IMAGE ANALYSIS  — THREE PROVIDERS, ZERO COST
# ═══════════════════════════════════════════════════════════════════

HAZARD_KEYWORDS = {
    "FLOOD":     ["flood", "water", "inundat", "submerge", "river overflow", "standing water", "waterlog"],
    "FIRE":      ["fire", "flame", "burn", "smoke", "blaze", "inferno", "charred", "ember"],
    "COLLAPSE":  ["collapse", "debris", "rubble", "ruin", "destruction", "crumble", "fallen building"],
    "EXPLOSION": ["explosion", "blast", "crater", "bomb", "detonat", "wreckage"],
    "HAZMAT":    ["chemical", "hazmat", "spill", "toxic", "gas cloud", "industrial", "yellow cloud"],
    "TORNADO":   ["tornado", "cyclone", "twister", "vortex", "funnel", "storm damage"],
    "LANDSLIDE": ["landslide", "mudslide", "avalanche", "mud flow", "buried road"],
}

EVAC_TEMPLATES = {
    "FLOOD":     ["Avoid all low-lying underpasses and subways — use elevated arterials.",
                  "Emergency vehicles: approach from uphill/upwind quadrant only.",
                  "Route evacuees to high-ground assembly points (>10 m elevation).",
                  "Monitor water rise every 15 min; trigger dynamic rerouting if >0.5 m/hr."],
    "FIRE":      ["Maintain strict upwind approach corridors for all responders.",
                  "Establish 300 m exclusion perimeter; 800 m for chemical fires.",
                  "Evacuate downwind population in 1 km radius immediately.",
                  "Pre-position water tankers at nearest hydrant grid intersections."],
    "COLLAPSE":  ["Restrict heavy vehicles (>3.5 t) from blocks with visible structural damage.",
                  "Mark secondary collapse risk buildings with GPS waypoints.",
                  "Search-and-rescue ingress via widest unobstructed route only.",
                  "Deploy shoring equipment before committing rescue teams inside structures."],
    "EXPLOSION": ["Treat 500 m radius as blast/fragmentation zone — no civilian entry.",
                  "Check for secondary devices before routing responders in.",
                  "Establish triage point at 600 m minimum from epicentre.",
                  "Coordinate with EOD before establishing forward command post."],
    "HAZMAT":    ["Identify chemical class before any approach — consult ERG guide.",
                  "Full SCBA required within 400 m of visible spill/cloud.",
                  "Establish hot/warm/cold zones per NFPA 472 protocol.",
                  "Decon corridor mandatory at warm-zone boundary before casualty transport."],
    "TORNADO":   ["Damage corridor may extend several km — verify entire route is clear.",
                  "Check for downed power lines across all planned ingress roads.",
                  "Prioritise search of reinforced structures (basements, interior rooms).",
                  "Aerial drone recon recommended before ground team deployment."],
    "LANDSLIDE": ["Route must bypass affected slope — use parallel ridge or valley road.",
                  "Monitor for secondary slides — do not pause vehicles under unstable terrain.",
                  "Heavy machinery (excavators) required before route can be reopened.",
                  "GPS track all responder positions due to rapid terrain change."],
}

GENERIC_RECS = [
    "Establish 500 m exclusion perimeter around visible hazard epicentre.",
    "Route all emergency vehicles via alternative arterial roads.",
    "Deploy search-and-rescue teams to structures showing visible damage.",
    "Set up staging area at least 800 m upwind of any smoke source.",
    "Coordinate with local fire/flood control agencies before entry.",
    "Maintain radio comms check-in every 10 minutes for all field teams.",
]


def _parse_caption(caption: str) -> dict:
    """Parse a raw caption into structured disaster intelligence."""
    cl = caption.lower()
    detected = [h for h, kws in HAZARD_KEYWORDS.items() if any(k in cl for k in kws)]

    severity = "LOW"
    if any(w in cl for w in ["destroyed", "collapsed", "massive", "severe", "devastating", "catastrophic", "engulfed"]):
        severity = "CRITICAL"
    elif any(w in cl for w in ["damage", "fire", "flood", "smoke", "rubble", "debris", "broken", "burning", "flooded"]):
        severity = "HIGH"

    recs = []
    for h in detected:
        recs.extend(EVAC_TEMPLATES.get(h, []))
    if not recs:
        recs = GENERIC_RECS.copy()
    else:
        recs = list(dict.fromkeys(recs))  # deduplicate, preserve order
        recs.extend(GENERIC_RECS[-2:])    # always append comms + agency lines

    return {
        "success":    True,
        "caption":    caption,
        "severity":   severity,
        "hazards":    detected if detected else ["GENERAL DISASTER"],
        "recommendations": recs,
        "provider":   "unknown",
    }


# ── Provider 1: Hugging Face Inference API (fixed URL) ───────────
def analyze_with_huggingface(image_file, hf_token: str, model_url: str) -> dict:
    """
    Uses the correct HF Inference API v2 URL format.
    Free tier: ~30 req/min with a free token.
    """
    img_bytes = resize_image_for_api(image_file)
    headers   = {"Authorization": f"Bearer {hf_token}", "Content-Type": "application/octet-stream"}

    for attempt in range(4):
        try:
            resp = requests.post(model_url, headers=headers, data=img_bytes, timeout=90)
        except requests.exceptions.Timeout:
            return {"success": False, "error": "Request timed out. The model may be cold — try again in 30 s."}
        except requests.exceptions.ConnectionError as e:
            return {"success": False, "error": f"Connection error: {e}"}

        if resp.status_code == 200:
            break
        elif resp.status_code == 503:
            try:
                wait = resp.json().get("estimated_time", 25)
            except Exception:
                wait = 25
            st.toast(f"⏳ Model loading… retrying in {min(int(wait), 30)} s (attempt {attempt+1}/4)")
            time.sleep(min(int(wait), 30))
        elif resp.status_code == 401:
            return {"success": False, "error": "Invalid HF token. Check your token at huggingface.co/settings/tokens"}
        elif resp.status_code == 404:
            return {"success": False,
                    "error": "Model endpoint not found (404). Try selecting 'BLIP Large' from the model dropdown."}
        else:
            return {"success": False, "error": f"HF API error {resp.status_code}: {resp.text[:300]}"}
    else:
        return {"success": False, "error": "Model still loading after 4 attempts. Please wait 1 minute and retry."}

    try:
        raw = resp.json()
        if isinstance(raw, list) and raw:
            caption = raw[0].get("generated_text") or raw[0].get("caption", "")
        elif isinstance(raw, dict):
            caption = raw.get("generated_text") or raw.get("caption", "")
        else:
            caption = str(raw)
    except Exception:
        return {"success": False, "error": f"Unexpected response format: {resp.text[:300]}"}

    if not caption:
        return {"success": False, "error": "Model returned an empty caption. Try a different model or image."}

    result = _parse_caption(caption)
    result["provider"] = "Hugging Face"
    return result


# ── Provider 2: Google Gemini (free tier, best quality) ──────────
def analyze_with_gemini(image_file, api_key: str) -> dict:
    """
    Uses Google Gemini 1.5 Flash — free tier = 15 req/min, 1500 req/day.
    No credit card needed. Get key at: aistudio.google.com/app/apikey
    """
    try:
        import google.generativeai as genai
    except ImportError:
        return {"success": False,
                "error": "google-generativeai not installed. Run: pip install google-generativeai"}

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")

        img_bytes = resize_image_for_api(image_file, max_size=1024)
        img_part  = {"mime_type": "image/jpeg", "data": img_bytes}

        prompt = """You are an AI assistant for emergency management and disaster response.
Analyse this image and provide:
1. A detailed description of what you see (2-3 sentences)
2. Disaster type(s) detected (flood, fire, collapse, explosion, hazmat, tornado, landslide, or none)
3. Severity assessment: CRITICAL / HIGH / MEDIUM / LOW
4. Specific safety hazards visible
5. Immediate evacuation/rescue routing recommendations (3-5 bullet points)

Format your response as JSON:
{
  "description": "...",
  "disaster_types": ["FLOOD", "..."],
  "severity": "HIGH",
  "hazards_visible": ["...", "..."],
  "routing_recommendations": ["...", "..."]
}
Respond ONLY with the JSON, no markdown fences."""

        response = model.generate_content([img_part, prompt])
        raw_text = response.text.strip()
        # Strip any accidental markdown fences
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]

        data = json.loads(raw_text)

        severity = data.get("severity", "HIGH").upper()
        if severity not in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
            severity = "HIGH"

        detected = [h.upper() for h in data.get("disaster_types", [])]
        caption  = data.get("description", "No description provided.")
        recs     = data.get("routing_recommendations", GENERIC_RECS)
        hazards  = data.get("hazards_visible", [])

        # Enrich recs with template knowledge
        for h in detected:
            recs.extend(EVAC_TEMPLATES.get(h, []))
        recs = list(dict.fromkeys(recs))[:8]  # deduplicate, cap at 8

        return {
            "success":          True,
            "caption":          caption,
            "severity":         severity,
            "hazards":          detected if detected else ["GENERAL DISASTER"],
            "hazards_visible":  hazards,
            "recommendations":  recs,
            "provider":         "Google Gemini 1.5 Flash",
        }
    except json.JSONDecodeError:
        # Fallback: treat Gemini response as plain text caption
        caption = response.text if hasattr(response, "text") else "Unable to parse."
        result  = _parse_caption(caption)
        result["provider"] = "Google Gemini 1.5 Flash"
        return result
    except Exception as e:
        err = str(e)
        if "API_KEY_INVALID" in err or "API key not valid" in err:
            return {"success": False, "error": "Invalid Gemini API key. Get a free key at aistudio.google.com/app/apikey"}
        return {"success": False, "error": f"Gemini error: {err}"}


# ── Provider 3: Ollama (fully local, no key) ─────────────────────
def analyze_with_ollama(image_file, model_name: str = "llava") -> dict:
    """
    Calls a local Ollama instance with a vision model (llava, bakllava, etc.)
    Zero cost, fully private. Requires: ollama pull llava
    """
    img_bytes = resize_image_for_api(image_file, max_size=768)
    b64       = base64.standard_b64encode(img_bytes).decode()

    prompt = (
        "You are a disaster response expert. Analyse this image and describe: "
        "1) What disaster/hazard is visible, 2) Severity (CRITICAL/HIGH/MEDIUM/LOW), "
        "3) Key dangers present, 4) Evacuation/routing recommendations. "
        "Be specific and concise."
    )

    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model_name, "prompt": prompt, "images": [b64], "stream": False},
            timeout=120,
        )
        if resp.status_code != 200:
            return {"success": False, "error": f"Ollama error {resp.status_code}: {resp.text[:200]}"}
        caption = resp.json().get("response", "")
        result  = _parse_caption(caption)
        result["provider"] = f"Ollama ({model_name})"
        return result
    except requests.exceptions.ConnectionError:
        return {"success": False,
                "error": "Cannot connect to Ollama. Make sure it's running: `ollama serve`"}
    except Exception as e:
        return {"success": False, "error": f"Ollama error: {e}"}


# ── Dispatcher ────────────────────────────────────────────────────
def analyze_disaster_image(image_file, provider: str, **kwargs) -> dict:
    """Route to the selected provider."""
    if provider == "huggingface":
        return analyze_with_huggingface(
            image_file, kwargs["hf_token"], kwargs["model_url"])
    elif provider == "gemini":
        return analyze_with_gemini(image_file, kwargs["api_key"])
    elif provider == "ollama":
        return analyze_with_ollama(image_file, kwargs.get("model_name", "llava"))
    return {"success": False, "error": f"Unknown provider: {provider}"}


# ═══════════════════════════════════════════════════════════════════
# MAP BUILDER
# ═══════════════════════════════════════════════════════════════════
def build_map(engine: CrisisVectorEngine,
              start_coords, end_coords,
              show_baseline: bool = True,
              analysis_zones: list = None) -> folium.Map:
    clat, clon = engine.map_centre()
    m = folium.Map(location=[clat, clon], zoom_start=14, tiles="CartoDB dark_matter")
    Fullscreen(position="topright").add_to(m)
    MiniMap(toggle_display=True, position="bottomright").add_to(m)
    LocateControl(auto_start=False).add_to(m)

    if not engine.G:
        return m

    # ── Disaster Zones ───────────────────────────────────────────
    zone_group = folium.FeatureGroup(name="⚠️ Disaster Zones", show=True)
    for i, obs in enumerate(engine.obstructions):
        folium.Circle(
            location=[obs["lat"], obs["lon"]], radius=obs["radius"],
            color=obs["color"], fill=True, fill_color=obs["fill"], fill_opacity=0.40,
            tooltip=f"<b>{obs['label']}</b><br>Radius: {obs['radius']} m",
            popup=folium.Popup(
                f"<b style='color:{obs['color']}'>{obs['icon']} {obs['label']}</b><br>"
                f"Centre: {obs['lat']:.5f}, {obs['lon']:.5f}<br>"
                f"Blocked radius: {obs['radius']} m", max_width=220),
        ).add_to(zone_group)
        folium.Circle(
            location=[obs["lat"], obs["lon"]], radius=obs["radius"] * 2.5,
            color=obs["color"], fill=False, dash_array="8 6", opacity=0.30,
            tooltip="Hazard influence buffer",
        ).add_to(zone_group)
        folium.Marker(
            location=[obs["lat"], obs["lon"]],
            icon=folium.DivIcon(
                html=f"<div style='font-size:22px;text-align:center;margin-top:-10px'>{obs['icon']}</div>",
                icon_size=(30, 30), icon_anchor=(15, 15)),
            tooltip=f"Zone {i+1}: {obs['label']}",
        ).add_to(zone_group)
    zone_group.add_to(m)

    # ── AI-detected zones (from image analysis) ──────────────────
    if analysis_zones:
        ai_group = folium.FeatureGroup(name="🤖 AI-Detected Zones", show=True)
        for az in analysis_zones:
            folium.Circle(
                location=[az["lat"], az["lon"]], radius=az["radius"],
                color="#FF00FF", fill=True, fill_color="#FF00FF", fill_opacity=0.25,
                tooltip=f"<b>AI DETECTED: {az['label']}</b><br>Confidence: {az.get('confidence','—')}",
                dash_array="5 3",
            ).add_to(ai_group)
        ai_group.add_to(m)

    # ── Routes ───────────────────────────────────────────────────
    route_group = folium.FeatureGroup(name="🗺️ Routes", show=True)
    route_keys  = ["baseline", "emergency", "safest"] if show_baseline else ["emergency", "safest"]
    for r_key in route_keys:
        route = engine.routes.get(r_key)
        style = ROUTE_STYLES[r_key]
        if not route:
            continue
        coords   = [(engine.G.nodes[n]["y"], engine.G.nodes[n]["x"]) for n in route]
        dist_km, time_min = route_stats(engine.G, route)
        kw = dict(locations=coords, color=style["color"], weight=style["weight"],
                  opacity=style["opacity"],
                  tooltip=f"<b>{style['label']}</b><br>📏 {dist_km} km  ⏱ {time_min} min")
        if style["dash"]:
            kw["dash_array"] = style["dash"]
        folium.PolyLine(**kw).add_to(route_group)
        if coords:
            mid = coords[len(coords) // 2]
            folium.Marker(
                mid,
                icon=folium.DivIcon(
                    html=f"<div style='background:{style['color']};color:#000;"
                         f"padding:2px 6px;border-radius:4px;font-size:11px;"
                         f"font-weight:bold;white-space:nowrap;'>"
                         f"{style['label']}<br>{dist_km} km · {time_min} min</div>",
                    icon_size=(160, 36), icon_anchor=(80, 18)),
            ).add_to(route_group)
    route_group.add_to(m)

    # ── Endpoint Markers ─────────────────────────────────────────
    pin_group = folium.FeatureGroup(name="📍 Waypoints", show=True)
    if start_coords:
        folium.Marker(start_coords,
            icon=folium.Icon(color="green", icon="play", prefix="fa"),
            tooltip="<b>🟢 START</b>",
            popup=f"Start: {start_coords[0]:.5f}, {start_coords[1]:.5f}",
        ).add_to(pin_group)
    if end_coords:
        folium.Marker(end_coords,
            icon=folium.Icon(color="red", icon="flag", prefix="fa"),
            tooltip="<b>🔴 DESTINATION</b>",
            popup=f"End: {end_coords[0]:.5f}, {end_coords[1]:.5f}",
        ).add_to(pin_group)
    pin_group.add_to(m)

    folium.LayerControl(position="topright", collapsed=False).add_to(m)
    return m


# ═══════════════════════════════════════════════════════════════════
# STREAMLIT PAGE CONFIG & GLOBAL CSS
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Crisis Vector AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Inter:wght@400;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.cv-header{background:linear-gradient(135deg,#0a0a0a 0%,#0d1f1f 50%,#0a0a0a 100%);border-bottom:2px solid #00FFCC;padding:18px 32px;margin:-1rem -1rem 1rem -1rem;display:flex;align-items:center;gap:16px;}
.cv-logo{font-family:'Share Tech Mono',monospace;font-size:28px;color:#00FFCC;letter-spacing:3px;}
.cv-sub{font-size:12px;color:#7faaaa;letter-spacing:2px;text-transform:uppercase;}
.cv-ver{margin-left:auto;font-size:11px;color:#3a5a5a;font-family:monospace;}
.metric-card{background:#0d1f1f;border:1px solid #1a3a3a;border-radius:8px;padding:14px 18px;text-align:center;}
.metric-val{font-size:28px;font-weight:700;color:#00FFCC;font-family:'Share Tech Mono',monospace;}
.metric-lbl{font-size:11px;color:#7faaaa;text-transform:uppercase;letter-spacing:1px;margin-top:4px;}
.badge{display:inline-block;padding:3px 10px;border-radius:12px;font-size:11px;font-weight:600;letter-spacing:0.5px;}
.badge-green{background:#0a2a1a;color:#00FF99;border:1px solid #00FF99;}
.badge-red{background:#2a0a0a;color:#FF4444;border:1px solid #FF4444;}
.badge-yellow{background:#2a2a0a;color:#FFD700;border:1px solid #FFD700;}
.badge-blue{background:#0a1a2a;color:#1E90FF;border:1px solid #1E90FF;}
.badge-purple{background:#1a0a2a;color:#CC77FF;border:1px solid #CC77FF;}
.section-header{color:#00FFCC;font-size:13px;font-weight:600;text-transform:uppercase;letter-spacing:2px;border-bottom:1px solid #1a3a3a;padding-bottom:6px;margin:16px 0 10px 0;}
.sev-critical{background:#2a0000;border-left:4px solid #FF0000;color:#FF6666;padding:10px 14px;border-radius:4px;margin-bottom:8px;}
.sev-high{background:#2a1a00;border-left:4px solid #FF8C00;color:#FFB347;padding:10px 14px;border-radius:4px;margin-bottom:8px;}
.sev-medium{background:#1a1a00;border-left:4px solid #FFD700;color:#FFE066;padding:10px 14px;border-radius:4px;margin-bottom:8px;}
.sev-low{background:#001a0a;border-left:4px solid #00FF99;color:#00CC77;padding:10px 14px;border-radius:4px;margin-bottom:8px;}
.provider-badge{background:#0a0a1a;border:1px solid #3344aa;color:#7799ff;padding:3px 8px;border-radius:4px;font-size:11px;font-family:monospace;}
.instr-box{background:#0d1f1f;border:1px dashed #00FFCC55;border-radius:6px;padding:10px 14px;font-size:12px;color:#99bbbb;line-height:1.6;}
.rec-item{background:#0a1a0a;border-left:3px solid #00FF99;padding:7px 12px;border-radius:0 4px 4px 0;margin-bottom:6px;font-size:13px;color:#cceecc;}
.api-help{background:#0d0d1a;border:1px solid #2233aa44;border-radius:6px;padding:10px 14px;font-size:12px;color:#8899cc;margin-top:8px;}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════
def _init_state():
    defaults = {
        "engine":             CrisisVectorEngine(),
        "start_coords":       None,
        "end_coords":         None,
        "last_click_id":      None,
        "click_mode":         "🟢 Start Point",
        "obs_type":           list(DISASTER_TYPES.keys())[0],
        "obs_radius":         400,
        "routes_computed":    False,
        "analysis_result":    None,
        "show_baseline":      True,
        "ai_provider":        "gemini",
        "analysis_zones":     [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()
engine: CrisisVectorEngine = st.session_state.engine


# ═══════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="cv-header">
  <div>
    <div class="cv-logo">⚡ CRISIS VECTOR</div>
    <div class="cv-sub">AI-Powered Disaster Routing Platform</div>
  </div>
  <div class="cv-ver">v{APP_VERSION} &nbsp;|&nbsp; Pune, Maharashtra</div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# TOP METRICS BAR
# ═══════════════════════════════════════════════════════════════════
m0, m1, m2, m3, m4, m5 = st.columns(6)
with m0:
    net_status = "LOADED" if engine.G else "OFFLINE"
    badge_cls  = "badge-green" if engine.G else "badge-red"
    st.markdown(f"<div class='metric-card'><div class='metric-val'><span class='badge {badge_cls}'>{net_status}</span></div><div class='metric-lbl'>Network</div></div>", unsafe_allow_html=True)
with m1:
    st.markdown(f"<div class='metric-card'><div class='metric-val'>{len(engine.G.nodes) if engine.G else 0:,}</div><div class='metric-lbl'>Road Nodes</div></div>", unsafe_allow_html=True)
with m2:
    st.markdown(f"<div class='metric-card'><div class='metric-val'>{len(engine.G.edges) if engine.G else 0:,}</div><div class='metric-lbl'>Road Edges</div></div>", unsafe_allow_html=True)
with m3:
    st.markdown(f"<div class='metric-card'><div class='metric-val'>{len(engine.obstructions)}</div><div class='metric-lbl'>Active Hazards</div></div>", unsafe_allow_html=True)
with m4:
    st.markdown(f"<div class='metric-card'><div class='metric-val'>{engine.blocked_node_count():,}</div><div class='metric-lbl'>Blocked Nodes</div></div>", unsafe_allow_html=True)
with m5:
    n_routes = sum(1 for v in engine.routes.values() if v)
    st.markdown(f"<div class='metric-card'><div class='metric-val'>{n_routes}</div><div class='metric-lbl'>Routes Found</div></div>", unsafe_allow_html=True)

st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ═══════════════════════════════════════════════════════════════════
col_map, col_ctrl = st.columns([2.4, 1], gap="medium")

# ─────────────────────────── RIGHT PANEL ──────────────────────────
with col_ctrl:
    tab_routing, tab_analysis, tab_legend = st.tabs(
        ["🗺️ Routing", "🛰️ AI Analysis", "ℹ️ Legend"])

    # ══════════════ TAB 1: ROUTING ════════════════════════════════
    with tab_routing:
        st.markdown("<div class='section-header'>① Load Area Network</div>", unsafe_allow_html=True)
        preset_key = st.selectbox("Target Area", list(PRESET_LOCATIONS.keys()), label_visibility="collapsed")
        custom_loc = st.text_input("Or enter custom location:", placeholder="e.g. Koregaon Park, Pune")

        if st.button("🔄 Load Network", use_container_width=True, type="primary"):
            query, dist = PRESET_LOCATIONS[preset_key]
            if custom_loc.strip():
                query, dist = custom_loc.strip(), 2000
            with st.spinner(f"Fetching road network for **{query}** …"):
                try:
                    n, e = engine.load_network(query, dist)
                    st.session_state.start_coords    = None
                    st.session_state.end_coords      = None
                    st.session_state.routes_computed = False
                    st.success(f"✅ {n:,} nodes · {e:,} edges loaded")
                    st.rerun()
                except Exception as err:
                    st.error(f"❌ Failed: {err}")

        st.markdown("<div class='section-header'>② Map Click Tool</div>", unsafe_allow_html=True)
        st.markdown("<div class='instr-box'>Select a tool, then <b>click anywhere on the map</b>.</div>", unsafe_allow_html=True)

        click_mode = st.radio(
            "Active tool:",
            ["🟢 Start Point", "🔴 End Point", "🚧 Add Disaster Zone"],
            index=["🟢 Start Point", "🔴 End Point", "🚧 Add Disaster Zone"]
                  .index(st.session_state.click_mode),
        )
        st.session_state.click_mode = click_mode

        if "Disaster" in click_mode:
            st.session_state.obs_type   = st.selectbox("Disaster type:", list(DISASTER_TYPES.keys()),
                index=list(DISASTER_TYPES.keys()).index(st.session_state.obs_type))
            st.session_state.obs_radius = st.slider("Exclusion radius (m)", 100, 1200,
                st.session_state.obs_radius, step=50)

        sc = st.session_state.start_coords
        ec = st.session_state.end_coords
        c1, c2 = st.columns(2)
        c1.markdown(f"🟢 **Start**<br><small>{'Set ✓' if sc else 'Not set'}</small>", unsafe_allow_html=True)
        c2.markdown(f"🔴 **End**<br><small>{'Set ✓' if ec else 'Not set'}</small>",   unsafe_allow_html=True)

        if engine.obstructions:
            st.markdown("<div class='section-header'>Active Hazard Zones</div>", unsafe_allow_html=True)
            for i, obs in enumerate(engine.obstructions):
                cols = st.columns([4, 1])
                cols[0].markdown(f"<small>{obs['icon']} <b>{obs['label']}</b> · {obs['radius']} m</small>", unsafe_allow_html=True)
                if cols[1].button("✕", key=f"del_obs_{i}"):
                    engine.remove_obstruction(i)
                    st.session_state.routes_computed = False
                    st.rerun()

        col_clr, col_rst = st.columns(2)
        if col_clr.button("🗑️ Clear Hazards", use_container_width=True):
            engine.clear_obstructions()
            st.session_state.routes_computed = False
            st.rerun()
        if col_rst.button("🔁 Reset All", use_container_width=True):
            st.session_state.start_coords    = None
            st.session_state.end_coords      = None
            engine.clear_obstructions()
            st.session_state.routes_computed = False
            st.rerun()

        st.markdown("<div class='section-header'>③ Compute Routes</div>", unsafe_allow_html=True)
        st.session_state.show_baseline = st.checkbox("Show original (blocked) route", value=True)

        if st.button("⚡ Compute Safe Routes", type="primary", use_container_width=True,
                     disabled=not (engine.G and sc and ec)):
            with st.spinner("Running routing algorithms…"):
                try:
                    engine.set_endpoints(sc[0], sc[1], ec[0], ec[1])
                    engine.compute_routes()
                    st.session_state.routes_computed = True
                    st.rerun()
                except Exception as err:
                    st.error(f"❌ Routing error: {err}")

        if not engine.G:
            st.caption("⬆ Load a network first")
        elif not sc or not ec:
            st.caption("⬆ Place start & end points on the map")

        if st.session_state.routes_computed and engine.routes:
            st.markdown("<div class='section-header'>Route Comparison</div>", unsafe_allow_html=True)
            for key, style in ROUTE_STYLES.items():
                route = engine.routes.get(key)
                if route:
                    dist, mins = route_stats(engine.G, route)
                    st.markdown(
                        f"<div style='margin-bottom:8px;padding:8px 12px;background:#0d1f1f;"
                        f"border-radius:6px;border-left:3px solid {style['color']}'>"
                        f"<b style='color:{style['color']}'>{style['label']}</b><br>"
                        f"<small>📏 {dist} km &nbsp; ⏱ {mins} min &nbsp; 📍 {len(route)} nodes</small>"
                        f"</div>", unsafe_allow_html=True)
                else:
                    st.markdown(
                        f"<div style='margin-bottom:6px;padding:8px 12px;background:#0d0d0d;"
                        f"border-radius:6px;border-left:3px solid #444;'>"
                        f"<b style='color:#555'>{style['label']}</b> &nbsp;"
                        f"<span class='badge badge-red'>BLOCKED</span></div>",
                        unsafe_allow_html=True)

    # ══════════════ TAB 2: AI ANALYSIS ════════════════════════════
    with tab_analysis:
        st.markdown("<div class='section-header'>🛰️ Satellite / Drone Image Analysis</div>", unsafe_allow_html=True)

        # ── Provider selector ────────────────────────────────────
        provider_labels = {
            "gemini":       "🌟 Google Gemini 1.5 Flash  (Recommended — Free)",
            "huggingface":  "🤗 Hugging Face BLIP  (Free token)",
            "ollama":       "🖥️ Ollama Local  (No internet needed)",
        }
        chosen_label = st.selectbox("AI Provider:", list(provider_labels.values()))
        provider_key = {v: k for k, v in provider_labels.items()}[chosen_label]
        st.session_state.ai_provider = provider_key

        # ── Provider-specific credentials ────────────────────────
        api_key = hf_token = model_url = ollama_model = None

        if provider_key == "gemini":
            api_key = st.text_input("Google AI Studio API Key:", type="password",
                placeholder="AIza...",
                help="Free at aistudio.google.com/app/apikey — no credit card needed.")
            st.markdown("""<div class='api-help'>
            🔑 <b>How to get a free Gemini key (2 minutes):</b><br>
            1. Go to <b>aistudio.google.com/app/apikey</b><br>
            2. Click "Create API key" → copy it<br>
            3. Free tier: 15 req/min · 1,500 req/day · No billing needed
            </div>""", unsafe_allow_html=True)

        elif provider_key == "huggingface":
            hf_token = st.text_input("Hugging Face Token:", type="password",
                placeholder="hf_...",
                help="Free at huggingface.co/settings/tokens")
            hf_model_name = st.selectbox("Model:", list(HF_MODELS.keys()))
            model_url = HF_MODELS[hf_model_name]
            st.markdown("""<div class='api-help'>
            🔑 <b>How to get a free HF token:</b><br>
            1. Sign up at <b>huggingface.co</b> (free)<br>
            2. Go to Settings → Access Tokens → New token<br>
            3. Select "Read" role → copy token<br>
            ⚠️ First request may take 20–40 s (cold start)
            </div>""", unsafe_allow_html=True)

        elif provider_key == "ollama":
            ollama_model = st.text_input("Ollama model name:", value="llava",
                help="Must have vision capability: llava, bakllava, llava-phi3")
            st.markdown("""<div class='api-help'>
            🖥️ <b>Setup Ollama (one-time):</b><br>
            1. Install: <b>ollama.com/download</b><br>
            2. Pull model: <code>ollama pull llava</code><br>
            3. Start server: <code>ollama serve</code><br>
            ✅ Fully local — no API key, no data sent online
            </div>""", unsafe_allow_html=True)

        # ── Image upload ─────────────────────────────────────────
        st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Upload disaster image (JPG / PNG):",
            type=["jpg", "jpeg", "png"],
            help="Drone footage, satellite imagery, or ground-level disaster photos.",
        )

        if uploaded:
            st.image(uploaded, use_container_width=True, caption="📷 Uploaded feed")

        # ── Analyse button ────────────────────────────────────────
        can_analyse = uploaded and (
            (provider_key == "gemini"       and api_key)  or
            (provider_key == "huggingface"  and hf_token) or
            (provider_key == "ollama")
        )

        if st.button("🧠 Analyse Disaster Image", type="primary",
                     use_container_width=True, disabled=not can_analyse):
            if not uploaded:
                st.warning("Upload an image first.")
            else:
                with st.spinner("AI processing visual feed…"):
                    kwargs = {}
                    if provider_key == "gemini":
                        kwargs = {"api_key": api_key}
                    elif provider_key == "huggingface":
                        kwargs = {"hf_token": hf_token, "model_url": model_url}
                    elif provider_key == "ollama":
                        kwargs = {"model_name": ollama_model}

                    result = analyze_disaster_image(uploaded, provider_key, **kwargs)
                    st.session_state.analysis_result = result
                    st.rerun()

        if not can_analyse and not uploaded:
            st.caption("⬆ Upload an image and enter credentials to analyse.")

        # ── Display results ───────────────────────────────────────
        res = st.session_state.analysis_result
        if res:
            st.markdown("---")
            if not res["success"]:
                st.error(f"**Analysis failed:** {res.get('error', 'Unknown error')}")
                st.markdown("""
**Troubleshooting tips:**
- For Gemini: verify your key at aistudio.google.com
- For HF: check token at huggingface.co/settings/tokens
- For Ollama: confirm `ollama serve` is running in a terminal
                """)
            else:
                # Provider badge
                st.markdown(
                    f"<span class='provider-badge'>📡 Powered by {res['provider']}</span>",
                    unsafe_allow_html=True)

                # Severity banner
                sev = res["severity"]
                sev_cls = {"CRITICAL": "sev-critical", "HIGH": "sev-high",
                           "MEDIUM": "sev-medium", "LOW": "sev-low"}.get(sev, "sev-low")
                sev_icons = {"CRITICAL": "🚨", "HIGH": "⚠️", "MEDIUM": "⚡", "LOW": "ℹ️"}
                st.markdown(
                    f"<div class='{sev_cls}'>"
                    f"<b>{sev_icons.get(sev,'⚠️')} THREAT LEVEL: {sev}</b></div>",
                    unsafe_allow_html=True)

                # Caption
                st.markdown("**📝 Scene Description:**")
                st.info(res["caption"])

                # Detected hazards
                st.markdown("**🔴 Detected Hazard Types:**")
                badges = " &nbsp; ".join(
                    f"<span class='badge badge-red'>{h}</span>"
                    for h in res["hazards"])
                st.markdown(badges, unsafe_allow_html=True)

                # Visible hazards (Gemini only)
                if res.get("hazards_visible"):
                    st.markdown("**👁️ Visible Hazards:**")
                    for hv in res["hazards_visible"]:
                        st.markdown(f"• {hv}")

                # Tactical recommendations
                st.markdown("**🗺️ Evacuation & Routing Recommendations:**")
                for i, rec in enumerate(res["recommendations"], 1):
                    st.markdown(
                        f"<div class='rec-item'><b>{i}.</b> {rec}</div>",
                        unsafe_allow_html=True)

                # Action buttons
                col_a, col_b = st.columns(2)
                if col_a.button("🗑️ Clear Result", use_container_width=True):
                    st.session_state.analysis_result = None
                    st.rerun()
                if col_b.button("📋 Copy to Clipboard ↓", use_container_width=True):
                    report = (
                        f"CRISIS VECTOR AI — ANALYSIS REPORT\n"
                        f"Provider: {res['provider']}\n"
                        f"Threat Level: {res['severity']}\n"
                        f"Scene: {res['caption']}\n"
                        f"Hazards: {', '.join(res['hazards'])}\n\n"
                        f"RECOMMENDATIONS:\n" +
                        "\n".join(f"{i}. {r}" for i, r in enumerate(res["recommendations"], 1))
                    )
                    st.code(report, language=None)

    # ══════════════ TAB 3: LEGEND ═════════════════════════════════
    with tab_legend:
        st.markdown("<div class='section-header'>Route Legend</div>", unsafe_allow_html=True)
        for key, style in ROUTE_STYLES.items():
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:8px'>"
                f"<div style='width:40px;height:4px;background:{style['color']};border-radius:2px'></div>"
                f"<span style='color:{style['color']};font-weight:600'>{style['label']}</span>"
                f"</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-header'>Disaster Zone Types</div>", unsafe_allow_html=True)
        for dtype, meta in DISASTER_TYPES.items():
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:6px'>"
                f"<div style='width:14px;height:14px;border-radius:50%;background:{meta['color']}'></div>"
                f"{meta['icon']} <b>{dtype}</b> — default {meta['radius']} m"
                f"</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-header'>AI Providers</div>", unsafe_allow_html=True)
        st.markdown("""
| Provider | Cost | Best For |
|---|---|---|
| Gemini Flash | Free (1500/day) | Best accuracy |
| HF BLIP | Free token | Image captioning |
| Ollama | Local / free | Privacy-first |
""")

        st.markdown("<div class='section-header'>How It Works</div>", unsafe_allow_html=True)
        st.markdown("""
1. **Load Network** — real OSM road data via OSMnx  
2. **Place Hazards** — click map to mark disaster zones  
3. **Set Endpoints** — click Start / End  
4. **Compute Routes** — three algorithms:
   - **Baseline**: shortest time (ignores hazards)
   - **Emergency**: Dijkstra on trimmed graph
   - **Safest**: A\\* with proximity penalty scores  
5. **AI Analysis** — upload satellite/drone image for tactical assessment
""")


# ─────────────────────────── LEFT PANEL (MAP) ─────────────────────
with col_map:
    m_folium = build_map(
        engine,
        st.session_state.start_coords,
        st.session_state.end_coords,
        show_baseline=st.session_state.get("show_baseline", True),
        analysis_zones=st.session_state.get("analysis_zones", []),
    )

    map_data = st_folium(
        m_folium,
        width="100%",
        height=680,
        returned_objects=["last_clicked"],
        key="main_map",
    )

    # ── Process map clicks ────────────────────────────────────────
    if map_data and map_data.get("last_clicked"):
        click    = map_data["last_clicked"]
        click_id = hashlib.md5(
            f"{click['lat']:.6f}_{click['lng']:.6f}_{st.session_state.click_mode}".encode()
        ).hexdigest()

        if click_id != st.session_state.last_click_id:
            st.session_state.last_click_id = click_id
            lat, lon = click["lat"], click["lng"]
            mode = st.session_state.click_mode

            if "Start" in mode:
                st.session_state.start_coords    = [lat, lon]
                st.session_state.routes_computed = False
                st.toast(f"🟢 Start set: {lat:.4f}, {lon:.4f}")
            elif "End" in mode:
                st.session_state.end_coords      = [lat, lon]
                st.session_state.routes_computed = False
                st.toast(f"🔴 End set: {lat:.4f}, {lon:.4f}")
            elif "Disaster" in mode:
                if engine.G:
                    engine.add_obstruction(lat, lon,
                        st.session_state.obs_radius, st.session_state.obs_type)
                    st.session_state.routes_computed = False
                    meta = DISASTER_TYPES[st.session_state.obs_type]
                    st.toast(f"{meta['icon']} {meta['label']} added at {lat:.4f}, {lon:.4f}")
                else:
                    st.toast("⚠️ Load a network before placing hazard zones!", icon="⚠️")
            st.rerun()

    # ── Status bar ────────────────────────────────────────────────
    status_parts = []
    if engine.G:
        status_parts.append(f"🗺 **{engine.location_name}** loaded")
    if engine.blocked_node_count():
        status_parts.append(f"🚫 {engine.blocked_node_count()} nodes blocked ({engine.coverage_percent()}%)")
    if st.session_state.routes_computed:
        n = sum(1 for v in engine.routes.values() if v)
        status_parts.append(f"⚡ {n}/3 routes computed")
    if st.session_state.analysis_result and st.session_state.analysis_result.get("success"):
        res = st.session_state.analysis_result
        status_parts.append(f"🛰 AI: {res['severity']} · {', '.join(res['hazards'][:2])}")
    st.caption("  ·  ".join(status_parts) if status_parts else "Load a network to begin")
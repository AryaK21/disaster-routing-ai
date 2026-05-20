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
    "🌊 Flood":      {"color": "#38BDF8", "fill": "#38BDF8", "radius": 600,  "label": "FLOOD ZONE",    "speed_factor": 0.0, "icon": "💧", "accent": "#0EA5E9"},
    "🔥 Fire":       {"color": "#FB923C", "fill": "#FB923C", "radius": 400,  "label": "FIRE ZONE",     "speed_factor": 0.0, "icon": "🔥", "accent": "#F97316"},
    "🏚️ Collapse":   {"color": "#A78BFA", "fill": "#8B5CF6", "radius": 300,  "label": "DEBRIS ZONE",   "speed_factor": 0.0, "icon": "🏚️", "accent": "#7C3AED"},
    "💣 Explosion":  {"color": "#FBBF24", "fill": "#F59E0B", "radius": 350,  "label": "BLAST ZONE",    "speed_factor": 0.0, "icon": "💥", "accent": "#D97706"},
    "☣️ Hazmat":     {"color": "#4ADE80", "fill": "#22C55E", "radius": 500,  "label": "HAZMAT ZONE",   "speed_factor": 0.0, "icon": "☣️", "accent": "#16A34A"},
    "🌪️ Tornado":    {"color": "#C084FC", "fill": "#A855F7", "radius": 450,  "label": "TORNADO ZONE",  "speed_factor": 0.0, "icon": "🌪️", "accent": "#9333EA"},
}

ROUTE_STYLES = {
    "baseline":  {"color": "#64748B", "weight": 4, "opacity": 0.6, "dash": "10 5",   "label": "Original Route"},
    "emergency": {"color": "#22D3EE", "weight": 5, "opacity": 0.9, "dash": None,     "label": "Emergency Detour"},
    "safest":    {"color": "#F0CF65", "weight": 6, "opacity": 1.0, "dash": None,     "label": "Safest Path"},
}

HF_API_BASE = "https://api-inference.huggingface.co/models"
HF_MODELS = {
    "BLIP Large (Best for disasters)":  f"{HF_API_BASE}/Salesforce/blip-image-captioning-large",
    "BLIP Base (Faster)":               f"{HF_API_BASE}/Salesforce/blip-image-captioning-base",
    "ViT-GPT2 Scene":                   f"{HF_API_BASE}/nlpconnect/vit-gpt2-image-captioning",
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
    image_file.seek(0)
    return base64.standard_b64encode(image_file.read()).decode("utf-8")


def resize_image_for_api(image_file, max_size: int = 1024) -> bytes:
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
# AI IMAGE ANALYSIS
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
        recs = list(dict.fromkeys(recs))
        recs.extend(GENERIC_RECS[-2:])
    return {
        "success":    True,
        "caption":    caption,
        "severity":   severity,
        "hazards":    detected if detected else ["GENERAL DISASTER"],
        "recommendations": recs,
        "provider":   "unknown",
    }


def analyze_with_huggingface(image_file, hf_token: str, model_url: str) -> dict:
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
            return {"success": False, "error": "Invalid HF token."}
        elif resp.status_code == 404:
            return {"success": False, "error": "Model endpoint not found (404)."}
        else:
            return {"success": False, "error": f"HF API error {resp.status_code}: {resp.text[:300]}"}
    else:
        return {"success": False, "error": "Model still loading after 4 attempts."}
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
        return {"success": False, "error": "Model returned an empty caption."}
    result = _parse_caption(caption)
    result["provider"] = "Hugging Face"
    return result


def analyze_with_gemini(image_file, api_key: str) -> dict:
    try:
        import google.generativeai as genai
    except ImportError:
        return {"success": False, "error": "google-generativeai not installed."}
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
        for h in detected:
            recs.extend(EVAC_TEMPLATES.get(h, []))
        recs = list(dict.fromkeys(recs))[:8]
        return {
            "success":          True,
            "caption":          caption,
            "severity":         severity,
            "hazards":          detected if detected else ["GENERAL DISASTER"],
            "hazards_visible":  hazards,
            "recommendations":  recs,
            "provider":         "Google Gemini 2.5 Flash",
        }
    except json.JSONDecodeError:
        caption = response.text if hasattr(response, "text") else "Unable to parse."
        result  = _parse_caption(caption)
        result["provider"] = "Google Gemini 2.5 Flash"
        return result
    except Exception as e:
        err = str(e)
        if "API_KEY_INVALID" in err or "API key not valid" in err:
            return {"success": False, "error": "Invalid Gemini API key."}
        return {"success": False, "error": f"Gemini error: {err}"}


def analyze_with_ollama(image_file, model_name: str = "llava") -> dict:
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
            return {"success": False, "error": f"Ollama error {resp.status_code}"}
        caption = resp.json().get("response", "")
        result  = _parse_caption(caption)
        result["provider"] = f"Ollama ({model_name})"
        return result
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Cannot connect to Ollama. Make sure it's running: `ollama serve`"}
    except Exception as e:
        return {"success": False, "error": f"Ollama error: {e}"}


def analyze_disaster_image(image_file, provider: str, **kwargs) -> dict:
    if provider == "huggingface":
        return analyze_with_huggingface(image_file, kwargs["hf_token"], kwargs["model_url"])
    elif provider == "gemini":
        return analyze_with_gemini(image_file, kwargs["api_key"])
    elif provider == "ollama":
        return analyze_with_ollama(image_file, kwargs.get("model_name", "llava"))
    return {"success": False, "error": f"Unknown provider: {provider}"}


# ═══════════════════════════════════════════════════════════════════
# MAP BUILDER
# ═══════════════════════════════════════════════════════════════════
def build_map(engine, start_coords, end_coords, show_baseline=True, analysis_zones=None):
    clat, clon = engine.map_centre()
    m = folium.Map(location=[clat, clon], zoom_start=14, tiles="CartoDB dark_matter")
    Fullscreen(position="topright").add_to(m)
    MiniMap(toggle_display=True, position="bottomright").add_to(m)
    LocateControl(auto_start=False).add_to(m)

    if not engine.G:
        return m

    zone_group = folium.FeatureGroup(name="⚠️ Disaster Zones", show=True)
    for i, obs in enumerate(engine.obstructions):
        folium.Circle(
            location=[obs["lat"], obs["lon"]], radius=obs["radius"],
            color=obs["color"], fill=True, fill_color=obs["fill"], fill_opacity=0.35,
            tooltip=f"<b>{obs['label']}</b><br>Radius: {obs['radius']} m",
            popup=folium.Popup(
                f"<b style='color:{obs['color']}'>{obs['icon']} {obs['label']}</b><br>"
                f"Centre: {obs['lat']:.5f}, {obs['lon']:.5f}<br>"
                f"Blocked radius: {obs['radius']} m", max_width=220),
        ).add_to(zone_group)
        folium.Circle(
            location=[obs["lat"], obs["lon"]], radius=obs["radius"] * 2.5,
            color=obs["color"], fill=False, dash_array="8 6", opacity=0.25,
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

    if analysis_zones:
        ai_group = folium.FeatureGroup(name="🤖 AI-Detected Zones", show=True)
        for az in analysis_zones:
            folium.Circle(
                location=[az["lat"], az["lon"]], radius=az["radius"],
                color="#FF00FF", fill=True, fill_color="#FF00FF", fill_opacity=0.20,
                tooltip=f"<b>AI DETECTED: {az['label']}</b>",
                dash_array="5 3",
            ).add_to(ai_group)
        ai_group.add_to(m)

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
                         f"padding:2px 8px;border-radius:20px;font-size:11px;"
                         f"font-weight:700;white-space:nowrap;letter-spacing:0.3px'>"
                         f"{style['label']} · {dist_km}km · {time_min}min</div>",
                    icon_size=(200, 28), icon_anchor=(100, 14)),
            ).add_to(route_group)
    route_group.add_to(m)

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

# ── Master CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&family=Bebas+Neue&display=swap');

/* ── Reset & Base ─────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
  font-family: 'Space Grotesk', sans-serif;
  background: #060B14;
  color: #C8D8E8;
}

/* Remove streamlit default padding */
.block-container { padding: 0 !important; max-width: 100% !important; }
.stApp { background: #060B14; }
header[data-testid="stHeader"] { background: transparent; }

/* ── Header ───────────────────────────────────────────── */
.cv-masthead {
  background: linear-gradient(180deg, #08111F 0%, #060B14 100%);
  border-bottom: 1px solid rgba(34,211,238,0.15);
  padding: 14px 28px 14px 28px;
  display: flex;
  align-items: center;
  gap: 20px;
  margin-bottom: 0;
  position: relative;
  overflow: hidden;
}
.cv-masthead::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, #22D3EE 40%, #F0CF65 60%, transparent);
}
.cv-logo-text {
  font-family: 'Bebas Neue', sans-serif;
  font-size: 36px;
  letter-spacing: 4px;
  color: #E2F4FE;
  line-height: 1;
}
.cv-logo-accent { color: #22D3EE; }
.cv-tagline {
  font-size: 10px;
  font-weight: 500;
  letter-spacing: 3px;
  text-transform: uppercase;
  color: #4A7A8A;
  margin-top: 3px;
}
.cv-version {
  margin-left: auto;
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  color: #2A4A5A;
  letter-spacing: 1px;
}
.cv-status-dot {
  width: 7px; height: 7px;
  border-radius: 50%;
  background: #22D3EE;
  box-shadow: 0 0 8px #22D3EE;
  animation: pulse-dot 2s ease-in-out infinite;
  display: inline-block;
  margin-right: 6px;
}
@keyframes pulse-dot {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.5; transform: scale(0.8); }
}

/* ── Metric Cards ─────────────────────────────────────── */
.metric-strip {
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: 1px;
  background: rgba(34,211,238,0.06);
  border-bottom: 1px solid rgba(34,211,238,0.1);
  padding: 0;
  margin-bottom: 0;
}
.metric-cell {
  background: #080F1C;
  padding: 14px 16px 12px;
  text-align: center;
  position: relative;
  transition: background 0.2s;
}
.metric-cell:hover { background: #0C1828; }
.metric-cell::after {
  content: '';
  position: absolute;
  bottom: 0; left: 20%; right: 20%;
  height: 1px;
  background: rgba(34,211,238,0.15);
}
.metric-num {
  font-family: 'JetBrains Mono', monospace;
  font-size: 22px;
  font-weight: 700;
  color: #22D3EE;
  line-height: 1;
  margin-bottom: 5px;
}
.metric-num.danger { color: #FB7185; }
.metric-num.warn { color: #F0CF65; }
.metric-num.ok { color: #4ADE80; }
.metric-lbl {
  font-size: 9px;
  font-weight: 600;
  letter-spacing: 2px;
  text-transform: uppercase;
  color: #2A5060;
}
.net-online {
  display: inline-block;
  background: rgba(74,222,128,0.12);
  color: #4ADE80;
  border: 1px solid rgba(74,222,128,0.3);
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 1px;
  padding: 2px 10px;
  border-radius: 20px;
}
.net-offline {
  display: inline-block;
  background: rgba(251,113,133,0.12);
  color: #FB7185;
  border: 1px solid rgba(251,113,133,0.3);
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  font-weight: 700;
  padding: 2px 10px;
  border-radius: 20px;
}

/* ── Layout ───────────────────────────────────────────── */
.main-layout {
  display: grid;
  grid-template-columns: 1fr 380px;
  height: calc(100vh - 130px);
  overflow: hidden;
}

/* ── Panel (right sidebar) ────────────────────────────── */
.cv-panel {
  background: #07101E;
  border-left: 1px solid rgba(34,211,238,0.1);
  overflow-y: auto;
  padding: 0;
  scrollbar-width: thin;
  scrollbar-color: #1A3A4A transparent;
}
.cv-panel::-webkit-scrollbar { width: 4px; }
.cv-panel::-webkit-scrollbar-track { background: transparent; }
.cv-panel::-webkit-scrollbar-thumb { background: #1A3A4A; border-radius: 2px; }

/* ── Streamlit Tab Overrides ──────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
  background: #060B14;
  border-bottom: 1px solid rgba(34,211,238,0.12);
  gap: 0;
  padding: 0 16px;
}
.stTabs [data-baseweb="tab"] {
  color: #3A6070;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  padding: 12px 16px;
  border-bottom: 2px solid transparent;
  transition: all 0.2s;
  background: transparent;
}
.stTabs [aria-selected="true"] {
  color: #22D3EE !important;
  border-bottom: 2px solid #22D3EE !important;
  background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] {
  padding: 16px;
  background: transparent;
}

/* ── Section Labels ───────────────────────────────────── */
.sec-label {
  font-size: 9px;
  font-weight: 700;
  letter-spacing: 3px;
  text-transform: uppercase;
  color: #22D3EE;
  opacity: 0.7;
  margin: 18px 0 10px 0;
  display: flex;
  align-items: center;
  gap: 8px;
}
.sec-label::after {
  content: '';
  flex: 1;
  height: 1px;
  background: rgba(34,211,238,0.15);
}

/* ── Streamlit Widgets ────────────────────────────────── */
.stSelectbox > div > div,
.stTextInput > div > div > input {
  background: #0C1828 !important;
  border: 1px solid rgba(34,211,238,0.15) !important;
  border-radius: 6px !important;
  color: #A0C0D0 !important;
  font-size: 13px !important;
}
.stSelectbox > div > div:hover,
.stTextInput > div > div > input:hover {
  border-color: rgba(34,211,238,0.35) !important;
}
.stTextInput > div > div > input:focus {
  border-color: #22D3EE !important;
  box-shadow: 0 0 0 2px rgba(34,211,238,0.1) !important;
}
.stSlider > div { padding: 4px 0 !important; }
.stSlider [data-baseweb="slider"] .rc-slider-rail { background: #1A3A4A !important; }
.stSlider [data-baseweb="slider"] .rc-slider-track { background: #22D3EE !important; }
.stSlider [data-baseweb="slider"] .rc-slider-handle {
  border-color: #22D3EE !important;
  background: #060B14 !important;
}
.stRadio > div { gap: 6px; }
.stRadio [data-testid="stMarkdownContainer"] { font-size: 13px; color: #7090A0; }
.stCheckbox > label { font-size: 13px; color: #7090A0; }

/* ── Primary Button ───────────────────────────────────── */
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #0F4A5A 0%, #0C3A4A 100%) !important;
  border: 1px solid rgba(34,211,238,0.4) !important;
  color: #22D3EE !important;
  font-family: 'Space Grotesk', sans-serif !important;
  font-size: 12px !important;
  font-weight: 600 !important;
  letter-spacing: 1.5px !important;
  text-transform: uppercase !important;
  border-radius: 6px !important;
  padding: 10px 16px !important;
  transition: all 0.2s !important;
  position: relative !important;
  overflow: hidden !important;
}
.stButton > button[kind="primary"]:hover {
  background: linear-gradient(135deg, #1A6A7A 0%, #125060 100%) !important;
  border-color: #22D3EE !important;
  box-shadow: 0 0 20px rgba(34,211,238,0.2) !important;
  transform: translateY(-1px) !important;
}
.stButton > button[kind="primary"]:active { transform: translateY(0) !important; }

/* ── Secondary Button ─────────────────────────────────── */
.stButton > button[kind="secondary"] {
  background: rgba(255,255,255,0.03) !important;
  border: 1px solid rgba(34,211,238,0.12) !important;
  color: #4A7080 !important;
  font-size: 11px !important;
  font-weight: 600 !important;
  letter-spacing: 1px !important;
  border-radius: 6px !important;
  transition: all 0.2s !important;
}
.stButton > button[kind="secondary"]:hover {
  background: rgba(34,211,238,0.06) !important;
  border-color: rgba(34,211,238,0.3) !important;
  color: #22D3EE !important;
}

/* ── Hazard Zone Pills ────────────────────────────────── */
.hazard-row {
  display: flex;
  align-items: center;
  gap: 8px;
  background: rgba(251,113,133,0.06);
  border: 1px solid rgba(251,113,133,0.15);
  border-radius: 6px;
  padding: 7px 10px;
  margin-bottom: 6px;
}
.hazard-icon { font-size: 16px; }
.hazard-info { flex: 1; }
.hazard-type { font-size: 11px; font-weight: 700; letter-spacing: 0.5px; color: #FB7185; }
.hazard-meta { font-size: 10px; color: #4A6070; margin-top: 1px; font-family: 'JetBrains Mono', monospace; }

/* ── Endpoint Status ──────────────────────────────────── */
.endpoint-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
  margin: 8px 0;
}
.endpoint-card {
  background: rgba(255,255,255,0.02);
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 6px;
  padding: 8px 10px;
  text-align: center;
}
.endpoint-card.set { border-color: rgba(74,222,128,0.25); background: rgba(74,222,128,0.05); }
.endpoint-dot { font-size: 14px; display: block; margin-bottom: 2px; }
.endpoint-status { font-size: 10px; font-weight: 600; letter-spacing: 1px; color: #4A6070; }
.endpoint-card.set .endpoint-status { color: #4ADE80; }

/* ── Route Comparison Cards ───────────────────────────── */
.route-card {
  border-radius: 6px;
  padding: 10px 14px;
  margin-bottom: 8px;
  border-left: 3px solid;
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.route-name { font-size: 12px; font-weight: 700; letter-spacing: 0.5px; }
.route-stats { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #4A7080; text-align: right; }
.route-blocked {
  border-radius: 6px;
  padding: 8px 14px;
  margin-bottom: 8px;
  background: rgba(255,255,255,0.02);
  border: 1px solid rgba(255,255,255,0.05);
  font-size: 11px;
  color: #3A5060;
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.badge-blocked {
  font-size: 9px;
  font-weight: 700;
  letter-spacing: 1px;
  color: #FB7185;
  background: rgba(251,113,133,0.1);
  border: 1px solid rgba(251,113,133,0.2);
  padding: 2px 6px;
  border-radius: 10px;
}

/* ── Instruction Box ──────────────────────────────────── */
.hint-box {
  background: rgba(34,211,238,0.04);
  border: 1px dashed rgba(34,211,238,0.2);
  border-radius: 6px;
  padding: 10px 12px;
  font-size: 12px;
  color: #4A7080;
  line-height: 1.6;
  margin-bottom: 12px;
}

/* ── AI Analysis UI ───────────────────────────────────── */
.provider-tag {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: rgba(34,211,238,0.08);
  border: 1px solid rgba(34,211,238,0.2);
  border-radius: 20px;
  padding: 4px 12px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  color: #22D3EE;
  margin-bottom: 14px;
}
.api-callout {
  background: rgba(240,207,101,0.05);
  border: 1px solid rgba(240,207,101,0.15);
  border-radius: 6px;
  padding: 10px 14px;
  font-size: 11px;
  color: #8A7040;
  margin-top: 8px;
  line-height: 1.6;
}
.api-callout b { color: #F0CF65; }
.api-callout code {
  font-family: 'JetBrains Mono', monospace;
  background: rgba(240,207,101,0.1);
  padding: 1px 4px;
  border-radius: 3px;
}

/* Severity Banner */
.sev-banner {
  border-radius: 8px;
  padding: 14px 16px;
  margin-bottom: 14px;
  position: relative;
  overflow: hidden;
}
.sev-banner::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0; bottom: 0;
  background: repeating-linear-gradient(
    45deg, transparent, transparent 8px,
    rgba(255,255,255,0.015) 8px, rgba(255,255,255,0.015) 9px
  );
}
.sev-critical { background: rgba(239,68,68,0.12); border: 1px solid rgba(239,68,68,0.35); }
.sev-high     { background: rgba(251,146,60,0.10); border: 1px solid rgba(251,146,60,0.30); }
.sev-medium   { background: rgba(234,179,8,0.08);  border: 1px solid rgba(234,179,8,0.25); }
.sev-low      { background: rgba(74,222,128,0.08);  border: 1px solid rgba(74,222,128,0.25); }

.sev-level-label {
  font-family: 'Bebas Neue', sans-serif;
  font-size: 26px;
  letter-spacing: 4px;
  line-height: 1;
}
.sev-critical .sev-level-label { color: #EF4444; }
.sev-high     .sev-level-label { color: #FB923C; }
.sev-medium   .sev-level-label { color: #EAB308; }
.sev-low      .sev-level-label { color: #4ADE80; }

.sev-sublabel {
  font-size: 9px;
  font-weight: 600;
  letter-spacing: 2.5px;
  text-transform: uppercase;
  opacity: 0.5;
  margin-top: 2px;
}

/* Hazard Chips */
.hazard-chip {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  background: rgba(251,113,133,0.1);
  border: 1px solid rgba(251,113,133,0.25);
  color: #FB7185;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 1px;
  padding: 4px 10px;
  border-radius: 20px;
  margin: 0 4px 4px 0;
}

/* Recommendation Items */
.rec-item {
  display: flex;
  gap: 10px;
  align-items: flex-start;
  background: rgba(74,222,128,0.04);
  border-left: 2px solid rgba(74,222,128,0.3);
  border-radius: 0 6px 6px 0;
  padding: 8px 12px;
  margin-bottom: 6px;
  font-size: 12px;
  color: #90B090;
  line-height: 1.5;
}
.rec-num {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  font-weight: 700;
  color: #4ADE80;
  min-width: 18px;
  margin-top: 1px;
}

/* Scene Description */
.scene-desc {
  background: rgba(34,211,238,0.04);
  border: 1px solid rgba(34,211,238,0.12);
  border-radius: 6px;
  padding: 12px 14px;
  font-size: 12px;
  color: #7AAABB;
  line-height: 1.65;
  font-style: italic;
  margin-bottom: 14px;
}

/* Visible Hazard Pills */
.vis-hazard {
  display: inline-block;
  background: rgba(240,207,101,0.08);
  border: 1px solid rgba(240,207,101,0.2);
  color: #D4A830;
  font-size: 10px;
  font-weight: 600;
  padding: 3px 8px;
  border-radius: 4px;
  margin: 0 4px 4px 0;
}

/* ── Legend ───────────────────────────────────────────── */
.legend-route {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 10px;
  padding: 8px 10px;
  border-radius: 6px;
  background: rgba(255,255,255,0.02);
}
.legend-line {
  width: 36px;
  height: 3px;
  border-radius: 2px;
  flex-shrink: 0;
}
.legend-name { font-size: 12px; font-weight: 600; }
.legend-zone {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 8px;
  padding: 6px 10px;
  border-radius: 6px;
}
.zone-dot { width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; }

/* ── Status Bar ───────────────────────────────────────── */
.status-bar {
  background: rgba(6,11,20,0.95);
  border-top: 1px solid rgba(34,211,238,0.08);
  padding: 7px 16px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  color: #2A5060;
  letter-spacing: 0.5px;
}
.status-bar span { color: #22D3EE; margin-right: 2px; }

/* ── Streamlit Caption ────────────────────────────────── */
.stCaption { font-family: 'JetBrains Mono', monospace !important; font-size: 10px !important; color: #2A5060 !important; }

/* Scrollable map column */
[data-testid="column"]:first-child { overflow: hidden; }

/* ── Chart Container ──────────────────────────────────── */
.chart-wrapper {
  background: rgba(255,255,255,0.02);
  border: 1px solid rgba(34,211,238,0.08);
  border-radius: 8px;
  padding: 14px;
  margin-bottom: 14px;
}
.chart-title {
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 2px;
  text-transform: uppercase;
  color: #2A6070;
  margin-bottom: 10px;
}
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
net_ok = engine.G is not None
st.markdown(f"""
<div class="cv-masthead">
  <div>
    <div class="cv-logo-text">⚡ <span class="cv-logo-accent">CRISIS</span> VECTOR</div>
    <div class="cv-tagline">AI-Powered Disaster Routing Platform &nbsp;·&nbsp; Pune, MH</div>
  </div>
  <div style="margin-left: 24px; display: flex; align-items: center; gap: 8px;">
    <span class="cv-status-dot"></span>
    <span style="font-size:11px; color:#2A6070; font-family:'JetBrains Mono',monospace;">
      SYSTEM {'NOMINAL' if net_ok else 'STANDBY'}
    </span>
  </div>
  <div class="cv-version">v{APP_VERSION} &nbsp;|&nbsp; CRISIS VECTOR AI</div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# METRICS STRIP
# ═══════════════════════════════════════════════════════════════════
net_label = f"<span class='net-online'>ONLINE</span>" if engine.G else "<span class='net-offline'>OFFLINE</span>"
nodes_cls = "metric-num ok" if engine.G else "metric-num"
blocked_pct = engine.coverage_percent()
blocked_cls = "metric-num danger" if blocked_pct > 20 else ("metric-num warn" if blocked_pct > 5 else "metric-num")
n_routes = sum(1 for v in engine.routes.values() if v)
routes_cls = "metric-num ok" if n_routes == 3 else ("metric-num warn" if n_routes > 0 else "metric-num danger" if st.session_state.routes_computed else "metric-num")

st.markdown(f"""
<div class="metric-strip">
  <div class="metric-cell">
    <div class="metric-num">{net_label}</div>
    <div class="metric-lbl">Network</div>
  </div>
  <div class="metric-cell">
    <div class="{nodes_cls}">{len(engine.G.nodes) if engine.G else 0:,}</div>
    <div class="metric-lbl">Road Nodes</div>
  </div>
  <div class="metric-cell">
    <div class="metric-num">{len(engine.G.edges) if engine.G else 0:,}</div>
    <div class="metric-lbl">Road Edges</div>
  </div>
  <div class="metric-cell">
    <div class="metric-num {'warn' if engine.obstructions else ''}">{len(engine.obstructions)}</div>
    <div class="metric-lbl">Active Hazards</div>
  </div>
  <div class="metric-cell">
    <div class="{blocked_cls}">{engine.blocked_node_count():,}</div>
    <div class="metric-lbl">Blocked Nodes</div>
  </div>
  <div class="metric-cell">
    <div class="{routes_cls}">{n_routes}/3</div>
    <div class="metric-lbl">Routes Found</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ═══════════════════════════════════════════════════════════════════
col_map, col_ctrl = st.columns([2.5, 1], gap="small")

# ─────────────────────────── RIGHT PANEL ──────────────────────────
with col_ctrl:
    tab_routing, tab_analysis, tab_legend = st.tabs(["Routing", "AI Analysis", "Legend"])

    # ══════════════ TAB 1: ROUTING ════════════════════════════════
    with tab_routing:
        st.markdown("<div class='sec-label'>① Area Network</div>", unsafe_allow_html=True)
        preset_key = st.selectbox("Target Area", list(PRESET_LOCATIONS.keys()), label_visibility="collapsed")
        custom_loc = st.text_input("Custom location", placeholder="e.g. Koregaon Park, Pune", label_visibility="collapsed")

        if st.button("⟳  Load Road Network", use_container_width=True, type="primary"):
            query, dist = PRESET_LOCATIONS[preset_key]
            if custom_loc.strip():
                query, dist = custom_loc.strip(), 2000
            with st.spinner(f"Fetching OSM data for {query} …"):
                try:
                    n, e = engine.load_network(query, dist)
                    st.session_state.start_coords    = None
                    st.session_state.end_coords      = None
                    st.session_state.routes_computed = False
                    st.success(f"✓ {n:,} nodes · {e:,} edges")
                    st.rerun()
                except Exception as err:
                    st.error(f"Failed: {err}")

        st.markdown("<div class='sec-label'>② Map Interaction</div>", unsafe_allow_html=True)
        st.markdown("<div class='hint-box'>Select a tool mode, then <b>click anywhere on the map</b> to place it.</div>", unsafe_allow_html=True)

        click_mode = st.radio(
            "Tool",
            ["🟢 Start Point", "🔴 End Point", "🚧 Add Disaster Zone"],
            index=["🟢 Start Point", "🔴 End Point", "🚧 Add Disaster Zone"].index(st.session_state.click_mode),
            label_visibility="collapsed",
        )
        st.session_state.click_mode = click_mode

        if "Disaster" in click_mode:
            st.session_state.obs_type   = st.selectbox("Type", list(DISASTER_TYPES.keys()),
                index=list(DISASTER_TYPES.keys()).index(st.session_state.obs_type),
                label_visibility="collapsed")
            st.session_state.obs_radius = st.slider("Exclusion radius (m)", 100, 1200,
                st.session_state.obs_radius, step=50)

        sc = st.session_state.start_coords
        ec = st.session_state.end_coords
        st.markdown(f"""
        <div class="endpoint-grid">
          <div class="endpoint-card {'set' if sc else ''}">
            <span class="endpoint-dot">🟢</span>
            <div class="endpoint-status">{'SET ✓' if sc else 'NOT SET'}</div>
          </div>
          <div class="endpoint-card {'set' if ec else ''}">
            <span class="endpoint-dot">🔴</span>
            <div class="endpoint-status">{'SET ✓' if ec else 'NOT SET'}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Active hazard zones
        if engine.obstructions:
            st.markdown("<div class='sec-label'>Active Zones</div>", unsafe_allow_html=True)
            for i, obs in enumerate(engine.obstructions):
                col_info, col_del = st.columns([5, 1])
                with col_info:
                    st.markdown(f"""
                    <div class="hazard-row">
                      <span class="hazard-icon">{obs['icon']}</span>
                      <div class="hazard-info">
                        <div class="hazard-type">{obs['label']}</div>
                        <div class="hazard-meta">r={obs['radius']}m · {obs['lat']:.3f},{obs['lon']:.3f}</div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_del:
                    if st.button("✕", key=f"del_obs_{i}"):
                        engine.remove_obstruction(i)
                        st.session_state.routes_computed = False
                        st.rerun()

        col_clr, col_rst = st.columns(2)
        if col_clr.button("Clear Hazards", use_container_width=True):
            engine.clear_obstructions()
            st.session_state.routes_computed = False
            st.rerun()
        if col_rst.button("Reset All", use_container_width=True):
            st.session_state.start_coords    = None
            st.session_state.end_coords      = None
            engine.clear_obstructions()
            st.session_state.routes_computed = False
            st.rerun()

        st.markdown("<div class='sec-label'>③ Compute Routes</div>", unsafe_allow_html=True)
        st.session_state.show_baseline = st.checkbox("Show blocked baseline route", value=True)

        if st.button("⚡  Run Routing Algorithms", type="primary", use_container_width=True,
                     disabled=not (engine.G and sc and ec)):
            with st.spinner("Computing routes …"):
                try:
                    engine.set_endpoints(sc[0], sc[1], ec[0], ec[1])
                    engine.compute_routes()
                    st.session_state.routes_computed = True
                    st.rerun()
                except Exception as err:
                    st.error(f"Routing error: {err}")

        if not engine.G:
            st.caption("↑ Load a network first")
        elif not sc or not ec:
            st.caption("↑ Place start & end points on the map")

        if st.session_state.routes_computed and engine.routes:
            st.markdown("<div class='sec-label'>Route Analysis</div>", unsafe_allow_html=True)
            for key, style in ROUTE_STYLES.items():
                route = engine.routes.get(key)
                if route:
                    dist, mins = route_stats(engine.G, route)
                    st.markdown(f"""
                    <div class="route-card" style="background:rgba(255,255,255,0.02);border-left-color:{style['color']}">
                      <div>
                        <div class="route-name" style="color:{style['color']}">{style['label']}</div>
                        <div style="font-size:10px;color:#2A5060;letter-spacing:0.5px;margin-top:2px">{len(route)} waypoints</div>
                      </div>
                      <div class="route-stats">
                        <div>{dist} km</div>
                        <div>{mins} min</div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="route-blocked">
                      <span style="color:#3A5060">{style['label']}</span>
                      <span class="badge-blocked">BLOCKED</span>
                    </div>
                    """, unsafe_allow_html=True)

    # ══════════════ TAB 2: AI ANALYSIS ════════════════════════════
    with tab_analysis:
        st.markdown("<div class='sec-label'>AI Provider</div>", unsafe_allow_html=True)

        provider_labels = {
            "gemini":       "✦ Google Gemini Flash  (Recommended)",
            "huggingface":  "⬡ Hugging Face BLIP",
            "ollama":       "◈ Ollama Local",
        }
        chosen_label = st.selectbox("Provider", list(provider_labels.values()), label_visibility="collapsed")
        provider_key = {v: k for k, v in provider_labels.items()}[chosen_label]
        st.session_state.ai_provider = provider_key

        api_key = hf_token = model_url = ollama_model = None

        if provider_key == "gemini":
            api_key = st.text_input("API Key", type="password", placeholder="AIza…", label_visibility="collapsed")
            st.markdown("""<div class="api-callout">
            <b>Free Gemini Key (2 min):</b><br>
            1. Visit <b>aistudio.google.com/app/apikey</b><br>
            2. Click "Create API key" → paste above<br>
            Free tier: 15 req/min · 1,500/day · No billing
            </div>""", unsafe_allow_html=True)

        elif provider_key == "huggingface":
            hf_token = st.text_input("HF Token", type="password", placeholder="hf_…", label_visibility="collapsed")
            hf_model_name = st.selectbox("Model", list(HF_MODELS.keys()), label_visibility="collapsed")
            model_url = HF_MODELS[hf_model_name]
            st.markdown("""<div class="api-callout">
            <b>Free HF Token:</b> huggingface.co → Settings → Access Tokens<br>
            ⚠ Cold start: first request may take 30–40 s
            </div>""", unsafe_allow_html=True)

        elif provider_key == "ollama":
            ollama_model = st.text_input("Model name", value="llava", label_visibility="collapsed")
            st.markdown("""<div class="api-callout">
            <b>Setup:</b> <code>ollama pull llava</code> then <code>ollama serve</code><br>
            Fully local · No API key · Privacy-first
            </div>""", unsafe_allow_html=True)

        st.markdown("<div class='sec-label'>Upload Image</div>", unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Drag & drop satellite / drone image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        if uploaded:
            st.image(uploaded, use_container_width=True)

        can_analyse = uploaded and (
            (provider_key == "gemini"       and api_key)  or
            (provider_key == "huggingface"  and hf_token) or
            (provider_key == "ollama")
        )

        if st.button("◈  Analyse Image", type="primary", use_container_width=True, disabled=not can_analyse):
            with st.spinner("Processing visual feed …"):
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

        # ── Display Analysis Results ──────────────────────────────
        res = st.session_state.analysis_result
        if res:
            st.markdown("---")
            if not res["success"]:
                st.error(f"**Analysis failed:** {res.get('error', 'Unknown')}")
            else:
                # Provider badge
                st.markdown(f"<div class='provider-tag'>◈ {res['provider']}</div>", unsafe_allow_html=True)

                # ── Severity banner
                sev = res["severity"]
                sev_cfg = {
                    "CRITICAL": ("sev-critical", "🔴 THREAT LEVEL", "CRITICAL — IMMEDIATE ACTION"),
                    "HIGH":     ("sev-high",     "🟠 THREAT LEVEL", "HIGH — URGENT RESPONSE"),
                    "MEDIUM":   ("sev-medium",   "🟡 THREAT LEVEL", "MEDIUM — ELEVATED RISK"),
                    "LOW":      ("sev-low",       "🟢 THREAT LEVEL", "LOW — MONITOR"),
                }
                cls, eyebrow, label = sev_cfg.get(sev, sev_cfg["HIGH"])
                st.markdown(f"""
                <div class="sev-banner {cls}">
                  <div class="sev-sublabel">{eyebrow}</div>
                  <div class="sev-level-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)

                # ── Threat gauge chart
                sev_scores = {"CRITICAL": 95, "HIGH": 70, "MEDIUM": 40, "LOW": 15}
                sev_colors = {"CRITICAL": "#EF4444", "HIGH": "#FB923C", "MEDIUM": "#EAB308", "LOW": "#4ADE80"}
                gauge_val  = sev_scores.get(sev, 70)
                gauge_col  = sev_colors.get(sev, "#FB923C")

                # Hazard breakdown (bar chart data)
                hazards = res["hazards"]
                hazard_display = [h for h in hazards if h != "GENERAL DISASTER"]

                st.markdown(f"""
                <div class="chart-wrapper">
                  <div class="chart-title">Threat Level Gauge</div>
                  <canvas id="gaugeChart" role="img" aria-label="Threat level gauge showing {sev} at {gauge_val}%">Threat level: {sev} ({gauge_val}/100)</canvas>
                </div>
                """, unsafe_allow_html=True)

                # Build hazard bar data
                hazard_labels_js = json.dumps(hazard_display if hazard_display else ["GENERAL"])
                hazard_scores = [85 if h in ["FLOOD","FIRE","EXPLOSION"] else 70 if h in ["HAZMAT","TORNADO"] else 55 for h in (hazard_display if hazard_display else ["GENERAL"])]
                hazard_scores_js = json.dumps(hazard_scores)
                hazard_colors_js = json.dumps(["#FB7185"] * len(hazard_display or ["GENERAL"]))
                n_recs = len(res["recommendations"])

                st.markdown(f"""
                <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
                <script>
                (function() {{
                  const ctx = document.getElementById('gaugeChart');
                  if (!ctx || ctx._chartInstance) return;
                  const chart = new Chart(ctx, {{
                    type: 'doughnut',
                    data: {{
                      datasets: [{{
                        data: [{gauge_val}, {100 - gauge_val}],
                        backgroundColor: ['{gauge_col}', 'rgba(255,255,255,0.05)'],
                        borderWidth: 0,
                        circumference: 270,
                        rotation: 225,
                        borderRadius: 4,
                      }}]
                    }},
                    options: {{
                      responsive: true,
                      maintainAspectRatio: true,
                      cutout: '72%',
                      plugins: {{
                        legend: {{ display: false }},
                        tooltip: {{ enabled: false }},
                      }},
                    }},
                    plugins: [{{
                      id: 'centerText',
                      afterDraw(chart) {{
                        const {{ ctx, chartArea: {{ left, right, top, bottom }} }} = chart;
                        const cx = (left + right) / 2, cy = (top + bottom) / 2 + 10;
                        ctx.save();
                        ctx.font = 'bold 28px JetBrains Mono, monospace';
                        ctx.fillStyle = '{gauge_col}';
                        ctx.textAlign = 'center';
                        ctx.textBaseline = 'middle';
                        ctx.fillText('{gauge_val}', cx, cy - 8);
                        ctx.font = '500 10px Space Grotesk, sans-serif';
                        ctx.fillStyle = 'rgba(255,255,255,0.3)';
                        ctx.letterSpacing = '2px';
                        ctx.fillText('RISK SCORE', cx, cy + 14);
                        ctx.restore();
                      }}
                    }}]
                  }});
                  ctx._chartInstance = chart;
                }})();
                </script>
                """, unsafe_allow_html=True)

                # Hazard breakdown bar
                if hazard_display:
                    st.markdown(f"""
                    <div class="chart-wrapper" style="margin-top:8px">
                      <div class="chart-title">Detected Hazard Types</div>
                      <div style="position:relative;height:{max(80, len(hazard_display)*44)}px">
                        <canvas id="hazardChart" role="img" aria-label="Horizontal bar chart of detected hazard types">
                          Detected hazards: {', '.join(hazard_display)}
                        </canvas>
                      </div>
                    </div>
                    <script>
                    (function() {{
                      const ctx2 = document.getElementById('hazardChart');
                      if (!ctx2 || ctx2._chartInstance) return;
                      const c = new Chart(ctx2, {{
                        type: 'bar',
                        data: {{
                          labels: {hazard_labels_js},
                          datasets: [{{
                            data: {hazard_scores_js},
                            backgroundColor: {hazard_colors_js},
                            borderRadius: 3,
                            borderSkipped: false,
                          }}]
                        }},
                        options: {{
                          indexAxis: 'y',
                          responsive: true,
                          maintainAspectRatio: false,
                          plugins: {{
                            legend: {{ display: false }},
                            tooltip: {{
                              callbacks: {{
                                label: (ctx) => ` Risk score: ${{ctx.raw}}/100`
                              }}
                            }}
                          }},
                          scales: {{
                            x: {{
                              min: 0, max: 100,
                              grid: {{ color: 'rgba(255,255,255,0.04)' }},
                              ticks: {{ color: '#2A5060', font: {{ size: 10 }}, callback: v => v + '%' }},
                            }},
                            y: {{
                              grid: {{ display: false }},
                              ticks: {{ color: '#FB7185', font: {{ size: 11, weight: '700' }}, letterSpacing: '1px' }}
                            }}
                          }}
                        }}
                      }});
                      ctx2._chartInstance = c;
                    }})();
                    </script>
                    """, unsafe_allow_html=True)

                # Recommendation coverage mini-chart
                st.markdown(f"""
                <div class="chart-wrapper" style="margin-top:8px">
                  <div class="chart-title">Response Coverage</div>
                  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;text-align:center">
                    <div style="background:rgba(251,113,133,0.08);border-radius:6px;padding:10px 6px">
                      <div style="font-family:'JetBrains Mono',monospace;font-size:20px;font-weight:700;color:#FB7185">{len(hazards)}</div>
                      <div style="font-size:9px;letter-spacing:1.5px;color:#4A3040;text-transform:uppercase;margin-top:3px">Hazards</div>
                    </div>
                    <div style="background:rgba(240,207,101,0.08);border-radius:6px;padding:10px 6px">
                      <div style="font-family:'JetBrains Mono',monospace;font-size:20px;font-weight:700;color:#F0CF65">{len(res.get('hazards_visible',[]))}</div>
                      <div style="font-size:9px;letter-spacing:1.5px;color:#4A4030;text-transform:uppercase;margin-top:3px">Visible</div>
                    </div>
                    <div style="background:rgba(74,222,128,0.08);border-radius:6px;padding:10px 6px">
                      <div style="font-family:'JetBrains Mono',monospace;font-size:20px;font-weight:700;color:#4ADE80">{n_recs}</div>
                      <div style="font-size:9px;letter-spacing:1.5px;color:#2A4030;text-transform:uppercase;margin-top:3px">Actions</div>
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # Scene description
                st.markdown("<div class='sec-label'>Scene Description</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='scene-desc'>{res['caption']}</div>", unsafe_allow_html=True)

                # Detected hazards chips
                st.markdown("<div class='sec-label'>Detected Hazards</div>", unsafe_allow_html=True)
                chips = " ".join(f"<span class='hazard-chip'>⬡ {h}</span>" for h in res["hazards"])
                st.markdown(chips, unsafe_allow_html=True)

                # Visible hazards (Gemini)
                if res.get("hazards_visible"):
                    st.markdown("<div class='sec-label' style='margin-top:10px'>Visible Dangers</div>", unsafe_allow_html=True)
                    pills = " ".join(f"<span class='vis-hazard'>{hv}</span>" for hv in res["hazards_visible"])
                    st.markdown(pills, unsafe_allow_html=True)

                # Recommendations
                st.markdown("<div class='sec-label' style='margin-top:10px'>Tactical Recommendations</div>", unsafe_allow_html=True)
                for i, rec in enumerate(res["recommendations"], 1):
                    st.markdown(f"""
                    <div class="rec-item">
                      <span class="rec-num">#{i:02d}</span>
                      <span>{rec}</span>
                    </div>
                    """, unsafe_allow_html=True)

                col_a, col_b = st.columns(2)
                if col_a.button("Clear Result", use_container_width=True):
                    st.session_state.analysis_result = None
                    st.rerun()
                if col_b.button("Export Report", use_container_width=True):
                    report = (
                        f"CRISIS VECTOR AI — ANALYSIS REPORT\n"
                        f"{'='*50}\n"
                        f"Provider:     {res['provider']}\n"
                        f"Threat Level: {res['severity']}\n"
                        f"Scene:        {res['caption']}\n"
                        f"Hazards:      {', '.join(res['hazards'])}\n\n"
                        f"TACTICAL RECOMMENDATIONS:\n" +
                        "\n".join(f"  {i:02d}. {r}" for i, r in enumerate(res["recommendations"], 1))
                    )
                    st.code(report, language=None)

    # ══════════════ TAB 3: LEGEND ═════════════════════════════════
    with tab_legend:
        st.markdown("<div class='sec-label'>Route Types</div>", unsafe_allow_html=True)
        for key, style in ROUTE_STYLES.items():
            dash_style = f"border-top: 3px dashed {style['color']};" if style.get("dash") else f"background:{style['color']};"
            st.markdown(f"""
            <div class="legend-route">
              <div class="legend-line" style="{dash_style} height:{'0' if style.get('dash') else '3px'}"></div>
              <div>
                <div class="legend-name" style="color:{style['color']}">{style['label']}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div class='sec-label'>Disaster Zones</div>", unsafe_allow_html=True)
        for dtype, meta in DISASTER_TYPES.items():
            st.markdown(f"""
            <div class="legend-zone">
              <div class="zone-dot" style="background:{meta['color']};box-shadow:0 0 6px {meta['color']}40"></div>
              <span style="font-size:13px">{meta['icon']}</span>
              <span style="font-size:12px;font-weight:600;color:#8090A0">{dtype}</span>
              <span style="font-size:10px;color:#2A5060;margin-left:auto;font-family:'JetBrains Mono',monospace">{meta['radius']}m</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div class='sec-label'>AI Providers</div>", unsafe_allow_html=True)
        providers_info = [
            ("✦", "Gemini Flash", "Best accuracy · Free 1500/day", "#22D3EE"),
            ("⬡", "HF BLIP", "Image captioning · Free token", "#A78BFA"),
            ("◈", "Ollama Local", "Private · No internet · Free", "#4ADE80"),
        ]
        for icon, name, desc, col in providers_info:
            st.markdown(f"""
            <div style="display:flex;gap:10px;align-items:center;margin-bottom:8px;padding:8px 10px;background:rgba(255,255,255,0.02);border-radius:6px">
              <span style="color:{col};font-size:16px">{icon}</span>
              <div>
                <div style="font-size:12px;font-weight:600;color:{col}">{name}</div>
                <div style="font-size:10px;color:#2A5060;margin-top:1px">{desc}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div class='sec-label'>How It Works</div>", unsafe_allow_html=True)
        steps = [
            ("1", "Load Network", "Real OSM road data via OSMnx"),
            ("2", "Place Hazards", "Click map to mark disaster zones"),
            ("3", "Set Endpoints", "Click Start / End on map"),
            ("4", "Compute Routes", "Dijkstra + A* with penalty scoring"),
            ("5", "AI Analysis", "Upload image for tactical assessment"),
        ]
        for num, title, desc in steps:
            st.markdown(f"""
            <div style="display:flex;gap:10px;align-items:flex-start;margin-bottom:8px">
              <div style="width:20px;height:20px;border-radius:50%;background:rgba(34,211,238,0.15);border:1px solid rgba(34,211,238,0.3);display:flex;align-items:center;justify-content:center;font-size:9px;font-weight:700;color:#22D3EE;flex-shrink:0;margin-top:1px">{num}</div>
              <div>
                <div style="font-size:12px;font-weight:600;color:#8090A0">{title}</div>
                <div style="font-size:10px;color:#2A5060;margin-top:1px">{desc}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)


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
        height=660,
        returned_objects=["last_clicked"],
        key="main_map",
    )

    # ── Process map clicks ─────────────────────────────────────
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
                    st.toast(f"{meta['icon']} {meta['label']} placed")
                else:
                    st.toast("⚠️ Load a network before placing zones!", icon="⚠️")
            st.rerun()

    # ── Status bar ─────────────────────────────────────────────
    status_parts = []
    if engine.G:
        status_parts.append(f"<span>◈</span> {engine.location_name}")
    if engine.blocked_node_count():
        status_parts.append(f"<span>⬡</span> {engine.blocked_node_count()} nodes blocked ({engine.coverage_percent()}%)")
    if st.session_state.routes_computed:
        n = sum(1 for v in engine.routes.values() if v)
        status_parts.append(f"<span>⚡</span> {n}/3 routes active")
    if st.session_state.analysis_result and st.session_state.analysis_result.get("success"):
        res = st.session_state.analysis_result
        status_parts.append(f"<span>🛰</span> AI: {res['severity']} · {res['hazards'][0]}")

    status_html = " &nbsp;·&nbsp; ".join(status_parts) if status_parts else "<span>◈</span> Load a network to begin"
    st.markdown(f"<div class='status-bar'>{status_html}</div>", unsafe_allow_html=True)

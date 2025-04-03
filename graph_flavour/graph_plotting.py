import json
import os
import numpy as np
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, ImageClip
from PIL import Image, ImageDraw

def build_graph_from_json(data, G=None, counter=None):
    """
    Recursively builds a graph from JSON. Ensures each node gets a unique ID
    by appending a counter if a video_path is duplicated.
    """
    if G is None:
        G = nx.DiGraph()
    if counter is None:
        counter = {}

    # Normalize the video path (strip whitespace)
    base_id = data["video_path"].strip()
    # Create a unique ID by appending a counter if needed
    if base_id in counter:
        counter[base_id] += 1
        unique_id = f"{base_id}_{counter[base_id]}"
    else:
        counter[base_id] = 0
        unique_id = base_id

    # Add the node with the unique ID and label.
    G.add_node(unique_id, label=data.get("name", ""))
    # Store the unique ID in the data for children reference (optional)
    data["_unique_id"] = unique_id

    for child in data.get("children", []):
        # Build the child and get its unique ID
        child_unique_id, counter = build_graph_from_json(child, G, counter)
        # Add an edge from the current unique node to the child's unique node.
        G.add_edge(unique_id, child_unique_id)

    return unique_id, counter

# Then, when calling it:



def create_node_composite(video_path, label, video_size=(300,300), gap=20, font_size=8):
    """
    Returns (node_clip, cw, ch):
      node_clip = a CompositeVideoClip with the video + text,
      cw, ch = bounding-box width/height of that composite.
    """
    if not os.path.isfile(video_path):
        print(f"Warning: Missing video {video_path}")
        return None, 0, 0
    try:
        vid = VideoFileClip(video_path).resize(video_size)
    except Exception as e:
        print(f"Error loading {video_path}: {e}")
        return None, 0, 0
    
    txt = TextClip(
        label,
        font="Arial",
        fontsize=font_size,
        color="black"
    ).set_duration(vid.duration)
    
    # We'll put text to the right of the video
    cw = video_size[0] + gap + txt.w
    ch = max(video_size[1], txt.h)
    
    text_x = video_size[0] + gap
    text_y = (ch - txt.h)/2  # center text vertically
    node_clip = CompositeVideoClip(
        [
            vid.set_position((0, 0)),
            txt.set_position((text_x, text_y))
        ],
        size=(cw, ch)
    ).set_duration(vid.duration)
    
    return node_clip, cw, ch

def draw_edges_center_to_center(G, centers, size, line_color=(0,0,0,255), line_width=4):
    """
    Creates a transparent Pillow image of 'size' and draws lines from
    center(u) to center(v) for each edge (u->v).
    """
    w, h = size
    img = Image.new("RGBA", (w, h), (0,0,0,0))
    draw = ImageDraw.Draw(img)
    for u,v in G.edges():
        if u in centers and v in centers:
            x1,y1 = centers[u]
            x2,y2 = centers[v]
            draw.line((x1,y1,x2,y2), fill=line_color, width=line_width)
    return img

def scale_and_center_positions(raw_positions, final_width, final_height, margin=200):
    """
    Scales the raw positions from graphviz_layout so that all nodes fit within
    the final_width x final_height canvas with a given margin, then shifts them
    so that the centroid of all positions is at the center of the canvas.
    """
    xs = [p[0] for p in raw_positions.values()]
    ys = [p[1] for p in raw_positions.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    # Compute the available width/height (inside margins)
    avail_w = final_width - 2 * margin
    avail_h = final_height - 2 * margin
    
    range_x = max_x - min_x if max_x != min_x else 1
    range_y = max_y - min_y if max_y != min_y else 1
    
    # Uniform scaling factor
    scale = min(avail_w / range_x, avail_h / range_y)
    
    # Scale positions
    scaled = {node: (margin + (x - min_x) * scale, margin + (y - min_y) * scale)
              for node, (x, y) in raw_positions.items()}
    
    # Compute centroid of scaled positions
    avg_x = sum(pos[0] for pos in scaled.values()) / len(scaled)
    avg_y = sum(pos[1] for pos in scaled.values()) / len(scaled)
    
    # Desired center
    center_x = final_width / 2
    center_y = final_height / 2
    
    # Compute shift needed
    shift_x = center_x - avg_x
    shift_y = center_y - avg_y
    
    # Shift all positions
    centered = {node: (x + shift_x, y + shift_y) for node, (x, y) in scaled.items()}
    return centered

def create_graph_composite(json_data,
                           final_size=(2080,1720),
                           video_size=(50,50),
                           margin=200):
    # Build graph
    G = nx.DiGraph()
    _, _ = build_graph_from_json(json_data, G)  
    for node in G.nodes():
        print(repr(node))
    
    # Add Graphviz attributes for spacing
    G.graph["graph"] = {
        "rankdir": "TB",
        "ranksep": "1000",   # more vertical space
        "nodesep": "1000"  # more horizontal space
    }
    
    # Layout: 'dot' for a hierarchical tree
    raw_positions = graphviz_layout(G, prog="dot")
    
    # Scale positions to final_size
    scaled_centers = scale_and_center_positions(raw_positions, final_size[0], final_size[1], margin=200)
    
    # Build each node's composite, measure bounding box
    node_info = {}
    durations = []
    for node in G.nodes():
        label = G.nodes[node].get("label","")
        clip, cw, ch = create_node_composite(node, label, video_size, gap=20, font_size=24)
        node_info[node] = {"clip": clip, "cw": cw, "ch": ch}
        if clip is not None:
            durations.append(clip.duration)
    
    if not durations:
        print("No valid nodes.")
        return None
    final_duration = max(durations)
    
    # Position each node so its center is at scaled_centers[node]
    placed_clips = []
    for node, data in node_info.items():
        clip = data["clip"]
        cw = data["cw"]
        ch = data["ch"]
        if clip is None or cw==0 or ch==0:
            continue
        cx, cy = scaled_centers[node]
        # top-left corner
        x0 = cx - cw/2
        y0 = cy - ch/2
        placed_clips.append(clip.set_position((x0, y0)))
    
    # Draw edges center-to-center
    edges_img = draw_edges_center_to_center(G, scaled_centers, final_size, line_color=(0,0,0,255), line_width=4)
    edges_clip = ImageClip(np.array(edges_img)).set_duration(final_duration)
    
    # Composite: edges behind, nodes on top
    final_clip = CompositeVideoClip(
        [edges_clip] + placed_clips,
        size=final_size,
        bg_color=(255,255,255)  # white background
    ).set_duration(final_duration)
    
    return final_clip

def main():
    # Load JSON
    json_file = "branches.json"
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Tweak as needed:
    # - final_size = bigger if you have many nodes
    # - video_size = bigger or smaller nodes
    # - margin = how much space around the edges
    final = create_graph_composite(
        data,
        final_size=(1920,1080),
        video_size=(200,200),
        margin=100
    )
    

    if final is None:
        print("No final composite produced.")
        return
    
    outfile = "graph_composite.mp4"
    print(f"Rendering {outfile} ...")
    final.write_videofile(outfile, fps=24, codec="libx264")
    print("Done!")

if __name__ == "__main__":
    main()

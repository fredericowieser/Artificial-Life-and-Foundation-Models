import json
import os
import numpy as np
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, ImageClip
from PIL import Image, ImageDraw
from networkx.drawing.nx_pydot import to_pydot


def build_graph_from_json(data, G=None, counter=None):
    """
    Recursively builds a directed graph from the JSON.
    Uses the video_path if available as the node's base ID,
    otherwise falls back to using the name.
    Appends an index if the same base ID appears more than once.
    """
    if G is None:
        G = nx.DiGraph()
    if counter is None:
        counter = {}

    # Use video_path if available; if not, use the name.
    raw_id = data.get("video_path") or data.get("name", "unknown")
    # Normalize the raw_id: remove extra whitespace, replace spaces with underscores, remove quotes.
    base_id = str(raw_id).replace('“','').replace('”','').strip().replace(" ", "_")
    
    # Check the counter dictionary to ensure uniqueness.
    if base_id in counter:
        counter[base_id] += 1
        unique_id = f"{base_id}_{counter[base_id]}"
    else:
        counter[base_id] = 0
        unique_id = base_id

    # Add the node with the unique ID, and store the original label as an attribute.
    G.add_node(unique_id, label=data.get("name", ""))
    data["_unique_id"] = unique_id  # optionally store it in the JSON data

    for child in data.get("children", []):
        child_unique_id, counter = build_graph_from_json(child, G, counter)
        G.add_edge(unique_id, child_unique_id)

    return unique_id, counter

# Example usage:


import textwrap

def create_node_composite(video_path, label, video_size=(300,300), font_size=24, max_chars_per_line=20):
    """
    Loads a video from 'video_path', resizes it to video_size, and creates a composite clip
    that displays the video on top and a text label below. If the label is too long,
    it is wrapped onto multiple lines.
    
    Returns a tuple: (node_clip, composite_width, composite_height)
    """
    if not os.path.isfile(video_path):
        print(f"Warning: Missing video {video_path}")
        return None, 0, 0
    try:
        vid_clip = VideoFileClip(video_path).resize(newsize=video_size)
    except Exception as e:
        print(f"Error loading {video_path}: {e}")
        return None, 0, 0

    # Wrap the label text if too long:
    wrapped_label = "\n".join(textwrap.wrap(label, width=max_chars_per_line))
    
    txt_clip = TextClip(
        wrapped_label,
        font="Arial",       # Change if necessary to a font available on your system
        fontsize=font_size,
        color="black"
    ).set_duration(vid_clip.duration)
    
    # Composite size: video on top, text below
    composite_width = video_size[0]
    composite_height = video_size[1] + txt_clip.h
    video_pos = (0, 0)
    text_pos = ((composite_width - txt_clip.w) / 2, video_size[1])
    
    node_comp = CompositeVideoClip(
        [vid_clip.set_position(video_pos),
         txt_clip.set_position(text_pos)],
        size=(composite_width, composite_height)
    ).set_duration(vid_clip.duration)
    
    return node_comp, composite_width, composite_height


def scale_and_center_positions(raw_positions, final_width, final_height, margin=200):
    """
    Scales raw graphviz positions so that they fit within the final canvas with given margin.
    Then, shifts positions so that their centroid is at the center of the canvas.
    These positions represent node centers.
    """
    xs = [p[0] for p in raw_positions.values()]
    ys = [p[1] for p in raw_positions.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    range_x = max_x - min_x if max_x != min_x else 1
    range_y = max_y - min_y if max_y != min_y else 1

    avail_w = final_width - 2 * margin
    avail_h = final_height - 2 * margin
    scale = min(avail_w / range_x, avail_h / range_y)

    scaled = {node: (margin + (x - min_x) * scale,
                     margin + (y - min_y) * scale)
              for node, (x, y) in raw_positions.items()}

    # Compute centroid
    avg_x = sum(x for x, _ in scaled.values()) / len(scaled)
    avg_y = sum(y for _, y in scaled.values()) / len(scaled)
    shift_x = final_width / 2 - avg_x
    shift_y = final_height / 2 - avg_y
    centered = {node: (x + shift_x, y + shift_y) for node, (x, y) in scaled.items()}
    return centered

def draw_edges_center_to_center(G, centers, size, line_color=(0,0,0,255), line_width=4):
    """
    Draws lines on a transparent Pillow RGBA image between the centers of connected nodes.
    """
    w, h = size
    img = Image.new("RGBA", (w, h), (0,0,0,0))
    draw = ImageDraw.Draw(img)
    for u, v in G.edges():
        if u in centers and v in centers:
            x1, y1 = centers[u]
            x2, y2 = centers[v]
            draw.line((x1, y1, x2, y2), fill=line_color, width=line_width)
    return img

def create_graph_composite(json_data, final_size=(2280,1720), video_size=(400,400), margin=200):
    """
    1. Builds graph from JSON.
    2. Computes a graphviz layout and scales & centers positions.
    3. Creates a composite clip for each node (video + bottom text).
    4. Positions each node so that its center aligns with the layout.
    5. Draws edges (lines) from center to center.
    6. Composites everything on a white background.
    """
    G = nx.DiGraph()
    _, _ = build_graph_from_json(json_data, G)
    # for node in G.nodes(data=True):
    #     label = node[1].get("label", "")
    #     if "\n" in label or '"' in label:
    #         print("Found suspicious label with newline/quote:", label)
    # for u, v in G.edges():
    #     if u not in G.nodes():
    #         print(f"Edge references missing node: {u}")
    #     if v not in G.nodes():
    #         print(f"Edge references missing node: {v}")
    # print("All node IDs:")
    # print("Nodes in G:")
    # for node in G.nodes():
    #     print(repr(node))

    # try:
    #     p = to_pydot(G)
    #     p.write_raw("debug.dot")
    #     print("Wrote debug.dot successfully.")
    # except Exception as e:
    #     print("Error writing DOT file:", e)
    # for node in G.nodes():
    #     G.nodes[node]['label'] = node
    # raw_positions = graphviz_layout(G, prog="neato")



    




    G.graph["graph"] = {"rankdir": "TB", "ranksep": "10", "nodesep": "1.5"}
    raw_positions = graphviz_layout(G, prog="dot")
    centered_positions = scale_and_center_positions(raw_positions, final_size[0], final_size[1], margin=margin)

    node_info = {}
    durations = []
    for node in G.nodes():
        # Use the cleaned node ID.
        clip, cw, ch = create_node_composite(node, G.nodes[node].get("label", ""), video_size, font_size=30)
        if clip is not None:
            node_info[node] = {"clip": clip, "cw": cw, "ch": ch}
            durations.append(clip.duration)
    if not durations:
        print("No valid nodes loaded.")
        return None
    final_duration = max(durations)

    # Position each node so that its composite center is at the computed center.
    placed_clips = []
    for node, info in node_info.items():
        clip = info["clip"]
        cw, ch = info["cw"], info["ch"]
        cx, cy = centered_positions[node]
        # Place the top-left so that the center is at (cx,cy)
        top_left = (cx - cw / 2, cy - ch / 2)
        placed_clips.append(clip.set_position(top_left))

    # Draw edges using the computed centers.
    edges_img = draw_edges_center_to_center(G, centered_positions, final_size, line_color=(0,0,0,255), line_width=4)
    edges_clip = ImageClip(np.array(edges_img)).set_duration(final_duration)

    # Composite final video: edges behind, nodes on top.
    final_composite = CompositeVideoClip(
        [edges_clip] + placed_clips,
        size=final_size,
        bg_color=(255,255,255)
    ).set_duration(final_duration)

    return final_composite

def main():
    json_file = "branches.json"  # Update with your JSON file name/path.
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    final = create_graph_composite(data, final_size=(2780,2720), video_size=(250,250), margin=150)
    if final is None:
        print("Failed to create composite.")
        return
    out_filename = "graph_composite.mp4"
    print(f"Rendering to {out_filename} ...")
    final.write_videofile(out_filename, fps=24, codec="libx264")
    print("Done!")

if __name__ == "__main__":
    main()

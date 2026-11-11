from tqdm.notebook import tqdm
import graph_tool.all as gt 
from graph_tool.all import load_graph
import polars as pl


def extract_subgraph(
        g, 
        seed_nodes, 
        include_neighbors=True, 
        name_prop="name",
        weight_prop="weight",              # EDIT: added weight property name
        min_edge_weight=None,              # EDIT: added threshold argument
        ):
    
    name_prop = g.vp[name_prop]
    name_to_vid = {name_prop[v]: int(v) for v in g.vertices()}
    name_to_vid = {k.lower():v for k,v in name_to_vid.items()}

    seed_vids = []
    missing = []

    for nm in seed_nodes:
        vid = name_to_vid.get(nm.lower())
        if vid is None:
            missing.append(nm)
        else:
            seed_vids.append(vid)

    keep = set(seed_vids)
    weights = g.ep[weight_prop] if min_edge_weight is not None else None

    if include_neighbors:
        for vid in tqdm(seed_vids, desc="Extracting neighbors"):
            v = g.vertex(vid)

            for e in v.all_edges():
                if min_edge_weight is None or weights[e] >= min_edge_weight:

                    src = int(e.source())
                    tgt = int(e.target())
                    other = tgt if src == vid else src

                    keep.add(other)

    vfilt = g.new_vertex_property("bool", False)
    for vid in keep:
        vfilt[g.vertex(vid)] = True

    print("setting vertex filter...")
    gv = gt.GraphView(g, vfilt=vfilt)
    print("pruning graph...")
    g_sub = gt.Graph(gv, prune=True)

    print("Seeds found:", len(seed_vids), "Missing:", len(missing))
    print("Expanded subgraph:", g_sub.num_vertices(), "vertices;", g_sub.num_edges(), "edges")
    if missing:
        print("Missing seed names (not found):", missing[:20], ("..." if len(missing) > 20 else ""))

    return g_sub


def name_to_degree(g):
    name_prop = g.vp["name"]
    verts = g.get_vertices()
    degs = g.get_out_degrees(verts)          

    return {
        name_prop[g.vertex(v)].lower(): int(d)
        for v, d in zip(verts, degs)
    }

def name_to_strength(g):
    name_prop = g.vp["name"]
    strength  = g.degree_property_map("total", weight=g.ep["weight"])

    return {
        name_prop[v].lower(): float(strength[v])
        for v in g.vertices()
    }

def degree_and_strength_by_name(g, nodes, name_prop="name", weight_prop="weight", show_progress=True):
    # a relatively slower solution
    name = g.vp[name_prop]
    w = g.ep[weight_prop]

    # map name -> vertex index (unique names assumed)
    name_to_vid = {name[v]: int(v) for v in g.vertices()}
    name_to_vid = {k.lower():v for k,v in name_to_vid.items()}

    deg = {}
    strength = {}
    missing = []

    iterator = nodes
    if show_progress:
        iterator = tqdm(nodes, desc="Computing degree & strength")

    for n in iterator:
        vid = name_to_vid.get(n)
        if vid is None:
            missing.append(n)
            continue

        v = g.vertex(vid)
        deg[n] = v.out_degree()  
        s = 0
        for e in v.all_edges():   # incident edges
            s += w[e]
        strength[n] = s

    return deg, strength, missing


def _infer_gt_type(dtype) -> str:
    """
    Infer a graph-tool property-map type from a Polars dtype.
    """
    if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                 pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
        return "long"
    elif dtype in (pl.Float32, pl.Float64):
        return "double"
    elif dtype == pl.Boolean:
        return "bool"
    else:
        return "string"


def read_network_gt(
    node_df: pl.DataFrame,
    edge_df: pl.DataFrame,
    directed: bool = False,
    add_node_attrs: bool = True,
    add_edge_attrs: bool = True,
    n_sample_nodes: int = -1,
    print_info: bool = True,
    min_edge_weight: float = -1,
    edge_weight_col: str = "weight",
) -> gt.Graph:
    """
    Build a graph-tool graph from Polars node and edge dataframes.

    Args:
        node_df (pl.DataFrame): Must include an "Id" column.
        edge_df (pl.DataFrame): Must include "Source" and "Target" columns.
        directed (bool): Whether to build a directed graph.
        add_node_attrs (bool): Whether to add node attributes beyond "Id".
        add_edge_attrs (bool): Whether to add edge attributes beyond "Source" and "Target".
        n_sample_nodes (int): If > 0, randomly sample this many nodes and keep induced edges.
        print_info (bool): Whether to print graph size.
        min_edge_weight (float | None): If provided, keep only edges with weight >= this value.  # EDITED
        edge_weight_col (str): Column used for edge-weight filtering.   

    Returns:
        graph_tool.Graph: graph-tool graph object.
    """

    required_node_cols = {"Id"}
    required_edge_cols = {"Source", "Target"}

    missing_node_cols = required_node_cols - set(node_df.columns)
    missing_edge_cols = required_edge_cols - set(edge_df.columns)

    if missing_node_cols:
        raise ValueError(f"node_df missing required columns: {missing_node_cols}")
    if missing_edge_cols:
        raise ValueError(f"edge_df missing required columns: {missing_edge_cols}")
    
    # Optional edge-weight filter
    if min_edge_weight > 0:
        if edge_weight_col not in edge_df.columns:
            raise ValueError(
                f"edge_weight_col='{edge_weight_col}' not found in edge_df. "
                f"Available columns: {edge_df.columns}"
            )

        edge_df = edge_df.filter(
            pl.col(edge_weight_col) >= min_edge_weight
        )

        if print_info:
            print(
                f"Keeping edges with {edge_weight_col} >= {min_edge_weight}: "
                f"{edge_df.height} edges remaining"
            )

    # Optional induced-node sample
    if n_sample_nodes > 0:
        node_df = node_df.sample(n=n_sample_nodes, with_replacement=False, seed=42)

        sampled_nodes = node_df.select("Id")
        edge_df = edge_df.join(
            sampled_nodes.rename({"Id": "Source"}),
            on="Source",
            how="inner",
        ).join(
            sampled_nodes.rename({"Id": "Target"}),
            on="Target",
            how="inner",
        )

    node_attr_cols = []
    edge_attr_cols = []

    if add_node_attrs:
        node_attr_cols = [c for c in node_df.columns if c != "Id"]
        if len(node_attr_cols) == 0:
            print("Node attributes not found!")

    if add_edge_attrs:
        edge_attr_cols = [c for c in edge_df.columns if c not in {"Source", "Target"}]
        if len(edge_attr_cols) == 0:
            print("Edge attributes not found!")

    # ------------------------------------------------------------------
    # Build graph
    # ------------------------------------------------------------------
    g = gt.Graph(directed=directed)

    # graph-tool uses integer vertex indices internally.
    # We keep original node IDs in a vertex property map.
    id_dtype = node_df.schema["Id"]
    id_prop_type = _infer_gt_type(id_dtype)

    v_id = g.new_vertex_property(id_prop_type)
    g.vp["Id"] = v_id

    # Create vertex property maps for node attributes
    for col in node_attr_cols:
        prop_type = _infer_gt_type(node_df.schema[col])
        g.vp[col] = g.new_vertex_property(prop_type)

    # Create edge property maps
    for col in edge_attr_cols:
        prop_type = _infer_gt_type(edge_df.schema[col])
        g.ep[col] = g.new_edge_property(prop_type)

    # ------------------------------------------------------------------
    # Add nodes
    # ------------------------------------------------------------------
    node_to_vertex = {}

    node_rows = node_df.iter_rows(named=True)

    for row in tqdm(node_rows, total=node_df.height, desc="Adding nodes..."):
        node_id = row["Id"]

        v = g.add_vertex()
        node_to_vertex[node_id] = v

        g.vp["Id"][v] = node_id

        for col in node_attr_cols:
            val = row[col]
            if val is not None:
                g.vp[col][v] = val

    # ------------------------------------------------------------------
    # Add edges
    # ------------------------------------------------------------------
    edge_rows = edge_df.iter_rows(named=True)

    n_edges_added = 0

    for row in tqdm(edge_rows, total=edge_df.height, desc="Adding edges..."):
        src = row["Source"]
        tgt = row["Target"]

        if src not in node_to_vertex or tgt not in node_to_vertex:
            continue

        e = g.add_edge(node_to_vertex[src], node_to_vertex[tgt])
        n_edges_added += 1

        for col in edge_attr_cols:
            val = row[col]
            if val is not None:
                g.ep[col][e] = val

    if print_info:
        print("# of nodes:", g.num_vertices())
        print("# of edges:", g.num_edges())

    # ------------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------------
    assert node_df.height == g.num_vertices(), (
        f"Node exporting error; have {node_df.height} rows in the input "
        f"but have {g.num_vertices()} nodes in the network."
    )

    assert n_edges_added == g.num_edges(), (
        f"Edge exporting error; added {n_edges_added} edges "
        f"but graph has {g.num_edges()} edges."
    )

    return g


def remove_user_page_nodes(g, name_prop="name"):
    """
    Return an induced GraphView excluding vertices whose normalized
    name begins with 'u_'.

    Vertex IDs remain those of the original graph.
    """
    names = g.vp[name_prop]
    keep_vertex = g.new_vertex_property("bool")

    for v in g.vertices():
        normalized_name = str(names[v]).lower().strip()
        keep_vertex[v] = not normalized_name.startswith("u_")

    g_filtered = gt.GraphView(g, vfilt=keep_vertex)

    print(
        f"Removed {g.num_vertices() - g_filtered.num_vertices():,} u_ nodes "
        f"and {g.num_edges() - g_filtered.num_edges():,} associated edges."
    )

    return g_filtered
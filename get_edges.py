import os
import sys

# Ensure SUMO_HOME is set
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import sumolib

def extract_network_edges(net_file="cleanedindorenetwork.net.xml"):
    print(f"Loading network file: {net_file}...")
    net = sumolib.net.readNet(net_file)
    
    print("\n--- Valid Incoming Edges (Sample of first 20) ---")
    edge_list = []
    for edge in net.getEdges():
        edge_id = edge.getID()
        # Internal edges in SUMO always start with a colon ':'
        if not edge_id.startswith(':'):
            edge_list.append(edge_id)
            print(edge_id)
            
    print(f"\nTotal valid edges found: {len(edge_list)}")
    return edge_list

if __name__ == "__main__":
    extract_network_edges()
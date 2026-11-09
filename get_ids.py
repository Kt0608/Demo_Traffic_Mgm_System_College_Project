import sumolib

# Load your network file
net = sumolib.net.readNet('cleanedindorenetwork.net.xml')

print("--- TRAFFIC LIGHT IDS ---")
tls_list = net.getTrafficLights()
tls_ids = [tls.getID() for tls in tls_list]
print("tls_ids =", tls_ids)

print("\n--- INCOMING LANES PER TRAFFIC LIGHT ---")
all_incoming_lanes = []
for tls in tls_list:
    lanes = []
    for connection in tls.getConnections():
        in_lane = connection[0].getID()
        if in_lane not in lanes:
            lanes.append(in_lane)
    print(f"{tls.getID()} incoming lanes: {lanes}")
    all_incoming_lanes.extend(lanes)

# Remove duplicates while preserving order
all_incoming_lanes = list(dict.fromkeys(all_incoming_lanes))
print("\n--- ALL MONITORED LANES FOR STATE SPACE ---")
print("monitored_lanes =", all_incoming_lanes)
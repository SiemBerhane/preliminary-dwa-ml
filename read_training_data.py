import json
import os

def get_tension_data(processed_data_path, layers):
    """
    Extract tension data from the processed JSON file for specified layers and store it in a flat list of records.

    Args:
        processed_data_path (str): Path to the processed JSON file.
        layers (list): List of layer names to extract tension data from.
    
    Returns:
        list: A list of dictionaries containing tension data for each wire in the specified layers. 
        Each element contains the following keys:
  
            - Stage: The stage of the process (e.g., "Winding").
            - Layer: The layer name (e.g., "U" or "X").
            - Side: The side of the layer (e.g., "A" or "B").
            - Wire_ID: The unique identifier for the wire.
            - Name: The name associated with the tension data.
            - Tension: The tension value extracted from the processed data.
    """
    
    full_data = json.load(open(processed_data_path))
    winding_data = full_data["Winding"]
    flat_records = []
    
    for layer in layers:
        layer_data = winding_data[layer]

        for side, side_data in layer_data.items():  
            for segment_id, wire_data in side_data.items(): # gets first two indices of dict
                if len(wire_data['tension']) == 0:
                    continue

                try:
                    parent_folder = list(side_data[segment_id]['tension'].keys())[0] # long string: e.g 20230208T185357_U_A_<segment>
                    tension = side_data[segment_id]['tension'][parent_folder]['Auto']['tension'] 

                    if layer == "U" and side == "B" and int(segment_id) == 18:
                        print(f"Debug Info - Layer: {layer}, Side: {side}, Segment_ID: {segment_id}, parent_folder: {parent_folder}, Tension: {tension}")
                    
                    flat_records.append({
                                    "uid": str(layer)+str(side)+str(int(segment_id)), # int(segment_id) to remove leading zeros
                                    "stage": "Winding",
                                    # "layer": layer, # can loop through layer and side in final version to make it ful automatic
                                    # "side": side,
                                    # "segment_id": int(segment_id),
                                    # "parent_folder": parent_folder,
                                    "tension": tension 
                                    # "Frequency": freq - create function to do so
                                })
                except IndexError as e:
                    print(f"IndexError: No tension data found for Segment_ID {segment_id} in Layer {layer}, Side {side}.")

    return flat_records

def get_raw_data(path, keys_to_remove):
    """
    Returns a list of dictionaries containing the cleaned scan data from all subdirectories of the given path.
    
    Args:
        path (str): The root directory containing all the subdirectories with scan data.
        keys_to_remove (list): A list of keys to remove from each scan dictionary.
    
    Returns:
        list: A list of dictionaries containing the cleaned scan data.
        """
    
    list_of_dir = []
    [list_of_dir.append(x[0]) for x in os.walk(path)] # first element is current path, rest are subdirectories

    list_of_scans = []
    for dir in list_of_dir:
        if "amplitudeData.json" in os.listdir(dir):
            scan = json.load(open(dir + "/amplitudeData.json"))
            clean_scan = {k: v for k, v in scan.items() if k not in keys_to_remove}
            final_scan = map_wire_segments_to_channels(clean_scan, dir)
            list_of_scans.extend(final_scan)

    return list_of_scans


def map_wire_segments_to_channels(scan, folder_path):
    """
    Maps wire segments to their corresponding channel indices based on the APA channels in the scan data.
    
    Args:
        scan (dict): A dictionary containing the cleaned scan data. 
    
    Returns:
        list: A list of dictionaries containing the wire ID, channel index, frequency and amplitude data
        """

    apa_channels = scan['apaChannels']
    wire_segments = scan['wireSegments']

    channel_map = { # creates map of channels corresponding to freq/ampl indexing e.g. 0 -> channel 0
        (val%100): idx for idx, val in enumerate(apa_channels) if val is not None
    }

    extracted_scans = []

    for segment in wire_segments:
        tar_digits = segment % 100
        # print(f"Wire: {wire}, Target Digits: {tar_digits}, Channel: {channel_map[tar_digits]}")
        if tar_digits in channel_map:
            channel_idx = channel_map[tar_digits]
            channel_data = scan.get(str(channel_idx), {})

            
            
            # Only append if there is actual frequency data in that channel
            if channel_data.get("freq") and channel_data.get("ampl"):
                extracted_scans.append({
                    "uid": str(scan["layer"])+str(scan["side"])+str(int(segment)),
                    "segment_id": int(segment),
                    "channel_index": channel_idx,
                    "freq": channel_data["freq"],
                    "ampl": channel_data["ampl"],
                    "layer": scan["layer"],
                    "headboardNumber": scan["headboardNum"],
                    "side": scan["side"],
                    "parent_folder": folder_path
                })

    return extracted_scans

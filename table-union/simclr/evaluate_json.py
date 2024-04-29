import json

# read the json file
with open("top_50_results.json", "r") as file:
    json_data = json.load(file)

def calculate_precision_at_k(data, k):
    total_matches = 0
    total_possible = 0
    
    for entry in data:
        ground_truth = entry["Ground truth column"]
        top_k_columns = entry["Top k columns"]
        
        # Increment total possible count
        total_possible += 1
        
        # Check if ground truth is within the top k elements and count matches
        if ground_truth in top_k_columns[:k]:
            total_matches += 1
    
    # Calculate global precision for this k
    if total_possible > 0:
        return total_matches / total_possible
    else:
        return 0

# Calculate precision at 10, 20, 50
precision_at_10 = calculate_precision_at_k(json_data, 10)
precision_at_20 = calculate_precision_at_k(json_data, 20)
precision_at_50 = calculate_precision_at_k(json_data, 50)

# Print the results
print("Precision at 10:", precision_at_10)
print("Precision at 20:", precision_at_20)
print("Precision at 50:", precision_at_50)
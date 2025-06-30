import matplotlib.pyplot as plt
import json
import tqdm

with open("../../data/desertification/desertification_stats_precomputed.json", 'r') as f:
    data = json.load(f)

plt.figure(figsize=(5, 5))

for A_data, results_data in tqdm.tqdm(data.items()):
    A_data = float(A_data)
    avg_bio, max_bio, max_wat = results_data[0]
    if A_data != float(list(data.keys())[-1]):
        plt.scatter(A_data, avg_bio, marker='o', color='g')
        plt.scatter(A_data, max_bio, marker='o', color='y')
        plt.scatter(A_data, 0.45/A_data, marker='x', color='r')
    else:
        plt.scatter(A_data, avg_bio, marker='o', color='g', label='AVG BIOMASS')
        plt.scatter(A_data, max_bio, marker='o', color='y', label='MAX BIOMASS')
        plt.scatter(A_data, 0.45/A_data, marker='x', color='r', label='CRITICAL LEVEL')

plt.xlabel('WATER SUPPLY')
plt.ylabel('BIOMASS')
plt.title('Desertification in the regime of shrinking water resources')
plt.gca().invert_xaxis()
plt.grid(True)
plt.legend()
plt.show()

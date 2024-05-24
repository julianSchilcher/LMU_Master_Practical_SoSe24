import numpy as np
from collections import Counter

def dengrom_purity(clusters, labels):
    total_elements = sum([len(cluster) for cluster in clusters])
    purity_sum = 0
    
    for cluster in clusters:
        if len(cluster)==0:
            continue
        label_counts = Counter([labels[i] for i in cluster])
        max_count = max(label_counts.values())
        purity = max_count/len(cluster)
        purity_sum+=purity*len(cluster)
    
    return purity_sum/total_elements

def leaf_purity():
    return None
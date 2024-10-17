import heapq

class ThroughputPQ:
    def __init__(self):
        self._heap = []
    
    def add(self, element):
        # element is expected to be in the form ((cpu_freq, gpu_freq), power)
        heapq.heappush(self._heap, (element[1], element[0]))  # Heap is ordered by power, and element[0] is the tuple (cpu_freq, gpu_freq)
    
    def top(self):
        # Return the element with the lowest power without removing it
        if not self.is_empty():
            power, freqs = self._heap[0]
            return (freqs, power)
        return None
    
    def is_empty(self):
        # Check if the priority queue is empty
        return len(self._heap) == 0
    
    def print_elements(self):
        # Print the elements in the priority queue
        for power, freqs in sorted(self._heap):
            print(f"CPU Frequency: {freqs[0]}, GPU Frequency: {freqs[1]}, Power: {power}")
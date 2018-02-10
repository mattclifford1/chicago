import gmaps
import gmaps.datasets
gmaps.configure(api_key="AIzaSyCmrMmc8SKB8SBpGs42gxUfOy14s2UbGe8") # Your Google API key

# load a Numpy array of (latitude, longitude) pairs
locations = gmaps.datasets.load_dataset("taxi_rides")
fig = gmaps.figure()
fig.add_layer(gmaps.heatmap_layer(locations))
fig
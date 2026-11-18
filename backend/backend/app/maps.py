"""Folium inference map. Reference route files are never loaded here."""
import folium
import numpy as np
from .data import TripData


def render_route_map(trip: TripData, prediction: dict, reference=None) -> str:
    track = trip.track[['latitude', 'longitude']].to_numpy()
    view = folium.Map(location=track.mean(axis=0).tolist(), tiles='OpenStreetMap', prefer_canvas=True)
    # GeoJSON renders many segments as one layer and never joins disconnected roads.
    edges = trip.nodes_lonlat[trip.arcs]
    west, south = trip.track.longitude.min() - .002, trip.track.latitude.min() - .002
    east, north = trip.track.longitude.max() + .002, trip.track.latitude.max() + .002
    visible = ((edges[:, :, 0].max(axis=1) >= west) & (edges[:, :, 0].min(axis=1) <= east)
               & (edges[:, :, 1].max(axis=1) >= south) & (edges[:, :, 1].min(axis=1) <= north))
    def layer(name, coordinates, color, weight, show=True):
        group = folium.FeatureGroup(name=name, show=show)
        if coordinates:
            folium.GeoJson({'type': 'Feature', 'properties': {}, 'geometry': {
                'type': 'MultiLineString', 'coordinates': coordinates}},
                style_function=lambda _: {'color': color, 'weight': weight, 'opacity': .8}).add_to(group)
        group.add_to(view)
    layer('Road network', edges[visible].tolist(), '#9aa5b5', 2)
    layer('GPS observations', [trip.track[['longitude', 'latitude']].values.tolist()], '#e99b49', 2)
    classifier = list(dict.fromkeys(x for x in prediction.get('classifier_route_arc_ids', prediction['classifier_only_arc_ids']) if x is not None))
    matched = list(dict.fromkeys(x for x in prediction.get('predicted_route_arc_ids', prediction['predicted_arc_ids']) if x is not None))
    layer('Classifier only', edges[classifier].tolist(), '#b36ddd', 3, False)
    layer('HMM matched roads', edges[matched].tolist(), '#4768e9', 5)
    if reference is not None:
        layer('Reference route for development report', edges[reference].tolist(), '#13954a', 3)
    for index, label, color in [(0, 'Start', 'green'), (-1, 'End', 'red')]:
        folium.Marker(track[index].tolist(), tooltip=label, icon=folium.Icon(color=color)).add_to(view)
    view.fit_bounds([track.min(axis=0).tolist(), track.max(axis=0).tolist()])
    folium.LayerControl(collapsed=False).add_to(view)
    return view.get_root().render()

from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from backend.app.data import TripData
from backend.app.network import RoadNetwork
from backend.app.hmm import HMMViterbiDecoder
from backend.app.services import _route_continuity, expand_route


def tiny_network(tmp_path, connected=True):
    track=pd.DataFrame({'longitude':[.00002,.00028], 'latitude':[0.,0.], 'timestamp':[1.,2.]})
    nodes=np.array([[0.,0.],[.0001,0.],[.0002,0.],[.0003,0.]])
    arcs=np.array([[0,1],[1,2],[2,3]]) if connected else np.array([[0,1],[2,3]])
    trip=TripData('00000001','student_test',tmp_path,track,nodes,arcs)
    return track,RoadNetwork(trip)


def test_route_expansion_and_larger_search_cutoff(tmp_path):
    track,network=tiny_network(tmp_path)
    assert network.shortest_distance(0,3,5.) is None
    assert network.shortest_distance(0,3,100.) is not None
    assert network.shortest_path_arcs(0,3,100.)==[0,1,2]
    assert expand_route([0,2],network,track,HMMViterbiDecoder())==[0,1,2]


def test_missing_and_disconnected_roads_remain_breaks(tmp_path):
    track,network=tiny_network(tmp_path,connected=False)
    assert expand_route([0,1],network,track,HMMViterbiDecoder())==[0,None,1]
    assert _route_continuity([0,None,1],network,None,None)==0.
    assert _route_continuity([None,None],network,None,None)==0.


def test_partition_preparation_is_reference_free(tmp_path):
    from scripts.prepare_partition import prepare
    folder=tmp_path/'raw/evaluator_test/00000001';folder.mkdir(parents=True)
    (folder/'00000001.track').write_text('.00002 0 1\n.00008 0 2\n')
    (folder/'00000001.nodes').write_text('0 0\n.0001 0\n')
    (folder/'00000001.arcs').write_text('0 1\n')
    (folder/'00000001.route').write_text('invalid reference must not be read')
    output=tmp_path/'prepared'
    prepare(tmp_path/'raw','evaluator_test',output)
    candidates=pd.read_parquet(output/'candidates.parquet')
    assert len(candidates)>0
    assert 'reference_arc_id' not in candidates
    assert not (output/'labels.parquet').exists()


def test_tuning_rejects_evaluator_partition(monkeypatch):
    root=Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root/'backend/scripts'))
    from experiment_utils import settings_from_args
    with pytest.raises(ValueError,match='never evaluator'):
        settings_from_args(None,'evaluator_test')

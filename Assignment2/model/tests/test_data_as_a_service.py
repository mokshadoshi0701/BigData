from fastapi.testclient import TestClient 

from model.sevir_model.data_as_a_service import app

client = TestClient(app)

def test_index():
    
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}

def test_event_id():
    response = client.get("/event-id/835047")
    assert response.status_code == 200
    assert response.json() == {
  "5 related files found": [
    "vil/2019/SEVIR_VIL_STORMEVENTS_2019_0101_0630.h5",
    "ir107/2019/SEVIR_IR107_STORMEVENTS_2019_0101_0630.h5",
    "ir069/2019/SEVIR_IR069_STORMEVENTS_2019_0101_0630.h5",
    "vis/2019/SEVIR_VIS_STORMEVENTS_2019_0601_0630.h5",
    "lght/2019/SEVIR_LGHT_ALLEVENTS_2019_0601_0701.h5"
  ]
}

def test_modality():
    response = client.get("/ir069")
    assert response.status_code == 200
    assert response.json() == ["13552 related files found"]

def test_unique_values():
    response = client.get("/feature/event_type")
    assert response.status_code == 200
    assert response.json() == {
  "unique items": [
    "",
    "Tornado",
    "Thunderstorm Wind",
    "Hail",
    "Funnel Cloud",
    "Flash Flood",
    "Heavy Rain",
    "Flood",
    "Lightning"
  ]
}
